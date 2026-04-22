import path from "path"
import os from "os"
import z from "zod"
import { SessionID, MessageID, PartID } from "./schema"
import { MessageV2 } from "./message-v2"
import { Log } from "../util"
import { SessionRevert } from "./revert"
import * as Session from "./session"
import { Agent } from "../agent/agent"
import { Provider } from "../provider"
import { ModelID, ProviderID } from "../provider/schema"
import { type Tool as AITool, tool, jsonSchema, type ToolExecutionOptions, asSchema } from "ai"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { SessionCompaction } from "./compaction"
import { Bus } from "../bus"
import { ProviderTransform } from "../provider"
import { SystemPrompt } from "./system"
import { Instruction } from "./instruction"
import { Plugin } from "../plugin"
import PROMPT_PLAN from "../session/prompt/plan.txt"
import BUILD_SWITCH from "../session/prompt/build-switch.txt"
import MAX_STEPS from "../session/prompt/max-steps.txt"
import { ToolRegistry } from "../tool"
import { MCP } from "../mcp"
import { LSP } from "../lsp"
import { Flag } from "../flag/flag"
import { ulid } from "ulid"
import { ChildProcess, ChildProcessSpawner } from "effect/unstable/process"
import * as CrossSpawnSpawner from "@/effect/cross-spawn-spawner"
import * as Stream from "effect/Stream"
import { Command } from "../command"
import { pathToFileURL, fileURLToPath } from "url"
import { ConfigMarkdown } from "../config"
import { SessionSummary } from "./summary"
import { NamedError } from "@opencode-ai/shared/util/error"
import { SessionProcessor } from "./processor"
import { Tool } from "@/tool"
import { Permission } from "@/permission"
import { SessionStatus } from "./status"
import { LLM } from "./llm"
import { Shell } from "@/shell/shell"
import { AppFileSystem } from "@opencode-ai/shared/filesystem"
import { Truncate } from "@/tool"
import { decodeDataUrl } from "@/util/data-url"
import { Process } from "@/util"
import { Cause, Effect, Exit, Layer, Option, Scope, Context } from "effect"
import { EffectLogger } from "@/effect"
import { InstanceState } from "@/effect"
import { TaskTool, type TaskPromptOps } from "@/tool/task"
import { SessionRunState } from "./run-state"
import { EffectBridge } from "@/effect"

// --- LCM imports ---
import { isLcmReady } from "./lcm/runtime"
import { LcmDb } from "./lcm/db"
import { LcmContext } from "./lcm/context"
import { LcmContextSnapshot } from "./lcm/context-snapshot"
import type { LcmRetrieval } from "./lcm/retrieval"
import {
  LCM_PRE_RESPONSE_HOOK_MAX_DISTANCE,
  LCM_PRE_RESPONSE_HOOK_MIN_SCORE,
  LCM_PRE_RESPONSE_HOOK_TOP_K,
} from "./lcm/config"
import {
  compactUntilUnderHardLimit,
  ensureLcmRuntimeStrategyConfigured,
  getActiveLcmRuntimeStrategy,
  isThresholdCompactionInFlight,
  scheduleThresholdCompaction,
} from "./lcm/strategy"
import { TokenBudget } from "./token-budget"
import { Token } from "@/util"

// @ts-ignore
globalThis.AI_SDK_LOG_WARNINGS = false

const STRUCTURED_OUTPUT_DESCRIPTION = `Use this tool to return your final response in the requested structured format.

IMPORTANT:
- You MUST call this tool exactly once at the end of your response
- The input must be valid JSON matching the required schema
- Complete all necessary research and tool calls BEFORE calling this tool
- This tool provides your final answer - no further actions are taken after calling it`

const STRUCTURED_OUTPUT_SYSTEM_PROMPT = `IMPORTANT: The user has requested structured output. You MUST use the StructuredOutput tool to provide your final response. Do NOT respond with plain text - you MUST call the StructuredOutput tool with your answer formatted according to the schema.`

const log = Log.create({ service: "session.prompt" })
const elog = EffectLogger.create({ service: "session.prompt" })

// =========================================================================
// --- LCM: helper functions for context management ---
// =========================================================================

/** Per-session sync state: tracks the last message ID synced to LCM. */
const lcmSyncState = new Map<string, string>()

/**
 * Write a context snapshot, swallowing errors (best-effort for TUI display).
 */
async function writeLcmContextSnapshotBestEffort(input: {
  conversationId: number
  sessionID: string
  reason: string
  triggerMessageId?: number
}) {
  try {
    await LcmContextSnapshot.write(input)
  } catch (error) {
    log.warn("failed to write lcm context snapshot", {
      sessionID: input.sessionID,
      conversationId: input.conversationId,
      reason: input.reason,
      error,
    })
  }
}

/**
 * Get or create an LCM conversation ID for an OpenCode session.
 * Creates a new LCM conversation if one doesn't exist.
 * If the session has a parent, the new conversation will be linked to
 * the parent's, allowing the child to access files and summaries from ancestors.
 */
async function getOrCreateLcmConversation(
  sessionID: string,
  model: Provider.Model,
  sessionGet: (id: string) => Promise<Session.Info>,
): Promise<number | null> {
  const titlePrefix = `[OpenCode Session: ${sessionID}]`

  const conn = LcmDb.getConnection()
  const existing = await conn<{ conversation_id: number }[]>`
    SELECT conversation_id
    FROM conversations
    WHERE title LIKE ${titlePrefix + "%"}
    LIMIT 1
  `

  if (existing.length > 0) {
    return existing[0].conversation_id
  }

  // Check if this session has a parent for conversation linkage
  let parentConversationId: number | undefined
  try {
    const session = await sessionGet(sessionID)
    if (session.parentID) {
      parentConversationId =
        (await getOrCreateLcmConversation(session.parentID, model, sessionGet)) ?? undefined
    }
  } catch {
    // Session lookup failed, proceed without parent linkage
  }

  const conversationId = await LcmDb.createConversation({
    title: titlePrefix,
    modelName: model.id,
    modelCtxMaxTokens: model.limit.context,
    parentConversationId,
  })

  log.info("created LCM conversation for session", { sessionID, conversationId, parentConversationId })
  return conversationId
}

/**
 * Look up the LCM conversation ID for a session (if one exists).
 * Returns null if no conversation has been created for this session yet.
 */
export async function getLcmConversationId(sessionID: string): Promise<number | null> {
  const titlePrefix = `[OpenCode Session: ${sessionID}]`
  try {
    const conn = LcmDb.getConnection()
    const existing = await conn<{ conversation_id: number }[]>`
      SELECT conversation_id
      FROM conversations
      WHERE title LIKE ${titlePrefix + "%"}
      LIMIT 1
    `
    return existing[0]?.conversation_id ?? null
  } catch (e) {
    log.error("failed to look up LCM conversation", { sessionID, error: e })
    return null
  }
}

/**
 * Format a session message (with parts) into LCM-compatible structure.
 * Produces a content string and structured parts array for database storage.
 */
export function formatMessageForLcm(msg: MessageV2.WithParts): {
  role: LcmDb.MessageRole
  content: string
  tokenCount: number
  parts: LcmDb.MessagePartInput[]
} {
  const contentParts: string[] = []
  const structuredParts: LcmDb.MessagePartInput[] = []

  for (let ordinal = 0; ordinal < msg.parts.length; ordinal++) {
    const part = msg.parts[ordinal]
    const base = { partId: part.id, sessionId: part.sessionID, ordinal }

    switch (part.type) {
      case "text":
        structuredParts.push({
          ...base,
          partType: "text",
          textContent: part.text,
          isIgnored: part.ignored ?? null,
          isSynthetic: part.synthetic ?? null,
          metadata: part.metadata ?? null,
        })
        if (!part.ignored) {
          contentParts.push(part.text)
        }
        break
      case "reasoning":
        structuredParts.push({
          ...base,
          partType: "reasoning",
          textContent: part.text,
          metadata: part.metadata ?? null,
        })
        if (part.text) {
          contentParts.push(`<reasoning>\n${part.text}\n</reasoning>`)
        }
        break
      case "tool": {
        const toolPart: LcmDb.MessagePartInput = {
          ...base,
          partType: "tool",
          toolCallId: part.callID,
          toolName: part.tool,
          toolStatus: part.state.status,
          toolInput: part.state.input,
          metadata: part.metadata ?? null,
        }
        if (part.state.status === "completed") {
          const rawOutput = part.state.output
          const output =
            typeof rawOutput === "string" ? rawOutput : rawOutput == null ? "" : JSON.stringify(rawOutput)
          toolPart.toolOutput = output
          toolPart.toolTitle = part.state.title
          if (part.state.metadata) {
            toolPart.metadata = toolPart.metadata
              ? { ...toolPart.metadata, ...part.state.metadata }
              : part.state.metadata
          }
          contentParts.push(
            `<tool name="${part.tool}">\nInput: ${JSON.stringify(part.state.input)}\nOutput: ${output}\n</tool>`,
          )
        } else if (part.state.status === "error") {
          toolPart.toolError = part.state.error
          if (part.state.metadata) {
            toolPart.metadata = toolPart.metadata
              ? { ...toolPart.metadata, ...part.state.metadata }
              : part.state.metadata
          }
          contentParts.push(
            `<tool name="${part.tool}">\nInput: ${JSON.stringify(part.state.input)}\nError: ${part.state.error}\n</tool>`,
          )
        }
        structuredParts.push(toolPart)
        break
      }
      case "file":
        structuredParts.push({
          ...base,
          partType: "file",
          fileMime: part.mime,
          fileName: part.filename ?? null,
          fileUrl: part.url,
        })
        break
      case "subtask":
        structuredParts.push({
          ...base,
          partType: "subtask",
          subtaskPrompt: part.prompt,
          subtaskDesc: part.description,
          subtaskAgent: part.agent,
        })
        break
      case "compaction":
        structuredParts.push({
          ...base,
          partType: "compaction",
          compactionAuto: part.auto,
        })
        break
      default:
        break
    }
  }

  const content = contentParts.join("\n\n")
  const role: LcmDb.MessageRole = msg.info.role === "user" ? "user" : "assistant"
  return {
    role,
    content,
    tokenCount: Token.estimate(content),
    parts: structuredParts,
  }
}

/**
 * Sync session messages to LCM database (incremental when possible).
 */
async function syncSessionMessagesToLcm(
  conversationId: number,
  sessionID: string,
  sessionMessages: MessageV2.WithParts[],
) {
  const lastSyncedId = lcmSyncState.get(sessionID)
  if (lastSyncedId) {
    const lastIndex = sessionMessages.findIndex((msg) => msg.info.id === lastSyncedId)
    if (lastIndex >= 0) {
      const newMessages = sessionMessages.slice(lastIndex + 1)
      for (const msg of newMessages) {
        const formatted = formatMessageForLcm(msg)
        const messageId = await LcmDb.appendMessage({
          conversationId,
          role: formatted.role,
          content: formatted.content,
          tokenCount: formatted.tokenCount,
        })
        await LcmDb.insertMessageParts(messageId, formatted.parts)
        await writeLcmContextSnapshotBestEffort({
          conversationId,
          sessionID,
          reason: "leaf_appended",
          triggerMessageId: messageId,
        })
      }
      const lastMsg = sessionMessages.at(-1)
      if (lastMsg) lcmSyncState.set(sessionID, lastMsg.info.id)
      return
    }
  }

  // Full sync fallback
  const existingCount = await LcmDb.getMessageCount(conversationId)
  if (existingCount > sessionMessages.length) {
    log.warn("LCM message count exceeds session message count", { sessionID, conversationId, existingCount })
    return
  }
  const newMessages = sessionMessages.slice(existingCount)
  for (const msg of newMessages) {
    const formatted = formatMessageForLcm(msg)
    const messageId = await LcmDb.appendMessage({
      conversationId,
      role: formatted.role,
      content: formatted.content,
      tokenCount: formatted.tokenCount,
    })
    await LcmDb.insertMessageParts(messageId, formatted.parts)
    await writeLcmContextSnapshotBestEffort({
      conversationId,
      sessionID,
      reason: "leaf_appended",
      triggerMessageId: messageId,
    })
  }
  const lastMsg = sessionMessages.at(-1)
  if (lastMsg) lcmSyncState.set(sessionID, lastMsg.info.id)
}

function mapLcmRoleToModel(role: string): "user" | "assistant" {
  if (role === "user" || role === "system") return "user"
  return "assistant"
}

/**
 * Parse <tool> XML tags from LCM content and extract tool call information.
 */
export function parseToolTagsFromLcm(content: string): {
  name: string
  input: unknown
  output: string
  isError: boolean
}[] {
  const tools: { name: string; input: unknown; output: string; isError: boolean }[] = []
  const openTagPattern = /<tool name="([^"]+)">/g
  let openMatch
  while ((openMatch = openTagPattern.exec(content)) !== null) {
    const name = openMatch[1]
    const openTagEnd = openMatch.index + openMatch[0].length
    const nextOpenMatch = content.slice(openTagEnd).match(/<tool name="[^"]+">\s*Input:/)
    const searchEndPos =
      nextOpenMatch && nextOpenMatch.index !== undefined ? openTagEnd + nextOpenMatch.index : content.length
    let closeTagStart = -1
    let searchPos = searchEndPos
    while (searchPos > openTagEnd) {
      const lastCloseInRange = content.lastIndexOf("</tool>", searchPos - 1)
      if (lastCloseInRange === -1 || lastCloseInRange < openTagEnd) break
      const candidateContent = content.slice(openTagEnd, lastCloseInRange)
      if (/^\s*Input:\s*/.test(candidateContent) && /\n(Output|Error):/s.test(candidateContent)) {
        closeTagStart = lastCloseInRange
        break
      }
      searchPos = lastCloseInRange
    }
    if (closeTagStart === -1) continue
    const innerContent = content.slice(openTagEnd, closeTagStart)
    const inputMatch = innerContent.match(/^\s*Input:\s*/)
    if (!inputMatch) continue
    const afterInput = innerContent.slice(inputMatch[0].length)
    const lastOutputIndex = afterInput.lastIndexOf("\nOutput:")
    const lastErrorIndex = afterInput.lastIndexOf("\nError:")
    let resultType: "Output" | "Error"
    let splitIndex: number
    if (lastOutputIndex === -1 && lastErrorIndex === -1) continue
    if (lastOutputIndex === -1) {
      resultType = "Error"
      splitIndex = lastErrorIndex
    } else if (lastErrorIndex === -1) {
      resultType = "Output"
      splitIndex = lastOutputIndex
    } else if (lastOutputIndex > lastErrorIndex) {
      resultType = "Output"
      splitIndex = lastOutputIndex
    } else {
      resultType = "Error"
      splitIndex = lastErrorIndex
    }
    const inputStr = afterInput.slice(0, splitIndex)
    const markerLength = resultType === "Output" ? "\nOutput:".length : "\nError:".length
    const resultStr = afterInput.slice(splitIndex + markerLength)
    let input: unknown
    try {
      const parsed = JSON.parse(inputStr.trim())
      input = typeof parsed === "object" && parsed !== null ? parsed : { value: parsed }
    } catch {
      input = { value: inputStr.trim() }
    }
    tools.push({ name, input, output: resultStr.trim(), isError: resultType === "Error" })
  }
  return tools
}

/**
 * Strip all <tool name="...">...</tool> tags from content.
 */
export function stripToolTagsFromLcm(content: string): string {
  const ranges: { start: number; end: number }[] = []
  const openTagPattern = /<tool name="[^"]+">[\s]*/g
  let openMatch
  while ((openMatch = openTagPattern.exec(content)) !== null) {
    const openTagStart = openMatch.index
    const openTagEnd = openMatch.index + openMatch[0].length
    const nextOpenMatch = content.slice(openTagEnd).match(/<tool name="[^"]+">\s*Input:/)
    const searchEndPos =
      nextOpenMatch && nextOpenMatch.index !== undefined ? openTagEnd + nextOpenMatch.index : content.length
    let closeTagEnd = -1
    let searchPos = searchEndPos
    while (searchPos > openTagEnd) {
      const lastCloseInRange = content.lastIndexOf("</tool>", searchPos - 1)
      if (lastCloseInRange === -1 || lastCloseInRange < openTagEnd) break
      const candidateContent = content.slice(openTagEnd, lastCloseInRange)
      if (/^\s*Input:\s*/.test(candidateContent) && /\n(Output|Error):/s.test(candidateContent)) {
        closeTagEnd = lastCloseInRange + "</tool>".length
        while (closeTagEnd < content.length && /\s/.test(content[closeTagEnd])) closeTagEnd++
        break
      }
      searchPos = lastCloseInRange
    }
    if (closeTagEnd !== -1) ranges.push({ start: openTagStart, end: closeTagEnd })
  }
  let result = content
  for (let i = ranges.length - 1; i >= 0; i--) {
    const range = ranges[i]
    result = result.slice(0, range.start) + result.slice(range.end)
  }
  return result.trim()
}

/**
 * Strip LCM-only markers that aren't tool tags but shouldn't appear in text parts.
 */
export function stripLcmMarkers(content: string): string {
  return content
    .replace(/\[Patch:[^\]]*\]/g, "")
    .replace(/<file\s+path="[^"]*"\s+mime="[^"]*"\s*\/>/g, "")
    .replace(/<compaction\s*\/>/g, "")
    .replace(/<subtask\s+agent="[^"]*">[\s\S]*?<\/subtask>/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

const SUMMARY_ID_IN_CONTEXT_RE = /\[Summary ID: (sum_[a-f0-9]{16})\]/g

/**
 * Build a retrieval query string from the latest user message text parts.
 */
function buildPreResponseRetrievalQuery(message: MessageV2.WithParts | undefined): string {
  if (!message || message.info.role !== "user") return ""
  const chunks: string[] = []
  for (const part of message.parts) {
    if (part.type === "text" && !part.ignored && !part.synthetic) {
      const text = part.text.trim()
      if (text) chunks.push(text)
    }
    if (part.type === "subtask") {
      const prompt = part.prompt.trim()
      if (prompt) chunks.push(prompt)
    }
  }
  return chunks.join("\n").trim()
}

/**
 * Extract active summary IDs already present in the in-context summary lane.
 */
function collectActiveSummaryIdsFromContext(
  context: Array<{ item_type: string; content: string }>,
): Set<string> {
  const ids = new Set<string>()
  for (const item of context) {
    if (item.item_type !== "summary") continue
    for (const match of item.content.matchAll(SUMMARY_ID_IN_CONTEXT_RE)) {
      if (match[1]) ids.add(match[1])
    }
  }
  return ids
}

/**
 * Format retrieval hits into ultra-short pre-response memory cue lines.
 */
export function formatPreResponseMemoryCueBlock(input: {
  hits: LcmRetrieval.QueryHit[]
  activeSummaryIds: Iterable<string>
  topK?: number
}): string | null {
  const topK = Math.max(1, Math.floor(input.topK ?? LCM_PRE_RESPONSE_HOOK_TOP_K))
  const active = new Set(input.activeSummaryIds)
  const cues = input.hits.filter((hit) => !active.has(hit.summaryId)).slice(0, topK)
  if (cues.length === 0) return null
  const lines = ["<memory-cues>"]
  for (const [index, cue] of cues.entries()) {
    const pointerIds = cue.pointerSummaryIds.length > 0 ? cue.pointerSummaryIds.join(",") : "-"
    const lineageIds = cue.lineageSummaryIds.length > 0 ? cue.lineageSummaryIds.join(",") : "-"
    const archived = cue.summaryType === "archive_stub" ? "yes" : "no"
    lines.push(
      `[cue ${index + 1}] summaryId=${cue.summaryId} summaryType=${cue.summaryType} archived=${archived} score=${cue.score.toFixed(3)} distance=${cue.distance.toFixed(3)} pointerIds=${pointerIds} lineageIds=${lineageIds} cue=${JSON.stringify(cue.cueText)}`,
    )
  }
  lines.push("</memory-cues>")
  return lines.join("\n")
}

/**
 * Insert the cue block before the latest user message so the current query remains last.
 */
export function injectPreResponseMemoryCueBlock(
  messages: Array<{ role: string; content: any }>,
  cueBlock: string | null,
): Array<{ role: string; content: any }> {
  if (!cueBlock) return messages
  const lastUserIndex = [...messages].reverse().findIndex((message) => message.role === "user")
  if (lastUserIndex === -1) {
    return [...messages, { role: "user", content: cueBlock }]
  }
  const insertAt = messages.length - 1 - lastUserIndex
  return [...messages.slice(0, insertAt), { role: "user", content: cueBlock }, ...messages.slice(insertAt)]
}

/**
 * Build LCM-managed model messages, replacing the standard message conversion.
 * Syncs session messages to LCM, handles threshold compaction, assembles
 * context from the strategy, and injects ghost cues from off-context retrieval.
 */
async function buildLcmModelMessages(input: {
  sessionID: string
  user: MessageV2.User
  model: Provider.Model
  sessionMessages: MessageV2.WithParts[]
  assistantMessageID?: string
  toolTokenEstimate?: number
  systemPromptTokens?: number
  sessionGet: (id: string) => Promise<Session.Info>
  setLcm: (sessionID: string, lcm: { inputTokens: number; threshold: number }) => Promise<void>
  updatePart: (part: MessageV2.TextPart) => Promise<any>
}): Promise<Array<{ role: string; content: any }>> {
  const conversationId = await getOrCreateLcmConversation(input.sessionID, input.model, input.sessionGet)
  const strategy = getActiveLcmRuntimeStrategy()
  if (conversationId === null) {
    throw new Error("failed to get or create LCM conversation for session " + input.sessionID)
  }

  try {
    await syncSessionMessagesToLcm(conversationId, input.sessionID, input.sessionMessages)

    // Two-tier threshold: measure actual overhead from system prompt + tools
    const toolTokens = input.toolTokenEstimate ?? 0
    const systemTokens = input.systemPromptTokens ?? 0
    const budget = TokenBudget.computeBudget({
      model: input.model,
      systemPromptTokens: systemTokens,
      toolTokens,
      softThresholdOverride: Number(process.env.VOLTCODE_LCM_CONTEXT_THRESHOLD || process.env.OPENCODE_LCM_CONTEXT_THRESHOLD) || undefined,
    })
    TokenBudget.storeSessionBudget(input.sessionID, budget)
    const overhead = budget.overhead
    const reserve = budget.reserve
    const contextWindow = input.model.limit.context
    const softThresholdOverride = Number(process.env.VOLTCODE_LCM_CONTEXT_THRESHOLD || process.env.OPENCODE_LCM_CONTEXT_THRESHOLD) || undefined
    const thresholdCheck = await LcmContext.isOverThreshold({
      conversationId,
      overhead,
      reserve,
      contextWindow,
      softThresholdOverride,
    })
    const compactionInFlight = isThresholdCompactionInFlight(conversationId)

    log.info("building LCM context", {
      sessionID: input.sessionID,
      conversationId,
      currentTokens: thresholdCheck.currentTokens,
      softThreshold: thresholdCheck.softThreshold,
      hardLimit: thresholdCheck.hardLimit,
      overSoft: thresholdCheck.overSoft,
      overHard: thresholdCheck.overHard,
      compactionInFlight,
      overhead,
      reserve,
      strategy: strategy.name,
    })
    // DEBUG: temporary file output to diagnose compaction trigger
    import("fs").then(fs => fs.appendFileSync(
      (process.env.HOME || process.env.USERPROFILE) + "/lcm-trace.log",
      `[${new Date().toISOString()}] tokens=${thresholdCheck.currentTokens} soft=${thresholdCheck.softThreshold} hard=${thresholdCheck.hardLimit} overSoft=${thresholdCheck.overSoft} overHard=${thresholdCheck.overHard} overhead=${overhead} reserve=${reserve} ctxWindow=${contextWindow} threshold=${softThresholdOverride} strategy=${strategy.name}\n`
    )).catch(() => {})

    // --- LCM: publish metrics to session for TUI display ---
    const flagThreshold = softThresholdOverride ?? 0
    await input.setLcm(input.sessionID, {
      inputTokens: thresholdCheck.currentTokens + overhead,
      threshold: flagThreshold,
    })

    // --- LCM: threshold compaction ---
    if (thresholdCheck.overHard) {
      log.info("context exceeds hard limit, blocking on compaction", {
        sessionID: input.sessionID,
        conversationId,
        currentTokens: thresholdCheck.currentTokens,
        hardLimit: thresholdCheck.hardLimit,
        strategy: strategy.name,
      })
      LcmContext.setCompactionState(input.sessionID, conversationId, true)
      try {
        const compactResult = await compactUntilUnderHardLimit({
          conversationId,
          sessionID: input.sessionID,
          user: input.user,
          model: input.model,
          overhead,
          reserve,
          contextWindow,
          softThresholdOverride,
        })
        if (!compactResult.success) {
          log.error("hard-limit compaction failed, proceeding anyway", {
            sessionID: input.sessionID,
            conversationId,
            finalTokens: compactResult.finalTokens,
            hardLimit: compactResult.hardLimit,
            strategy: strategy.name,
          })
        }
        await writeLcmContextSnapshotBestEffort({
          conversationId,
          sessionID: input.sessionID,
          reason: "compaction_hard_limit",
        })
      } finally {
        LcmContext.clearCompactionState(input.sessionID)
      }
    } else if (thresholdCheck.overSoft || strategy.name === "upward") {
      // Tier 1 (soft threshold): schedule async compaction, proceed immediately
      import("fs").then(fs => fs.appendFileSync(
        (process.env.HOME || process.env.USERPROFILE) + "/lcm-trace.log",
        `[${new Date().toISOString()}] COMPACTION TRIGGERED overSoft=${thresholdCheck.overSoft} strategy=${strategy.name} tokens=${thresholdCheck.currentTokens}\n`
      )).catch(() => {})
      const job = scheduleThresholdCompaction({
        conversationId,
        sessionID: input.sessionID,
        user: input.user,
        model: input.model,
        overhead,
        reserve,
        contextWindow,
        softThresholdOverride,
      })

      if (job) {
        LcmContext.setCompactionState(input.sessionID, conversationId, false)
        const clearCompacting = () => LcmContext.clearCompactionState(input.sessionID)

        if (input.assistantMessageID) {
          const assistantMessageID = input.assistantMessageID
          void job
            .then(async (result) => {
              if (!result?.actionTaken || !result.createdSummary) return

              const afterTokens = result.newTokenCount ?? (await LcmDb.getContextTokenCount(conversationId))
              const beforeTokens = result.beforeTokenCount ?? afterTokens

              log.info("async compaction completed", {
                sessionID: input.sessionID,
                conversationId,
                summaryId: result.createdSummary.summaryId,
                summaryKind: result.createdSummary.kind,
                beforeTokens,
                afterTokens,
                strategy: strategy.name,
              })

              // Build an LCM event part for the TUI
              const event: MessageV2.TextPart = {
                id: PartID.ascending(),
                messageID: MessageID.make(assistantMessageID),
                sessionID: SessionID.make(input.sessionID),
                type: "text",
                text: `LCM summary created: ${result.createdSummary.summaryId}`,
                synthetic: true,
                ignored: true,
                metadata: {
                  lcm: {
                    type: "summary",
                    summaryId: result.createdSummary.summaryId,
                    summaryKind: result.createdSummary.kind,
                    beforeTokens,
                    afterTokens,
                    strategy: strategy.name,
                  },
                },
              }
              await input.updatePart(event)

              // Update session.lcm so TUI displays the new context size
              await input.setLcm(input.sessionID, {
                inputTokens: afterTokens + overhead,
                threshold: flagThreshold,
              })

              await writeLcmContextSnapshotBestEffort({
                conversationId,
                sessionID: input.sessionID,
                reason: "compaction_async_complete",
              })
            })
            .catch((error: unknown) => {
              log.warn("failed to publish async LCM summary event", {
                sessionID: input.sessionID,
                conversationId,
                error,
                strategy: strategy.name,
              })
            })
            .finally(clearCompacting)
        } else {
          void job.finally(clearCompacting)
        }
      }
    }

    // --- LCM: assemble context from strategy ---
    const context = await strategy.assembleContext(conversationId)

    log.debug("LCM context fetched", {
      conversationId,
      entryCount: context.length,
      byType: {
        message: context.filter((e) => e.item_type === "message").length,
        summary: context.filter((e) => e.item_type === "summary").length,
      },
      totalTokens: context.reduce((s, e) => s + e.token_count, 0),
    })

    // Pre-fetch structured parts for message entries
    const messageIds = context
      .filter((e) => e.item_type === "message" && e.message_id !== null)
      .map((e) => e.message_id!)
    const partsMap =
      messageIds.length > 0
        ? await LcmDb.getMessagePartsForMessages(messageIds)
        : new Map<number, LcmDb.MessagePart[]>()

    // --- LCM: pre-response ghost cue retrieval ---
    const activeSummaryIds = collectActiveSummaryIdsFromContext(context)
    const currentUserMessage = input.sessionMessages.find((message) => message.info.id === input.user.id)
    const retrievalQuery = buildPreResponseRetrievalQuery(currentUserMessage)

    let preResponseCueBlock: string | null = null
    if (retrievalQuery) {
      try {
        const retrieval = await strategy.resolveRetrieval({
          conversationId,
          query: retrievalQuery,
          topK: LCM_PRE_RESPONSE_HOOK_TOP_K,
          minScore: LCM_PRE_RESPONSE_HOOK_MIN_SCORE,
          maxDistance: LCM_PRE_RESPONSE_HOOK_MAX_DISTANCE,
        })
        preResponseCueBlock = formatPreResponseMemoryCueBlock({
          hits: retrieval.hits,
          activeSummaryIds,
          topK: LCM_PRE_RESPONSE_HOOK_TOP_K,
        })
        if (preResponseCueBlock) {
          log.debug("prepared pre-response memory cues", {
            sessionID: input.sessionID,
            conversationId,
            cueCount: retrieval.hits.filter((hit) => !activeSummaryIds.has(hit.summaryId)).length,
            strategy: strategy.name,
          })
        }
      } catch (error) {
        log.warn("failed pre-response off-context retrieval", {
          sessionID: input.sessionID,
          conversationId,
          error,
          strategy: strategy.name,
        })
      }
    }

    // --- LCM: convert context entries to model messages ---
    const messages: Array<{ role: string; content: any }> = context.flatMap((entry, idx): Array<{ role: string; content: any }> => {
      if (!entry.content.trim()) return []
      const role = mapLcmRoleToModel(entry.role)

      if (role === "user" && entry.message_id !== null) {
        const dbParts = partsMap.get(entry.message_id)
        const fileParts = dbParts?.filter((p) => p.part_type === "file" && p.file_url && p.file_mime)
        if (fileParts && fileParts.length > 0) {
          return [
            {
              role,
              content: [
                { type: "text", text: entry.content },
                ...fileParts.map((p) => ({
                  type: "image" as const,
                  image: new URL(p.file_url!),
                  mediaType: p.file_mime!,
                })),
              ],
            },
          ]
        }
      }

      // For assistant messages, parse tool XML into structured tool-call/tool-result messages
      if (role === "assistant") {
        const tools = parseToolTagsFromLcm(entry.content)
        if (tools.length > 0) {
          const textContent = stripLcmMarkers(stripToolTagsFromLcm(entry.content))
          const assistantParts: any[] = []
          if (textContent) assistantParts.push({ type: "text", text: textContent })
          for (let j = 0; j < tools.length; j++) {
            assistantParts.push({
              type: "tool-call",
              toolCallId: `lcm_${idx}_${j}`,
              toolName: tools[j].name,
              input: tools[j].input,
            })
          }
          return [
            { role: "assistant", content: assistantParts },
            {
              role: "tool",
              content: tools.map((t, j) => ({
                type: "tool-result" as const,
                toolCallId: `lcm_${idx}_${j}`,
                toolName: t.name,
                output: { type: "text" as const, value: t.output },
              })),
            },
          ]
        }
      }

      return [{ role, content: entry.content }]
    })

    return injectPreResponseMemoryCueBlock(messages, preResponseCueBlock)
  } catch (e) {
    log.error("failed to build LCM context", { sessionID: input.sessionID, error: e })
    throw e
  }
}

// =========================================================================
// --- End LCM helper functions ---
// =========================================================================

export interface Interface {
  readonly cancel: (sessionID: SessionID) => Effect.Effect<void>
  readonly prompt: (input: PromptInput) => Effect.Effect<MessageV2.WithParts>
  readonly loop: (input: z.infer<typeof LoopInput>) => Effect.Effect<MessageV2.WithParts>
  readonly shell: (input: ShellInput) => Effect.Effect<MessageV2.WithParts>
  readonly command: (input: CommandInput) => Effect.Effect<MessageV2.WithParts>
  readonly resolvePromptParts: (template: string) => Effect.Effect<PromptInput["parts"]>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionPrompt") {}

export const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const bus = yield* Bus.Service
    const status = yield* SessionStatus.Service
    const sessions = yield* Session.Service
    const agents = yield* Agent.Service
    const provider = yield* Provider.Service
    const processor = yield* SessionProcessor.Service
    const compaction = yield* SessionCompaction.Service
    const plugin = yield* Plugin.Service
    const commands = yield* Command.Service
    const permission = yield* Permission.Service
    const fsys = yield* AppFileSystem.Service
    const mcp = yield* MCP.Service
    const lsp = yield* LSP.Service
    const registry = yield* ToolRegistry.Service
    const truncate = yield* Truncate.Service
    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner
    const scope = yield* Scope.Scope
    const instruction = yield* Instruction.Service
    const state = yield* SessionRunState.Service
    const revert = yield* SessionRevert.Service
    const summary = yield* SessionSummary.Service
    const sys = yield* SystemPrompt.Service
    const llm = yield* LLM.Service
    const runner = Effect.fn("SessionPrompt.runner")(function* () {
      return yield* EffectBridge.make()
    })
    const ops = Effect.fn("SessionPrompt.ops")(function* () {
      const run = yield* runner()
      return {
        cancel: (sessionID: SessionID) => run.fork(cancel(sessionID)),
        resolvePromptParts: (template: string) => resolvePromptParts(template),
        prompt: (input: PromptInput) => prompt(input),
      } satisfies TaskPromptOps
    })

    const cancel = Effect.fn("SessionPrompt.cancel")(function* (sessionID: SessionID) {
      yield* elog.info("cancel", { sessionID })
      yield* state.cancel(sessionID)
    })

    const resolvePromptParts = Effect.fn("SessionPrompt.resolvePromptParts")(function* (template: string) {
      const ctx = yield* InstanceState.context
      const parts: PromptInput["parts"] = [{ type: "text", text: template }]
      const files = ConfigMarkdown.files(template)
      const seen = new Set<string>()
      yield* Effect.forEach(
        files,
        Effect.fnUntraced(function* (match) {
          const name = match[1]
          if (seen.has(name)) return
          seen.add(name)
          const filepath = name.startsWith("~/")
            ? path.join(os.homedir(), name.slice(2))
            : path.resolve(ctx.worktree, name)

          const info = yield* fsys.stat(filepath).pipe(Effect.option)
          if (Option.isNone(info)) {
            const found = yield* agents.get(name)
            if (found) parts.push({ type: "agent", name: found.name })
            return
          }
          const stat = info.value
          parts.push({
            type: "file",
            url: pathToFileURL(filepath).href,
            filename: name,
            mime: stat.type === "Directory" ? "application/x-directory" : "text/plain",
          })
        }),
        { concurrency: "unbounded", discard: true },
      )
      return parts
    })

    const title = Effect.fn("SessionPrompt.ensureTitle")(function* (input: {
      session: Session.Info
      history: MessageV2.WithParts[]
      providerID: ProviderID
      modelID: ModelID
    }) {
      if (input.session.parentID) return
      if (!Session.isDefaultTitle(input.session.title)) return

      const real = (m: MessageV2.WithParts) =>
        m.info.role === "user" && !m.parts.every((p) => "synthetic" in p && p.synthetic)
      const idx = input.history.findIndex(real)
      if (idx === -1) return
      if (input.history.filter(real).length !== 1) return

      const context = input.history.slice(0, idx + 1)
      const firstUser = context[idx]
      if (!firstUser || firstUser.info.role !== "user") return
      const firstInfo = firstUser.info

      const subtasks = firstUser.parts.filter((p): p is MessageV2.SubtaskPart => p.type === "subtask")
      const onlySubtasks = subtasks.length > 0 && firstUser.parts.every((p) => p.type === "subtask")

      const ag = yield* agents.get("title")
      if (!ag) return
      const mdl = ag.model
        ? yield* provider.getModel(ag.model.providerID, ag.model.modelID)
        : ((yield* provider.getSmallModel(input.providerID)) ??
          (yield* provider.getModel(input.providerID, input.modelID)))
      const msgs = onlySubtasks
        ? [{ role: "user" as const, content: subtasks.map((p) => p.prompt).join("\n") }]
        : yield* MessageV2.toModelMessagesEffect(context, mdl)
      const text = yield* llm
        .stream({
          agent: ag,
          user: firstInfo,
          system: [],
          small: true,
          tools: {},
          model: mdl,
          sessionID: input.session.id,
          retries: 2,
          messages: [{ role: "user", content: "Generate a title for this conversation:\n" }, ...msgs],
        })
        .pipe(
          Stream.filter((e): e is Extract<LLM.Event, { type: "text-delta" }> => e.type === "text-delta"),
          Stream.map((e) => e.text),
          Stream.mkString,
          Effect.orDie,
        )
      const cleaned = text
        .replace(/<think>[\s\S]*?<\/think>\s*/g, "")
        .split("\n")
        .map((line) => line.trim())
        .find((line) => line.length > 0)
      if (!cleaned) return
      const t = cleaned.length > 100 ? cleaned.substring(0, 97) + "..." : cleaned
      yield* sessions
        .setTitle({ sessionID: input.session.id, title: t })
        .pipe(Effect.catchCause((cause) => elog.error("failed to generate title", { error: Cause.squash(cause) })))
    })

    const insertReminders = Effect.fn("SessionPrompt.insertReminders")(function* (input: {
      messages: MessageV2.WithParts[]
      agent: Agent.Info
      session: Session.Info
    }) {
      const userMessage = input.messages.findLast((msg) => msg.info.role === "user")
      if (!userMessage) return input.messages

      if (!Flag.OPENCODE_EXPERIMENTAL_PLAN_MODE) {
        if (input.agent.name === "plan") {
          userMessage.parts.push({
            id: PartID.ascending(),
            messageID: userMessage.info.id,
            sessionID: userMessage.info.sessionID,
            type: "text",
            text: PROMPT_PLAN,
            synthetic: true,
          })
        }
        const wasPlan = input.messages.some((msg) => msg.info.role === "assistant" && msg.info.agent === "plan")
        if (wasPlan && input.agent.name === "build") {
          userMessage.parts.push({
            id: PartID.ascending(),
            messageID: userMessage.info.id,
            sessionID: userMessage.info.sessionID,
            type: "text",
            text: BUILD_SWITCH,
            synthetic: true,
          })
        }
        return input.messages
      }

      const assistantMessage = input.messages.findLast((msg) => msg.info.role === "assistant")
      if (input.agent.name !== "plan" && assistantMessage?.info.agent === "plan") {
        const plan = Session.plan(input.session)
        if (!(yield* fsys.existsSafe(plan))) return input.messages
        const part = yield* sessions.updatePart({
          id: PartID.ascending(),
          messageID: userMessage.info.id,
          sessionID: userMessage.info.sessionID,
          type: "text",
          text: `${BUILD_SWITCH}\n\nA plan file exists at ${plan}. You should execute on the plan defined within it`,
          synthetic: true,
        })
        userMessage.parts.push(part)
        return input.messages
      }

      if (input.agent.name !== "plan" || assistantMessage?.info.agent === "plan") return input.messages

      const plan = Session.plan(input.session)
      const exists = yield* fsys.existsSafe(plan)
      if (!exists) yield* fsys.ensureDir(path.dirname(plan)).pipe(Effect.catch(Effect.die))
      const part = yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: userMessage.info.id,
        sessionID: userMessage.info.sessionID,
        type: "text",
        text: `<system-reminder>
Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits (with the exception of the plan file mentioned below), run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supersedes any other instructions you have received.

## Plan File Info:
${exists ? `A plan file already exists at ${plan}. You can read it and make incremental edits using the edit tool.` : `No plan file exists yet. You should create your plan at ${plan} using the write tool.`}
You should build your plan incrementally by writing to or editing this file. NOTE that this is the only file you are allowed to edit - other than this you are only allowed to take READ-ONLY actions.

## Plan Workflow

### Phase 1: Initial Understanding
Goal: Gain a comprehensive understanding of the user's request by reading through code and asking them questions. Critical: In this phase you should only use the explore subagent type.

1. Focus on understanding the user's request and the code associated with their request

2. **Launch up to 3 explore agents IN PARALLEL** (single message, multiple tool calls) to efficiently explore the codebase.
 - Use 1 agent when the task is isolated to known files, the user provided specific file paths, or you're making a small targeted change.
 - Use multiple agents when: the scope is uncertain, multiple areas of the codebase are involved, or you need to understand existing patterns before planning.
 - Quality over quantity - 3 agents maximum, but you should try to use the minimum number of agents necessary (usually just 1)
 - If using multiple agents: Provide each agent with a specific search focus or area to explore. Example: One agent searches for existing implementations, another explores related components, a third investigates testing patterns

3. After exploring the code, use the question tool to clarify ambiguities in the user request up front.

### Phase 2: Design
Goal: Design an implementation approach.

Launch general agent(s) to design the implementation based on the user's intent and your exploration results from Phase 1.

You can launch up to 1 agent(s) in parallel.

**Guidelines:**
- **Default**: Launch at least 1 Plan agent for most tasks - it helps validate your understanding and consider alternatives
- **Skip agents**: Only for truly trivial tasks (typo fixes, single-line changes, simple renames)

Examples of when to use multiple agents:
- The task touches multiple parts of the codebase
- It's a large refactor or architectural change
- There are many edge cases to consider
- You'd benefit from exploring different approaches

Example perspectives by task type:
- New feature: simplicity vs performance vs maintainability
- Bug fix: root cause vs workaround vs prevention
- Refactoring: minimal change vs clean architecture

In the agent prompt:
- Provide comprehensive background context from Phase 1 exploration including filenames and code path traces
- Describe requirements and constraints
- Request a detailed implementation plan

### Phase 3: Review
Goal: Review the plan(s) from Phase 2 and ensure alignment with the user's intentions.
1. Read the critical files identified by agents to deepen your understanding
2. Ensure that the plans align with the user's original request
3. Use question tool to clarify any remaining questions with the user

### Phase 4: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- Include only your recommended approach, not all alternatives
- Ensure that the plan file is concise enough to scan quickly, but detailed enough to execute effectively
- Include the paths of critical files to be modified
- Include a verification section describing how to test the changes end-to-end (run the code, use MCP tools, run tests)

### Phase 5: Call plan_exit tool
At the very end of your turn, once you have asked the user questions and are happy with your final plan file - you should always call plan_exit to indicate to the user that you are done planning.
This is critical - your turn should only end with either asking the user a question or calling plan_exit. Do not stop unless it's for these 2 reasons.

**Important:** Use question tool to clarify requirements/approach, use plan_exit to request plan approval. Do NOT use question tool to ask "Is this plan okay?" - that's what plan_exit does.

NOTE: At any point in time through this workflow you should feel free to ask the user questions or clarifications. Don't make large assumptions about user intent. The goal is to present a well researched plan to the user, and tie any loose ends before implementation begins.
</system-reminder>`,
        synthetic: true,
      })
      userMessage.parts.push(part)
      return input.messages
    })

    const resolveTools = Effect.fn("SessionPrompt.resolveTools")(function* (input: {
      agent: Agent.Info
      model: Provider.Model
      session: Session.Info
      tools?: Record<string, boolean>
      processor: Pick<SessionProcessor.Handle, "message" | "updateToolCall" | "completeToolCall">
      bypassAgentCheck: boolean
      messages: MessageV2.WithParts[]
    }) {
      using _ = log.time("resolveTools")
      const tools: Record<string, AITool> = {}
      const run = yield* runner()
      const promptOps = yield* ops()

      const context = (args: any, options: ToolExecutionOptions): Tool.Context => ({
        sessionID: input.session.id,
        abort: options.abortSignal!,
        messageID: input.processor.message.id,
        callID: options.toolCallId,
        extra: { model: input.model, bypassAgentCheck: input.bypassAgentCheck, promptOps },
        agent: input.agent.name,
        messages: input.messages,
        metadata: (val) =>
          input.processor.updateToolCall(options.toolCallId, (match) => {
            if (!["running", "pending"].includes(match.state.status)) return match
            return {
              ...match,
              state: {
                title: val.title,
                metadata: val.metadata,
                status: "running",
                input: args,
                time: { start: Date.now() },
              },
            }
          }),
        ask: (req) =>
          permission
            .ask({
              ...req,
              sessionID: input.session.id,
              tool: { messageID: input.processor.message.id, callID: options.toolCallId },
              ruleset: Permission.merge(input.agent.permission, input.session.permission ?? []),
            })
            .pipe(Effect.orDie),
      })

      for (const item of yield* registry.tools({
        modelID: ModelID.make(input.model.api.id),
        providerID: input.model.providerID,
        agent: input.agent,
      })) {
        const schema = ProviderTransform.schema(input.model, z.toJSONSchema(item.parameters))
        tools[item.id] = tool({
          description: item.description,
          inputSchema: jsonSchema(schema),
          execute(args, options) {
            return run.promise(
              Effect.gen(function* () {
                const ctx = context(args, options)
                yield* plugin.trigger(
                  "tool.execute.before",
                  { tool: item.id, sessionID: ctx.sessionID, callID: ctx.callID },
                  { args },
                )
                const result = yield* item.execute(args, ctx)
                const output = {
                  ...result,
                  attachments: result.attachments?.map((attachment) => ({
                    ...attachment,
                    id: PartID.ascending(),
                    sessionID: ctx.sessionID,
                    messageID: input.processor.message.id,
                  })),
                }
                yield* plugin.trigger(
                  "tool.execute.after",
                  { tool: item.id, sessionID: ctx.sessionID, callID: ctx.callID, args },
                  output,
                )
                if (options.abortSignal?.aborted) {
                  yield* input.processor.completeToolCall(options.toolCallId, output)
                }
                return output
              }),
            )
          },
        })
      }

      for (const [key, item] of Object.entries(yield* mcp.tools())) {
        const execute = item.execute
        if (!execute) continue

        const schema = yield* Effect.promise(() => Promise.resolve(asSchema(item.inputSchema).jsonSchema))
        const transformed = ProviderTransform.schema(input.model, schema)
        item.inputSchema = jsonSchema(transformed)
        item.execute = (args, opts) =>
          run.promise(
            Effect.gen(function* () {
              const ctx = context(args, opts)
              yield* plugin.trigger(
                "tool.execute.before",
                { tool: key, sessionID: ctx.sessionID, callID: opts.toolCallId },
                { args },
              )
              yield* ctx.ask({ permission: key, metadata: {}, patterns: ["*"], always: ["*"] })
              const result: Awaited<ReturnType<NonNullable<typeof execute>>> = yield* Effect.promise(() =>
                execute(args, opts),
              )
              yield* plugin.trigger(
                "tool.execute.after",
                { tool: key, sessionID: ctx.sessionID, callID: opts.toolCallId, args },
                result,
              )

              const textParts: string[] = []
              const attachments: Omit<MessageV2.FilePart, "id" | "sessionID" | "messageID">[] = []
              for (const contentItem of result.content) {
                if (contentItem.type === "text") textParts.push(contentItem.text)
                else if (contentItem.type === "image") {
                  attachments.push({
                    type: "file",
                    mime: contentItem.mimeType,
                    url: `data:${contentItem.mimeType};base64,${contentItem.data}`,
                  })
                } else if (contentItem.type === "resource") {
                  const { resource } = contentItem
                  if (resource.text) textParts.push(resource.text)
                  if (resource.blob) {
                    attachments.push({
                      type: "file",
                      mime: resource.mimeType ?? "application/octet-stream",
                      url: `data:${resource.mimeType ?? "application/octet-stream"};base64,${resource.blob}`,
                      filename: resource.uri,
                    })
                  }
                }
              }

              const truncated = yield* truncate.output(textParts.join("\n\n"), {}, input.agent)
              const metadata = {
                ...result.metadata,
                truncated: truncated.truncated,
                ...(truncated.truncated && { outputPath: truncated.outputPath }),
              }

              const output = {
                title: "",
                metadata,
                output: truncated.content,
                attachments: attachments.map((attachment) => ({
                  ...attachment,
                  id: PartID.ascending(),
                  sessionID: ctx.sessionID,
                  messageID: input.processor.message.id,
                })),
                content: result.content,
              }
              if (opts.abortSignal?.aborted) {
                yield* input.processor.completeToolCall(opts.toolCallId, output)
              }
              return output
            }),
          )
        tools[key] = item
      }

      return tools
    })

    const handleSubtask = Effect.fn("SessionPrompt.handleSubtask")(function* (input: {
      task: MessageV2.SubtaskPart
      model: Provider.Model
      lastUser: MessageV2.User
      sessionID: SessionID
      session: Session.Info
      msgs: MessageV2.WithParts[]
    }) {
      const { task, model, lastUser, sessionID, session, msgs } = input
      const ctx = yield* InstanceState.context
      const promptOps = yield* ops()
      const { task: taskTool } = yield* registry.named()
      const taskModel = task.model ? yield* getModel(task.model.providerID, task.model.modelID, sessionID) : model
      const assistantMessage: MessageV2.Assistant = yield* sessions.updateMessage({
        id: MessageID.ascending(),
        role: "assistant",
        parentID: lastUser.id,
        sessionID,
        mode: task.agent,
        agent: task.agent,
        variant: lastUser.model.variant,
        path: { cwd: ctx.directory, root: ctx.worktree },
        cost: 0,
        tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        modelID: taskModel.id,
        providerID: taskModel.providerID,
        time: { created: Date.now() },
      })
      let part: MessageV2.ToolPart = yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: assistantMessage.id,
        sessionID: assistantMessage.sessionID,
        type: "tool",
        callID: ulid(),
        tool: TaskTool.id,
        state: {
          status: "running",
          input: {
            prompt: task.prompt,
            description: task.description,
            subagent_type: task.agent,
            command: task.command,
          },
          time: { start: Date.now() },
        },
      })
      const taskArgs = {
        prompt: task.prompt,
        description: task.description,
        subagent_type: task.agent,
        command: task.command,
      }
      yield* plugin.trigger(
        "tool.execute.before",
        { tool: TaskTool.id, sessionID, callID: part.id },
        { args: taskArgs },
      )

      const taskAgent = yield* agents.get(task.agent)
      if (!taskAgent) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${task.agent}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID, error: error.toObject() })
        throw error
      }

      let error: Error | undefined
      const taskAbort = new AbortController()
      const result = yield* taskTool
        .execute(taskArgs, {
          agent: task.agent,
          messageID: assistantMessage.id,
          sessionID,
          abort: taskAbort.signal,
          callID: part.callID,
          extra: { bypassAgentCheck: true, promptOps },
          messages: msgs,
          metadata: (val: { title?: string; metadata?: Record<string, any> }) =>
            Effect.gen(function* () {
              part = yield* sessions.updatePart({
                ...part,
                type: "tool",
                state: { ...part.state, ...val },
              } satisfies MessageV2.ToolPart)
            }),
          ask: (req: any) =>
            permission
              .ask({
                ...req,
                sessionID,
                ruleset: Permission.merge(taskAgent.permission, session.permission ?? []),
              })
              .pipe(Effect.orDie),
        })
        .pipe(
          Effect.catchCause((cause) => {
            const defect = Cause.squash(cause)
            error = defect instanceof Error ? defect : new Error(String(defect))
            log.error("subtask execution failed", { error, agent: task.agent, description: task.description })
            return Effect.void
          }),
          Effect.onInterrupt(() =>
            Effect.gen(function* () {
              taskAbort.abort()
              assistantMessage.finish = "tool-calls"
              assistantMessage.time.completed = Date.now()
              yield* sessions.updateMessage(assistantMessage)
              if (part.state.status === "running") {
                yield* sessions.updatePart({
                  ...part,
                  state: {
                    status: "error",
                    error: "Cancelled",
                    time: { start: part.state.time.start, end: Date.now() },
                    metadata: part.state.metadata,
                    input: part.state.input,
                  },
                } satisfies MessageV2.ToolPart)
              }
            }),
          ),
        )

      const attachments = result?.attachments?.map((attachment) => ({
        ...attachment,
        id: PartID.ascending(),
        sessionID,
        messageID: assistantMessage.id,
      }))

      yield* plugin.trigger(
        "tool.execute.after",
        { tool: TaskTool.id, sessionID, callID: part.id, args: taskArgs },
        result,
      )

      assistantMessage.finish = "tool-calls"
      assistantMessage.time.completed = Date.now()
      yield* sessions.updateMessage(assistantMessage)

      if (result && part.state.status === "running") {
        yield* sessions.updatePart({
          ...part,
          state: {
            status: "completed",
            input: part.state.input,
            title: result.title,
            metadata: result.metadata,
            output: result.output,
            attachments,
            time: { ...part.state.time, end: Date.now() },
          },
        } satisfies MessageV2.ToolPart)
      }

      if (!result) {
        yield* sessions.updatePart({
          ...part,
          state: {
            status: "error",
            error: error ? `Tool execution failed: ${error.message}` : "Tool execution failed",
            time: {
              start: part.state.status === "running" ? part.state.time.start : Date.now(),
              end: Date.now(),
            },
            metadata: part.state.status === "pending" ? undefined : part.state.metadata,
            input: part.state.input,
          },
        } satisfies MessageV2.ToolPart)
      }

      if (!task.command) return

      const summaryUserMsg: MessageV2.User = {
        id: MessageID.ascending(),
        sessionID,
        role: "user",
        time: { created: Date.now() },
        agent: lastUser.agent,
        model: lastUser.model,
      }
      yield* sessions.updateMessage(summaryUserMsg)
      yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: summaryUserMsg.id,
        sessionID,
        type: "text",
        text: "Summarize the task tool output above and continue with your task.",
        synthetic: true,
      } satisfies MessageV2.TextPart)
    })

    const shellImpl = Effect.fn("SessionPrompt.shellImpl")(function* (input: ShellInput) {
      const ctx = yield* InstanceState.context
      const run = yield* runner()
      const session = yield* sessions.get(input.sessionID)
      if (session.revert) {
        yield* revert.cleanup(session)
      }
      const agent = yield* agents.get(input.agent)
      if (!agent) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${input.agent}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }
      const model = input.model ?? agent.model ?? (yield* lastModel(input.sessionID))
      const userMsg: MessageV2.User = {
        id: input.messageID ?? MessageID.ascending(),
        sessionID: input.sessionID,
        time: { created: Date.now() },
        role: "user",
        agent: input.agent,
        model: { providerID: model.providerID, modelID: model.modelID },
      }
      yield* sessions.updateMessage(userMsg)
      const userPart: MessageV2.Part = {
        type: "text",
        id: PartID.ascending(),
        messageID: userMsg.id,
        sessionID: input.sessionID,
        text: "The following tool was executed by the user",
        synthetic: true,
      }
      yield* sessions.updatePart(userPart)

      const msg: MessageV2.Assistant = {
        id: MessageID.ascending(),
        sessionID: input.sessionID,
        parentID: userMsg.id,
        mode: input.agent,
        agent: input.agent,
        cost: 0,
        path: { cwd: ctx.directory, root: ctx.worktree },
        time: { created: Date.now() },
        role: "assistant",
        tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        modelID: model.modelID,
        providerID: model.providerID,
      }
      yield* sessions.updateMessage(msg)
      const part: MessageV2.ToolPart = {
        type: "tool",
        id: PartID.ascending(),
        messageID: msg.id,
        sessionID: input.sessionID,
        tool: "bash",
        callID: ulid(),
        state: {
          status: "running",
          time: { start: Date.now() },
          input: { command: input.command },
        },
      }
      yield* sessions.updatePart(part)

      const sh = Shell.preferred()
      const shellName = (
        process.platform === "win32" ? path.win32.basename(sh, ".exe") : path.basename(sh)
      ).toLowerCase()
      const invocations: Record<string, { args: string[] }> = {
        nu: { args: ["-c", input.command] },
        fish: { args: ["-c", input.command] },
        zsh: {
          args: [
            "-l",
            "-c",
            `
              __oc_cwd=$PWD
              [[ -f ~/.zshenv ]] && source ~/.zshenv >/dev/null 2>&1 || true
              [[ -f "\${ZDOTDIR:-$HOME}/.zshrc" ]] && source "\${ZDOTDIR:-$HOME}/.zshrc" >/dev/null 2>&1 || true
              cd "$__oc_cwd"
              eval ${JSON.stringify(input.command)}
            `,
          ],
        },
        bash: {
          args: [
            "-l",
            "-c",
            `
              __oc_cwd=$PWD
              shopt -s expand_aliases
              [[ -f ~/.bashrc ]] && source ~/.bashrc >/dev/null 2>&1 || true
              cd "$__oc_cwd"
              eval ${JSON.stringify(input.command)}
            `,
          ],
        },
        cmd: { args: ["/c", input.command] },
        powershell: { args: ["-NoProfile", "-Command", input.command] },
        pwsh: { args: ["-NoProfile", "-Command", input.command] },
        "": { args: ["-c", input.command] },
      }

      const args = (invocations[shellName] ?? invocations[""]).args
      const cwd = ctx.directory
      const shellEnv = yield* plugin.trigger(
        "shell.env",
        { cwd, sessionID: input.sessionID, callID: part.callID },
        { env: {} },
      )

      const cmd = ChildProcess.make(sh, args, {
        cwd,
        extendEnv: true,
        env: { ...shellEnv.env, TERM: "dumb" },
        stdin: "ignore",
        forceKillAfter: "3 seconds",
      })

      let output = ""
      let aborted = false

      const finish = Effect.uninterruptible(
        Effect.gen(function* () {
          if (aborted) {
            output += "\n\n" + ["<metadata>", "User aborted the command", "</metadata>"].join("\n")
          }
          if (!msg.time.completed) {
            msg.time.completed = Date.now()
            yield* sessions.updateMessage(msg)
          }
          if (part.state.status === "running") {
            part.state = {
              status: "completed",
              time: { ...part.state.time, end: Date.now() },
              input: part.state.input,
              title: "",
              metadata: { output, description: "" },
              output,
            }
            yield* sessions.updatePart(part)
          }
        }),
      )

      const exit = yield* Effect.gen(function* () {
        const handle = yield* spawner.spawn(cmd)
        yield* Stream.runForEach(Stream.decodeText(handle.all), (chunk) =>
          Effect.sync(() => {
            output += chunk
            if (part.state.status === "running") {
              part.state.metadata = { output, description: "" }
              void run.fork(sessions.updatePart(part))
            }
          }),
        )
        yield* handle.exitCode
      }).pipe(
        Effect.scoped,
        Effect.onInterrupt(() =>
          Effect.sync(() => {
            aborted = true
          }),
        ),
        Effect.orDie,
        Effect.ensuring(finish),
        Effect.exit,
      )

      if (Exit.isFailure(exit) && !Cause.hasInterruptsOnly(exit.cause)) {
        return yield* Effect.failCause(exit.cause)
      }

      return { info: msg, parts: [part] }
    })

    const getModel = Effect.fn("SessionPrompt.getModel")(function* (
      providerID: ProviderID,
      modelID: ModelID,
      sessionID: SessionID,
    ) {
      const exit = yield* provider.getModel(providerID, modelID).pipe(Effect.exit)
      if (Exit.isSuccess(exit)) return exit.value
      const err = Cause.squash(exit.cause)
      if (Provider.ModelNotFoundError.isInstance(err)) {
        const hint = err.data.suggestions?.length ? ` Did you mean: ${err.data.suggestions.join(", ")}?` : ""
        yield* bus.publish(Session.Event.Error, {
          sessionID,
          error: new NamedError.Unknown({
            message: `Model not found: ${err.data.providerID}/${err.data.modelID}.${hint}`,
          }).toObject(),
        })
      }
      return yield* Effect.failCause(exit.cause)
    })

    const lastModel = Effect.fnUntraced(function* (sessionID: SessionID) {
      const match = yield* sessions.findMessage(sessionID, (m) => m.info.role === "user" && !!m.info.model)
      if (Option.isSome(match) && match.value.info.role === "user") return match.value.info.model
      return yield* provider.defaultModel()
    })

    const createUserMessage = Effect.fn("SessionPrompt.createUserMessage")(function* (input: PromptInput) {
      const agentName = input.agent || (yield* agents.defaultAgent())
      const ag = yield* agents.get(agentName)
      if (!ag) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${agentName}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }

      const model = input.model ?? ag.model ?? (yield* lastModel(input.sessionID))
      const same = ag.model && model.providerID === ag.model.providerID && model.modelID === ag.model.modelID
      const full =
        !input.variant && ag.variant && same
          ? yield* provider.getModel(model.providerID, model.modelID).pipe(Effect.catchDefect(() => Effect.void))
          : undefined
      const variant = input.variant ?? (ag.variant && full?.variants?.[ag.variant] ? ag.variant : undefined)

      const info: MessageV2.User = {
        id: input.messageID ?? MessageID.ascending(),
        role: "user",
        sessionID: input.sessionID,
        time: { created: Date.now() },
        tools: input.tools,
        agent: ag.name,
        model: {
          providerID: model.providerID,
          modelID: model.modelID,
          variant,
        },
        system: input.system,
        format: input.format,
      }

      yield* Effect.addFinalizer(() => instruction.clear(info.id))

      type Draft<T> = T extends MessageV2.Part ? Omit<T, "id"> & { id?: string } : never
      const assign = (part: Draft<MessageV2.Part>): MessageV2.Part => ({
        ...part,
        id: part.id ? PartID.make(part.id) : PartID.ascending(),
      })

      const resolvePart: (part: PromptInput["parts"][number]) => Effect.Effect<Draft<MessageV2.Part>[]> = Effect.fn(
        "SessionPrompt.resolveUserPart",
      )(function* (part) {
        if (part.type === "file") {
          if (part.source?.type === "resource") {
            const { clientName, uri } = part.source
            log.info("mcp resource", { clientName, uri, mime: part.mime })
            const pieces: Draft<MessageV2.Part>[] = [
              {
                messageID: info.id,
                sessionID: input.sessionID,
                type: "text",
                synthetic: true,
                text: `Reading MCP resource: ${part.filename} (${uri})`,
              },
            ]
            const exit = yield* mcp.readResource(clientName, uri).pipe(Effect.exit)
            if (Exit.isSuccess(exit)) {
              const content = exit.value
              if (!content) throw new Error(`Resource not found: ${clientName}/${uri}`)
              const items = Array.isArray(content.contents) ? content.contents : [content.contents]
              for (const c of items) {
                if ("text" in c && c.text) {
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: c.text,
                  })
                } else if ("blob" in c && c.blob) {
                  const mime = "mimeType" in c ? c.mimeType : part.mime
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `[Binary content: ${mime}]`,
                  })
                }
              }
              pieces.push({ ...part, messageID: info.id, sessionID: input.sessionID })
            } else {
              const error = Cause.squash(exit.cause)
              log.error("failed to read MCP resource", { error, clientName, uri })
              const message = error instanceof Error ? error.message : String(error)
              pieces.push({
                messageID: info.id,
                sessionID: input.sessionID,
                type: "text",
                synthetic: true,
                text: `Failed to read MCP resource ${part.filename}: ${message}`,
              })
            }
            return pieces
          }
          const url = new URL(part.url)
          switch (url.protocol) {
            case "data:":
              if (part.mime === "text/plain") {
                return [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify({ filePath: part.filename })}`,
                  },
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: decodeDataUrl(part.url),
                  },
                  { ...part, messageID: info.id, sessionID: input.sessionID },
                ]
              }
              break
            case "file:": {
              log.info("file", { mime: part.mime })
              const filepath = fileURLToPath(part.url)
              if (yield* fsys.isDir(filepath)) part.mime = "application/x-directory"

              const { read } = yield* registry.named()
              const execRead = (args: Parameters<typeof read.execute>[0], extra?: Tool.Context["extra"]) => {
                const controller = new AbortController()
                return read
                  .execute(args, {
                    sessionID: input.sessionID,
                    abort: controller.signal,
                    agent: input.agent!,
                    messageID: info.id,
                    extra: { bypassCwdCheck: true, ...extra },
                    messages: [],
                    metadata: () => Effect.void,
                    ask: () => Effect.void,
                  })
                  .pipe(Effect.onInterrupt(() => Effect.sync(() => controller.abort())))
              }

              if (part.mime === "text/plain") {
                let offset: number | undefined
                let limit: number | undefined
                const range = { start: url.searchParams.get("start"), end: url.searchParams.get("end") }
                if (range.start != null) {
                  const filePathURI = part.url.split("?")[0]
                  let start = parseInt(range.start)
                  let end = range.end ? parseInt(range.end) : undefined
                  if (start === end) {
                    const symbols = yield* lsp.documentSymbol(filePathURI).pipe(Effect.catch(() => Effect.succeed([])))
                    for (const symbol of symbols) {
                      let r: LSP.Range | undefined
                      if ("range" in symbol) r = symbol.range
                      else if ("location" in symbol) r = symbol.location.range
                      if (r?.start?.line && r?.start?.line === start) {
                        start = r.start.line
                        end = r?.end?.line ?? start
                        break
                      }
                    }
                  }
                  offset = Math.max(start, 1)
                  if (end) limit = end - (offset - 1)
                }
                const args = { filePath: filepath, offset, limit }
                const pieces: Draft<MessageV2.Part>[] = [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify(args)}`,
                  },
                ]
                const exit = yield* provider.getModel(info.model.providerID, info.model.modelID).pipe(
                  Effect.flatMap((mdl) => execRead(args, { model: mdl })),
                  Effect.exit,
                )
                if (Exit.isSuccess(exit)) {
                  const result = exit.value
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: result.output,
                  })
                  if (result.attachments?.length) {
                    pieces.push(
                      ...result.attachments.map((a) => ({
                        ...a,
                        synthetic: true,
                        filename: a.filename ?? part.filename,
                        messageID: info.id,
                        sessionID: input.sessionID,
                      })),
                    )
                  } else {
                    pieces.push({ ...part, messageID: info.id, sessionID: input.sessionID })
                  }
                } else {
                  const error = Cause.squash(exit.cause)
                  log.error("failed to read file", { error })
                  const message = error instanceof Error ? error.message : String(error)
                  yield* bus.publish(Session.Event.Error, {
                    sessionID: input.sessionID,
                    error: new NamedError.Unknown({ message }).toObject(),
                  })
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Read tool failed to read ${filepath} with the following error: ${message}`,
                  })
                }
                return pieces
              }

              if (part.mime === "application/x-directory") {
                const args = { filePath: filepath }
                const exit = yield* execRead(args).pipe(Effect.exit)
                if (Exit.isFailure(exit)) {
                  const error = Cause.squash(exit.cause)
                  log.error("failed to read directory", { error })
                  const message = error instanceof Error ? error.message : String(error)
                  yield* bus.publish(Session.Event.Error, {
                    sessionID: input.sessionID,
                    error: new NamedError.Unknown({ message }).toObject(),
                  })
                  return [
                    {
                      messageID: info.id,
                      sessionID: input.sessionID,
                      type: "text",
                      synthetic: true,
                      text: `Read tool failed to read ${filepath} with the following error: ${message}`,
                    },
                  ]
                }
                return [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify(args)}`,
                  },
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: exit.value.output,
                  },
                  { ...part, messageID: info.id, sessionID: input.sessionID },
                ]
              }

              return [
                {
                  messageID: info.id,
                  sessionID: input.sessionID,
                  type: "text",
                  synthetic: true,
                  text: `Called the Read tool with the following input: {"filePath":"${filepath}"}`,
                },
                {
                  id: part.id,
                  messageID: info.id,
                  sessionID: input.sessionID,
                  type: "file",
                  url:
                    `data:${part.mime};base64,` +
                    Buffer.from(yield* fsys.readFile(filepath).pipe(Effect.catch(Effect.die))).toString("base64"),
                  mime: part.mime,
                  filename: part.filename!,
                  source: part.source,
                },
              ]
            }
          }
        }

        if (part.type === "agent") {
          const perm = Permission.evaluate("task", part.name, ag.permission)
          const hint = perm.action === "deny" ? " . Invoked by user; guaranteed to exist." : ""
          return [
            { ...part, messageID: info.id, sessionID: input.sessionID },
            {
              messageID: info.id,
              sessionID: input.sessionID,
              type: "text",
              synthetic: true,
              text:
                " Use the above message and context to generate a prompt and call the task tool with subagent: " +
                part.name +
                hint,
            },
          ]
        }

        return [{ ...part, messageID: info.id, sessionID: input.sessionID }]
      })

      const parts = yield* Effect.forEach(input.parts, resolvePart, { concurrency: "unbounded" }).pipe(
        Effect.map((x) => x.flat().map(assign)),
      )

      yield* plugin.trigger(
        "chat.message",
        {
          sessionID: input.sessionID,
          agent: input.agent,
          model: input.model,
          messageID: input.messageID,
          variant: input.variant,
        },
        { message: info, parts },
      )

      const parsed = MessageV2.Info.zod.safeParse(info)
      if (!parsed.success) {
        log.error("invalid user message before save", {
          sessionID: input.sessionID,
          messageID: info.id,
          agent: info.agent,
          model: info.model,
          issues: parsed.error.issues,
        })
      }
      parts.forEach((part, index) => {
        const p = MessageV2.Part.zod.safeParse(part)
        if (p.success) return
        log.error("invalid user part before save", {
          sessionID: input.sessionID,
          messageID: info.id,
          partID: part.id,
          partType: part.type,
          index,
          issues: p.error.issues,
          part,
        })
      })

      yield* sessions.updateMessage(info)
      for (const part of parts) yield* sessions.updatePart(part)

      return { info, parts }
    }, Effect.scoped)

    const prompt: (input: PromptInput) => Effect.Effect<MessageV2.WithParts> = Effect.fn("SessionPrompt.prompt")(
      function* (input: PromptInput) {
        const session = yield* sessions.get(input.sessionID)
        yield* revert.cleanup(session)
        const message = yield* createUserMessage(input)
        yield* sessions.touch(input.sessionID)

        const permissions: Permission.Ruleset = []
        for (const [t, enabled] of Object.entries(input.tools ?? {})) {
          permissions.push({ permission: t, action: enabled ? "allow" : "deny", pattern: "*" })
        }
        if (permissions.length > 0) {
          session.permission = permissions
          yield* sessions.setPermission({ sessionID: session.id, permission: permissions })
        }

        if (input.noReply === true) return message
        return yield* loop({ sessionID: input.sessionID })
      },
    )

    const lastAssistant = Effect.fnUntraced(function* (sessionID: SessionID) {
      const match = yield* sessions.findMessage(sessionID, (m) => m.info.role !== "user")
      if (Option.isSome(match)) return match.value
      const msgs = yield* sessions.messages({ sessionID, limit: 1 })
      if (msgs.length > 0) return msgs[0]
      throw new Error("Impossible")
    })

    const runLoop: (sessionID: SessionID) => Effect.Effect<MessageV2.WithParts> = Effect.fn("SessionPrompt.run")(
      function* (sessionID: SessionID) {
        const ctx = yield* InstanceState.context
        const slog = elog.with({ sessionID })
        let structured: unknown | undefined
        let step = 0
        const session = yield* sessions.get(sessionID)

        while (true) {
          yield* status.set(sessionID, { type: "busy" })
          yield* slog.info("loop", { step })

          let msgs = yield* MessageV2.filterCompactedEffect(sessionID)

          let lastUser: MessageV2.User | undefined
          let lastAssistant: MessageV2.Assistant | undefined
          let lastFinished: MessageV2.Assistant | undefined
          let tasks: (MessageV2.CompactionPart | MessageV2.SubtaskPart)[] = []
          for (let i = msgs.length - 1; i >= 0; i--) {
            const msg = msgs[i]
            if (!lastUser && msg.info.role === "user") lastUser = msg.info
            if (!lastAssistant && msg.info.role === "assistant") lastAssistant = msg.info
            if (!lastFinished && msg.info.role === "assistant" && msg.info.finish) lastFinished = msg.info
            if (lastUser && lastFinished) break
            const task = msg.parts.filter((part) => part.type === "compaction" || part.type === "subtask")
            if (task && !lastFinished) tasks.push(...task)
          }

          if (!lastUser) throw new Error("No user message found in stream. This should never happen.")

          const lastAssistantMsg = msgs.findLast(
            (msg) => msg.info.role === "assistant" && msg.info.id === lastAssistant?.id,
          )
          // Some providers return "stop" even when the assistant message contains tool calls.
          // Keep the loop running so tool results can be sent back to the model.
          // Skip provider-executed tool parts — those were fully handled within the
          // provider's stream (e.g. DWS Agent Platform) and don't need a re-loop.
          const hasToolCalls =
            lastAssistantMsg?.parts.some((part) => part.type === "tool" && !part.metadata?.providerExecuted) ?? false

          if (
            lastAssistant?.finish &&
            !["tool-calls"].includes(lastAssistant.finish) &&
            !hasToolCalls &&
            lastUser.id < lastAssistant.id
          ) {
            yield* slog.info("exiting loop")
            break
          }

          step++
          if (step === 1)
            yield* title({
              session,
              modelID: lastUser.model.modelID,
              providerID: lastUser.model.providerID,
              history: msgs,
            }).pipe(Effect.ignore, Effect.forkIn(scope))

          const model = yield* getModel(lastUser.model.providerID, lastUser.model.modelID, sessionID)
          const task = tasks.pop()

          if (task?.type === "subtask") {
            yield* handleSubtask({ task, model, lastUser, sessionID, session, msgs })
            continue
          }

          // --- LCM: compaction is handled by LCM context management ---
          // When LCM is active, stale compaction parts are removed.
          // When LCM is not active, fall back to legacy compaction.
          if (task?.type === "compaction") {
            if (isLcmReady()) {
              // LCM handles compaction — just remove the stale compaction part
              yield* sessions.removePart({
                sessionID: task.sessionID,
                messageID: task.messageID,
                partID: task.id,
              })
              yield* slog.info("removed stale compaction task (LCM active)")
              continue
            }
            const result = yield* compaction.process({
              messages: msgs,
              parentID: lastUser.id,
              sessionID,
              auto: task.auto,
              overflow: task.overflow,
            })
            if (result === "stop") break
            continue
          }

          // Legacy overflow check — skipped when LCM is active (LCM manages thresholds)
          if (
            !isLcmReady() &&
            lastFinished &&
            lastFinished.summary !== true &&
            (yield* compaction.isOverflow({ tokens: lastFinished.tokens, model }))
          ) {
            yield* compaction.create({ sessionID, agent: lastUser.agent, model: lastUser.model, auto: true })
            continue
          }

          const agent = yield* agents.get(lastUser.agent)
          if (!agent) {
            const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
            const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
            const error = new NamedError.Unknown({ message: `Agent not found: "${lastUser.agent}".${hint}` })
            yield* bus.publish(Session.Event.Error, { sessionID, error: error.toObject() })
            throw error
          }
          const maxSteps = agent.steps ?? Infinity
          const isLastStep = step >= maxSteps
          msgs = yield* insertReminders({ messages: msgs, agent, session })

          const msg: MessageV2.Assistant = {
            id: MessageID.ascending(),
            parentID: lastUser.id,
            role: "assistant",
            mode: agent.name,
            agent: agent.name,
            variant: lastUser.model.variant,
            path: { cwd: ctx.directory, root: ctx.worktree },
            cost: 0,
            tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
            modelID: model.id,
            providerID: model.providerID,
            time: { created: Date.now() },
            sessionID,
          }
          yield* sessions.updateMessage(msg)
          const handle = yield* processor.create({
            assistantMessage: msg,
            sessionID,
            model,
          })

          const outcome: "break" | "continue" = yield* Effect.gen(function* () {
            const run = yield* runner()
            const lastUserMsg = msgs.findLast((m) => m.info.role === "user")
            const bypassAgentCheck = lastUserMsg?.parts.some((p) => p.type === "agent") ?? false

            const tools = yield* resolveTools({
              agent,
              session,
              model,
              tools: lastUser.tools,
              processor: handle,
              bypassAgentCheck,
              messages: msgs,
            })

            if (lastUser.format?.type === "json_schema") {
              tools["StructuredOutput"] = createStructuredOutputTool({
                schema: lastUser.format.schema,
                onSuccess(output) {
                  structured = output
                },
              })
            }

            if (step === 1)
              yield* summary.summarize({ sessionID, messageID: lastUser.id }).pipe(Effect.ignore, Effect.forkIn(scope))

            if (step > 1 && lastFinished) {
              for (const m of msgs) {
                if (m.info.role !== "user" || m.info.id <= lastFinished.id) continue
                for (const p of m.parts) {
                  if (p.type !== "text" || p.ignored || p.synthetic) continue
                  if (!p.text.trim()) continue
                  p.text = [
                    "<system-reminder>",
                    "The user sent the following message:",
                    p.text,
                    "",
                    "Please address this message and continue with your tasks.",
                    "</system-reminder>",
                  ].join("\n")
                }
              }
            }

            yield* plugin.trigger("experimental.chat.messages.transform", {}, { messages: msgs })

            const [skills, env, instructions] = yield* Effect.all([
              sys.skills(agent),
              Effect.sync(() => sys.environment(model)),
              instruction.system().pipe(Effect.orDie),
            ])

            // --- LCM: context management ---
            // When LCM is active, replace standard message conversion with
            // LCM-managed context window (sync, threshold compaction, ghost cues).
            // Estimate tool + system prompt tokens for accurate budget computation
            const toolTokenEstimate = Object.values(tools).reduce((sum, t) => {
              const desc = (t as any).description ?? ""
              const params = (t as any).parameters ? JSON.stringify((t as any).parameters) : ""
              return sum + Token.estimate(desc + params)
            }, 0)
            const systemPromptEstimate = [
              ...env,
              ...(skills ? [skills] : []),
              ...instructions,
            ].reduce((sum, part) => sum + Token.estimate(part), 0)

            import("fs").then(fs => fs.appendFileSync(
              (process.env.HOME || process.env.USERPROFILE) + "/lcm-trace.log",
              `[${new Date().toISOString()}] isLcmReady=${isLcmReady()} toolTokens=${toolTokenEstimate} systemTokens=${systemPromptEstimate} total_overhead=${toolTokenEstimate + systemPromptEstimate}\n`
            )).catch(() => {})
            const modelMsgs: any[] = isLcmReady()
              ? yield* Effect.promise(() => {
                  return buildLcmModelMessages({
                    sessionID,
                    user: lastUser,
                    model,
                    sessionMessages: msgs,
                    assistantMessageID: handle.message.id,
                    toolTokenEstimate,
                    systemPromptTokens: systemPromptEstimate,
                    sessionGet: (id) =>
                      run.promise(sessions.get(SessionID.make(id))),
                    setLcm: (sid, lcm) =>
                      run.promise(sessions.setLcm({ sessionID: SessionID.make(sid), lcm })),
                    updatePart: (part) =>
                      run.promise(sessions.updatePart(part)),
                  })
                })
              : yield* MessageV2.toModelMessagesEffect(msgs, model)

            const system = [...env, ...(skills ? [skills] : []), ...instructions]
            const format = lastUser.format ?? { type: "text" as const }
            if (format.type === "json_schema") system.push(STRUCTURED_OUTPUT_SYSTEM_PROMPT)
            const result = yield* handle.process({
              user: lastUser,
              agent,
              permission: session.permission,
              sessionID,
              parentSessionID: session.parentID,
              system,
              messages: [...modelMsgs, ...(isLastStep ? [{ role: "assistant" as const, content: MAX_STEPS }] : [])],
              tools,
              model,
              toolChoice: format.type === "json_schema" ? "required" : undefined,
            })

            if (structured !== undefined) {
              handle.message.structured = structured
              handle.message.finish = handle.message.finish ?? "stop"
              yield* sessions.updateMessage(handle.message)
              return "break" as const
            }

            const finished = handle.message.finish && !["tool-calls", "unknown"].includes(handle.message.finish)
            if (finished && !handle.message.error) {
              if (format.type === "json_schema") {
                handle.message.error = new MessageV2.StructuredOutputError({
                  message: "Model did not produce structured output",
                  retries: 0,
                }).toObject()
                yield* sessions.updateMessage(handle.message)
                return "break" as const
              }
            }

            if (result === "stop") return "break" as const
            if (result === "compact") {
              if (isLcmReady()) {
                // LCM handles compaction — trigger blocking compaction
                yield* Effect.promise(async () => {
                  const convId = await getLcmConversationId(sessionID)
                  if (convId) {
                    const strategy = ensureLcmRuntimeStrategyConfigured()
                    const budget = TokenBudget.computeBudget({ model, systemPromptTokens: 0, toolTokens: 0 })
                    await strategy.compactManual({
                      sessionID,
                      conversationId: convId,
                      user: lastUser,
                      model,
                      overhead: budget.overhead,
                      reserve: budget.reserve,
                      contextWindow: model.limit.context,
                    })
                  }
                })
              } else {
                yield* compaction.create({
                  sessionID,
                  agent: lastUser.agent,
                  model: lastUser.model,
                  auto: true,
                  overflow: !handle.message.finish,
                })
              }
            }
            return "continue" as const
          }).pipe(Effect.ensuring(instruction.clear(handle.message.id)))
          if (outcome === "break") break
          continue
        }

        yield* compaction.prune({ sessionID }).pipe(Effect.ignore, Effect.forkIn(scope))
        return yield* lastAssistant(sessionID)
      },
    )

    const loop: (input: z.infer<typeof LoopInput>) => Effect.Effect<MessageV2.WithParts> = Effect.fn(
      "SessionPrompt.loop",
    )(function* (input: z.infer<typeof LoopInput>) {
      return yield* state.ensureRunning(input.sessionID, lastAssistant(input.sessionID), runLoop(input.sessionID))
    })

    const shell: (input: ShellInput) => Effect.Effect<MessageV2.WithParts> = Effect.fn("SessionPrompt.shell")(
      function* (input: ShellInput) {
        return yield* state.startShell(input.sessionID, lastAssistant(input.sessionID), shellImpl(input))
      },
    )

    const command = Effect.fn("SessionPrompt.command")(function* (input: CommandInput) {
      yield* elog.info("command", { sessionID: input.sessionID, command: input.command, agent: input.agent })
      const cmd = yield* commands.get(input.command)
      if (!cmd) {
        const available = (yield* commands.list()).map((c) => c.name)
        const hint = available.length ? ` Available commands: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Command not found: "${input.command}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }
      const agentName = cmd.agent ?? input.agent ?? (yield* agents.defaultAgent())

      const raw = input.arguments.match(argsRegex) ?? []
      const args = raw.map((arg) => arg.replace(quoteTrimRegex, ""))
      const templateCommand = yield* Effect.promise(async () => cmd.template)

      const placeholders = templateCommand.match(placeholderRegex) ?? []
      let last = 0
      for (const item of placeholders) {
        const value = Number(item.slice(1))
        if (value > last) last = value
      }

      const withArgs = templateCommand.replaceAll(placeholderRegex, (_, index) => {
        const position = Number(index)
        const argIndex = position - 1
        if (argIndex >= args.length) return ""
        if (position === last) return args.slice(argIndex).join(" ")
        return args[argIndex]
      })
      const usesArgumentsPlaceholder = templateCommand.includes("$ARGUMENTS")
      let template = withArgs.replaceAll("$ARGUMENTS", input.arguments)

      if (placeholders.length === 0 && !usesArgumentsPlaceholder && input.arguments.trim()) {
        template = template + "\n\n" + input.arguments
      }

      const shellMatches = ConfigMarkdown.shell(template)
      if (shellMatches.length > 0) {
        const sh = Shell.preferred()
        const results = yield* Effect.promise(() =>
          Promise.all(
            shellMatches.map(async ([, cmd]) => (await Process.text([cmd], { shell: sh, nothrow: true })).text),
          ),
        )
        let index = 0
        template = template.replace(bashRegex, () => results[index++])
      }
      template = template.trim()

      const taskModel = yield* Effect.gen(function* () {
        if (cmd.model) return Provider.parseModel(cmd.model)
        if (cmd.agent) {
          const cmdAgent = yield* agents.get(cmd.agent)
          if (cmdAgent?.model) return cmdAgent.model
        }
        if (input.model) return Provider.parseModel(input.model)
        return yield* lastModel(input.sessionID)
      })

      yield* getModel(taskModel.providerID, taskModel.modelID, input.sessionID)

      const agent = yield* agents.get(agentName)
      if (!agent) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${agentName}".${hint}` })
        yield* bus.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }

      const templateParts = yield* resolvePromptParts(template)
      const isSubtask = (agent.mode === "subagent" && cmd.subtask !== false) || cmd.subtask === true
      const parts = isSubtask
        ? [
            {
              type: "subtask" as const,
              agent: agent.name,
              description: cmd.description ?? "",
              command: input.command,
              model: { providerID: taskModel.providerID, modelID: taskModel.modelID },
              prompt: templateParts.find((y) => y.type === "text")?.text ?? "",
            },
          ]
        : [...templateParts, ...(input.parts ?? [])]

      const userAgent = isSubtask ? (input.agent ?? (yield* agents.defaultAgent())) : agentName
      const userModel = isSubtask
        ? input.model
          ? Provider.parseModel(input.model)
          : yield* lastModel(input.sessionID)
        : taskModel

      yield* plugin.trigger(
        "command.execute.before",
        { command: input.command, sessionID: input.sessionID, arguments: input.arguments },
        { parts },
      )

      const result = yield* prompt({
        sessionID: input.sessionID,
        messageID: input.messageID,
        model: userModel,
        agent: userAgent,
        parts,
        variant: input.variant,
      })
      yield* bus.publish(Command.Event.Executed, {
        name: input.command,
        sessionID: input.sessionID,
        arguments: input.arguments,
        messageID: result.info.id,
      })
      return result
    })

    return Service.of({
      cancel,
      prompt,
      loop,
      shell,
      command,
      resolvePromptParts,
    })
  }),
)

export const defaultLayer = Layer.suspend(() =>
  layer.pipe(
    Layer.provide(SessionRunState.defaultLayer),
    Layer.provide(SessionStatus.defaultLayer),
    Layer.provide(SessionCompaction.defaultLayer),
    Layer.provide(SessionProcessor.defaultLayer),
    Layer.provide(Command.defaultLayer),
    Layer.provide(Permission.defaultLayer),
    Layer.provide(MCP.defaultLayer),
    Layer.provide(LSP.defaultLayer),
    Layer.provide(ToolRegistry.defaultLayer),
    Layer.provide(Truncate.defaultLayer),
    Layer.provide(Provider.defaultLayer),
    Layer.provide(Instruction.defaultLayer),
    Layer.provide(AppFileSystem.defaultLayer),
    Layer.provide(Plugin.defaultLayer),
    Layer.provide(Session.defaultLayer),
    Layer.provide(SessionRevert.defaultLayer),
    Layer.provide(SessionSummary.defaultLayer),
    Layer.provide(
      Layer.mergeAll(
        Agent.defaultLayer,
        SystemPrompt.defaultLayer,
        LLM.defaultLayer,
        Bus.layer,
        CrossSpawnSpawner.defaultLayer,
      ),
    ),
  ),
)
export const PromptInput = z.object({
  sessionID: SessionID.zod,
  messageID: MessageID.zod.optional(),
  model: z
    .object({
      providerID: ProviderID.zod,
      modelID: ModelID.zod,
    })
    .optional(),
  agent: z.string().optional(),
  noReply: z.boolean().optional(),
  tools: z
    .record(z.string(), z.boolean())
    .optional()
    .describe("@deprecated tools and permissions have been merged, you can set permissions on the session itself now"),
  format: MessageV2.Format.zod.optional(),
  system: z.string().optional(),
  variant: z.string().optional(),
  parts: z.array(
    z.discriminatedUnion("type", [
      MessageV2.TextPartInput.zod as unknown as z.ZodObject<any>,
      MessageV2.FilePartInput.zod as unknown as z.ZodObject<any>,
      MessageV2.AgentPartInput.zod as unknown as z.ZodObject<any>,
      MessageV2.SubtaskPartInput.zod as unknown as z.ZodObject<any>,
    ]),
  ),
})
// `z.discriminatedUnion` erases the discriminated members' shapes back to
// `{}` because the derived `.zod` on each input is typed as an opaque
// `z.ZodType`. Restore the precise `parts` type from the exported Schema
// input types so callers see a proper tagged union.
type PartInputUnion =
  | MessageV2.TextPartInput
  | MessageV2.FilePartInput
  | MessageV2.AgentPartInput
  | MessageV2.SubtaskPartInput
export type PromptInput = Omit<z.infer<typeof PromptInput>, "parts"> & {
  parts: PartInputUnion[]
}

export const LoopInput = z.object({
  sessionID: SessionID.zod,
})

export const ShellInput = z.object({
  sessionID: SessionID.zod,
  messageID: MessageID.zod.optional(),
  agent: z.string(),
  model: z
    .object({
      providerID: ProviderID.zod,
      modelID: ModelID.zod,
    })
    .optional(),
  command: z.string(),
})
export type ShellInput = z.infer<typeof ShellInput>

export const CommandInput = z.object({
  messageID: MessageID.zod.optional(),
  sessionID: SessionID.zod,
  agent: z.string().optional(),
  model: z.string().optional(),
  arguments: z.string(),
  command: z.string(),
  variant: z.string().optional(),
  // Inlined (no `.meta({ ref })`) to keep the original SDK output — the
  // PromptInput call site below references FilePartInput by ref via the
  // Schema export in message-v2.ts.
  parts: z
    .array(
      z.discriminatedUnion("type", [
        z.object({
          id: PartID.zod.optional(),
          type: z.literal("file"),
          mime: z.string(),
          filename: z.string().optional(),
          url: z.string(),
          source: MessageV2.FilePartSource.zod.optional(),
        }),
      ]),
    )
    .optional(),
})
export type CommandInput = z.infer<typeof CommandInput>

/** @internal Exported for testing */
export function createStructuredOutputTool(input: {
  schema: Record<string, any>
  onSuccess: (output: unknown) => void
}): AITool {
  // Remove $schema property if present (not needed for tool input)
  const { $schema: _, ...toolSchema } = input.schema

  return tool({
    description: STRUCTURED_OUTPUT_DESCRIPTION,
    inputSchema: jsonSchema(toolSchema as JSONSchema7),
    async execute(args) {
      // AI SDK validates args against inputSchema before calling execute()
      input.onSuccess(args)
      return {
        output: "Structured output captured successfully.",
        title: "Structured Output",
        metadata: { valid: true },
      }
    },
    toModelOutput({ output }) {
      return {
        type: "text",
        value: output.output,
      }
    },
  })
}
const bashRegex = /!`([^`]+)`/g
// Match [Image N] as single token, quoted strings, or non-space sequences
const argsRegex = /(?:\[Image\s+\d+\]|"[^"]*"|'[^']*'|[^\s"']+)/gi
const placeholderRegex = /\$(\d+)/g
const quoteTrimRegex = /^["']|["']$/g

export * as SessionPrompt from "./prompt"
