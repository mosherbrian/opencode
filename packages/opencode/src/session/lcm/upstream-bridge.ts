/**
 * Bridge between LCM's async/await code and upstream's Effect service pattern.
 *
 * Upstream migrated from static async functions to Effect services. This module
 * re-exports thin async wrappers so LCM business logic stays in plain async/await.
 *
 * When rebasing on upstream, this is the only file that needs updating if
 * Effect service APIs change.
 */
import { Effect } from "effect"
import { AppRuntime } from "@/effect/app-runtime"
import { Provider } from "@/provider"
import { Session } from "@/session"
import { SessionPrompt } from "@/session/prompt"
import { Config } from "@/config"

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export async function getLanguage(model: Provider.Model) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const p = yield* Provider.Service
      return yield* p.getLanguage(model)
    }),
  )
}

export async function getModel(providerID: string, modelID: string) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const p = yield* Provider.Service
      return yield* p.getModel(providerID as any, modelID as any)
    }),
  )
}

export async function getSmallModel(providerID: string) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const p = yield* Provider.Service
      return yield* p.getSmallModel(providerID as any)
    }),
  )
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

export async function sessionCreate(input?: Parameters<Session.Interface["create"]>[0]) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const s = yield* Session.Service
      return yield* s.create(input)
    }),
  )
}

export async function sessionGet(id: Session.Info["id"]) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const s = yield* Session.Service
      return yield* s.get(id)
    }),
  )
}

export async function sessionMessages(input: Parameters<Session.Interface["messages"]>[0]) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const s = yield* Session.Service
      return yield* s.messages(input)
    }),
  )
}

// ---------------------------------------------------------------------------
// SessionPrompt
// ---------------------------------------------------------------------------

export async function promptCancel(sessionID: Parameters<SessionPrompt.Interface["cancel"]>[0]) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const p = yield* SessionPrompt.Service
      return yield* p.cancel(sessionID)
    }),
  )
}

export async function promptPrompt(input: Parameters<SessionPrompt.Interface["prompt"]>[0]) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const p = yield* SessionPrompt.Service
      return yield* p.prompt(input)
    }),
  )
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export async function configGet() {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const c = yield* Config.Service
      return yield* c.get()
    }),
  )
}

// ---------------------------------------------------------------------------
// Plugin (lazy to avoid circular imports)
// ---------------------------------------------------------------------------

export async function pluginTrigger<A, B>(name: string, input: A, output: B): Promise<B> {
  const { Plugin } = await import("@/plugin")
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const p = yield* Plugin.Service
      return yield* p.trigger(name as any, input, output)
    }),
  )
}

// ---------------------------------------------------------------------------
// Task execution (replaces the old executeTask function)
// ---------------------------------------------------------------------------

export interface ExecuteTaskInput {
  description: string
  prompt: string
  subagent_type: string
  agent?: string
  model?: { modelID: string; providerID: string }
  parentSessionID?: string
  permission?: any[]
}

export interface ExecuteTaskResult {
  output: string
  sessionID: string
}

/**
 * Simplified task execution: creates a sub-agent session and runs a prompt.
 * Replaces the old `executeTask` that was removed in the Effect migration.
 */
export async function executeTask(input: ExecuteTaskInput): Promise<ExecuteTaskResult> {
  const session = await sessionCreate({
    parentID: input.parentSessionID as any,
    title: input.description,
    permission: input.permission,
  })

  const result = await promptPrompt({
    sessionID: session.id,
    model: input.model ? { modelID: input.model.modelID as any, providerID: input.model.providerID as any } : { modelID: "" as any, providerID: "" as any },
    agent: input.agent ?? input.subagent_type,
    parts: [{ type: "text" as const, text: input.prompt }],
  })

  const output = result.parts.findLast((p) => p.type === "text")
  return {
    output: output && "text" in output ? output.text : "",
    sessionID: session.id,
  }
}
