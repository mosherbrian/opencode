import type { Argv } from "yargs"
import { cmd } from "./cmd"
import { Session } from "../../session"
import { SessionID } from "../../session/schema"
import { bootstrap } from "../bootstrap"
import { UI } from "../ui"
import { Locale } from "../../util"
import { Flag } from "../../flag/flag"
import { Filesystem } from "../../util"
import { Process } from "../../util"
import { EOL } from "os"
import path from "path"
import { which } from "../../util/which"
import { AppRuntime } from "@/effect/app-runtime"
import { ensureLcmReady } from "../../session/lcm/runtime"
import { LcmDb } from "../../session/lcm/db"
import { LCM_CONTEXT_SNAPSHOT_PATH } from "../../session/lcm/config"
import { LcmContextSnapshot } from "../../session/lcm/context-snapshot"

function pagerCmd(): string[] {
  const lessOptions = ["-R", "-S"]
  if (process.platform !== "win32") {
    return ["less", ...lessOptions]
  }

  // user could have less installed via other options
  const lessOnPath = which("less")
  if (lessOnPath) {
    if (Filesystem.stat(lessOnPath)?.size) return [lessOnPath, ...lessOptions]
  }

  if (Flag.OPENCODE_GIT_BASH_PATH) {
    const less = path.join(Flag.OPENCODE_GIT_BASH_PATH, "..", "..", "usr", "bin", "less.exe")
    if (Filesystem.stat(less)?.size) return [less, ...lessOptions]
  }

  const git = which("git")
  if (git) {
    const less = path.join(git, "..", "..", "usr", "bin", "less.exe")
    if (Filesystem.stat(less)?.size) return [less, ...lessOptions]
  }

  // Fall back to Windows built-in more (via cmd.exe)
  return ["cmd", "/c", "more"]
}

export const SessionCommand = cmd({
  command: "session",
  describe: "manage sessions",
  builder: (yargs: Argv) =>
    yargs.command(SessionListCommand).command(SessionDeleteCommand).command(SessionLcmWatchCommand).demandCommand(),
  async handler() {},
})

export const SessionDeleteCommand = cmd({
  command: "delete <sessionID>",
  describe: "delete a session",
  builder: (yargs: Argv) => {
    return yargs.positional("sessionID", {
      describe: "session ID to delete",
      type: "string",
      demandOption: true,
    })
  },
  handler: async (args) => {
    await bootstrap(process.cwd(), async () => {
      const sessionID = SessionID.make(args.sessionID)
      try {
        await AppRuntime.runPromise(Session.Service.use((svc) => svc.get(sessionID)))
      } catch {
        UI.error(`Session not found: ${args.sessionID}`)
        process.exit(1)
      }
      await AppRuntime.runPromise(Session.Service.use((svc) => svc.remove(sessionID)))
      UI.println(UI.Style.TEXT_SUCCESS_BOLD + `Session ${args.sessionID} deleted` + UI.Style.TEXT_NORMAL)
    })
  },
})

export const SessionListCommand = cmd({
  command: "list",
  describe: "list sessions",
  builder: (yargs: Argv) => {
    return yargs
      .option("max-count", {
        alias: "n",
        describe: "limit to N most recent sessions",
        type: "number",
      })
      .option("format", {
        describe: "output format",
        type: "string",
        choices: ["table", "json"],
        default: "table",
      })
  },
  handler: async (args) => {
    await bootstrap(process.cwd(), async () => {
      const sessions = [...Session.list({ roots: true, limit: args.maxCount })]

      if (sessions.length === 0) {
        return
      }

      let output: string
      if (args.format === "json") {
        output = formatSessionJSON(sessions)
      } else {
        output = formatSessionTable(sessions)
      }

      const shouldPaginate = process.stdout.isTTY && !args.maxCount && args.format === "table"

      if (shouldPaginate) {
        const proc = Process.spawn(pagerCmd(), {
          stdin: "pipe",
          stdout: "inherit",
          stderr: "inherit",
        })

        if (!proc.stdin) {
          console.log(output)
          return
        }

        proc.stdin.write(output)
        proc.stdin.end()
        await proc.exited
      } else {
        console.log(output)
      }
    })
  },
})

export const SessionLcmWatchCommand = cmd({
  command: "lcm-watch",
  describe: "watch live LCM leaves/sprigs/bindles from context snapshot + database",
  builder: (yargs: Argv) =>
    yargs
      .option("snapshot", {
        type: "string",
        describe: "path to context snapshot json",
        default: LCM_CONTEXT_SNAPSHOT_PATH,
      })
      .option("conversation-id", {
        type: "number",
        describe: "override conversation id instead of reading from snapshot",
      })
      .option("interval-ms", {
        type: "number",
        describe: "refresh interval in milliseconds",
        default: 1200,
      }),
  handler: async (args) => {
    await bootstrap(process.cwd(), async () => {
      const ready = await ensureLcmReady()
      if (!ready) {
        UI.error("LCM is not available on this platform")
        return
      }

      const snapshotPath = path.resolve(String(args.snapshot))
      const intervalMs = Math.max(250, Math.floor(Number(args.intervalMs ?? 1200)))
      let stop = false
      const onStop = () => {
        stop = true
      }

      process.on("SIGINT", onStop)
      process.on("SIGTERM", onStop)

      try {
        while (!stop) {
          const view = await buildLcmWatchView({
            snapshotPath,
            conversationIdOverride:
              typeof args.conversationId === "number" && Number.isFinite(args.conversationId)
                ? Math.floor(args.conversationId)
                : undefined,
          }).catch((error) =>
            [
              "OpenCode LCM Watch",
              "",
              `watch error: ${error instanceof Error ? error.message : String(error)}`,
              "",
              "Ctrl-C to exit.",
            ].join(EOL),
          )
          process.stdout.write("\u001bc")
          process.stdout.write(view + EOL)
          await new Promise((resolve) => setTimeout(resolve, intervalMs))
        }
      } finally {
        process.off("SIGINT", onStop)
        process.off("SIGTERM", onStop)
      }
    })
  },
})

async function buildLcmWatchView(input: { snapshotPath: string; conversationIdOverride?: number }): Promise<string> {
  const snapshot = await LcmContextSnapshot.read(input.snapshotPath)
  const conversationId = input.conversationIdOverride ?? snapshot?.conversationId
  if (!conversationId || conversationId <= 0) {
    return [
      "OpenCode LCM Watch",
      "",
      `Waiting for snapshot at ${input.snapshotPath}`,
      "No conversation id found yet. Start/continue a session with LCM enabled.",
      "",
      "Ctrl-C to exit.",
    ].join(EOL)
  }

  const rows = await LcmDb.getCurrentContextWithRefs(conversationId)
  const laneTokens = await LcmDb.getContextLaneTokenCounts(conversationId)
  const totalTokens = await LcmDb.getContextTokenCount(conversationId)

  const bindles = rows.filter((row) => {
    if (row.item_type !== "summary") return false
    return (
      LcmDb.classifySummaryForDoltLane({
        condensationOrder: row.condensation_order,
        summaryLevel: row.summary_level,
        summaryType: row.summary_type,
        kind: null,
      }) === "bindle"
    )
  })
  const sprigs = rows.filter((row) => {
    if (row.item_type !== "summary") return false
    return (
      LcmDb.classifySummaryForDoltLane({
        condensationOrder: row.condensation_order,
        summaryLevel: row.summary_level,
        summaryType: row.summary_type,
        kind: null,
      }) === "sprig"
    )
  })
  const leaves = rows.filter((row) => row.item_type === "message")
  const ghosts = (await LcmDb.getOffContextSummaries({ conversationId, summaryLevel: "bindle", limit: 20 })).filter(
    (summary) => summary.summary_type === "archive_stub",
  )

  const lines: string[] = []
  lines.push("OpenCode LCM Watch")
  lines.push(
    `conversation=${conversationId} session=${snapshot?.sessionID ?? "unknown"} snapshot=${snapshot?.writtenAt ?? "n/a"} reason=${snapshot?.reason ?? "n/a"}`,
  )
  lines.push(
    `tokens total=${totalTokens} bindles=${laneTokens.bindles} sprigs=${laneTokens.sprigs} leaves=${laneTokens.leaves}`,
  )
  lines.push(`counts bindles=${bindles.length} sprigs=${sprigs.length} leaves=${leaves.length} ghosts=${ghosts.length}`)
  lines.push("")

  lines.push("Bindles (active, in-context)")
  if (bindles.length === 0) lines.push("  (none)")
  for (const row of bindles) {
    lines.push(`  [pos ${row.position}] ${row.summary_id} tok=${row.token_count} ${singleLine(row.content, 120)}`)
  }
  lines.push("")

  lines.push("Sprigs (active, in-context)")
  if (sprigs.length === 0) lines.push("  (none)")
  for (const row of sprigs) {
    lines.push(`  [pos ${row.position}] ${row.summary_id} tok=${row.token_count} ${singleLine(row.content, 120)}`)
  }
  lines.push("")

  lines.push("Leaves (active turns, in-context)")
  if (leaves.length === 0) lines.push("  (none)")
  for (const row of leaves) {
    lines.push(
      `  [pos ${row.position}] msg=${row.message_id} role=${row.role} tok=${row.token_count} ${singleLine(row.content, 120)}`,
    )
  }
  lines.push("")

  lines.push("Ghost Cues (off-context archive stubs, most recent first)")
  if (ghosts.length === 0) lines.push("  (none)")
  for (const summary of ghosts.slice(0, 12)) {
    lines.push(`  ${summary.summary_id} tok=${summary.token_count} ${singleLine(summary.content, 120)}`)
  }
  lines.push("")
  lines.push("Ctrl-C to exit.")
  return lines.join(EOL)
}

function singleLine(value: string, maxLen: number): string {
  const oneLine = value.replace(/\s+/g, " ").trim()
  if (oneLine.length <= maxLen) return oneLine
  return `${oneLine.slice(0, maxLen - 3).trimEnd()}...`
}

function formatSessionTable(sessions: Session.Info[]): string {
  const lines: string[] = []

  const maxIdWidth = Math.max(20, ...sessions.map((s) => s.id.length))
  const maxTitleWidth = Math.max(25, ...sessions.map((s) => s.title.length))

  const header = `Session ID${" ".repeat(maxIdWidth - 10)}  Title${" ".repeat(maxTitleWidth - 5)}  Updated`
  lines.push(header)
  lines.push("─".repeat(header.length))
  for (const session of sessions) {
    const truncatedTitle = Locale.truncate(session.title, maxTitleWidth)
    const timeStr = Locale.todayTimeOrDateTime(session.time.updated)
    const line = `${session.id.padEnd(maxIdWidth)}  ${truncatedTitle.padEnd(maxTitleWidth)}  ${timeStr}`
    lines.push(line)
  }

  return lines.join(EOL)
}

function formatSessionJSON(sessions: Session.Info[]): string {
  const jsonData = sessions.map((session) => ({
    id: session.id,
    title: session.title,
    updated: session.time.updated,
    created: session.time.created,
    projectId: session.projectID,
    directory: session.directory,
  }))
  return JSON.stringify(jsonData, null, 2)
}
