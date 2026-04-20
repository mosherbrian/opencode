# LCM Typecheck Fix — Code Review Summary

## Problem

After merging the LCM fork onto the current upstream `opencode` codebase, `bun run typecheck` produced ~40 errors in source files and ~187 in test files. The LCM code was written against an older upstream API that has since been refactored.

## Root Cause

Upstream reorganized module exports into **barrel re-exports** using `export * as X from "./y"` and migrated from **static async functions** to **Effect service pattern** (accessed via `yield* Service`). The LCM code used the old direct-import, async/await style throughout.

## Strategy

Two principles guided the fixes:

1. **Mechanical import fixes** for renamed/reorganized exports
2. **A facade bridge module** (`upstream-bridge.ts`) for the Effect service migration — LCM business logic stays in plain async/await, insulating it from future upstream Effect changes

The bridge module is the key design decision: when rebasing on upstream in the future, API changes in Effect services only require updating one file, not the entire LCM codebase.

## Changes (64 source files, +220/-212 lines, 1 new file)

### 1. New file: `src/session/lcm/upstream-bridge.ts` (170 lines)

A thin async wrapper layer that bridges LCM's async/await code to upstream's Effect services using `AppRuntime.runPromise()`. Provides:

- **Provider**: `getLanguage(model)`, `getModel(providerID, modelID)`, `getSmallModel(providerID)`
- **Session**: `sessionCreate(input)`, `sessionGet(id)`, `sessionMessages(input)`
- **SessionPrompt**: `promptCancel(sessionID)`, `promptPrompt(input)`
- **Config**: `configGet()`
- **Plugin**: `pluginTrigger(name, input, output)`
- **Task**: `executeTask(input)` — replaces the removed `executeTask` function with a simplified sub-agent delegation

### 2. Barrel import migration (~50 files)

Changed direct leaf-module imports to barrel re-exports:

| Before | After |
|--------|-------|
| `import { Log } from "@/util/log"` | `import { Log } from "@/util"` |
| `import { Token } from "@/util/token"` | `import { Token } from "@/util"` |
| `import { Archive } from "@/util/archive"` | `import { Archive } from "@/util"` |
| `import { Provider } from "@/provider/provider"` | `import { Provider } from "@/provider"` |

The barrel at `src/util/index.ts` does `export * as Log from "./log"`, etc. Same pattern for `src/provider/index.ts`.

Affected: all `src/session/lcm/` files, all `src/session/lcm/explore/` files, all `src/tool/lcm-*` files, `src/tool/agentic-map.ts`, `src/tool/llm-map.ts`.

### 3. Effect service bridge calls (~12 files)

LCM code that called upstream functions as static async methods now calls bridge equivalents:

```typescript
// Before (old upstream API):
const language = await Provider.getLanguage(model)
const session = await Session.create(input)
await Plugin.trigger(name, input, output)

// After (via bridge):
import * as Bridge from "./upstream-bridge"
const language = await Bridge.getLanguage(model)
const session = await Bridge.sessionCreate(input)
await Bridge.pluginTrigger(name, input, output)
```

### 4. Removed function replacements

| Old function | Replacement |
|-------------|-------------|
| `Provider.getLanguage(model)` | `Bridge.getLanguage(model)` |
| `Provider.getModel(pid, mid)` | `Bridge.getModel(pid, mid)` |
| `Provider.getSmallModel(pid)` | `Bridge.getSmallModel(pid)` |
| `Session.create/get/messages()` | `Bridge.sessionCreate/Get/Messages()` |
| `SessionPrompt.cancel/prompt()` | `Bridge.promptCancel/promptPrompt()` |
| `SessionPrompt.getOrCreateLcmConversation()` | `SessionPrompt.getLcmConversationId()` (exported function, no bridge needed) |
| `Config.Config` / `Config.get()` | `Bridge.configGet()` |
| `Plugin.trigger()` | `Bridge.pluginTrigger()` |
| `executeTask()` from `./task` | `Bridge.executeTask()` |
| `SystemPrompt.header()` / `buildSections()` | Removed — inlined upstream `llm.ts` pattern (empty array + push) |

### 5. Branded type fixes

Upstream introduced branded string types (`SessionID`, `MessageID`, `PartID`, `ProviderID`, `ModelID`) using `Schema.brand()`. LCM code that constructed IDs as plain strings now uses type assertions (`as any`) where the IDs are synthetic/ephemeral (e.g., LCM-internal messages that don't live in the real session store).

Files: `context.ts`, `agent-summary.ts`, `migration.ts`, `upstream-bridge.ts`.

### 6. API signature fixes

- **`z.record(z.unknown())`** → **`z.record(z.string(), z.unknown())`** in `session.ts` (Zod now requires explicit key schema)
- **`ManualCompactionInput`** — made `user`, `model`, `overhead`, `reserve`, `contextWindow` optional with early-return guards in `context.ts`
- **`Tool.define`** — changed `satisfies Tool.DefWithoutID<...>` to `as Tool.DefWithoutID<...>` for `Effect.succeed({...})` to satisfy the Init type constraint
- **`Log.Default.clone()`** → **`Log.create()`** (upstream removed `Default.clone()`)
- **`@opencode-ai/util/error`** → **`@opencode-ai/shared/util/error`** (module moved to shared package)
- **`ProviderTransform`** import fixed to use barrel `../provider` instead of `../provider/transform`
- **`flatmap` callback** in `prompt.ts` — added explicit return type annotation to help TypeScript unify the complex union
- Added `ajv` dependency to `package.json` (needed by `map-shared.ts`)

### 7. Test files (NOT fixed — 14 test files, 187 errors)

The test files under `test/session/lcm/` and `test/tool/` have the same categories of issues (barrel imports, branded types, removed functions, changed Tool.define signatures). These are deferred to a follow-up since they don't block building.

## Verification

```
$ bun run typecheck
# Zero errors in source code
# 187 errors in test/ files (deferred)
```

## Future Maintenance

When rebasing on upstream:
1. **Barrel imports** should generally survive rebases unchanged (the barrel pattern is stable)
2. **Effect API changes** only require updating `upstream-bridge.ts`
3. **Test files** should be updated with the same patterns applied to source
