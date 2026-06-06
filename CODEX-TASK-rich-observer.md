# Task: Enhance Loop Pilot Observer to Collect Rich Episode Schema

## Context

Loop Pilot is transitioning from a KNN-based prediction system to a data collection phase for training a neural network. The observer currently collects basic prediction-vs-actual data. We need to **enrich the episode schema** to capture all available context signals that a future neural net can learn from.

**Principle:** Log greedily. We don't know which variables the net will find useful. Storage is cheap. Missing signals are expensive.

## Current State

- Observer: `src/core/observer.ts` — tails `events.jsonl`, makes predictions, records outcomes
- Events source: Cal Gateway writes to `events.jsonl` with these event types:
  - `message_received` (has: `runId`, `sessionId`, `source`, `payload.text`)
  - `run_started` (has: `runId`, `sessionId`, `source`)
  - `tool_call_started` (has: `runId`, `payload.tool`, `payload.input`)
  - `tool_call_finished` (has: `runId`, `payload.tool`, `payload.error?`)
  - `run_finished` (has: `runId`, `sessionId`)
  - `run_error` (has: `runId`, `payload.error`)
  - `response_complete` (has: `runId`, `payload.text`)
  - `status_changed` (has: `runId`, `payload.state`)
  - `steer_ack`, `proactive`, `lex`, `vec`, `hyde` (less common)

- Output: `data/shadow-observations.jsonl` — currently writes `prediction` and `outcome` records

## What Needs to Change

### 1. Enrich the outcome record with ALL derivable context signals

When a run finishes, the observer has access to all events for that run. Extract these additional fields:

```typescript
interface RichEpisode {
  // === Already captured ===
  runId: string;
  sessionId: string;
  source: string;  // "pwa" | "tmux" | "scheduled" | "api"
  task: string;    // from message_received payload.text
  
  // === Outcome (already partially captured) ===
  actualToolCalls: number;
  actualToolChain: string[];  // ordered list of tool names used
  hitMaxIterations: boolean;  // toolCalls >= 10
  durationMs: number;         // run_finished.timestamp - run_started.timestamp
  outcome: "success" | "failure" | "partial";
  
  // === NEW: Context signals to add ===
  
  // Temporal
  timeOfDay: number;          // decimal hours UTC (e.g., 17.5 = 5:30 PM)
  dayOfWeek: number;          // 1=Mon ... 7=Sun (from timestamp)
  
  // Task characteristics (derived from task string)
  taskWordCount: number;
  taskCharCount: number;
  taskQuestionMarks: number;  // number of ? in task
  taskHasCodeBlock: boolean;  // contains ``` 
  taskHasUrl: boolean;        // contains http:// or https://
  isScheduledJob: boolean;    // task starts with "[Scheduled Job:"
  
  // Session context (derived from tracking runs within same sessionId)
  sessionRunIndex: number;    // which run is this within the session (1st, 2nd, 5th...)
  previousRunToolCount: number | null;   // tool count of the previous run in same session
  previousRunOutcome: string | null;     // outcome of previous run in same session
  previousRunDurationMs: number | null;  // duration of previous run in same session
  
  // Tool details
  uniqueToolsUsed: string[];  // deduplicated tool list
  uniqueToolCount: number;    // number of distinct tools
  toolErrors: number;         // count of tool_call_finished with error
  toolErrorNames: string[];   // which tools errored
  
  // Response characteristics
  responseCharCount: number | null;  // length of response_complete payload.text
  
  // Prediction comparison (from Loop Pilot's shadow prediction)
  predictedBudget: number | null;
  predictedConfidence: string | null;  // "high" | "medium" | "low" | "none"
  predictedTools: string[] | null;
  knnTopSimilarity: number | null;     // similarity score of best neighbor
  knnNeighborCount: number | null;     // how many neighbors were found
  
  // Derived labels (automatic, no human input needed)
  outcomeLabel: "efficient" | "wasteful" | "hit-cap" | "error";
  // efficient: success + didn't hit max iterations + toolCalls <= predictedBudget+2
  // wasteful: success but toolCalls > predictedBudget+3 (or > 2x average for similar tasks)
  // hit-cap: hitMaxIterations = true
  // error: outcome = "failure"
}
```

### 2. Track session state across runs

The observer needs to maintain a per-session memory of previous runs to populate `sessionRunIndex`, `previousRunToolCount`, `previousRunOutcome`, `previousRunDurationMs`.

Add a `Map<sessionId, PreviousRunInfo[]>` that stores the last N runs per session. When a new run completes, look up what came before it.

### 3. Update the output format

The `outcome` record in `shadow-observations.jsonl` should include ALL the rich fields above. Keep backward compatibility — add fields, don't remove existing ones.

### 4. Add an `experiment` version bump

Change experiment identifier from `loop_pilot_standalone_shadow_v1` to `loop_pilot_standalone_shadow_v2` so we can distinguish old vs new schema data.

### 5. Add a summary stats command

Add a CLI command or method: `looppilot observe --stats` that reads `shadow-observations.jsonl` and prints:
- Total episodes collected (v2 schema)
- Date range
- Task type distribution (scheduled vs interactive)
- Average tool calls
- Hit-cap rate
- Prediction accuracy (MAE) if predictions exist
- Unique tools seen

## Files to Modify

- `src/core/observer.ts` — main changes here
- `src/core/types.ts` — add RichEpisode interface
- `src/cli/` — add stats command if CLI exists
- Tests — update/add tests for new fields

## What NOT to Change

- Don't modify the prediction logic (KNN stays as-is for now)
- Don't modify how events.jsonl is read (the tail/poll mechanism works fine)
- Don't change the embedding bridge or SQLite store
- Don't add any prompt injection — we're in observation-only mode

## Acceptance Criteria

1. Observer captures all fields in the RichEpisode schema above
2. Session tracking works (previousRun* fields populated correctly)
3. Automatic labeling works (outcomeLabel derived correctly)
4. Old `shadow-observations.jsonl` data still readable (backward compatible)
5. New records have `experiment: "loop_pilot_standalone_shadow_v2"`
6. Stats command shows useful summary of collected data
7. All existing tests still pass
8. New tests cover the enriched fields

## Reference Files

- Current observer: `src/core/observer.ts`
- Types: `src/core/types.ts`  
- Events parser: `src/adapters/jsonl-events/parser.ts`
- Cal Gateway events format: Each event is `{seq, timestamp, type, sessionId, runId, source, payload?}`
