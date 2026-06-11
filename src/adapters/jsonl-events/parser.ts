import { readFile } from 'node:fs/promises';
import type { ToolCallEpisode, ToolCallRecord, ToolErrorCategory, ToolInputSummary } from '../../core/types.js';

export interface JsonlRunEvent {
  seq?: number;
  timestamp?: string;
  type: string;
  sessionId?: string;
  runId?: string;
  source?: string;
  payload?: Record<string, unknown>;
}

interface MaxIterationLog {
  ts: string;
  session: string;
  message: string;
  iterations?: string;
}

export async function parseJsonlEventFiles(options: {
  eventsPath: string;
  errorsPath?: string;
  harness?: string;
}): Promise<ToolCallEpisode[]> {
  const eventsText = await readFile(options.eventsPath, 'utf8');
  const events = parseJsonlRunEvents(eventsText);
  const maxIterationLogs = options.errorsPath
    ? parseMaxIterationLogs(await readFile(options.errorsPath, 'utf8'))
    : [];
  return buildEpisodesFromJsonlEvents(events, maxIterationLogs, {
    harness: options.harness ?? 'jsonl-events',
  });
}

export function parseJsonlRunEvents(text: string): JsonlRunEvent[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as JsonlRunEvent);
}

export function parseMaxIterationLogs(text: string): MaxIterationLog[] {
  const logs: MaxIterationLog[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const maybeJson = safeJson(trimmed);
    const message = typeof maybeJson?.message === 'string' ? maybeJson.message.trim() : trimmed;
    const embedded = safeJson(message);
    if (embedded?.event === 'max_iterations') {
      logs.push({
        ts: String(embedded.ts ?? ''),
        session: String(embedded.session ?? ''),
        message: String(embedded.message ?? ''),
        iterations: embedded.iterations == null ? undefined : String(embedded.iterations),
      });
    }
  }
  return logs;
}

export function buildEpisodesFromJsonlEvents(
  events: JsonlRunEvent[],
  maxIterationLogs: MaxIterationLog[] = [],
  options: { harness?: string } = {},
): ToolCallEpisode[] {
  const byRun = new Map<string, JsonlRunEvent[]>();
  for (const event of events) {
    if (!event.runId) continue;
    if (!byRun.has(event.runId)) byRun.set(event.runId, []);
    byRun.get(event.runId)?.push(event);
  }

  const episodes: ToolCallEpisode[] = [];
  for (const [runId, runEvents] of byRun) {
    runEvents.sort((a, b) => String(a.timestamp ?? '').localeCompare(String(b.timestamp ?? '')));
    const message = runEvents.find((event) => event.type === 'message_received');
    if (typeof message?.payload?.text !== 'string') continue;

    const runStarted = runEvents.find((event) => event.type === 'run_started') ?? runEvents[0];
    const terminal = [...runEvents].reverse().find((event) => event.type === 'run_finished' || event.type === 'run_error');
    const toolStarts = runEvents.filter((event) => event.type === 'tool_call_started');
    const toolFinishes = runEvents.filter((event) => event.type === 'tool_call_finished');
    const runError = runEvents.find((event) => event.type === 'run_error');
    const shadowPlan = runEvents.find((event) => event.type === 'loop_pilot_shadow_plan');
    const responseComplete = [...runEvents].reverse().find((event) => event.type === 'response_complete');
    const task = message.payload.text;
    const sessionId = message.sessionId ?? runStarted?.sessionId ?? null;
    const matchedMaxIteration = findMaxIterationMatch(task, sessionId, maxIterationLogs);
    const toolCalls = buildToolCalls(toolStarts, toolFinishes);
    const toolErrors = toolCalls
      .filter((call) => call.isError)
      .map((call) => call.tool);
    const toolErrorCategories = toolCalls
      .filter((call) => call.isError && call.errorCategory)
      .map((call) => call.errorCategory as ToolErrorCategory);

    const failureLabels = [
      ...(runError ? ['run_error'] : []),
      ...(matchedMaxIteration ? ['max_iterations'] : []),
      ...toolCalls
        .filter((call) => call.isError)
        .map((call) => `tool_error:${call.tool}${call.errorCategory ? `:${call.errorCategory}` : ''}`),
    ];

    episodes.push({
      episodeId: runId,
      harness: options.harness ?? 'jsonl-events',
      task,
      source: message.source ?? runStarted?.source ?? 'unknown',
      sessionId,
      startedAt: runStarted?.timestamp ?? null,
      finishedAt: terminal?.timestamp ?? null,
      durationMs: durationMs(runStarted?.timestamp, terminal?.timestamp),
      toolsUsed: toolCalls.map((call) => call.tool),
      toolCalls,
      toolCallCount: toolStarts.length,
      toolErrors,
      toolErrorCategories,
      outcome: runError || matchedMaxIteration ? 'error' : terminal?.type === 'run_finished' ? 'success' : 'unknown',
      hitMaxIterations: Boolean(matchedMaxIteration),
      neededContinuation: /\b(continue|keep going|go on)\b/i.test(task),
      failureLabels,
      rawSource: {
        runId,
        eventCount: runEvents.length,
        maxIterationLog: matchedMaxIteration ?? null,
        loopPilotShadowPlan: shadowPlan?.payload ?? null,
        responseCharCount: typeof responseComplete?.payload?.text === 'string'
          ? responseComplete.payload.text.length
          : null,
      },
    });
  }

  return episodes.sort((a, b) => String(a.startedAt ?? '').localeCompare(String(b.startedAt ?? '')));
}

export function buildToolCalls(toolStarts: JsonlRunEvent[], toolFinishes: JsonlRunEvent[]): ToolCallRecord[] {
  const remainingFinishes = [...toolFinishes];

  return toolStarts.map((start, index) => {
    const id = readToolUseId(start);
    let finishIndex = -1;
    if (id) {
      finishIndex = remainingFinishes.findIndex((event) => readToolUseId(event) === id);
    }
    if (finishIndex < 0) {
      const startTool = String(start.payload?.tool ?? 'unknown');
      finishIndex = remainingFinishes.findIndex((event) => String(event.payload?.tool ?? 'unknown') === startTool);
    }

    const finish = finishIndex >= 0 ? remainingFinishes.splice(finishIndex, 1)[0] : null;
    const payload = finish?.payload ?? {};
    const startedAt = start.timestamp ?? null;
    const finishedAt = finish?.timestamp ?? null;
    const computedDuration = durationMs(startedAt ?? undefined, finishedAt ?? undefined);
    const payloadDuration = typeof payload.durationMs === 'number' ? payload.durationMs : null;
    const errorCategory = normalizeErrorCategory(payload.errorCategory);
    const errorMessage = typeof payload.errorMessage === 'string'
      ? truncate(payload.errorMessage, 200)
      : typeof payload.error === 'string'
        ? truncate(payload.error, 200)
        : null;

    return {
      index,
      id: id ?? readToolUseId(finish) ?? null,
      tool: String(start.payload?.tool ?? finish?.payload?.tool ?? 'unknown'),
      startedAt,
      finishedAt,
      durationMs: payloadDuration ?? computedDuration,
      isError: payload.isError === true || payload.error != null || payload.errorMessage != null,
      errorCategory,
      errorMessage,
      inputSummary: readInputSummary(start.payload),
      iteration: normalizeIteration(payload.iteration ?? start.payload?.iteration),
    };
  });
}

function findMaxIterationMatch(task: string, sessionId: string | null, logs: MaxIterationLog[]): MaxIterationLog | null {
  const normalizedTask = normalize(task);
  for (const log of logs) {
    if (sessionId && log.session && sessionId !== log.session) continue;
    const normalizedMessage = normalize(log.message);
    if (!normalizedMessage) continue;
    if (normalizedTask.startsWith(normalizedMessage) || normalizedMessage.startsWith(normalizedTask.slice(0, 80))) {
      return log;
    }
  }
  return null;
}

function durationMs(start?: string, end?: string): number | null {
  if (!start || !end) return null;
  const value = new Date(end).getTime() - new Date(start).getTime();
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function readToolUseId(event: JsonlRunEvent | null | undefined): string | null {
  const id = event?.payload?.toolUseId ?? event?.payload?.tool_use_id ?? event?.payload?.id;
  return typeof id === 'string' && id ? id : null;
}

function readInputSummary(payload: Record<string, unknown> | undefined): ToolInputSummary | null {
  const existing = payload?.inputSummary;
  if (existing && typeof existing === 'object' && !Array.isArray(existing)) {
    const summary = existing as Record<string, unknown>;
    return {
      kind: typeof summary.kind === 'string' ? summary.kind : 'unknown',
      charCount: typeof summary.charCount === 'number' ? summary.charCount : 0,
      preview: typeof summary.preview === 'string' ? truncate(summary.preview, 50) : '',
      commandLength: typeof summary.commandLength === 'number' ? summary.commandLength : undefined,
      commandPreview: typeof summary.commandPreview === 'string' ? truncate(summary.commandPreview, 50) : undefined,
    };
  }

  const input = payload?.input;
  if (input == null) return null;
  if (typeof input === 'string') {
    return {
      kind: 'string',
      charCount: input.length,
      preview: truncate(input, 50),
    };
  }
  if (typeof input === 'object' && !Array.isArray(input)) {
    const inputRecord = input as Record<string, unknown>;
    const command = inputRecord.command;
    const serialized = safeStringify(inputRecord);
    return {
      kind: 'object',
      charCount: serialized.length,
      preview: truncate(serialized, 50),
      commandLength: typeof command === 'string' ? command.length : undefined,
      commandPreview: typeof command === 'string' ? truncate(command, 50) : undefined,
    };
  }
  const serialized = safeStringify(input);
  return {
    kind: Array.isArray(input) ? 'array' : typeof input,
    charCount: serialized.length,
    preview: truncate(serialized, 50),
  };
}

function normalizeErrorCategory(value: unknown): ToolErrorCategory | null {
  if (
    value === 'timeout' ||
    value === 'command_error' ||
    value === 'network' ||
    value === 'corruption' ||
    value === 'context_exhaustion' ||
    value === 'unknown'
  ) {
    return value;
  }
  return null;
}

function normalizeIteration(value: unknown): string | number | null {
  return typeof value === 'string' || typeof value === 'number' ? value : null;
}

function truncate(value: string, max: number): string {
  return value.length <= max ? value : value.slice(0, max);
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

function normalize(value: string): string {
  return value.toLowerCase().replace(/\s+/g, ' ').trim();
}

function safeJson(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, unknown> : null;
  } catch {
    return null;
  }
}
