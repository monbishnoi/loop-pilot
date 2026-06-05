import { readFile } from 'node:fs/promises';
import type { ToolCallEpisode } from '../../core/types.js';

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
    const task = message.payload.text;
    const sessionId = message.sessionId ?? runStarted?.sessionId ?? null;
    const matchedMaxIteration = findMaxIterationMatch(task, sessionId, maxIterationLogs);
    const toolErrors = toolFinishes
      .filter((event) => event.payload?.isError === true)
      .map((event) => String(event.payload?.tool ?? 'unknown'));

    const failureLabels = [
      ...(runError ? ['run_error'] : []),
      ...(matchedMaxIteration ? ['max_iterations'] : []),
      ...toolErrors.map((tool) => `tool_error:${tool}`),
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
      toolsUsed: toolStarts.map((event) => String(event.payload?.tool ?? 'unknown')),
      toolCallCount: toolStarts.length,
      toolErrors,
      outcome: runError || matchedMaxIteration ? 'error' : terminal?.type === 'run_finished' ? 'success' : 'unknown',
      hitMaxIterations: Boolean(matchedMaxIteration),
      neededContinuation: /\b(continue|keep going|go on)\b/i.test(task),
      failureLabels,
      rawSource: {
        runId,
        eventCount: runEvents.length,
        maxIterationLog: matchedMaxIteration ?? null,
      },
    });
  }

  return episodes.sort((a, b) => String(a.startedAt ?? '').localeCompare(String(b.startedAt ?? '')));
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
