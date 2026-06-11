import { readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { buildToolCalls, parseJsonlRunEvents, type JsonlRunEvent } from '../adapters/jsonl-events/parser.js';
import type { ToolCallRecord, ToolErrorCategory } from './types.js';

export interface BackfillOptions {
  observationsPath: string;
  eventsPath: string;
  errorsPath?: string;
  outputPath: string;
}

export interface BackfillResult {
  inputRecords: number;
  outputRecords: number;
  outcomeRecords: number;
  matchedOutcomeRecords: number;
  enrichedToolCalls: number;
  categorizedToolErrors: number;
  outputPath: string;
}

interface StructuredErrorLog {
  ts: string;
  event: string;
  session: string;
  tool?: string;
  error?: string;
  message?: string;
}

interface ToolErrorLog {
  ts: string;
  session: string;
  error: string;
}

export async function backfillShadowObservations(options: BackfillOptions): Promise<BackfillResult> {
  const observationLines = (await readFile(options.observationsPath, 'utf8'))
    .split('\n')
    .filter((line) => line.trim());
  const observations = observationLines.map((line) => JSON.parse(line) as Record<string, unknown>);
  const events = parseJsonlRunEvents(await readFile(options.eventsPath, 'utf8'));
  const eventsByRun = groupEventsByRun(events);
  const errorLogs = options.errorsPath ? parsePm2ErrorLogs(await readFile(options.errorsPath, 'utf8')) : {
    structured: [],
    toolErrors: [],
  };

  let outcomeRecords = 0;
  let matchedOutcomeRecords = 0;
  let enrichedToolCalls = 0;
  let categorizedToolErrors = 0;

  const enriched = observations.map((record) => {
    if (record.type !== 'outcome' || typeof record.runId !== 'string') return record;
    outcomeRecords++;

    const runEvents = eventsByRun.get(record.runId);
    if (!runEvents) return record;
    matchedOutcomeRecords++;

    const sortedEvents = [...runEvents].sort((a, b) => String(a.timestamp ?? '').localeCompare(String(b.timestamp ?? '')));
    const toolStarts = sortedEvents.filter((event) => event.type === 'tool_call_started');
    const toolFinishes = sortedEvents.filter((event) => event.type === 'tool_call_finished');
    const runStarted = sortedEvents.find((event) => event.type === 'run_started') ?? sortedEvents[0];
    const terminal = [...sortedEvents].reverse().find((event) => event.type === 'run_finished' || event.type === 'run_error');
    const sessionId = String(record.sessionId ?? runStarted?.sessionId ?? '');

    const toolCalls = enrichToolCallsWithLogs(buildToolCalls(toolStarts, toolFinishes), sessionId, errorLogs);
    enrichedToolCalls += toolCalls.filter((call) => call.durationMs != null).length;
    categorizedToolErrors += toolCalls.filter((call) => call.isError && call.errorCategory).length;

    const toolErrorRecords = toolCalls.filter((call) => call.isError);
    const toolErrorNames = toolErrorRecords.map((call) => call.tool);
    const toolErrorCategories = toolErrorRecords
      .map((call) => call.errorCategory)
      .filter((category): category is ToolErrorCategory => Boolean(category));

    const actual = isRecord(record.actual) ? record.actual : {};
    return {
      ...record,
      actualToolCalls: typeof record.actualToolCalls === 'number' ? record.actualToolCalls : toolCalls.length,
      actualToolChain: Array.isArray(record.actualToolChain) ? record.actualToolChain : toolCalls.map((call) => call.tool),
      toolCalls,
      toolErrors: toolErrorNames.length,
      toolErrorNames,
      toolErrorCategories,
      durationMs: typeof record.durationMs === 'number'
        ? record.durationMs
        : durationMs(runStarted?.timestamp, terminal?.timestamp),
      actual: {
        ...actual,
        toolCalls,
        toolErrorCategories,
      },
    };
  });

  await mkdir(dirname(options.outputPath), { recursive: true });
  await writeFile(options.outputPath, enriched.map((record) => JSON.stringify(record)).join('\n') + '\n', 'utf8');

  return {
    inputRecords: observations.length,
    outputRecords: enriched.length,
    outcomeRecords,
    matchedOutcomeRecords,
    enrichedToolCalls,
    categorizedToolErrors,
    outputPath: options.outputPath,
  };
}

export function parsePm2ErrorLogs(text: string): { structured: StructuredErrorLog[]; toolErrors: ToolErrorLog[] } {
  const structured: StructuredErrorLog[] = [];
  const toolErrors: ToolErrorLog[] = [];

  for (const line of text.split('\n')) {
    const outer = safeJson(line.trim());
    if (!outer) continue;
    const timestamp = typeof outer.timestamp === 'string' ? outer.timestamp : '';
    const message = typeof outer.message === 'string' ? outer.message : '';
    const embedded = extractStructuredError(message);
    if (embedded) {
      structured.push(embedded);
    }

    const match = message.match(/\[Session ([^\]]+)\] Tool error:\s*([\s\S]+)/);
    if (match) {
      toolErrors.push({
        ts: embedded?.ts ?? parsePm2Timestamp(timestamp),
        session: match[1],
        error: match[2].trim(),
      });
    }
  }

  return { structured, toolErrors };
}

function groupEventsByRun(events: JsonlRunEvent[]): Map<string, JsonlRunEvent[]> {
  const byRun = new Map<string, JsonlRunEvent[]>();
  for (const event of events) {
    if (!event.runId) continue;
    if (!byRun.has(event.runId)) byRun.set(event.runId, []);
    byRun.get(event.runId)?.push(event);
  }
  return byRun;
}

function enrichToolCallsWithLogs(
  toolCalls: ToolCallRecord[],
  sessionId: string,
  logs: { structured: StructuredErrorLog[]; toolErrors: ToolErrorLog[] },
): ToolCallRecord[] {
  return toolCalls.map((call) => {
    if (!call.isError) return call;

    const structured = findStructuredToolError(call, sessionId, logs.structured);
    const unstructured = findToolErrorLog(call, sessionId, logs.toolErrors);
    const errorMessage = call.errorMessage ?? structured?.error ?? unstructured?.error ?? null;
    const category = call.errorCategory ?? categoryFromStructuredEvent(structured?.event) ?? categorizeError(errorMessage);

    return {
      ...call,
      errorCategory: category,
      errorMessage: errorMessage ? truncate(errorMessage, 200) : null,
    };
  });
}

function findStructuredToolError(
  call: ToolCallRecord,
  sessionId: string,
  logs: StructuredErrorLog[],
): StructuredErrorLog | null {
  const finishedMs = call.finishedAt ? new Date(call.finishedAt).getTime() : NaN;
  return logs.find((log) => {
    if (log.session !== sessionId) return false;
    if (log.tool && log.tool !== call.tool) return false;
    if (!Number.isFinite(finishedMs)) return false;
    const logMs = new Date(log.ts).getTime();
    return Number.isFinite(logMs) && Math.abs(logMs - finishedMs) <= 5 * 60 * 1000;
  }) ?? null;
}

function findToolErrorLog(
  call: ToolCallRecord,
  sessionId: string,
  logs: ToolErrorLog[],
): ToolErrorLog | null {
  const finishedMs = call.finishedAt ? new Date(call.finishedAt).getTime() : NaN;
  if (!Number.isFinite(finishedMs)) return null;
  return logs.find((log) => {
    if (log.session !== sessionId) return false;
    const logMs = new Date(log.ts).getTime();
    return Number.isFinite(logMs) && Math.abs(logMs - finishedMs) <= 2_000;
  }) ?? null;
}

function categoryFromStructuredEvent(event: string | undefined): ToolErrorCategory | null {
  if (event === 'tool_timeout') return 'timeout';
  if (event === 'session_corruption') return 'corruption';
  if (event === 'context_exhausted') return 'context_exhaustion';
  return null;
}

function categorizeError(message: string | null): ToolErrorCategory {
  if (!message) return 'unknown';
  const value = message.toLowerCase();
  if (/\b(etimedout|timed out|timeout|request timed out)\b/.test(value)) return 'timeout';
  if (/\b(enotfound|econnreset|econnrefused|ehostunreach|network|dns|socket hang up|fetch failed)\b/.test(value)) return 'network';
  if (value.includes('context_length') || value.includes('context window') || value.includes('too many tokens') || value.includes('request too large') || value.includes('maximum context length')) {
    return 'context_exhaustion';
  }
  if (value.includes('tool_use') || value.includes('tool_result') || value.includes('tool_use_id') || value.includes('orphaned')) {
    return 'corruption';
  }
  if (/\b(exit code|enoent|eacces|eperm|permission denied|file not found|text not found|syntax error|command failed|spawn)\b/.test(value)) {
    return 'command_error';
  }
  return 'unknown';
}

function extractStructuredError(message: string): StructuredErrorLog | null {
  const match = message.match(/\{"ts":"[^"]+"[\s\S]*\}/);
  if (!match) return null;
  const parsed = safeJson(match[0]);
  if (!parsed || typeof parsed.event !== 'string' || typeof parsed.ts !== 'string') return null;
  return {
    ts: parsed.ts,
    event: parsed.event,
    session: typeof parsed.session === 'string' ? parsed.session : 'unknown',
    tool: typeof parsed.tool === 'string' ? parsed.tool : undefined,
    error: typeof parsed.error === 'string' ? parsed.error : undefined,
    message: typeof parsed.message === 'string' ? parsed.message : undefined,
  };
}

function parsePm2Timestamp(value: string): string {
  const timestamp = new Date(value);
  return Number.isFinite(timestamp.getTime()) ? timestamp.toISOString() : '';
}

function durationMs(start?: string, end?: string): number | null {
  if (!start || !end) return null;
  const value = new Date(end).getTime() - new Date(start).getTime();
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function safeJson(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function truncate(value: string, max: number): string {
  return value.length <= max ? value : value.slice(0, max);
}
