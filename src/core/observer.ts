import { appendFile, mkdir, open, readFile, stat } from 'node:fs/promises';
import { dirname } from 'node:path';
import type { LoopPilot } from './loop-pilot.js';
import type { BudgetPrediction, RichEpisode, RichEpisodeOutcome } from './types.js';
import { buildEpisodesFromJsonlEvents, type JsonlRunEvent } from '../adapters/jsonl-events/parser.js';

const SHADOW_EXPERIMENT_V2 = 'loop_pilot_standalone_shadow_v2';
const SESSION_HISTORY_LIMIT = 50;

export interface ObserverOptions {
  eventsPath: string;
  outputPath: string;
  pollMs?: number;
  fromStart?: boolean;
  harness?: string;
  maxBudget?: number;
}

export class LoopPilotObserver {
  private readonly loopPilot: LoopPilot;
  private readonly options: Required<Pick<ObserverOptions, 'outputPath' | 'harness'>> & ObserverOptions;
  private readonly eventsByRun = new Map<string, JsonlRunEvent[]>();
  private readonly predictionsByRun = new Map<string, ReturnType<typeof summarizePrediction>>();
  private readonly sessionRuns = new Map<string, PreviousRunInfo[]>();
  private readonly plannedRuns = new Set<string>();
  private readonly completedRuns = new Set<string>();

  constructor(loopPilot: LoopPilot, options: ObserverOptions) {
    this.loopPilot = loopPilot;
    this.options = {
      ...options,
      outputPath: options.outputPath,
      harness: options.harness ?? 'jsonl-events',
    };
  }

  async processEvent(event: JsonlRunEvent): Promise<void> {
    if (!event.runId) return;

    const runEvents = this.eventsByRun.get(event.runId) ?? [];
    runEvents.push(event);
    this.eventsByRun.set(event.runId, runEvents);

    if (event.type === 'message_received') {
      await this.recordPrediction(event);
    }

    if (event.type === 'run_finished' || event.type === 'run_error') {
      await this.recordOutcome(event.runId);
    }
  }

  private async recordPrediction(event: JsonlRunEvent): Promise<void> {
    if (!event.runId || this.plannedRuns.has(event.runId)) return;
    if (typeof event.payload?.text !== 'string') return;

    this.plannedRuns.add(event.runId);
    const plan = await this.loopPilot.plan({
      task: event.payload.text,
      maxBudget: this.options.maxBudget,
    });
    const prediction = summarizePrediction(plan.prediction);
    this.predictionsByRun.set(event.runId, prediction);

    await appendJsonl(this.options.outputPath, {
      type: 'prediction',
      experiment: SHADOW_EXPERIMENT_V2,
      timestamp: new Date().toISOString(),
      runId: event.runId,
      sessionId: event.sessionId ?? null,
      source: event.source ?? 'unknown',
      injected: false,
      task: event.payload.text,
      prediction,
      promptGuidanceAvailable: Boolean(plan.promptGuidance),
      promptGuidance: plan.promptGuidance ? plan.promptGuidance.slice(0, 2000) : '',
      similarEpisodes: plan.similarEpisodes.slice(0, 5).map((item) => ({
        similarity: item.similarity,
        task: item.episode.task,
        outcome: item.episode.outcome,
        toolCallCount: item.episode.toolCallCount,
        toolsUsed: item.episode.toolsUsed,
        hitMaxIterations: item.episode.hitMaxIterations,
      })),
    });
  }

  private async recordOutcome(runId: string): Promise<void> {
    if (this.completedRuns.has(runId)) return;

    const runEvents = this.eventsByRun.get(runId) ?? [];
    const episodes = buildEpisodesFromJsonlEvents(runEvents, [], { harness: this.options.harness });
    const episode = episodes[0];
    if (!episode) return;
    this.completedRuns.add(runId);

    await this.loopPilot.importEpisodes([episode]);
    await this.loopPilot.indexEpisodes();

    const prediction = this.predictionsByRun.get(runId) ?? null;
    const richEpisode = buildRichEpisode(runId, runEvents, episode, prediction, this.sessionRuns);
    rememberSessionRun(this.sessionRuns, richEpisode);

    await appendJsonl(this.options.outputPath, {
      type: 'outcome',
      experiment: SHADOW_EXPERIMENT_V2,
      timestamp: new Date().toISOString(),
      injected: false,
      ...richEpisode,
      actual: {
        toolCallCount: episode.toolCallCount,
        toolsUsed: episode.toolsUsed,
        outcome: episode.outcome,
        hitMaxIterations: episode.hitMaxIterations,
        durationMs: episode.durationMs,
        failureLabels: episode.failureLabels,
        responseCharCount: episode.rawSource?.responseCharCount ?? null,
      },
      prediction,
      reward: prediction ? scorePrediction(prediction, episode.toolCallCount, episode.outcome, episode.hitMaxIterations) : null,
    });
  }
}

interface PreviousRunInfo {
  toolCount: number;
  outcome: string;
  durationMs: number;
}

export interface ShadowObservationStats {
  totalEpisodes: number;
  dateRange: {
    start: string | null;
    end: string | null;
  };
  taskTypeDistribution: {
    scheduled: number;
    interactive: number;
  };
  averageToolCalls: number;
  hitCapRate: number;
  predictionMae: number | null;
  uniqueToolsSeen: string[];
}

function buildRichEpisode(
  runId: string,
  runEvents: JsonlRunEvent[],
  episode: ReturnType<typeof buildEpisodesFromJsonlEvents>[number],
  prediction: ReturnType<typeof summarizePrediction> | null,
  sessionRuns: Map<string, PreviousRunInfo[]>,
): RichEpisode {
  const sortedEvents = [...runEvents].sort((a, b) => String(a.timestamp ?? '').localeCompare(String(b.timestamp ?? '')));
  const message = sortedEvents.find((event) => event.type === 'message_received');
  const runStarted = sortedEvents.find((event) => event.type === 'run_started') ?? sortedEvents[0];
  const responseComplete = [...sortedEvents].reverse().find((event) => event.type === 'response_complete');
  const toolFinishes = sortedEvents.filter((event) => event.type === 'tool_call_finished');
  const timestamp = parseTimestamp(runStarted?.timestamp ?? message?.timestamp ?? episode.startedAt ?? new Date().toISOString());
  const sessionId = episode.sessionId ?? message?.sessionId ?? runStarted?.sessionId ?? 'unknown';
  const previousRuns = sessionRuns.get(sessionId) ?? [];
  const previousRun = previousRuns.at(-1) ?? null;
  const task = episode.task;
  const actualToolChain = episode.toolsUsed;
  const uniqueToolsUsed = [...new Set(actualToolChain)];
  const toolErrorNames = toolFinishes
    .filter(hasToolError)
    .map((event) => String(event.payload?.tool ?? 'unknown'));
  const outcome = normalizeOutcome(episode.outcome);
  const predictedBudget = prediction?.suggestedBudget ?? null;

  return {
    runId,
    sessionId,
    source: episode.source,
    task,
    actualToolCalls: episode.toolCallCount,
    actualToolChain,
    hitMaxIterations: episode.hitMaxIterations || episode.toolCallCount >= 10,
    durationMs: episode.durationMs ?? 0,
    outcome,
    timeOfDay: timestamp.getUTCHours() + timestamp.getUTCMinutes() / 60 + timestamp.getUTCSeconds() / 3600,
    dayOfWeek: dayOfWeekUtc(timestamp),
    taskWordCount: wordCount(task),
    taskCharCount: task.length,
    taskQuestionMarks: (task.match(/\?/g) ?? []).length,
    taskHasCodeBlock: task.includes('```'),
    taskHasUrl: /https?:\/\//i.test(task),
    isScheduledJob: task.startsWith('[Scheduled Job:'),
    sessionRunIndex: previousRuns.length + 1,
    previousRunToolCount: previousRun?.toolCount ?? null,
    previousRunOutcome: previousRun?.outcome ?? null,
    previousRunDurationMs: previousRun?.durationMs ?? null,
    uniqueToolsUsed,
    uniqueToolCount: uniqueToolsUsed.length,
    toolErrors: toolErrorNames.length,
    toolErrorNames,
    responseCharCount: typeof responseComplete?.payload?.text === 'string' ? responseComplete.payload.text.length : null,
    predictedBudget,
    predictedConfidence: prediction?.confidence ?? null,
    predictedTools: prediction?.likelyTools ?? null,
    knnTopSimilarity: prediction?.evidence.topSimilarity ?? null,
    knnNeighborCount: prediction?.neighborCount ?? null,
    outcomeLabel: labelOutcome(outcome, episode.toolCallCount, episode.hitMaxIterations || episode.toolCallCount >= 10, predictedBudget),
  };
}

function rememberSessionRun(sessionRuns: Map<string, PreviousRunInfo[]>, episode: RichEpisode): void {
  const runs = sessionRuns.get(episode.sessionId) ?? [];
  runs.push({
    toolCount: episode.actualToolCalls,
    outcome: episode.outcome,
    durationMs: episode.durationMs,
  });
  sessionRuns.set(episode.sessionId, runs.slice(-SESSION_HISTORY_LIMIT));
}

export async function collectShadowObservationStats(path: string): Promise<ShadowObservationStats> {
  let text = '';
  try {
    text = await readFile(path, 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return emptyStats();
    }
    throw error;
  }

  const records = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>)
    .filter((record) => record.type === 'outcome' && record.experiment === SHADOW_EXPERIMENT_V2);

  if (records.length === 0) return emptyStats();

  const timestamps = records
    .map((record) => typeof record.timestamp === 'string' ? record.timestamp : null)
    .filter((value): value is string => Boolean(value))
    .sort();
  const toolCounts = records.map(readActualToolCalls).filter((value): value is number => value != null);
  const predictionErrors = records
    .map((record) => {
      const actual = readActualToolCalls(record);
      const predicted = typeof record.predictedBudget === 'number' ? record.predictedBudget : null;
      return actual != null && predicted != null ? Math.abs(actual - predicted) : null;
    })
    .filter((value): value is number => value != null);
  const uniqueToolsSeen = [...new Set(records.flatMap(readUniqueTools))].sort();
  const hitCaps = records.filter((record) => record.hitMaxIterations === true).length;
  const scheduled = records.filter((record) => record.isScheduledJob === true).length;

  return {
    totalEpisodes: records.length,
    dateRange: {
      start: timestamps[0] ?? null,
      end: timestamps.at(-1) ?? null,
    },
    taskTypeDistribution: {
      scheduled,
      interactive: records.length - scheduled,
    },
    averageToolCalls: average(toolCounts),
    hitCapRate: records.length === 0 ? 0 : hitCaps / records.length,
    predictionMae: predictionErrors.length === 0 ? null : average(predictionErrors),
    uniqueToolsSeen,
  };
}

export function formatShadowObservationStats(stats: ShadowObservationStats): string {
  return [
    'Loop Pilot observation stats',
    '',
    `Total episodes collected (v2): ${stats.totalEpisodes}`,
    `Date range: ${stats.dateRange.start ?? 'n/a'} to ${stats.dateRange.end ?? 'n/a'}`,
    `Task type distribution: ${stats.taskTypeDistribution.scheduled} scheduled, ${stats.taskTypeDistribution.interactive} interactive`,
    `Average tool calls: ${formatNumber(stats.averageToolCalls)}`,
    `Hit-cap rate: ${formatPercent(stats.hitCapRate)}`,
    `Prediction accuracy (MAE): ${stats.predictionMae == null ? 'n/a' : formatNumber(stats.predictionMae)}`,
    `Unique tools seen: ${stats.uniqueToolsSeen.length === 0 ? 'none' : stats.uniqueToolsSeen.join(', ')}`,
  ].join('\n');
}

export async function observeEventLog(loopPilot: LoopPilot, options: ObserverOptions): Promise<void> {
  const observer = new LoopPilotObserver(loopPilot, options);
  const pollMs = options.pollMs ?? 2000;
  let offset = options.fromStart ? 0 : await fileSize(options.eventsPath);
  let stopped = false;

  const stop = () => {
    stopped = true;
  };
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);

  while (!stopped) {
    const next = await readNewBytes(options.eventsPath, offset);
    offset = next.offset;
    for (const event of parseEventLines(next.text)) {
      await observer.processEvent(event);
    }
    await sleep(pollMs);
  }
}

function summarizePrediction(prediction: BudgetPrediction) {
  return {
    suggestedBudget: prediction.suggestedBudget,
    budgetRange: prediction.budgetRange,
    confidence: prediction.confidence,
    risk: prediction.risk,
    guidanceMode: prediction.guidanceMode,
    likelyTools: prediction.likelyTools,
    toolBudget: prediction.toolBudget,
    evidence: prediction.evidence,
    neighborCount: prediction.neighborCount,
    reason: prediction.reason,
  };
}

function scorePrediction(
  prediction: ReturnType<typeof summarizePrediction>,
  actualToolCalls: number,
  outcome: string,
  hitMaxIterations: boolean,
) {
  const underBudget = actualToolCalls > prediction.budgetRange.max;
  const overBudget = actualToolCalls < prediction.budgetRange.min;
  const withinRange = !underBudget && !overBudget;
  const success = outcome === 'success' && !hitMaxIterations;

  return {
    success,
    withinRange,
    underBudget,
    overBudget,
    missDistance: underBudget
      ? actualToolCalls - prediction.budgetRange.max
      : overBudget
        ? prediction.budgetRange.min - actualToolCalls
        : 0,
    label: success && withinRange ? 'good' : underBudget ? 'penalize_under_budget' : overBudget ? 'penalize_over_budget' : 'needs_review',
  };
}

function normalizeOutcome(outcome: string): RichEpisodeOutcome {
  if (outcome === 'success') return 'success';
  if (outcome === 'error') return 'failure';
  return 'partial';
}

function labelOutcome(
  outcome: RichEpisodeOutcome,
  actualToolCalls: number,
  hitMaxIterations: boolean,
  predictedBudget: number | null,
): RichEpisode['outcomeLabel'] {
  if (hitMaxIterations) return 'hit-cap';
  if (outcome === 'failure') return 'error';
  if (outcome === 'success' && predictedBudget != null && actualToolCalls > predictedBudget + 3) return 'wasteful';
  if (outcome === 'success' && (predictedBudget == null || actualToolCalls <= predictedBudget + 2)) return 'efficient';
  return 'wasteful';
}

function hasToolError(event: JsonlRunEvent): boolean {
  return event.payload?.isError === true || event.payload?.error != null;
}

function parseTimestamp(value: string): Date {
  const timestamp = new Date(value);
  return Number.isFinite(timestamp.getTime()) ? timestamp : new Date(0);
}

function dayOfWeekUtc(timestamp: Date): number {
  const day = timestamp.getUTCDay();
  return day === 0 ? 7 : day;
}

function wordCount(value: string): number {
  const trimmed = value.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

function readActualToolCalls(record: Record<string, unknown>): number | null {
  if (typeof record.actualToolCalls === 'number') return record.actualToolCalls;
  const actual = record.actual;
  if (actual && typeof actual === 'object' && !Array.isArray(actual)) {
    const toolCallCount = (actual as Record<string, unknown>).toolCallCount;
    if (typeof toolCallCount === 'number') return toolCallCount;
  }
  return null;
}

function readUniqueTools(record: Record<string, unknown>): string[] {
  if (Array.isArray(record.uniqueToolsUsed)) {
    return record.uniqueToolsUsed.map(String);
  }
  const actual = record.actual;
  if (actual && typeof actual === 'object' && !Array.isArray(actual)) {
    const toolsUsed = (actual as Record<string, unknown>).toolsUsed;
    if (Array.isArray(toolsUsed)) return [...new Set(toolsUsed.map(String))];
  }
  return [];
}

function average(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function emptyStats(): ShadowObservationStats {
  return {
    totalEpisodes: 0,
    dateRange: {
      start: null,
      end: null,
    },
    taskTypeDistribution: {
      scheduled: 0,
      interactive: 0,
    },
    averageToolCalls: 0,
    hitCapRate: 0,
    predictionMae: null,
    uniqueToolsSeen: [],
  };
}

async function appendJsonl(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await appendFile(path, `${JSON.stringify(value)}\n`, 'utf8');
}

async function fileSize(path: string): Promise<number> {
  try {
    return (await stat(path)).size;
  } catch {
    return 0;
  }
}

async function readNewBytes(path: string, offset: number): Promise<{ text: string; offset: number }> {
  const size = await fileSize(path);
  if (size < offset) offset = 0;
  if (size === offset) return { text: '', offset };

  const length = size - offset;
  const buffer = Buffer.alloc(length);
  const handle = await open(path, 'r');
  try {
    await handle.read(buffer, 0, length, offset);
  } finally {
    await handle.close();
  }
  return { text: buffer.toString('utf8'), offset: size };
}

function parseEventLines(text: string): JsonlRunEvent[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as JsonlRunEvent);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
