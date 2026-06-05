import { appendFile, mkdir, open, stat } from 'node:fs/promises';
import { dirname } from 'node:path';
import type { LoopPilot } from './loop-pilot.js';
import type { BudgetPrediction } from './types.js';
import { buildEpisodesFromJsonlEvents, type JsonlRunEvent } from '../adapters/jsonl-events/parser.js';

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
      experiment: 'loop_pilot_standalone_shadow_v1',
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
    this.completedRuns.add(runId);

    const runEvents = this.eventsByRun.get(runId) ?? [];
    const episodes = buildEpisodesFromJsonlEvents(runEvents, [], { harness: this.options.harness });
    const episode = episodes[0];
    if (!episode) return;

    await this.loopPilot.importEpisodes([episode]);
    await this.loopPilot.indexEpisodes();

    const prediction = this.predictionsByRun.get(runId) ?? null;
    await appendJsonl(this.options.outputPath, {
      type: 'outcome',
      experiment: 'loop_pilot_standalone_shadow_v1',
      timestamp: new Date().toISOString(),
      runId,
      sessionId: episode.sessionId,
      source: episode.source,
      injected: false,
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
