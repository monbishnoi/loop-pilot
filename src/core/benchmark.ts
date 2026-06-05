import type { BenchmarkResult, EmbeddingProvider, ToolCallEpisode } from './types.js';
import { findSimilarEpisodes } from './similarity.js';
import { predictBudget } from './budget-predictor.js';

export async function runRetrospectiveBenchmark(
  episodes: ToolCallEpisode[],
  embeddings: EmbeddingProvider,
  options: {
    k?: number;
    fixedBudget?: number;
    minHistory?: number;
    maxBudget?: number;
    minSimilarity?: number;
    relativeMinSimilarity?: number;
  } = {},
): Promise<BenchmarkResult> {
  const fixedBudget = options.fixedBudget ?? 10;
  const minHistory = options.minHistory ?? 5;
  const k = options.k ?? 5;
  const maxBudget = options.maxBudget ?? 25;
  const sorted = [...episodes].sort((a, b) => String(a.startedAt ?? '').localeCompare(String(b.startedAt ?? '')));
  const embeddedHistory: Array<ToolCallEpisode & { embedding: number[] }> = [];

  let evaluated = 0;
  let fixedSuccess = 0;
  let loopPilotSuccess = 0;
  let underBudget = 0;
  let overBudget = 0;
  let maxIterationCases = 0;
  let maxIterationAvoided = 0;
  let suggestedBudgetTotal = 0;

  for (const episode of sorted) {
    const queryEmbedding = embeddings.embedQuery
      ? await embeddings.embedQuery(episode.task)
      : await embeddings.embed(episode.task);

    if (embeddedHistory.length >= minHistory) {
      const similar = findSimilarEpisodes(embeddedHistory, queryEmbedding, {
        k,
        minSimilarity: options.minSimilarity,
        relativeMinSimilarity: options.relativeMinSimilarity,
      });
      const prediction = predictBudget(similar, { maxBudget, defaultBudget: fixedBudget });
      const actualCalls = episode.toolCallCount;
      const actualSuccess = episode.outcome === 'success' && !episode.hitMaxIterations;
      const fixedWouldCover = actualCalls <= fixedBudget && actualSuccess;
      const loopPilotWouldCover = actualCalls <= prediction.suggestedBudget && actualSuccess;

      evaluated += 1;
      suggestedBudgetTotal += prediction.suggestedBudget;
      if (fixedWouldCover) fixedSuccess += 1;
      if (loopPilotWouldCover) loopPilotSuccess += 1;
      if (prediction.suggestedBudget < actualCalls) underBudget += 1;
      if (prediction.suggestedBudget > actualCalls + 3) overBudget += 1;
      if (episode.hitMaxIterations) {
        maxIterationCases += 1;
        if (prediction.suggestedBudget > fixedBudget) maxIterationAvoided += 1;
      }
    }

    const documentEmbedding = embeddings.embedDocument
      ? await embeddings.embedDocument(episode.task, episode.harness)
      : queryEmbedding;
    embeddedHistory.push({ ...episode, embedding: documentEmbedding });
  }

  return {
    totalEvaluated: evaluated,
    skippedColdStart: Math.min(sorted.length, minHistory),
    fixedBudget,
    fixedSuccessRate: rate(fixedSuccess, evaluated),
    loopPilotSuccessRate: rate(loopPilotSuccess, evaluated),
    underBudgetRate: rate(underBudget, evaluated),
    overBudgetRate: rate(overBudget, evaluated),
    maxIterationAvoidanceRate: rate(maxIterationAvoided, maxIterationCases),
    averageSuggestedBudget: evaluated ? suggestedBudgetTotal / evaluated : 0,
  };
}

function rate(count: number, total: number): number {
  return total === 0 ? 0 : count / total;
}
