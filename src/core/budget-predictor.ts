import type { BudgetPrediction, SimilarEpisode } from './types.js';

export function predictBudget(
  similarEpisodes: SimilarEpisode[],
  options: {
    defaultBudget?: number;
    minBudget?: number;
    maxBudget?: number;
    safetyMargin?: number;
  } = {},
): BudgetPrediction {
  const defaultBudget = options.defaultBudget ?? 10;
  const minBudget = options.minBudget ?? 2;
  const maxBudget = options.maxBudget ?? 25;
  const safetyMargin = options.safetyMargin ?? 1;

  if (similarEpisodes.length === 0) {
    return {
      suggestedBudget: defaultBudget,
      toolBudget: {},
      confidence: 'low',
      risk: 'unknown',
      likelyTools: [],
      repeatedToolPatterns: [],
      failureHints: [],
      reason: 'No similar episodes found. Using default budget.',
      neighborCount: 0,
    };
  }

  const neighbors = similarEpisodes.map(({ episode }) => episode);
  const successful = neighbors.filter((episode) => episode.outcome === 'success' && !episode.hitMaxIterations);
  const budgetSource = successful.length > 0 ? successful : neighbors;
  const averageCalls = average(budgetSource.map((episode) => episode.toolCallCount));
  const suggestedBudget = clamp(Math.ceil(averageCalls + safetyMargin), minBudget, maxBudget);
  const failureCount = neighbors.filter((episode) => episode.outcome !== 'success' || episode.hitMaxIterations).length;
  const failureRate = failureCount / neighbors.length;

  return {
    suggestedBudget,
    toolBudget: allocateToolBudget(budgetSource, suggestedBudget),
    confidence: confidenceFor(similarEpisodes, successful.length),
    risk: riskFor(failureRate, suggestedBudget, maxBudget),
    likelyTools: rankTools(budgetSource),
    repeatedToolPatterns: repeatedTools(budgetSource),
    failureHints: failureHints(neighbors),
    reason: `Based on ${budgetSource.length} similar ${successful.length ? 'successful ' : ''}episode(s), average tool calls were ${averageCalls.toFixed(1)}.`,
    neighborCount: neighbors.length,
  };
}

function average(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function rankTools(episodes: { toolsUsed: string[] }[]): string[] {
  const counts = new Map<string, number>();
  for (const episode of episodes) {
    for (const tool of episode.toolsUsed) {
      counts.set(tool, (counts.get(tool) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([tool]) => tool);
}

function repeatedTools(episodes: { toolsUsed: string[] }[]): string[] {
  const counts = new Map<string, number>();
  for (const episode of episodes) {
    for (let index = 1; index < episode.toolsUsed.length; index += 1) {
      if (episode.toolsUsed[index] === episode.toolsUsed[index - 1]) {
        counts.set(episode.toolsUsed[index], (counts.get(episode.toolsUsed[index]) ?? 0) + 1);
      }
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([tool]) => tool);
}

function allocateToolBudget(episodes: { toolsUsed: string[] }[], suggestedBudget: number): Record<string, number> {
  const counts = new Map<string, number>();
  let total = 0;
  for (const episode of episodes) {
    for (const tool of episode.toolsUsed) {
      counts.set(tool, (counts.get(tool) ?? 0) + 1);
      total += 1;
    }
  }
  if (total === 0) return {};

  const allocation = new Map<string, number>();
  let allocated = 0;
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  for (const [tool, count] of ranked) {
    const value = Math.max(1, Math.round((count / total) * suggestedBudget));
    allocation.set(tool, value);
    allocated += value;
  }

  while (allocated > suggestedBudget && allocation.size > 0) {
    const adjustable = [...allocation.entries()]
      .filter(([, value]) => value > 1)
      .sort((a, b) => b[1] - a[1])[0];
    if (!adjustable) break;
    allocation.set(adjustable[0], adjustable[1] - 1);
    allocated -= 1;
  }

  return Object.fromEntries(allocation);
}

function failureHints(episodes: { toolErrors: string[]; toolsUsed: string[] }[]): string[] {
  const errorCounts = new Map<string, number>();
  const useCounts = new Map<string, number>();
  for (const episode of episodes) {
    for (const tool of episode.toolsUsed) {
      useCounts.set(tool, (useCounts.get(tool) ?? 0) + 1);
    }
    for (const tool of episode.toolErrors) {
      errorCounts.set(tool, (errorCounts.get(tool) ?? 0) + 1);
    }
  }

  return [...errorCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .filter(([tool, errors]) => errors > 0 && (errors / Math.max(1, useCounts.get(tool) ?? 0)) >= 0.2)
    .slice(0, 3)
    .map(([tool, errors]) => `${tool} failed in ${errors} similar episode(s); if it fails once, pivot rather than retrying repeatedly.`);
}

function confidenceFor(similarEpisodes: SimilarEpisode[], successfulCount: number): BudgetPrediction['confidence'] {
  const avgSimilarity = average(similarEpisodes.map((item) => item.similarity));
  if (similarEpisodes.length >= 5 && successfulCount >= 3 && avgSimilarity >= 0.25) return 'high';
  if (similarEpisodes.length >= 3 && successfulCount >= 1) return 'medium';
  return 'low';
}

function riskFor(failureRate: number, suggestedBudget: number, maxBudget: number): BudgetPrediction['risk'] {
  if (failureRate >= 0.4 || suggestedBudget >= maxBudget * 0.8) return 'high';
  if (failureRate >= 0.15) return 'medium';
  return 'low';
}
