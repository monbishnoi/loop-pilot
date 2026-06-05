import type { BudgetPrediction, SimilarEpisode } from './types.js';

export function predictBudget(
  similarEpisodes: SimilarEpisode[],
  options: {
    task?: string;
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
    const evidence = evidenceFor([], 0, options.task);
    return {
      suggestedBudget: defaultBudget,
      budgetRange: budgetRangeFor(defaultBudget, 'low', evidence.taskComplexity, minBudget, maxBudget),
      toolBudget: {},
      confidence: 'low',
      risk: 'unknown',
      guidanceMode: 'rough-prior',
      evidence,
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
  const evidence = evidenceFor(similarEpisodes, successful.length, options.task);
  const confidence = confidenceFor(evidence, suggestedBudget);

  return {
    suggestedBudget,
    budgetRange: budgetRangeFor(suggestedBudget, confidence, evidence.taskComplexity, minBudget, maxBudget),
    toolBudget: allocateToolBudget(budgetSource, suggestedBudget),
    confidence,
    risk: riskFor(failureRate, suggestedBudget, maxBudget),
    guidanceMode: guidanceModeFor(confidence),
    evidence,
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

function standardDeviation(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = average(values);
  return Math.sqrt(average(values.map((value) => (value - mean) ** 2)));
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

function evidenceFor(
  similarEpisodes: SimilarEpisode[],
  successfulCount: number,
  task?: string,
): BudgetPrediction['evidence'] {
  const similarities = similarEpisodes.map((item) => item.similarity);
  const taskTags = tagsFor(task ?? '');
  const matchedNeighbors = taskTags.length === 0
    ? similarEpisodes.length
    : similarEpisodes.filter((item) => hasTagOverlap(taskTags, tagsFor(item.episode.task))).length;

  return {
    topSimilarity: similarities.length ? Math.max(...similarities) : 0,
    averageSimilarity: average(similarities),
    similaritySpread: standardDeviation(similarities),
    successfulNeighbors: successfulCount,
    totalNeighbors: similarEpisodes.length,
    intentMatchRate: similarEpisodes.length === 0 ? 0 : matchedNeighbors / similarEpisodes.length,
    taskComplexity: task ? taskComplexityFor(task) : 'unknown',
  };
}

function confidenceFor(evidence: BudgetPrediction['evidence'], suggestedBudget: number): BudgetPrediction['confidence'] {
  const hasStrongEvidence = evidence.totalNeighbors >= 5
    && evidence.successfulNeighbors >= 3
    && evidence.topSimilarity >= 0.72
    && evidence.averageSimilarity >= 0.6
    && evidence.similaritySpread <= 0.18
    && evidence.intentMatchRate >= 0.6;

  if (hasStrongEvidence) return 'high';

  const hasUsableEvidence = evidence.totalNeighbors >= 3
    && evidence.successfulNeighbors >= 2
    && evidence.topSimilarity >= 0.55
    && evidence.averageSimilarity >= 0.42
    && evidence.intentMatchRate >= 0.3;

  if (hasUsableEvidence && !(evidence.taskComplexity === 'complex' && suggestedBudget <= 3)) return 'medium';
  return 'low';
}

function riskFor(failureRate: number, suggestedBudget: number, maxBudget: number): BudgetPrediction['risk'] {
  if (failureRate >= 0.4 || suggestedBudget >= maxBudget * 0.8) return 'high';
  if (failureRate >= 0.15) return 'medium';
  return 'low';
}

function budgetRangeFor(
  suggestedBudget: number,
  confidence: BudgetPrediction['confidence'],
  taskComplexity: BudgetPrediction['evidence']['taskComplexity'],
  minBudget: number,
  maxBudget: number,
): BudgetPrediction['budgetRange'] {
  if (confidence === 'high') {
    return { min: suggestedBudget, max: suggestedBudget };
  }

  if (confidence === 'medium') {
    return {
      min: clamp(suggestedBudget - 1, minBudget, maxBudget),
      max: clamp(suggestedBudget + (taskComplexity === 'complex' ? 4 : 2), minBudget, maxBudget),
    };
  }

  return {
    min: clamp(suggestedBudget - 1, minBudget, maxBudget),
    max: clamp(Math.max(suggestedBudget + 4, taskComplexity === 'complex' ? 8 : suggestedBudget + 3), minBudget, maxBudget),
  };
}

function guidanceModeFor(confidence: BudgetPrediction['confidence']): BudgetPrediction['guidanceMode'] {
  if (confidence === 'high') return 'optimize';
  if (confidence === 'medium') return 'planning-prior';
  return 'rough-prior';
}

function taskComplexityFor(task: string): BudgetPrediction['evidence']['taskComplexity'] {
  const text = task.toLowerCase();
  const complexPatterns = [
    /\bpublish\b/,
    /\bdeploy\b/,
    /\bcommit\b/,
    /\bpush\b/,
    /\brelease\b/,
    /\bship\b/,
    /\bwebsite\b/,
    /\bblog\b/,
    /\bhtml\b/,
    /\bupdate\b.*\bsite\b/,
    /\bcreate\b.*\bsite\b/,
    /\bcreate\b.*\bwebsite\b/,
  ];
  return complexPatterns.some((pattern) => pattern.test(text)) ? 'complex' : 'routine';
}

function tagsFor(task: string): string[] {
  const text = task.toLowerCase();
  const tags: string[] = [];
  const tagPatterns: Array<[string, RegExp]> = [
    ['publish', /\b(publish|post|blog|article)\b/],
    ['deploy', /\b(deploy|release|ship)\b/],
    ['git', /\b(commit|push|pull request|repo|github)\b/],
    ['website', /\b(website|site|html|page|readme)\b/],
    ['calendar', /\b(calendar|meeting|schedule)\b/],
    ['research', /\b(research|find|look up|search)\b/],
    ['code', /\b(code|implement|fix|test|build)\b/],
    ['file-edit', /\b(write|edit|create|update)\b/],
  ];
  for (const [tag, pattern] of tagPatterns) {
    if (pattern.test(text)) tags.push(tag);
  }
  return tags;
}

function hasTagOverlap(left: string[], right: string[]): boolean {
  return left.some((tag) => right.includes(tag));
}
