export type EpisodeOutcome = 'success' | 'error' | 'unknown';
export type Confidence = 'low' | 'medium' | 'high';
export type Risk = 'low' | 'medium' | 'high' | 'unknown';

export interface ToolCallEpisode {
  episodeId: string;
  harness: string;
  task: string;
  source: string;
  sessionId?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  durationMs?: number | null;
  toolsUsed: string[];
  toolCallCount: number;
  toolErrors: string[];
  outcome: EpisodeOutcome;
  hitMaxIterations: boolean;
  neededContinuation: boolean;
  failureLabels: string[];
  rawSource?: Record<string, unknown>;
}

export interface StoredEpisode extends ToolCallEpisode {
  embedding?: number[];
  embeddingProvider?: string;
}

export interface SimilarEpisode {
  episode: StoredEpisode;
  similarity: number;
}

export interface EmbeddingProvider {
  name: string;
  dimensions: number;
  embed(text: string): Promise<number[]>;
  embedQuery?(text: string): Promise<number[]>;
  embedDocument?(text: string, title?: string): Promise<number[]>;
  dispose?(): Promise<void>;
}

export interface EpisodeStore {
  init(): Promise<void>;
  upsertEpisodes(episodes: ToolCallEpisode[]): Promise<void>;
  upsertEmbedding(episodeId: string, provider: string, vector: number[]): Promise<void>;
  getEpisodes(): Promise<StoredEpisode[]>;
  getEpisodeCount(): Promise<number>;
}

export interface PlanRequest {
  task: string;
  availableTools?: string[];
  k?: number;
  minSimilarity?: number;
  relativeMinSimilarity?: number;
  defaultBudget?: number;
  minBudget?: number;
  maxBudget?: number;
}

export interface BudgetPrediction {
  suggestedBudget: number;
  budgetRange: {
    min: number;
    max: number;
  };
  toolBudget: Record<string, number>;
  confidence: Confidence;
  risk: Risk;
  guidanceMode: 'optimize' | 'planning-prior' | 'rough-prior';
  evidence: {
    topSimilarity: number;
    averageSimilarity: number;
    similaritySpread: number;
    successfulNeighbors: number;
    totalNeighbors: number;
    intentMatchRate: number;
    taskComplexity: 'routine' | 'complex' | 'unknown';
  };
  likelyTools: string[];
  repeatedToolPatterns: string[];
  failureHints: string[];
  reason: string;
  neighborCount: number;
}

export interface LoopPilotPlan {
  task: string;
  prediction: BudgetPrediction;
  similarEpisodes: SimilarEpisode[];
  promptGuidance: string;
}

export interface BenchmarkResult {
  totalEvaluated: number;
  skippedColdStart: number;
  fixedBudget: number;
  fixedSuccessRate: number;
  loopPilotSuccessRate: number;
  underBudgetRate: number;
  overBudgetRate: number;
  maxIterationAvoidanceRate: number;
  averageSuggestedBudget: number;
}
