import type { EmbeddingProvider, EpisodeStore, LoopPilotPlan, PlanRequest, ToolCallEpisode } from './types.js';
import { findSimilarEpisodes } from './similarity.js';
import { predictBudget } from './budget-predictor.js';
import { createPromptGuidance } from './prompt-guidance.js';

export class LoopPilot {
  store: EpisodeStore;
  embeddings: EmbeddingProvider;
  defaultK: number;

  constructor(options: { store: EpisodeStore; embeddings: EmbeddingProvider; defaultK?: number }) {
    this.store = options.store;
    this.embeddings = options.embeddings;
    this.defaultK = options.defaultK ?? 5;
  }

  async init(): Promise<void> {
    await this.store.init();
  }

  async importEpisodes(episodes: ToolCallEpisode[]): Promise<{ imported: number; total: number }> {
    await this.store.upsertEpisodes(episodes);
    return { imported: episodes.length, total: await this.store.getEpisodeCount() };
  }

  async indexEpisodes(): Promise<{ indexed: number; total: number }> {
    const episodes = await this.store.getEpisodes();
    let indexed = 0;
    for (const episode of episodes) {
      if (episode.embedding && episode.embeddingProvider === this.embeddings.name) continue;
      const vector = this.embeddings.embedDocument
        ? await this.embeddings.embedDocument(episode.task, episode.harness)
        : await this.embeddings.embed(episode.task);
      await this.store.upsertEmbedding(episode.episodeId, this.embeddings.name, vector);
      indexed += 1;
    }
    return { indexed, total: episodes.length };
  }

  async plan(request: PlanRequest): Promise<LoopPilotPlan> {
    const queryEmbedding = this.embeddings.embedQuery
      ? await this.embeddings.embedQuery(request.task)
      : await this.embeddings.embed(request.task);
    const episodes = await this.store.getEpisodes();
    const similarEpisodes = findSimilarEpisodes(episodes, queryEmbedding, {
      k: request.k ?? this.defaultK,
      minSimilarity: request.minSimilarity,
      relativeMinSimilarity: request.relativeMinSimilarity,
    });
    const prediction = predictBudget(similarEpisodes, {
      task: request.task,
      defaultBudget: request.defaultBudget,
      minBudget: request.minBudget,
      maxBudget: request.maxBudget,
    });
    return {
      task: request.task,
      prediction,
      similarEpisodes: similarEpisodes.map(({ episode, similarity }) => ({
        episode: {
          ...episode,
          embedding: undefined,
        },
        similarity,
      })),
      promptGuidance: createPromptGuidance(prediction),
    };
  }

  async recordOutcome(episode: ToolCallEpisode): Promise<void> {
    await this.store.upsertEpisodes([episode]);
  }

  async getStats(): Promise<{ episodes: number }> {
    return { episodes: await this.store.getEpisodeCount() };
  }
}
