import type { SimilarEpisode, StoredEpisode } from './types.js';

export function findSimilarEpisodes(
  episodes: StoredEpisode[],
  queryEmbedding: number[],
  options: { k?: number; includeUnknownOutcomes?: boolean; minSimilarity?: number; relativeMinSimilarity?: number } = {},
): SimilarEpisode[] {
  const k = options.k ?? 5;
  const ranked = episodes
    .filter((episode) => Array.isArray(episode.embedding))
    .filter((episode) => options.includeUnknownOutcomes || episode.outcome !== 'unknown')
    .map((episode) => ({
      episode,
      similarity: cosineSimilarity(queryEmbedding, episode.embedding ?? []),
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, k);

  const bestSimilarity = ranked[0]?.similarity ?? 0;
  const relativeCutoff = bestSimilarity > 0 ? bestSimilarity * (options.relativeMinSimilarity ?? 0.6) : Number.NEGATIVE_INFINITY;
  const absoluteCutoff = options.minSimilarity ?? Number.NEGATIVE_INFINITY;
  const cutoff = Math.max(relativeCutoff, absoluteCutoff);
  return ranked.filter((episode) => episode.similarity >= cutoff);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const length = Math.min(a.length, b.length);
  if (!length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < length; index += 1) {
    dot += a[index] * b[index];
    normA += a[index] * a[index];
    normB += b[index] * b[index];
  }
  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
