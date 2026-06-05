import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import type { LoopPilot } from './loop-pilot.js';
import type { ToolCallEpisode } from './types.js';
import { parseJsonlEventFiles } from '../adapters/jsonl-events/parser.js';

export type CollectionParser = 'jsonl-events';
export type CollectionCandidateType = 'events' | 'errors' | 'log' | 'unknown';

export interface BehaviorCollectionSource {
  events: string;
  errors?: string;
}

export interface BehaviorCollection {
  name: string;
  parser: CollectionParser;
  root: string;
  sources: BehaviorCollectionSource;
}

export interface BehaviorCollectionConfig {
  version: 1;
  collections: BehaviorCollection[];
}

export interface CollectionScanCandidate {
  path: string;
  type: CollectionCandidateType;
  parser?: CollectionParser;
  reason: string;
}

export interface CollectionScanResult {
  root: string;
  collections: BehaviorCollection[];
  candidates: CollectionScanCandidate[];
}

const SKIP_DIRS = new Set([
  '.git',
  '.next',
  '.turbo',
  '.venv',
  'build',
  'coverage',
  'dist',
  'node_modules',
]);

export async function scanBehaviorCollections(rootPath: string): Promise<CollectionScanResult> {
  const root = resolve(rootPath);
  const files = await walkFiles(root);
  const candidates = await detectCandidates(root, files);
  const eventStreams = candidates.filter((candidate) => candidate.parser === 'jsonl-events' && candidate.type === 'events');
  const errorCandidates = candidates.filter((candidate) => candidate.type === 'errors');

  const collections = eventStreams.map((eventsCandidate, index) => {
    const errors = findNearestErrorLog(eventsCandidate.path, errorCandidates.map((candidate) => candidate.path));
    return {
      name: index === 0 ? 'jsonl-events' : `jsonl-events-${index + 1}`,
      parser: 'jsonl-events' as const,
      root: '.',
      sources: {
        events: eventsCandidate.path,
        ...(errors ? { errors } : {}),
      },
    };
  });

  return { root, collections, candidates };
}

export async function writeBehaviorCollectionConfig(
  configPath: string,
  config: BehaviorCollectionConfig,
): Promise<void> {
  await mkdir(dirname(configPath), { recursive: true });
  await writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`, 'utf8');
}

export async function loadBehaviorCollectionConfig(configPath: string): Promise<BehaviorCollectionConfig> {
  const parsed = JSON.parse(await readFile(configPath, 'utf8')) as unknown;
  if (!isBehaviorCollectionConfig(parsed)) {
    throw new Error(`Invalid Loop Pilot collection config: ${configPath}`);
  }
  return parsed;
}

export async function parseBehaviorCollection(
  collection: BehaviorCollection,
  configDir: string,
): Promise<ToolCallEpisode[]> {
  const root = resolvePath(configDir, collection.root);

  if (collection.parser === 'jsonl-events') {
    return parseJsonlEventFiles({
      eventsPath: resolvePath(root, collection.sources.events),
      errorsPath: collection.sources.errors ? resolvePath(root, collection.sources.errors) : undefined,
      harness: collection.name,
    });
  }

  throw new Error(`Unsupported collection parser "${String(collection.parser)}"`);
}

export async function parseBehaviorCollections(configPath: string): Promise<ToolCallEpisode[]> {
  const config = await loadBehaviorCollectionConfig(configPath);
  const configDir = dirname(resolve(configPath));
  const batches = await Promise.all(
    config.collections.map((collection) => parseBehaviorCollection(collection, configDir)),
  );
  return batches.flat();
}

export async function importBehaviorCollections(
  loopPilot: LoopPilot,
  configPath: string,
): Promise<{ imported: number; parsed: number; total: number; collections: number }> {
  const config = await loadBehaviorCollectionConfig(configPath);
  const configDir = dirname(resolve(configPath));
  const batches = await Promise.all(
    config.collections.map((collection) => parseBehaviorCollection(collection, configDir)),
  );
  const episodes = batches.flat();
  const result = await loopPilot.importEpisodes(episodes);
  return {
    ...result,
    parsed: episodes.length,
    collections: config.collections.length,
  };
}

async function walkFiles(root: string): Promise<string[]> {
  const found: string[] = [];
  const stack = [root];

  while (stack.length) {
    const current = stack.pop();
    if (!current) continue;

    let entries;
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (entry.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) stack.push(join(current, entry.name));
      } else if (entry.isFile()) {
        found.push(join(current, entry.name));
      }
    }
  }

  return found;
}

async function detectCandidates(root: string, files: string[]): Promise<CollectionScanCandidate[]> {
  const candidates: CollectionScanCandidate[] = [];

  for (const file of files) {
    const rel = relative(root, file);
    const lower = rel.toLowerCase();

    if (lower.endsWith('.jsonl') && (lower.includes('event') || lower.endsWith('events.jsonl'))) {
      const parser = await looksLikeJsonlRunEvents(file) ? 'jsonl-events' : undefined;
      candidates.push({
        path: rel,
        type: 'events',
        ...(parser ? { parser } : {}),
        reason: parser ? 'JSONL file contains structured run events.' : 'JSONL filename looks like an event stream.',
      });
      continue;
    }

    if (lower.endsWith('.log') || lower.endsWith('.txt')) {
      const isError = lower.includes('error') || lower.includes('stderr');
      candidates.push({
        path: rel,
        type: isError ? 'errors' : 'log',
        reason: isError ? 'Filename looks like an error log.' : 'Filename looks like a runtime log.',
      });
    }
  }

  return candidates.sort((a, b) => a.path.localeCompare(b.path));
}

async function looksLikeJsonlRunEvents(file: string): Promise<boolean> {
  let text = '';
  try {
    text = await readFile(file, 'utf8');
  } catch {
    return false;
  }

  let inspected = 0;
  let matched = 0;
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    inspected += 1;
    if (inspected > 80) break;

    try {
      const parsed = JSON.parse(trimmed) as { runId?: unknown; type?: unknown; payload?: unknown };
      const hasRunId = typeof parsed.runId === 'string';
      const hasKnownType = typeof parsed.type === 'string' && [
        'message_received',
        'run_started',
        'tool_call_started',
        'tool_call_finished',
        'run_finished',
        'run_error',
      ].includes(parsed.type);
      if (hasRunId && hasKnownType) matched += 1;
    } catch {
      // A noisy line should not prevent other lines from being detected.
    }
  }

  return matched >= 2;
}

function findNearestErrorLog(eventsPath: string, errorPaths: string[]): string | undefined {
  if (errorPaths.length === 0) return undefined;
  const eventParts = eventsPath.split('/');
  return [...errorPaths].sort((a, b) => scoreErrorLog(b, eventParts) - scoreErrorLog(a, eventParts))[0];
}

function scoreErrorLog(errorPath: string, eventParts: string[]): number {
  const lower = errorPath.toLowerCase();
  let score = sharedPrefixLength(errorPath.split('/'), eventParts) * 10;
  if (lower.endsWith('error.log')) score += 15;
  if (lower.endsWith('errors.log')) score += 15;
  if (lower.includes('/archive/')) score -= 30;
  score -= errorPath.split('/').length;
  return score;
}

function sharedPrefixLength(left: string[], right: string[]): number {
  let count = 0;
  while (left[count] && left[count] === right[count]) count += 1;
  return count;
}

function resolvePath(base: string, target: string): string {
  return isAbsolute(target) ? target : resolve(base, target);
}

function isBehaviorCollectionConfig(value: unknown): value is BehaviorCollectionConfig {
  if (!value || typeof value !== 'object') return false;
  const config = value as BehaviorCollectionConfig;
  return config.version === 1 && Array.isArray(config.collections);
}
