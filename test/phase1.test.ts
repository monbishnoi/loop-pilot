import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  DeterministicEmbeddingProvider,
  HttpEmbeddingProvider,
  LoopPilot,
  SqliteEpisodeStore,
  buildEpisodesFromJsonlEvents,
  importBehaviorCollections,
  parseMaxIterationLogs,
  runRetrospectiveBenchmark,
  scanBehaviorCollections,
  startHttpServer,
  type ToolCallEpisode,
  writeBehaviorCollectionConfig,
} from '../src/index.js';

test('JSONL event parser groups events into behavior episodes and enriches max-iteration failures', () => {
  const maxLogs = parseMaxIterationLogs(JSON.stringify({
    message: JSON.stringify({
      ts: '2026-06-04T17:00:30.000Z',
      event: 'max_iterations',
      session: 'monika-main',
      message: 'Research the agent harness',
      iterations: '10/10',
    }),
  }));
  const episodes = buildEpisodesFromJsonlEvents([
    event('message_received', 'run-1', { text: 'Research the agent harness deeply' }),
    event('run_started', 'run-1'),
    event('tool_call_started', 'run-1', { tool: 'genai_search' }),
    event('tool_call_finished', 'run-1', { tool: 'genai_search', isError: false }),
    event('run_finished', 'run-1'),
  ], maxLogs);

  assert.equal(episodes.length, 1);
  assert.equal(episodes[0].episodeId, 'run-1');
  assert.equal(episodes[0].toolCallCount, 1);
  assert.deepEqual(episodes[0].toolsUsed, ['genai_search']);
  assert.equal(episodes[0].hitMaxIterations, true);
  assert.equal(episodes[0].outcome, 'error');
});

test('LoopPilot stores, indexes, plans, and generates guidance', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'looppilot-'));
  const dbPath = join(dir, 'looppilot.sqlite');
  try {
    const loopPilot = await createLoopPilot(dbPath);
    await loopPilot.importEpisodes([
      episode('1', 'Check my calendar for today', ['read_calendar'], 1),
      episode('2', 'Prepare me for my next meeting', ['read_calendar', 'semantic_search', 'read_file'], 3),
      episode('3', 'Research the agent harness deeply', ['genai_search', 'genai_search', 'write_file'], 3),
      episode('4', 'Prepare for tomorrow meeting with notes', ['read_calendar', 'semantic_search', 'read_file'], 3),
      episode('5', 'Calendar meeting preparation', ['read_calendar', 'read_file'], 2),
    ]);
    const indexResult = await loopPilot.indexEpisodes();
    assert.equal(indexResult.indexed, 5);

    const plan = await loopPilot.plan({ task: 'Can you prepare me for my next meeting?' });
    assert.ok(plan.prediction.suggestedBudget >= 2);
    assert.ok(Object.keys(plan.prediction.toolBudget).length > 0);
    assert.ok(plan.prediction.likelyTools.length > 0);
    assert.match(plan.promptGuidance, /Loop Pilot Guidance/);
    assert.match(plan.promptGuidance, /Suggested tool-call budget/);
    assert.match(plan.promptGuidance, /Likely allocation/);
    assert.equal(plan.similarEpisodes.some((item) => item.episode.embedding), false);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('benchmark compares fixed budget to Loop Pilot predictions', async () => {
  const episodes = [
    episode('1', 'Calendar lookup', ['read_calendar'], 1),
    episode('2', 'Calendar meeting prep', ['read_calendar', 'read_file'], 2),
    episode('3', 'Research agents', ['genai_search', 'genai_search', 'write_file'], 3),
    episode('4', 'Research memory systems', ['genai_search', 'genai_search', 'genai_search', 'write_file'], 4),
    episode('5', 'Prepare meeting from calendar', ['read_calendar', 'semantic_search', 'read_file'], 3),
    episode('6', 'Deep research agent protocol', ['genai_search', 'genai_search', 'genai_search', 'write_file'], 4),
  ];
  const result = await runRetrospectiveBenchmark(episodes, new DeterministicEmbeddingProvider(), { minHistory: 3 });
  assert.equal(result.totalEvaluated, 3);
  assert.equal(result.fixedBudget, 10);
  assert.ok(result.averageSuggestedBudget > 0);
});

test('HTTP embedding provider calls a shared embedding service', async () => {
  const server = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk.toString();
    });
    request.on('end', () => {
      const parsed = JSON.parse(body) as { text: string };
      response.setHeader('content-type', 'application/json');
      response.end(JSON.stringify({ embedding: [parsed.text.length, 1, 0] }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;

  try {
    const provider = new HttpEmbeddingProvider({
      endpoint: `http://127.0.0.1:${port}/embed`,
      dimensions: 3,
    });
    const vector = await provider.embed('calendar');
    assert.equal(vector.length, 3);
    assert.ok(vector[0] > vector[1]);
  } finally {
    server.close();
  }
});

test('HTTP service exposes health, stats, import, and plan endpoints', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'looppilot-http-'));
  const dbPath = join(dir, 'looppilot.sqlite');
  const loopPilot = await createLoopPilot(dbPath);
  const server = startHttpServer({ loopPilot, port: 0 });
  await new Promise<void>((resolve) => server.once('listening', resolve));
  const address = server.address();
  assert.equal(typeof address, 'object');
  const port = typeof address === 'object' && address ? address.port : 0;

  try {
    const health = await fetchJson(`http://127.0.0.1:${port}/health`);
    assert.equal(health.status, 'ok');

    const importResult = await fetchJson(`http://127.0.0.1:${port}/episodes/import`, {
      method: 'POST',
      body: JSON.stringify({ episodes: [episode('1', 'Check calendar', ['read_calendar'], 1)] }),
      headers: { 'content-type': 'application/json' },
    });
    assert.equal(importResult.imported, 1);

    await loopPilot.indexEpisodes();
    const plan = await fetchJson(`http://127.0.0.1:${port}/plan`, {
      method: 'POST',
      body: JSON.stringify({ task: 'Check my calendar' }),
      headers: { 'content-type': 'application/json' },
    });
    assert.ok(plan.prediction.suggestedBudget >= 2);
  } finally {
    server.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test('collection scanner discovers JSONL run logs and imports from generated config', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'looppilot-collection-'));
  const logsDir = join(dir, 'logs');
  const dataDir = join(dir, 'data');
  const configPath = join(dir, 'looppilot.collections.json');
  const dbPath = join(dir, 'looppilot.sqlite');

  try {
    await mkdir(logsDir, { recursive: true });
    await mkdir(dataDir, { recursive: true });
    await writeFile(join(dataDir, 'events.jsonl'), [
      JSON.stringify(event('message_received', 'run-1', { text: 'Prepare for my meeting' })),
      JSON.stringify(event('run_started', 'run-1')),
      JSON.stringify(event('tool_call_started', 'run-1', { tool: 'read_calendar' })),
      JSON.stringify(event('tool_call_finished', 'run-1', { tool: 'read_calendar', isError: false })),
      JSON.stringify(event('run_finished', 'run-1')),
    ].join('\n'), 'utf8');
    await writeFile(join(logsDir, 'error.log'), '', 'utf8');

    const scan = await scanBehaviorCollections(dir);
    assert.equal(scan.collections.length, 1);
    assert.equal(scan.collections[0].sources.events, 'data/events.jsonl');
    assert.equal(scan.collections[0].sources.errors, 'logs/error.log');
    assert.equal(scan.collections[0].parser, 'jsonl-events');

    await writeBehaviorCollectionConfig(configPath, { version: 1, collections: scan.collections });
    const loopPilot = await createLoopPilot(dbPath);
    const result = await importBehaviorCollections(loopPilot, configPath);
    assert.equal(result.parsed, 1);
    assert.equal(result.total, 1);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

async function createLoopPilot(dbPath: string): Promise<LoopPilot> {
  const loopPilot = new LoopPilot({
    store: new SqliteEpisodeStore({ dbPath }),
    embeddings: new DeterministicEmbeddingProvider(),
  });
  await loopPilot.init();
  return loopPilot;
}

function event(type: string, runId: string, payload: Record<string, unknown> = {}) {
  return {
    seq: 1,
    timestamp: '2026-06-04T17:00:00.000Z',
    type,
    sessionId: 'monika-main',
    runId,
    source: 'test',
    payload,
  };
}

function episode(episodeId: string, task: string, toolsUsed: string[], toolCallCount: number): ToolCallEpisode {
  return {
    episodeId,
    harness: 'test',
    task,
    source: 'test',
    sessionId: 'test-session',
    startedAt: `2026-06-04T17:00:0${episodeId}.000Z`,
    finishedAt: `2026-06-04T17:00:1${episodeId}.000Z`,
    durationMs: 10000,
    toolsUsed,
    toolCallCount,
    toolErrors: [],
    outcome: 'success',
    hitMaxIterations: false,
    neededContinuation: false,
    failureLabels: [],
  };
}

async function fetchJson(url: string, init?: RequestInit): Promise<any> {
  const response = await fetch(url, init);
  if (!response.ok) {
    assert.fail(await response.text());
  }
  return response.json();
}
