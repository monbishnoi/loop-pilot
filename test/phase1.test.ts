import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  DeterministicEmbeddingProvider,
  HttpEmbeddingProvider,
  LoopPilot,
  LoopPilotObserver,
  SqliteEpisodeStore,
  buildEpisodesFromJsonlEvents,
  collectShadowObservationStats,
  createPromptGuidance,
  importBehaviorCollections,
  parseMaxIterationLogs,
  predictBudget,
  runRetrospectiveBenchmark,
  scanBehaviorCollections,
  startHttpServer,
  type SimilarEpisode,
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
    event('loop_pilot_shadow_plan', 'run-1', {
      experiment: 'loop_pilot_shadow_v1',
      injected: false,
      suggestedBudget: 3,
      budgetRange: { min: 2, max: 5 },
      confidence: 'medium',
    }),
    event('tool_call_started', 'run-1', { tool: 'genai_search' }),
    event('tool_call_finished', 'run-1', { tool: 'genai_search', isError: false }),
    event('response_complete', 'run-1', { text: 'Done' }),
    event('run_finished', 'run-1'),
  ], maxLogs);

  assert.equal(episodes.length, 1);
  assert.equal(episodes[0].episodeId, 'run-1');
  assert.equal(episodes[0].toolCallCount, 1);
  assert.deepEqual(episodes[0].toolsUsed, ['genai_search']);
  assert.equal(episodes[0].hitMaxIterations, true);
  assert.equal(episodes[0].outcome, 'error');
  assert.equal(episodes[0].rawSource?.responseCharCount, 4);
  assert.deepEqual(episodes[0].rawSource?.loopPilotShadowPlan, {
    experiment: 'loop_pilot_shadow_v1',
    injected: false,
    suggestedBudget: 3,
    budgetRange: { min: 2, max: 5 },
    confidence: 'medium',
  });
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
    assert.ok(plan.prediction.budgetRange.max >= plan.prediction.suggestedBudget);
    assert.ok(plan.prediction.evidence.totalNeighbors > 0);
    assert.ok(Object.keys(plan.prediction.toolBudget).length > 0);
    assert.ok(plan.prediction.likelyTools.length > 0);
    if (plan.prediction.confidence === 'low') {
      assert.equal(plan.promptGuidance, '');
    } else {
      assert.match(plan.promptGuidance, /Loop Pilot Guidance/);
      assert.match(plan.promptGuidance, /Estimated budget range/);
    }
    assert.equal(plan.similarEpisodes.some((item) => item.episode.embedding), false);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('budget predictor does not call weakly similar neighbors high confidence', () => {
  const similar = [
    similarEpisode('1', 'Check calendar', ['read_calendar'], 1, 0.31),
    similarEpisode('2', 'Prepare meeting notes', ['read_calendar', 'read_file'], 2, 0.29),
    similarEpisode('3', 'Search notes for meeting', ['semantic_search'], 1, 0.27),
    similarEpisode('4', 'Summarize schedule', ['read_calendar'], 1, 0.26),
    similarEpisode('5', 'Find calendar conflicts', ['read_calendar'], 1, 0.25),
  ];

  const prediction = predictBudget(similar, { task: 'Publish the Loop Pilot blog post and push it to the website' });

  assert.equal(prediction.confidence, 'low');
  assert.equal(prediction.guidanceMode, 'rough-prior');
  assert.equal(prediction.evidence.taskComplexity, 'complex');
  assert.ok(prediction.budgetRange.max >= 8);

  const guidance = createPromptGuidance(prediction);
  assert.equal(guidance, '');
});

test('budget predictor requires strong evidence before high confidence', () => {
  const similar = [
    similarEpisode('1', 'Publish blog post to website and push changes', ['read_file', 'write_file', 'git'], 5, 0.86),
    similarEpisode('2', 'Create website post then commit and push', ['read_file', 'write_file', 'git'], 6, 0.82),
    similarEpisode('3', 'Update website HTML and push to GitHub', ['read_file', 'write_file', 'git'], 5, 0.8),
    similarEpisode('4', 'Publish article page and commit website', ['read_file', 'write_file', 'git'], 6, 0.78),
    similarEpisode('5', 'Post blog HTML update to website repo', ['read_file', 'write_file', 'git'], 5, 0.76),
  ];

  const prediction = predictBudget(similar, { task: 'Publish the Loop Pilot blog post and push it to the website' });

  assert.equal(prediction.confidence, 'high');
  assert.equal(prediction.guidanceMode, 'optimize');
  assert.deepEqual(prediction.budgetRange, {
    min: prediction.suggestedBudget,
    max: prediction.suggestedBudget,
  });
});

test('prompt guidance caveats suspiciously low budgets for complex tasks', () => {
  const similar = [
    similarEpisode('1', 'Publish short post to website and push changes', ['write_file'], 1, 0.86),
    similarEpisode('2', 'Create tiny website post then commit and push', ['write_file'], 1, 0.82),
    similarEpisode('3', 'Update website HTML and push to GitHub', ['write_file', 'git'], 2, 0.8),
    similarEpisode('4', 'Publish article page and commit website', ['write_file'], 1, 0.78),
    similarEpisode('5', 'Post blog HTML update to website repo', ['write_file'], 1, 0.76),
  ];

  const prediction = predictBudget(similar, { task: 'Publish the Loop Pilot blog post and push it to the website' });
  const guidance = createPromptGuidance(prediction);

  assert.equal(prediction.confidence, 'high');
  assert.ok(prediction.suggestedBudget <= 3);
  assert.match(guidance, /Suggested tool-call budget/);
  assert.match(guidance, /looks complex for such a small budget/);
  assert.match(guidance, /use more steps if the task requires them/);
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

test('standalone observer writes shadow predictions and scored outcomes', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'looppilot-observer-'));
  const dbPath = join(dir, 'looppilot.sqlite');
  const outputPath = join(dir, 'shadow.jsonl');

  try {
    const loopPilot = await createLoopPilot(dbPath);
    await loopPilot.importEpisodes([
      episode('hist-1', 'Prepare meeting notes', ['read_calendar', 'read_file'], 2),
      episode('hist-2', 'Prepare next meeting from calendar', ['read_calendar', 'semantic_search', 'read_file'], 3),
      episode('hist-3', 'Calendar meeting preparation', ['read_calendar', 'read_file'], 2),
    ]);
    await loopPilot.indexEpisodes();

    const observer = new LoopPilotObserver(loopPilot, { outputPath, eventsPath: 'unused', harness: 'test' });
    await observer.processEvent(event('message_received', 'run-1', { text: 'Prepare me for my next meeting' }));
    await observer.processEvent(event('run_started', 'run-1'));
    await observer.processEvent(event('tool_call_started', 'run-1', { tool: 'read_calendar' }));
    await observer.processEvent(event('tool_call_finished', 'run-1', { tool: 'read_calendar', isError: false }));
    await observer.processEvent(event('response_complete', 'run-1', { text: 'Prepared.' }));
    await observer.processEvent(event('run_finished', 'run-1'));

    const records = (await readFile(outputPath, 'utf8'))
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));

    assert.equal(records.length, 2);
    assert.equal(records[0].type, 'prediction');
    assert.equal(records[0].experiment, 'loop_pilot_standalone_shadow_v2');
    assert.equal(records[0].injected, false);
    assert.equal(records[1].type, 'outcome');
    assert.equal(records[1].experiment, 'loop_pilot_standalone_shadow_v2');
    assert.equal(records[1].actual.toolCallCount, 1);
    assert.ok(['good', 'penalize_under_budget', 'penalize_over_budget', 'needs_review'].includes(records[1].reward.label));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('standalone observer enriches v2 outcomes with rich episode fields and session history', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'looppilot-rich-observer-'));
  const dbPath = join(dir, 'looppilot.sqlite');
  const outputPath = join(dir, 'shadow.jsonl');

  try {
    const loopPilot = await createLoopPilot(dbPath);
    await loopPilot.importEpisodes([
      episode('hist-1', 'Check service docs', ['fetch'], 1),
      episode('hist-2', 'Run scheduled service check', ['fetch', 'notify'], 2),
      episode('hist-3', 'Debug failing tool call', ['shell'], 1),
    ]);
    await loopPilot.indexEpisodes();

    const observer = new LoopPilotObserver(loopPilot, { outputPath, eventsPath: 'unused', harness: 'test' });
    await observer.processEvent(eventAt('message_received', 'run-1', '2026-06-01T10:00:00.000Z', { text: 'Check service docs' }));
    await observer.processEvent(eventAt('run_started', 'run-1', '2026-06-01T10:00:01.000Z'));
    await observer.processEvent(eventAt('tool_call_started', 'run-1', '2026-06-01T10:00:02.000Z', { tool: 'fetch' }));
    await observer.processEvent(eventAt('tool_call_finished', 'run-1', '2026-06-01T10:00:03.000Z', { tool: 'fetch' }));
    await observer.processEvent(eventAt('response_complete', 'run-1', '2026-06-01T10:00:04.000Z', { text: 'Done.' }));
    await observer.processEvent(eventAt('run_finished', 'run-1', '2026-06-01T10:00:05.000Z'));

    const task = '[Scheduled Job: docs] Check https://example.com? ```ts\nconst ok = true;\n```';
    await observer.processEvent(eventAt('message_received', 'run-2', '2026-06-02T17:30:00.000Z', { text: task }));
    await observer.processEvent(eventAt('run_started', 'run-2', '2026-06-02T17:30:01.000Z'));
    await observer.processEvent(eventAt('tool_call_started', 'run-2', '2026-06-02T17:30:02.000Z', { tool: 'fetch' }));
    await observer.processEvent(eventAt('tool_call_finished', 'run-2', '2026-06-02T17:30:03.000Z', { tool: 'fetch', error: 'network' }));
    await observer.processEvent(eventAt('response_complete', 'run-2', '2026-06-02T17:30:04.000Z', { text: 'Could not fetch.' }));
    await observer.processEvent(eventAt('run_error', 'run-2', '2026-06-02T17:30:05.000Z', { error: 'network' }));

    const records = (await readFile(outputPath, 'utf8'))
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    const outcomes = records.filter((record) => record.type === 'outcome');
    const second = outcomes[1];

    assert.equal(outcomes.length, 2);
    assert.equal(second.runId, 'run-2');
    assert.equal(second.sessionRunIndex, 2);
    assert.equal(second.previousRunToolCount, 1);
    assert.equal(second.previousRunOutcome, 'success');
    assert.equal(second.previousRunDurationMs, 4000);
    assert.equal(second.actualToolCalls, 1);
    assert.deepEqual(second.actualToolChain, ['fetch']);
    assert.deepEqual(second.uniqueToolsUsed, ['fetch']);
    assert.equal(second.uniqueToolCount, 1);
    assert.equal(second.toolErrors, 1);
    assert.deepEqual(second.toolErrorNames, ['fetch']);
    assert.equal(second.outcome, 'failure');
    assert.equal(second.outcomeLabel, 'error');
    assert.equal(second.durationMs, 4000);
    assert.equal(second.timeOfDay, 17 + 30 / 60 + 1 / 3600);
    assert.equal(second.dayOfWeek, 2);
    assert.equal(second.taskCharCount, task.length);
    assert.equal(second.taskQuestionMarks, 1);
    assert.equal(second.taskHasCodeBlock, true);
    assert.equal(second.taskHasUrl, true);
    assert.equal(second.isScheduledJob, true);
    assert.equal(second.responseCharCount, 'Could not fetch.'.length);
    assert.equal(typeof second.predictedBudget, 'number');
    assert.ok(['high', 'medium', 'low'].includes(second.predictedConfidence));
    assert.ok(Array.isArray(second.predictedTools));
    assert.equal(typeof second.knnTopSimilarity, 'number');
    assert.equal(typeof second.knnNeighborCount, 'number');

    const stats = await collectShadowObservationStats(outputPath);
    assert.equal(stats.totalEpisodes, 2);
    assert.equal(stats.taskTypeDistribution.scheduled, 1);
    assert.equal(stats.taskTypeDistribution.interactive, 1);
    assert.equal(stats.averageToolCalls, 1);
    assert.equal(stats.hitCapRate, 0);
    assert.equal(stats.uniqueToolsSeen.includes('fetch'), true);
    assert.equal(typeof stats.predictionMae, 'number');
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

function eventAt(type: string, runId: string, timestamp: string, payload: Record<string, unknown> = {}) {
  return {
    ...event(type, runId, payload),
    timestamp,
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

function similarEpisode(
  episodeId: string,
  task: string,
  toolsUsed: string[],
  toolCallCount: number,
  similarity: number,
): SimilarEpisode {
  return {
    episode: episode(episodeId, task, toolsUsed, toolCallCount),
    similarity,
  };
}

async function fetchJson(url: string, init?: RequestInit): Promise<any> {
  const response = await fetch(url, init);
  if (!response.ok) {
    assert.fail(await response.text());
  }
  return response.json();
}
