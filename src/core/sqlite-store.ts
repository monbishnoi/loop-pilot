import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { spawn } from 'node:child_process';
import type { EpisodeStore, StoredEpisode, ToolCallEpisode } from './types.js';

export class SqliteEpisodeStore implements EpisodeStore {
  dbPath: string;
  sqliteCommand: string;

  constructor(options: { dbPath: string; sqliteCommand?: string }) {
    this.dbPath = options.dbPath;
    this.sqliteCommand = options.sqliteCommand ?? 'sqlite3';
  }

  async init(): Promise<void> {
    await mkdir(dirname(this.dbPath), { recursive: true });
    await this.exec(`
      PRAGMA journal_mode = WAL;
      CREATE TABLE IF NOT EXISTS episodes (
        episode_id TEXT PRIMARY KEY,
        harness TEXT NOT NULL,
        task TEXT NOT NULL,
        source TEXT NOT NULL,
        session_id TEXT,
        started_at TEXT,
        finished_at TEXT,
        duration_ms INTEGER,
        tool_call_count INTEGER NOT NULL,
        tools_used_json TEXT NOT NULL,
        tool_errors_json TEXT NOT NULL,
        outcome TEXT NOT NULL,
        hit_max_iterations INTEGER NOT NULL,
        needed_continuation INTEGER NOT NULL,
        failure_labels_json TEXT NOT NULL,
        raw_source_json TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
      CREATE TABLE IF NOT EXISTS tool_calls (
        episode_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        tool TEXT NOT NULL,
        is_error INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (episode_id, position)
      );
      CREATE TABLE IF NOT EXISTS episode_embeddings (
        episode_id TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        dimensions INTEGER NOT NULL,
        vector_json TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
    `);
  }

  async upsertEpisodes(episodes: ToolCallEpisode[]): Promise<void> {
    await this.init();
    const statements: string[] = ['BEGIN;'];
    for (const episode of episodes) {
      statements.push(`
        INSERT INTO episodes (
          episode_id, harness, task, source, session_id, started_at, finished_at, duration_ms,
          tool_call_count, tools_used_json, tool_errors_json, outcome, hit_max_iterations,
          needed_continuation, failure_labels_json, raw_source_json, updated_at
        ) VALUES (
          ${q(episode.episodeId)}, ${q(episode.harness)}, ${q(episode.task)}, ${q(episode.source)},
          ${q(episode.sessionId ?? null)}, ${q(episode.startedAt ?? null)}, ${q(episode.finishedAt ?? null)},
          ${n(episode.durationMs)}, ${n(episode.toolCallCount)}, ${q(JSON.stringify(episode.toolsUsed))},
          ${q(JSON.stringify(episode.toolErrors))}, ${q(episode.outcome)}, ${b(episode.hitMaxIterations)},
          ${b(episode.neededContinuation)}, ${q(JSON.stringify(episode.failureLabels))},
          ${q(JSON.stringify(episode.rawSource ?? {}))}, CURRENT_TIMESTAMP
        )
        ON CONFLICT(episode_id) DO UPDATE SET
          harness = excluded.harness,
          task = excluded.task,
          source = excluded.source,
          session_id = excluded.session_id,
          started_at = excluded.started_at,
          finished_at = excluded.finished_at,
          duration_ms = excluded.duration_ms,
          tool_call_count = excluded.tool_call_count,
          tools_used_json = excluded.tools_used_json,
          tool_errors_json = excluded.tool_errors_json,
          outcome = excluded.outcome,
          hit_max_iterations = excluded.hit_max_iterations,
          needed_continuation = excluded.needed_continuation,
          failure_labels_json = excluded.failure_labels_json,
          raw_source_json = excluded.raw_source_json,
          updated_at = CURRENT_TIMESTAMP;
      `);
      statements.push(`DELETE FROM tool_calls WHERE episode_id = ${q(episode.episodeId)};`);
      episode.toolsUsed.forEach((tool, index) => {
        const isError = episode.toolErrors.includes(tool);
        statements.push(`
          INSERT INTO tool_calls (episode_id, position, tool, is_error)
          VALUES (${q(episode.episodeId)}, ${n(index + 1)}, ${q(tool)}, ${b(isError)});
        `);
      });
    }
    statements.push('COMMIT;');
    await this.exec(statements.join('\n'));
  }

  async upsertEmbedding(episodeId: string, provider: string, vector: number[]): Promise<void> {
    await this.init();
    await this.exec(`
      INSERT INTO episode_embeddings (episode_id, provider, dimensions, vector_json, updated_at)
      VALUES (${q(episodeId)}, ${q(provider)}, ${n(vector.length)}, ${q(JSON.stringify(vector))}, CURRENT_TIMESTAMP)
      ON CONFLICT(episode_id) DO UPDATE SET
        provider = excluded.provider,
        dimensions = excluded.dimensions,
        vector_json = excluded.vector_json,
        updated_at = CURRENT_TIMESTAMP;
    `);
  }

  async getEpisodes(): Promise<StoredEpisode[]> {
    await this.init();
    const rows = await this.queryRows(`
      SELECT
        e.*,
        em.vector_json AS embedding_json,
        em.provider AS embedding_provider
      FROM episodes e
      LEFT JOIN episode_embeddings em ON e.episode_id = em.episode_id
      ORDER BY e.started_at ASC, e.episode_id ASC;
    `);
    return rows.map(rowToEpisode);
  }

  async getEpisodeCount(): Promise<number> {
    await this.init();
    const rows = await this.queryRows('SELECT COUNT(*) AS count FROM episodes;');
    return Number(rows[0]?.count ?? 0);
  }

  private async queryRows(sql: string): Promise<Record<string, unknown>[]> {
    const output = await this.exec(sql, ['-json']);
    if (!output.trim()) return [];
    return JSON.parse(output) as Record<string, unknown>[];
  }

  private exec(sql: string, extraArgs: string[] = []): Promise<string> {
    return new Promise((resolve, reject) => {
      const child = spawn(this.sqliteCommand, [...extraArgs, this.dbPath], { stdio: ['pipe', 'pipe', 'pipe'] });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (chunk: Buffer) => {
        stdout += chunk.toString();
      });
      child.stderr.on('data', (chunk: Buffer) => {
        stderr += chunk.toString();
      });
      child.on('error', reject);
      child.on('close', (code) => {
        if (code === 0) resolve(stdout);
        else reject(new Error(stderr || `${this.sqliteCommand} exited with ${code}`));
      });
      child.stdin.end(sql);
    });
  }
}

function rowToEpisode(row: Record<string, unknown>): StoredEpisode {
  return {
    episodeId: String(row.episode_id),
    harness: String(row.harness),
    task: String(row.task),
    source: String(row.source),
    sessionId: row.session_id == null ? null : String(row.session_id),
    startedAt: row.started_at == null ? null : String(row.started_at),
    finishedAt: row.finished_at == null ? null : String(row.finished_at),
    durationMs: row.duration_ms == null ? null : Number(row.duration_ms),
    toolsUsed: parseJsonArray(row.tools_used_json),
    toolCallCount: Number(row.tool_call_count),
    toolErrors: parseJsonArray(row.tool_errors_json),
    outcome: String(row.outcome) as StoredEpisode['outcome'],
    hitMaxIterations: Boolean(Number(row.hit_max_iterations)),
    neededContinuation: Boolean(Number(row.needed_continuation)),
    failureLabels: parseJsonArray(row.failure_labels_json),
    rawSource: parseJsonObject(row.raw_source_json),
    embedding: row.embedding_json ? parseJsonNumberArray(row.embedding_json) : undefined,
    embeddingProvider: row.embedding_provider == null ? undefined : String(row.embedding_provider),
  };
}

function parseJsonArray(value: unknown): string[] {
  if (typeof value !== 'string') return [];
  const parsed = JSON.parse(value) as unknown;
  return Array.isArray(parsed) ? parsed.map(String) : [];
}

function parseJsonNumberArray(value: unknown): number[] {
  if (typeof value !== 'string') return [];
  const parsed = JSON.parse(value) as unknown;
  return Array.isArray(parsed) ? parsed.map(Number) : [];
}

function parseJsonObject(value: unknown): Record<string, unknown> {
  if (typeof value !== 'string') return {};
  const parsed = JSON.parse(value) as unknown;
  return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {};
}

function q(value: string | null): string {
  if (value === null) return 'NULL';
  return `'${value.replaceAll("'", "''")}'`;
}

function n(value: number | null | undefined): string {
  return value == null || Number.isNaN(value) ? 'NULL' : String(value);
}

function b(value: boolean): string {
  return value ? '1' : '0';
}
