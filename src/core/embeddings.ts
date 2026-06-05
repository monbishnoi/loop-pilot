import { spawn } from 'node:child_process';
import type { EmbeddingProvider } from './types.js';

export class DeterministicEmbeddingProvider implements EmbeddingProvider {
  name = 'deterministic-hash-v1';
  dimensions: number;

  constructor(options: { dimensions?: number } = {}) {
    this.dimensions = options.dimensions ?? 256;
  }

  async embed(text: string): Promise<number[]> {
    const vector = new Array(this.dimensions).fill(0);
    for (const token of tokenize(text)) {
      const hash = hashToken(token);
      const index = Math.abs(hash) % this.dimensions;
      vector[index] += hash % 2 === 0 ? 1 : -1;
    }
    return normalize(vector);
  }
}

export class CommandEmbeddingProvider implements EmbeddingProvider {
  name: string;
  dimensions: number;
  command: string;
  args: string[];

  constructor(options: { name?: string; dimensions: number; command: string; args?: string[] }) {
    this.name = options.name ?? 'command-embedding';
    this.dimensions = options.dimensions;
    this.command = options.command;
    this.args = options.args ?? [];
  }

  async embed(text: string): Promise<number[]> {
    const output = await runCommand(this.command, this.args, text);
    return normalize(parseEmbeddingResponse(JSON.parse(output), this.dimensions, 'Embedding command'));
  }
}

export class HttpEmbeddingProvider implements EmbeddingProvider {
  name: string;
  dimensions: number;
  endpoint: string;
  headers: Record<string, string>;

  constructor(options: {
    name?: string;
    dimensions: number;
    endpoint: string;
    headers?: Record<string, string>;
  }) {
    this.name = options.name ?? 'http-embedding';
    this.dimensions = options.dimensions;
    this.endpoint = options.endpoint;
    this.headers = options.headers ?? {};
  }

  async embed(text: string): Promise<number[]> {
    return this.requestEmbedding({ text });
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.requestEmbedding({ text, isQuery: true });
  }

  async embedDocument(text: string, title?: string): Promise<number[]> {
    return this.requestEmbedding({ text, isQuery: false, title });
  }

  private async requestEmbedding(body: { text: string; isQuery?: boolean; title?: string }): Promise<number[]> {
    const response = await fetch(this.endpoint, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...this.headers,
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`Embedding HTTP provider returned ${response.status}: ${await response.text()}`);
    }
    return normalize(parseEmbeddingResponse(await response.json(), this.dimensions, 'Embedding HTTP provider'));
  }
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9_ -]/g, ' ')
    .split(/\s+/)
    .filter((token) => token.length > 1);
}

function hashToken(token: string): number {
  let hash = 2166136261;
  for (let i = 0; i < token.length; i += 1) {
    hash ^= token.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash | 0;
}

function normalize(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return norm === 0 ? vector : vector.map((value) => value / norm);
}

function parseEmbeddingResponse(value: unknown, dimensions: number, source: string): number[] {
  const embedding = Array.isArray(value)
    ? value
    : typeof value === 'object' && value !== null && Array.isArray((value as { embedding?: unknown }).embedding)
      ? (value as { embedding: unknown[] }).embedding
      : undefined;

  if (!embedding || embedding.some((item) => typeof item !== 'number')) {
    throw new Error(`${source} must return a JSON number array or {"embedding": number[]}`);
  }
  if (embedding.length !== dimensions) {
    throw new Error(`${source} returned ${embedding.length} dimensions, expected ${dimensions}`);
  }
  return embedding as number[];
}

function runCommand(command: string, args: string[], input: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve(stdout.trim());
      else reject(new Error(stderr.trim() || `${command} exited with ${code}`));
    });
    child.stdin.end(input);
  });
}
