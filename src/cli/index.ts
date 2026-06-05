#!/usr/bin/env node

import { mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import {
  type BehaviorCollectionConfig,
  CommandEmbeddingProvider,
  DeterministicEmbeddingProvider,
  HttpEmbeddingProvider,
  type EmbeddingProvider,
  LoopPilot,
  SqliteEpisodeStore,
  importBehaviorCollections,
  parseJsonlEventFiles,
  parseBehaviorCollections,
  runRetrospectiveBenchmark,
  scanBehaviorCollections,
  startHttpServer,
  startMcpServer,
  writeBehaviorCollectionConfig,
} from '../index.js';

const args = process.argv.slice(2);

try {
  await main(args);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}

async function main(argv: string[]): Promise<void> {
  const [command] = argv;
  const subcommand = command === 'import' || command === 'collection' ? argv[1] : undefined;
  const rest = command === 'import' || command === 'collection' ? argv.slice(2) : argv.slice(1);
  const parsed = parseCliValues(rest);
  const options = parsed.options;
  const dbPath = resolve(options.db ?? 'data/looppilot.sqlite');
  await mkdir(dirname(dbPath), { recursive: true });

  const loopPilot = new LoopPilot({
    store: new SqliteEpisodeStore({ dbPath }),
    embeddings: createEmbeddingProvider(options),
  });
  await loopPilot.init();

  if (command === 'import' && (subcommand === 'events' || subcommand === 'jsonl-events')) {
    const eventsPath = required(options.events, '--events');
    const episodes = await parseJsonlEventFiles({
      eventsPath,
      errorsPath: options.errors,
      harness: options.harness ?? 'jsonl-events',
    });
    const result = await loopPilot.importEpisodes(episodes);
    console.log(JSON.stringify({ ...result, parsed: episodes.length }, null, 2));
    return;
  }

  if (command === 'collection') {
    if (subcommand === 'scan') {
      const root = resolve(parsed.positionals[0] ?? options.root ?? '.');
      const scan = await scanBehaviorCollections(root);
      if (options.json === 'true') {
        console.log(JSON.stringify(scan, null, 2));
      } else {
        printCollectionScan(scan);
      }
      return;
    }

    if (subcommand === 'init') {
      const root = resolve(parsed.positionals[0] ?? options.root ?? '.');
      const scan = await scanBehaviorCollections(root);
      const config: BehaviorCollectionConfig = {
        version: 1,
        collections: scan.collections,
      };
      const output = resolve(options.output ?? options.config ?? `${root}/looppilot.collections.json`);
      await writeBehaviorCollectionConfig(output, config);
      console.log(JSON.stringify({
        output,
        collections: config.collections.length,
        candidates: scan.candidates.length,
      }, null, 2));
      return;
    }

    if (subcommand === 'parse') {
      const configPath = resolve(options.config ?? parsed.positionals[0] ?? 'looppilot.collections.json');
      const episodes = await parseBehaviorCollections(configPath);
      console.log(JSON.stringify({
        config: configPath,
        parsed: episodes.length,
        sample: episodes.slice(0, Number(options.sample ?? 3)),
      }, null, 2));
      return;
    }

    if (subcommand === 'import') {
      const configPath = resolve(options.config ?? parsed.positionals[0] ?? 'looppilot.collections.json');
      console.log(JSON.stringify(await importBehaviorCollections(loopPilot, configPath), null, 2));
      return;
    }

    throw new Error(`Unknown collection command "${subcommand ?? ''}". Use scan, init, parse, or import.`);
  }

  if (command === 'index') {
    console.log(JSON.stringify(await loopPilot.indexEpisodes(), null, 2));
    return;
  }

  if (command === 'plan') {
    const task = required(options.task, '--task');
    console.log(JSON.stringify(await loopPilot.plan({
      task,
      minSimilarity: optionalNumber(options.minSimilarity),
      relativeMinSimilarity: optionalNumber(options.relativeMinSimilarity),
    }), null, 2));
    return;
  }

  if (command === 'benchmark') {
    const eventsPath = required(options.events, '--events');
    const episodes = await parseJsonlEventFiles({
      eventsPath,
      errorsPath: options.errors,
      harness: options.harness ?? 'jsonl-events',
    });
    const result = await runRetrospectiveBenchmark(episodes, createEmbeddingProvider(options), {
      minSimilarity: optionalNumber(options.minSimilarity),
      relativeMinSimilarity: optionalNumber(options.relativeMinSimilarity),
    });
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (command === 'serve') {
    const transport = options.transport ?? 'http';
    const port = Number(options.port ?? '8191');
    if (transport === 'mcp') {
      startMcpServer({ loopPilot, port });
    } else {
      startHttpServer({ loopPilot, port });
    }
    console.log(`Loop Pilot ${transport} server listening on http://127.0.0.1:${port}`);
    return;
  }

  printHelp();
}

function createEmbeddingProvider(options: Record<string, string>): EmbeddingProvider {
  const provider = options.embedding ?? process.env.LOOPPILOT_EMBEDDING ?? detectEmbeddingProvider();

  if (provider === 'none') {
    return {
      name: 'unconfigured',
      dimensions: 0,
      async embed() {
        throw new Error(
          'No embedding provider configured. Use --embedding http with --embedding-url, --embedding command with --embedding-command, or --embedding deterministic for tests.',
        );
      },
    };
  }

  if (provider === 'deterministic') {
    return new DeterministicEmbeddingProvider({
      dimensions: options.dimensions ? Number(options.dimensions) : undefined,
    });
  }

  if (provider === 'command') {
    return new CommandEmbeddingProvider({
      name: options.embeddingName,
      dimensions: Number(options.dimensions ?? process.env.LOOPPILOT_EMBEDDING_DIMENSIONS ?? 768),
      command: options.embeddingCommand ?? required(process.env.LOOPPILOT_EMBEDDING_COMMAND, '--embedding-command'),
      args: (options.embeddingArgs ?? process.env.LOOPPILOT_EMBEDDING_ARGS ?? '').split(' ').filter(Boolean),
    });
  }

  if (provider === 'http') {
    return new HttpEmbeddingProvider({
      name: options.embeddingName,
      dimensions: Number(options.dimensions ?? process.env.LOOPPILOT_EMBEDDING_DIMENSIONS ?? 768),
      endpoint: options.embeddingUrl ?? required(process.env.LOOPPILOT_EMBEDDING_URL, '--embedding-url'),
      headers: parseJsonObject(options.embeddingHeaders ?? process.env.LOOPPILOT_EMBEDDING_HEADERS),
    });
  }

  throw new Error(`Unknown embedding provider "${provider}". Use http, command, or deterministic.`);
}

function detectEmbeddingProvider(): string {
  if (process.env.LOOPPILOT_EMBEDDING_URL) return 'http';
  if (process.env.LOOPPILOT_EMBEDDING_COMMAND) return 'command';
  return 'none';
}

function parseCliValues(values: string[]): { options: Record<string, string>; positionals: string[] } {
  const options: Record<string, string> = {};
  const positionals: string[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith('--')) {
      positionals.push(value);
      continue;
    }
    const key = value.slice(2);
    const next = values[index + 1];
    const optionValue = next && !next.startsWith('--') ? next : 'true';
    options[key] = optionValue;
    options[toCamelCase(key)] = optionValue;
    if (optionValue !== 'true') index += 1;
  }
  return { options, positionals };
}

function toCamelCase(value: string): string {
  return value.replace(/-([a-z])/g, (_, letter: string) => letter.toUpperCase());
}

function required(value: string | undefined, name: string): string {
  if (!value) throw new Error(`Missing required option ${name}`);
  return value;
}

function optionalNumber(value: string | undefined): number | undefined {
  return value == null ? undefined : Number(value);
}

function parseJsonObject(value: string | undefined): Record<string, string> | undefined {
  if (!value) return undefined;
  const parsed = JSON.parse(value) as unknown;
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('--embedding-headers must be a JSON object');
  }
  return parsed as Record<string, string>;
}

function printCollectionScan(scan: Awaited<ReturnType<typeof scanBehaviorCollections>>): void {
  console.log(`Loop Pilot collection scan

Root: ${scan.root}
Detected collections: ${scan.collections.length}
Detected log candidates: ${scan.candidates.length}
`);

  for (const collection of scan.collections) {
    console.log(`- ${collection.name} (${collection.parser})`);
    console.log(`  events: ${collection.sources.events}`);
    if (collection.sources.errors) console.log(`  errors: ${collection.sources.errors}`);
  }

  if (scan.collections.length === 0) {
    console.log('No importable collection was detected yet. Run with --json for agent-readable candidates.');
  }
}

function printHelp(): void {
  console.log(`Loop Pilot

Commands:
  looppilot collection scan [root] [--json]
  looppilot collection init [root] [--output looppilot.collections.json]
  looppilot collection parse [--config looppilot.collections.json] [--sample 3]
  looppilot collection import [--config looppilot.collections.json] [--db <path>]
  looppilot import events --events <events.jsonl> [--errors <error.log>] [--harness <name>] [--db <path>] [--embedding http|command|deterministic]
  looppilot index [--db <path>] [--embedding http|command|deterministic]
  looppilot plan --task <task> [--db <path>] [--embedding http|command|deterministic]
  looppilot benchmark --events <events.jsonl> [--errors <error.log>] [--harness <name>] [--embedding http|command|deterministic]
  looppilot serve --transport <http|mcp> [--port 8191] [--db <path>] [--embedding http|command|deterministic]

Embedding options:
  --embedding http                Use a shared local embedding HTTP service
  --embedding command             Use a local embedding command
  --embedding deterministic       Use the fast test embedder only, explicitly
  --embedding-url <url>           POST {"text": "..."} and expect number[] or {"embedding": number[]}
  --embedding-command <cmd>       Command reads task text on stdin and returns number[] or {"embedding": number[]}
  --dimensions <n>                Embedding dimensions, for example 768 for EmbeddingGemma
  --embedding-headers <json>      Optional JSON headers for HTTP embedding service
  --relative-min-similarity <n>   Keep neighbors at least this fraction of the top score (default 0.6)
  --min-similarity <n>            Optional absolute similarity cutoff

Environment:
  LOOPPILOT_EMBEDDING=http|command|deterministic
  LOOPPILOT_EMBEDDING_URL=http://127.0.0.1:8000/embed
  LOOPPILOT_EMBEDDING_COMMAND=/path/to/embed-one
  LOOPPILOT_EMBEDDING_DIMENSIONS=768

Loop Pilot will not install an embedding model. Configure HTTP or command to reuse an existing local model.

Collections:
  collection scan       Finds likely behavior logs in a harness repo.
  collection init       Writes looppilot.collections.json from detected logs.
  collection parse      Dry-runs parsing and prints a small episode sample.
  collection import     Imports all configured collections into local memory.
`);
}
