import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import express from 'express';
import { z } from 'zod';
import type { LoopPilot } from '../core/loop-pilot.js';
import type { ToolCallEpisode } from '../core/types.js';

export function createMcpServer(loopPilot: LoopPilot): McpServer {
  const server = new McpServer({ name: 'looppilot', version: '0.1.0' });

  server.registerTool('plan_task', {
    title: 'Plan Task',
    description: 'Suggest a tool-call budget and guidance for a new agent task.',
    inputSchema: {
      task: z.string(),
      availableTools: z.array(z.string()).optional(),
      k: z.number().optional(),
      defaultBudget: z.number().optional(),
      maxBudget: z.number().optional(),
    },
  }, async (input) => {
    const plan = await loopPilot.plan(input);
    return { content: [{ type: 'text', text: JSON.stringify(plan, null, 2) }] };
  });

  server.registerTool('record_episode', {
    title: 'Record Episode',
    description: 'Record one completed behavior episode.',
    inputSchema: {
      episode: z.record(z.string(), z.unknown()),
    },
  }, async ({ episode }) => {
    await loopPilot.recordOutcome(episode as unknown as ToolCallEpisode);
    return { content: [{ type: 'text', text: JSON.stringify({ ok: true }) }] };
  });

  server.registerTool('import_episodes', {
    title: 'Import Episodes',
    description: 'Import normalized behavior episodes.',
    inputSchema: {
      episodes: z.array(z.record(z.string(), z.unknown())),
    },
  }, async ({ episodes }) => {
    const result = await loopPilot.importEpisodes(episodes as unknown as ToolCallEpisode[]);
    return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
  });

  server.registerTool('get_stats', {
    title: 'Get Stats',
    description: 'Return Loop Pilot memory statistics.',
  }, async () => {
    return { content: [{ type: 'text', text: JSON.stringify(await loopPilot.getStats(), null, 2) }] };
  });

  return server;
}

export function startMcpServer(options: { loopPilot: LoopPilot; port: number; host?: string }) {
  const app = express();
  app.use(express.json({ limit: '10mb' }));

  app.get('/health', (_req, res) => {
    res.json({ status: 'ok' });
  });

  app.post('/mcp', async (req, res) => {
    const server = createMcpServer(options.loopPilot);
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    try {
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
      res.on('close', () => {
        void transport.close();
        void server.close();
      });
    } catch (error) {
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: '2.0',
          error: { code: -32603, message: error instanceof Error ? error.message : String(error) },
          id: null,
        });
      }
    }
  });

  return app.listen(options.port, options.host ?? '127.0.0.1');
}
