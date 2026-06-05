import http from 'node:http';
import { URL } from 'node:url';
import type { LoopPilot } from '../core/loop-pilot.js';
import type { ToolCallEpisode } from '../core/types.js';

export function startHttpServer(options: { loopPilot: LoopPilot; port: number; host?: string }): http.Server {
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
      if (req.method === 'GET' && url.pathname === '/health') {
        return json(res, 200, { status: 'ok' });
      }
      if (req.method === 'GET' && url.pathname === '/stats') {
        return json(res, 200, await options.loopPilot.getStats());
      }
      if (req.method === 'POST' && url.pathname === '/plan') {
        const body = await readJson(req);
        return json(res, 200, await options.loopPilot.plan({
          task: String(body.task ?? ''),
          availableTools: Array.isArray(body.availableTools) ? body.availableTools.map(String) : undefined,
          k: typeof body.k === 'number' ? body.k : undefined,
          defaultBudget: typeof body.defaultBudget === 'number' ? body.defaultBudget : undefined,
          maxBudget: typeof body.maxBudget === 'number' ? body.maxBudget : undefined,
        }));
      }
      if (req.method === 'POST' && url.pathname === '/episodes/import') {
        const body = await readJson(req);
        const episodes = Array.isArray(body.episodes) ? body.episodes as ToolCallEpisode[] : [];
        return json(res, 200, await options.loopPilot.importEpisodes(episodes));
      }
      if (req.method === 'POST' && url.pathname === '/outcome') {
        const body = await readJson(req);
        await options.loopPilot.recordOutcome(body as unknown as ToolCallEpisode);
        return json(res, 200, { ok: true });
      }
      return json(res, 404, { error: 'Not found' });
    } catch (error) {
      return json(res, 500, { error: error instanceof Error ? error.message : String(error) });
    }
  });

  server.listen(options.port, options.host ?? '127.0.0.1');
  return server;
}

function json(res: http.ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body, null, 2));
}

function readJson(req: http.IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => {
      data += chunk.toString();
    });
    req.on('end', () => {
      try {
        resolve(data.trim() ? JSON.parse(data) as Record<string, unknown> : {});
      } catch (error) {
        reject(error);
      }
    });
    req.on('error', reject);
  });
}
