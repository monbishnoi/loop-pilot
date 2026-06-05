# Cal Gateway Integration

This example shows how [Cal](https://github.com/monbishnoi/cal) — an open-source agent harness — integrates Loop Pilot as its trajectory advisor.

## Architecture

```
User message → Cal Gateway → Loop Pilot (plan_task) → guidance injected → Claude API → tool loop runs
```

Cal calls Loop Pilot once per user message, before the agent loop starts. The guidance is injected into the system prompt alongside Cal's other context (memory, user profile, calendar).

## Integration Code

```javascript
// src/loop-pilot.js — Cal Gateway's Loop Pilot integration

import { MCPClient } from './mcp-client.js';

const LOOP_PILOT_URL = 'http://127.0.0.1:8191';

/**
 * Get Loop Pilot guidance for a task before the agent loop starts.
 * Returns a prompt guidance block to inject into the system message.
 */
export async function getLoopPilotGuidance(task, options = {}) {
  try {
    const response = await fetch(`${LOOP_PILOT_URL}/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task,
        availableTools: options.availableTools,
        defaultBudget: options.defaultBudget ?? 5,
        maxBudget: options.maxBudget ?? 10,
      }),
    });

    if (!response.ok) return null;

    const plan = await response.json();
    return plan.promptGuidance;
  } catch {
    // Loop Pilot is advisory — if it's down, the harness continues without guidance
    return null;
  }
}
```

## Injection Point

```javascript
// In session.js — where Cal builds the system prompt before calling Claude

import { getLoopPilotGuidance } from './loop-pilot.js';

async function buildSystemPrompt(userMessage, context) {
  // Get Loop Pilot guidance (advisory, non-blocking)
  const loopGuidance = await getLoopPilotGuidance(userMessage);

  const systemPrompt = `
${context.identity}
${context.memory}
${context.userProfile}

${loopGuidance ? `[SYSTEM: Loop Pilot guidance]\n${loopGuidance}\n\nHarness note: Cal's configured max-iteration limit is ${MAX_ITERATIONS}. Loop Pilot is advisory only; do not treat the suggested budget as a hard limit.\n\nPrinciple: inform, then trust the model.` : ''}
  `;

  return systemPrompt;
}
```

## Key Design Decisions

1. **Advisory only.** Loop Pilot's budget suggestion doesn't change Cal's `maxIterations` (hard safety cap = 10). The model receives guidance as context and self-moderates.

2. **Fail-open.** If Loop Pilot is down or slow, Cal continues without guidance. The harness never blocks on an advisory service.

3. **Shared embedding model.** Cal already runs a local embedding model (EmbeddingGemma) for its knowledge search (QMD). Loop Pilot connects to the same model via HTTP — one model, two uses, no extra memory.

4. **No live recording.** Cal has rich structured logs already. Loop Pilot imports episodes on a schedule (nightly cron) rather than recording in real-time. This keeps the hot path simple.

## Running Loop Pilot Alongside Cal

```bash
# Start the embedding bridge (shared with QMD)
node scripts/looppilot-embedding-bridge.mjs &

# Start Loop Pilot MCP server
looppilot serve --transport mcp --port 8191 &

# Start Cal Gateway (Loop Pilot guidance auto-injected)
node src/gateway.js
```

Or with PM2:
```bash
pm2 start ecosystem.looppilot.config.cjs
pm2 start ecosystem.config.cjs
```

## Results

With 404 episodes of Cal behavior history:
- Budget predictions consistently accurate for conversational tasks
- KNN finds 5 relevant neighbors with high confidence on most queries
- Latency: ~300ms per plan (embedding model already warm from QMD usage)
- Model observably respects suggested budgets as operational guidance
