import type { BudgetPrediction } from './types.js';

export function createPromptGuidance(prediction: BudgetPrediction): string {
  const likelyTools = prediction.likelyTools.length > 0
    ? prediction.likelyTools.join(', ')
    : 'No strong tool pattern yet';
  const repeatedTools = prediction.repeatedToolPatterns.length > 0
    ? `\nRepeated-tool caution: similar tasks repeated ${prediction.repeatedToolPatterns.join(', ')}. Avoid repeating a tool once enough context is found.`
    : '';
  const toolBudget = Object.keys(prediction.toolBudget).length > 0
    ? `\nLikely allocation:\n${Object.entries(prediction.toolBudget)
      .map(([tool, budget]) => `- ${tool}: ${budget} call${budget === 1 ? '' : 's'}`)
      .join('\n')}`
    : '';
  const failureHints = prediction.failureHints.length > 0
    ? `\nFailure hints:\n${prediction.failureHints.map((hint) => `- ${hint}`).join('\n')}`
    : '';

  return `## Loop Pilot Guidance

Similar past behavior suggests:
- Suggested tool-call budget: ${prediction.suggestedBudget}
- Confidence: ${prediction.confidence}
- Risk: ${prediction.risk}
- Likely useful tools: ${likelyTools}

Reason: ${prediction.reason}${toolBudget}${repeatedTools}${failureHints}

Use this as operational guidance. Continue to reason normally.`;
}
