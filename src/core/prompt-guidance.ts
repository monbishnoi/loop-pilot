import type { BudgetPrediction } from './types.js';

export function createPromptGuidance(prediction: BudgetPrediction): string {
  const likelyTools = prediction.likelyTools.length > 0
    ? prediction.likelyTools.join(', ')
    : 'No strong tool pattern yet';
  const budgetRange = prediction.budgetRange.min === prediction.budgetRange.max
    ? `${prediction.budgetRange.min}`
    : `${prediction.budgetRange.min}-${prediction.budgetRange.max}`;
  const confidenceNote = confidenceNoteFor(prediction);
  const budgetLines = budgetLinesFor(prediction, budgetRange);
  const likelyToolsLabel = prediction.guidanceMode === 'rough-prior'
    ? 'Tools seen in weakly related runs'
    : 'Likely useful tools';
  const repeatedTools = prediction.repeatedToolPatterns.length > 0
    ? `\nRepeated-tool caution: similar tasks repeated ${prediction.repeatedToolPatterns.join(', ')}. Avoid repeating a tool once enough context is found.`
    : '';
  const toolBudget = prediction.guidanceMode !== 'rough-prior' && Object.keys(prediction.toolBudget).length > 0
    ? `\nLikely allocation:\n${Object.entries(prediction.toolBudget)
      .map(([tool, budget]) => `- ${tool}: ${budget} call${budget === 1 ? '' : 's'}`)
      .join('\n')}`
    : '';
  const failureHints = prediction.failureHints.length > 0
    ? `\nFailure hints:\n${prediction.failureHints.map((hint) => `- ${hint}`).join('\n')}`
    : '';

  return `## Loop Pilot Guidance

Similar past behavior suggests:
${budgetLines}
- Confidence: ${prediction.confidence}
- Risk: ${prediction.risk}
- ${likelyToolsLabel}: ${likelyTools}

How to use this: ${confidenceNote}

Evidence: top similarity ${prediction.evidence.topSimilarity.toFixed(2)}, average similarity ${prediction.evidence.averageSimilarity.toFixed(2)}, intent match ${(prediction.evidence.intentMatchRate * 100).toFixed(0)}%.

Reason: ${prediction.reason}${toolBudget}${repeatedTools}${failureHints}

Use this as operational guidance. Continue to reason normally.`;
}

function budgetLinesFor(prediction: BudgetPrediction, budgetRange: string): string {
  if (prediction.guidanceMode === 'rough-prior') {
    return `- Reliable budget suggestion: none; historical match is weak
- Rough budget range only: ${budgetRange}`;
  }

  return `- Suggested tool-call budget: ${prediction.suggestedBudget}
- Estimated budget range: ${budgetRange}`;
}

function confidenceNoteFor(prediction: BudgetPrediction): string {
  if (prediction.guidanceMode === 'optimize') {
    return 'Strong match. It is reasonable to optimize around this budget while still following the task requirements.';
  }

  if (prediction.guidanceMode === 'planning-prior') {
    return 'Partial match. Use this as a planning prior, then adjust from the actual task steps.';
  }

  return 'Weak match. Do not treat Loop Pilot as an instruction here. Decompose the task first and choose the number of steps from the actual requirements.';
}
