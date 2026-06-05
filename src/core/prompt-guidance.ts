import type { BudgetPrediction } from './types.js';

export function createPromptGuidance(prediction: BudgetPrediction): string {
  if (prediction.guidanceMode === 'rough-prior') {
    return '';
  }

  const likelyTools = prediction.likelyTools.length > 0
    ? prediction.likelyTools.join(', ')
    : 'No strong tool pattern yet';
  const budgetRange = prediction.budgetRange.min === prediction.budgetRange.max
    ? `${prediction.budgetRange.min}`
    : `${prediction.budgetRange.min}-${prediction.budgetRange.max}`;
  const confidenceNote = confidenceNoteFor(prediction);
  const caution = complexLowBudgetCaution(prediction);
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
- Estimated budget range: ${budgetRange}
- Confidence: ${prediction.confidence}
- Risk: ${prediction.risk}
- Likely useful tools: ${likelyTools}

How to use this: ${confidenceNote}${caution}

Evidence: top similarity ${prediction.evidence.topSimilarity.toFixed(2)}, average similarity ${prediction.evidence.averageSimilarity.toFixed(2)}, intent match ${(prediction.evidence.intentMatchRate * 100).toFixed(0)}%.

Reason: ${prediction.reason}${toolBudget}${repeatedTools}${failureHints}

Use this as operational guidance. Continue to reason normally.`;
}

function complexLowBudgetCaution(prediction: BudgetPrediction): string {
  if (prediction.evidence.taskComplexity === 'complex' && prediction.suggestedBudget <= 3) {
    return ' The task looks complex for such a small budget, so treat the number as a weak estimate and use more steps if the task requires them.';
  }
  return '';
}

function confidenceNoteFor(prediction: BudgetPrediction): string {
  if (prediction.guidanceMode === 'optimize') {
    return 'Strong match. It is reasonable to optimize around this budget while still following the task requirements.';
  }

  if (prediction.guidanceMode === 'planning-prior') {
    return 'Partial match. Use this as a planning prior, then adjust from the actual task steps.';
  }

  return 'Weak match. Do not inject guidance.';
}
