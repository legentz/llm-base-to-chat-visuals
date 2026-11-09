/**
 * ILLUSTRATIVE numbers for the chapter 2 SFT simulation. Not measured. They're shaped to match what
 * is documented: a base model gives special/template tokens almost no probability (it never saw
 * them, see the Olmo 3 finding in docs/research/base-vs-instruct.md), knows the facts (moderate
 * probability on content tokens), and SFT drives the assistant span toward the demonstration.
 */
import { hash01 } from '#lib/math/rng.js';
import type { Message, RenderedToken, TemplateId } from '#lib/templates.js';

export const conversation: Message[] = [
	{ role: 'system', content: 'You are a helpful assistant.' },
	{ role: 'user', content: 'What is the capital of France?' },
	{
		role: 'assistant',
		content: 'The capital of France is Paris.',
		reasoning: 'The user asks for the capital of France. That is Paris.'
	}
];

export interface TokenTarget {
	/** Probability the base model gives this token at this position. */
	p0: number;
	/** Probability after SFT has converged on this example. */
	p1: number;
	/** Learning time constant in steps (bigger = slower). */
	tau: number;
}

const SPECIAL = /^<\/?think>$|^<\|.*\|>$/;

/** Deterministic per-token targets from the token's class, plus a little hash-based jitter. */
export function targetFor(tok: RenderedToken, index: number): TokenTarget {
	const j = hash01(`${tok.text}#${index}`);
	if (SPECIAL.test(tok.text))
		return { p0: 0.002 + 0.008 * j, p1: 0.97 + 0.025 * j, tau: 110 + 50 * j };
	if (/^\n+$/.test(tok.text)) return { p0: 0.04 + 0.08 * j, p1: 0.95 + 0.04 * j, tau: 70 + 40 * j };
	if (tok.role === 'assistant')
		return { p0: 0.12 + 0.45 * j, p1: 0.86 + 0.12 * j, tau: 40 + 60 * j };
	// Prompt tokens (only trained when the mask is off): the model has to predict the user's words.
	return { p0: 0.08 + 0.3 * j, p1: 0.35 + 0.3 * j, tau: 80 + 60 * j };
}

export interface Probe {
	id: string;
	label: string;
	tokens: string[];
	p0: number[];
	p1: number[];
}

/** Next-token distributions at two positions of the assistant turn, before vs after SFT. */
export function probes(template: TemplateId): Probe[] {
	const eot = template === 'chatml' ? '<|im_end|>' : '<|eot_id|>';
	const eod = template === 'chatml' ? '<|endoftext|>' : '<|end_of_text|>';
	const withRest = (p: number[]) => [...p, Math.max(0, 1 - p.reduce((a, b) => a + b, 0))];
	return [
		{
			id: 'first',
			label: 'First token of the reply',
			tokens: ['The', 'Paris', ' What', '\n', 'I', eod, '(rest of vocab)'],
			p0: withRest([0.06, 0.03, 0.18, 0.22, 0.09, 0.05]),
			p1: withRest([0.78, 0.15, 0.004, 0.01, 0.02, 0.0005])
		},
		{
			id: 'stop',
			label: 'After “Paris.”: does it stop?',
			tokens: [eot, ' It', '\n\n', ' The', eod, '(rest of vocab)'],
			p0: withRest([0.004, 0.18, 0.31, 0.12, 0.06]),
			p1: withRest([0.93, 0.03, 0.02, 0.01, 0.0005])
		}
	];
}

/** Steps shown in the training scrubber (≈ 3 epochs of a small SFT run). */
export const TRAIN_STEPS = 300;
