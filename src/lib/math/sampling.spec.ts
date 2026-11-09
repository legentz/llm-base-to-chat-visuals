import { describe, expect, it } from 'vitest';
import { applySampler, defaultSampler, entropyBits, logitsFromProbs, softmax } from './sampling';

// Example distribution from docs/research/base-vs-instruct.md (top-10, sums to 1.0).
const probs = [0.42, 0.21, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01];
const logits = logitsFromProbs(probs);
const keptCount = (k: boolean[]) => k.filter(Boolean).length;

describe('softmax / temperature', () => {
	it('recovers the original probabilities at T = 1', () => {
		softmax(logits).forEach((p, i) => expect(p).toBeCloseTo(probs[i], 10));
	});
	it('sharpens for T < 1 and flattens for T > 1', () => {
		expect(softmax(logits, 0.5)[0]).toBeGreaterThan(0.42);
		expect(softmax(logits, 2)[0]).toBeLessThan(0.42);
		expect(entropyBits(softmax(logits, 2))).toBeGreaterThan(entropyBits(probs));
	});
	it('approaches greedy as T → 0', () => {
		expect(softmax(logits, 0.01)[0]).toBeGreaterThan(0.999);
	});
});

describe('truncation', () => {
	it('top-k keeps k tokens', () => {
		expect(keptCount(applySampler(logits, { ...defaultSampler, topK: 3 }).kept)).toBe(3);
	});
	it('top-p = 0.9 keeps 6 tokens (0.42+…+0.04 = 0.92 ≥ 0.9)', () => {
		expect(keptCount(applySampler(logits, { ...defaultSampler, topP: 0.9 }).kept)).toBe(6);
	});
	it('min-p = 0.1 keeps tokens ≥ 0.042, i.e. 5 tokens', () => {
		expect(keptCount(applySampler(logits, { ...defaultSampler, minP: 0.1 }).kept)).toBe(5);
	});
	it('renormalises the kept set and always keeps the argmax', () => {
		const { final } = applySampler(logits, { ...defaultSampler, topK: 2 });
		expect(final.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
		expect(final[0]).toBeCloseTo(0.42 / 0.63, 10);
		const extreme = applySampler(logits, { ...defaultSampler, topP: 0.01 });
		expect(extreme.final[0]).toBe(1);
	});
});
