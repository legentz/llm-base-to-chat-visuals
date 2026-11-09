import { describe, expect, it } from 'vitest';
import {
	bfdPack,
	interpolateDistribution,
	interpolateTokenProb,
	maskedNLL,
	progress,
	utilization
} from './sft';

describe('maskedNLL', () => {
	const probs = [0.01, 0.5, 0.25, 0.9];
	const mask = [false, true, true, false];
	it('ignores masked positions (label -100)', () => {
		expect(maskedNLL(probs, mask)).toBeCloseTo((Math.log(2) + Math.log(4)) / 2, 10);
	});
	it('sum reduction adds instead of averaging', () => {
		expect(maskedNLL(probs, mask, 'sum')).toBeCloseTo(Math.log(8), 10);
	});
	it('returns 0 when nothing is trainable', () => {
		expect(maskedNLL(probs, [false, false, false, false])).toBe(0);
	});
});

describe('learning simulation', () => {
	it('starts at p0 and converges to p1', () => {
		expect(interpolateTokenProb(0.02, 0.95, progress(0, 50))).toBeCloseTo(0.02, 6);
		expect(interpolateTokenProb(0.02, 0.95, progress(1e6, 50))).toBeCloseTo(0.95, 6);
	});
	it('is monotone in the number of steps', () => {
		const a = interpolateTokenProb(0.02, 0.95, progress(20, 50));
		const b = interpolateTokenProb(0.02, 0.95, progress(40, 50));
		expect(b).toBeGreaterThan(a);
	});
	it('interpolated distributions stay normalised', () => {
		const d = interpolateDistribution([0.7, 0.2, 0.1], [0.05, 0.05, 0.9], 0.5);
		expect(d.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 10);
	});
});

describe('BFD packing', () => {
	it('packs every sequence, never overflows, and beats one-sequence-per-row padding', () => {
		const lengths = [900, 120, 640, 300, 1500, 220, 760, 410];
		const bins = bfdPack(lengths, 2048);
		expect(bins.flatMap((b) => b.items).length).toBe(lengths.length);
		expect(bins.every((b) => b.used <= 2048)).toBe(true);
		const packed = utilization(
			bins.map((b) => b.used),
			2048
		);
		const padded = utilization(lengths, Math.max(...lengths));
		expect(packed).toBeGreaterThan(padded);
		expect(bins.length).toBe(3); // 4850 tokens need at least 3 rows of 2048
	});
	it('truncates sequences longer than the context', () => {
		expect(bfdPack([5000], 2048)[0].used).toBe(2048);
	});
});
