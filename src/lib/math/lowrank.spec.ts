import { describe, expect, it } from 'vitest';
import {
	bestRankError,
	frob2,
	learnedFractions,
	lowRankTarget,
	orthonormal,
	trainFactorization
} from './lowrank';

const n = 16;
const sigmas = Array.from({ length: n }, (_, i) => 3 * 0.8 ** i);
const tg = lowRankTarget(n, sigmas, 5);

describe('toy LoRA training', () => {
	it('orthonormal vectors are orthonormal', () => {
		const U = orthonormal(8, 4, 3);
		for (let a = 0; a < 4; a++)
			for (let b = 0; b < 4; b++) {
				const d = U[a].reduce((s, x, i) => s + x * U[b][i], 0);
				expect(d).toBeCloseTo(a === b ? 1 : 0, 10);
			}
	});
	it('ΔW is exactly zero at step 0 (B = 0)', () => {
		const tr = trainFactorization(tg, 4, 5, 0.05);
		expect(frob2(tr.snapshots[0])).toBe(0);
		expect(tr.relError[0]).toBeCloseTo(1, 10);
	});
	it('converges to the best rank-r error (Eckart–Young) and never beats it', () => {
		for (const r of [1, 4, 8]) {
			const tr = trainFactorization(tg, r, 600, 0.06);
			const best = bestRankError(sigmas, r);
			expect(tr.relError.at(-1)!).toBeGreaterThanOrEqual(best - 1e-6);
			expect(tr.relError.at(-1)!).toBeCloseTo(best, 2);
		}
	});
	it('learns the largest directions first and leaves those beyond rank r untouched', () => {
		const tr = trainFactorization(tg, 4, 600, 0.06);
		const early = learnedFractions(tg, tr.snapshots[40]);
		expect(early[0]).toBeGreaterThan(early[3]);
		const late = learnedFractions(tg, tr.snapshots[600]);
		late.slice(0, 4).forEach((f) => expect(f).toBeCloseTo(1, 2));
		late.slice(4).forEach((f) => expect(Math.abs(f)).toBeLessThan(0.02));
	});
});
