import { describe, expect, it } from 'vitest';
import {
	energyCaptured,
	INT4_LEVELS,
	loraParams,
	mse,
	NF4_LEVELS,
	presetModules,
	quantizeBlock,
	qwen3_8b,
	scaling,
	totalBytes,
	trainableParams,
	trainingMemory
} from './lora';
import { mulberry32 } from './rng';

describe('LoRA parameter counts', () => {
	it('r·(d_in + d_out) per matrix', () => {
		expect(loraParams(16, { dIn: 4096, dOut: 4096 })).toBe(131072);
	});
	it('Qwen3-8B all-linear: 75,776 r per layer × 36 layers', () => {
		expect(trainableParams(qwen3_8b, 1, presetModules['all-linear'])).toBe(75776 * 36);
		expect(trainableParams(qwen3_8b, 16, presetModules['all-linear'])).toBe(43_646_976);
	});
	it('q,v-only is a small fraction of all-linear', () => {
		const qv = trainableParams(qwen3_8b, 16, presetModules.qv);
		expect(qv).toBe(16 * (8192 + 5120) * 36);
	});
	it('scaling: α/r vs α/√r', () => {
		expect(scaling(32, 16)).toBe(2);
		expect(scaling(32, 16, 'rslora')).toBe(8);
	});
});

describe('training memory (no activations)', () => {
	const N = 8.2e9;
	it('full fine-tuning is 12 bytes/param, 16 with an fp32 master copy', () => {
		expect(totalBytes(trainingMemory('full', N, 0))).toBe(12 * N);
		expect(totalBytes(trainingMemory('full', N, 0, true))).toBe(16 * N);
	});
	it('LoRA ≈ bf16 base + 16 bytes per adapter param; QLoRA shrinks the base ~4×', () => {
		const a = 43.6e6;
		expect(totalBytes(trainingMemory('lora', N, a))).toBeCloseTo(2 * N + 16 * a, 0);
		const q = totalBytes(trainingMemory('qlora', N, a));
		expect(q).toBeLessThan(totalBytes(trainingMemory('lora', N, a)) / 3);
	});
});

describe('quantisation', () => {
	it('NF4 has 16 sorted levels including exact 0 and ±1', () => {
		expect(NF4_LEVELS).toHaveLength(16);
		expect(NF4_LEVELS).toContain(0);
		expect([...NF4_LEVELS].sort((a, b) => a - b)).toEqual(NF4_LEVELS);
		expect(INT4_LEVELS).toHaveLength(15);
	});
	it('NF4 has lower error than uniform int4 on Gaussian weights', () => {
		const r = mulberry32(1);
		const gauss = () => Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r());
		let eNf = 0;
		let eInt = 0;
		for (let b = 0; b < 200; b++) {
			const w = Array.from({ length: 64 }, gauss);
			eNf += mse(w, quantizeBlock(w, NF4_LEVELS));
			eInt += mse(w, quantizeBlock(w, INT4_LEVELS));
		}
		expect(eNf).toBeLessThan(eInt);
	});
});

describe('low-rank energy', () => {
	it('captures all energy at full rank and is monotone', () => {
		const s = [5, 3, 2, 1, 0.5];
		expect(energyCaptured(s, 5)).toBeCloseTo(1, 10);
		expect(energyCaptured(s, 2)).toBeGreaterThan(energyCaptured(s, 1));
	});
});
