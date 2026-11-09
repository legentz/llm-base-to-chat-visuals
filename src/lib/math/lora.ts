/**
 * LoRA / QLoRA arithmetic for chapter 3 (sources: docs/research/lora.md).
 * ΔW = (α/r)·B·A with B ∈ R^{d_out×r} (init 0) and A ∈ R^{r×d_in}; trainable params r·(d_in + d_out).
 */

export interface LinearShape {
	name: string;
	dIn: number;
	dOut: number;
	kind: 'attn' | 'mlp';
}

export interface ModelConfig {
	name: string;
	layers: number;
	/** Per-layer linear projections (the matrices LoRA can target and QLoRA quantises). */
	linears: LinearShape[];
	/** Everything else, kept in 16-bit by QLoRA: token embeddings, untied lm_head, norms. */
	otherParams: number;
}

/**
 * Qwen3-8B from its config.json: hidden 4096, 36 layers, 32 query heads / 8 KV heads × head_dim 128,
 * intermediate (MLP) size 12288, vocab 151,936, tie_word_embeddings false.
 * Embedding + lm_head = 2 × 151,936 × 4096; norms = 36 × (2·4096 + 2·128) + 4096.
 * Total 8,190,735,360 (model card: 8.2B, 6.95B non-embedding).
 */
export const qwen3_8b: ModelConfig = {
	name: 'Qwen3-8B',
	layers: 36,
	otherParams: 2 * 151_936 * 4096 + 36 * (2 * 4096 + 2 * 128) + 4096,
	linears: [
		{ name: 'q_proj', dIn: 4096, dOut: 4096, kind: 'attn' },
		{ name: 'k_proj', dIn: 4096, dOut: 1024, kind: 'attn' },
		{ name: 'v_proj', dIn: 4096, dOut: 1024, kind: 'attn' },
		{ name: 'o_proj', dIn: 4096, dOut: 4096, kind: 'attn' },
		{ name: 'gate_proj', dIn: 4096, dOut: 12288, kind: 'mlp' },
		{ name: 'up_proj', dIn: 4096, dOut: 12288, kind: 'mlp' },
		{ name: 'down_proj', dIn: 12288, dOut: 4096, kind: 'mlp' }
	]
};

export const linearParams = (cfg: ModelConfig) =>
	cfg.layers * cfg.linears.reduce((a, l) => a + l.dIn * l.dOut, 0);
export const totalParams = (cfg: ModelConfig) => linearParams(cfg) + cfg.otherParams;

export const loraParams = (r: number, s: Pick<LinearShape, 'dIn' | 'dOut'>) => r * (s.dIn + s.dOut);

export type TargetPreset = 'qv' | 'attn' | 'all-linear';
export const presetModules: Record<TargetPreset, string[]> = {
	qv: ['q_proj', 'v_proj'],
	attn: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
	'all-linear': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
};

export function trainableParams(cfg: ModelConfig, r: number, modules: string[]): number {
	const perLayer = cfg.linears
		.filter((l) => modules.includes(l.name))
		.reduce((a, l) => a + loraParams(r, l), 0);
	return perLayer * cfg.layers;
}

export type Scaling = 'lora' | 'rslora';
/** Multiplier applied to BA: α/r (LoRA, PEFT default) or α/√r (rsLoRA). */
export function scaling(alpha: number, r: number, kind: Scaling = 'lora'): number {
	return kind === 'lora' ? alpha / r : alpha / Math.sqrt(r);
}

export type Method = 'full' | 'lora' | 'qlora';
export interface MemoryBreakdown {
	weights: number;
	grads: number;
	optimizer: number;
	master: number;
}

/**
 * Bits per parameter for 4-bit block quantisation with blocks of 64: one fp32 absmax per block adds
 * 32/64 = 0.5 bits. Double quantisation stores those constants as 8-bit, in blocks of 256 with one
 * fp32 constant each: 8/64 + 32/(64·256) ≈ 0.127 bits (QLoRA §3).
 */
export const quantBits = (doubleQuant: boolean) =>
	4 + (doubleQuant ? 8 / 64 + 32 / (64 * 256) : 32 / 64);
export const NF4_BITS = quantBits(true);

/**
 * Training memory in bytes, excluding activations, KV and framework overhead.
 * Full FT: bf16 weights (2) + bf16 grads (2) + fp32 Adam m,v (8) [+ fp32 master copy (4)].
 * LoRA/QLoRA: frozen base (bf16, or NF4 for the block linears) + adapters in fp32 with grads and Adam (4 + 4 + 8).
 */
export function trainingMemory(
	method: Method,
	cfg: ModelConfig,
	adapterParams: number,
	fp32Master = false
): MemoryBreakdown {
	const N = totalParams(cfg);
	if (method === 'full') {
		return {
			weights: 2 * N,
			grads: 2 * N,
			optimizer: 8 * N,
			master: fp32Master ? 4 * N : 0
		};
	}
	// QLoRA quantises the transformer-block linears to NF4; embeddings, lm_head and norms stay bf16.
	const base = method === 'lora' ? 2 * N : (NF4_BITS / 8) * linearParams(cfg) + 2 * cfg.otherParams;
	return {
		weights: base + 4 * adapterParams,
		grads: 4 * adapterParams,
		optimizer: 8 * adapterParams,
		master: 0
	};
}

export const totalBytes = (m: MemoryBreakdown) => m.weights + m.grads + m.optimizer + m.master;

/** The 16 NF4 code values (QLoRA / bitsandbytes): quantiles of N(0,1) normalised to [-1, 1]. */
export const NF4_LEVELS = [
	-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
	-0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
	0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
	0.7229568362236023, 1.0
];
/** Symmetric absmax int4: integers −7..7 scaled to [−1, 1] (15 usable levels). */
export const INT4_LEVELS = Array.from({ length: 15 }, (_, i) => (i - 7) / 7);

/** Block-wise absmax quantisation to the nearest code value; returns dequantised weights. */
export function quantizeBlock(w: number[], levels: number[]): number[] {
	const absmax = Math.max(...w.map(Math.abs)) || 1;
	return w.map((x) => {
		const v = x / absmax;
		let best = levels[0];
		for (const l of levels) if (Math.abs(l - v) < Math.abs(best - v)) best = l;
		return best * absmax;
	});
}

export const mse = (a: number[], b: number[]) =>
	a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / a.length;

/** Fraction of a matrix's energy (Σσ²) captured by its top-r singular values. */
export function energyCaptured(singular: number[], r: number): number {
	const tot = singular.reduce((a, s) => a + s * s, 0);
	return singular.slice(0, r).reduce((a, s) => a + s * s, 0) / tot;
}
