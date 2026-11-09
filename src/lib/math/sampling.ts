/**
 * Decoding / sampling transforms applied to a next-token distribution.
 * Order follows common practice (HF `generate`): temperature, then truncation (top-k, top-p, min-p),
 * then renormalise over the kept set.
 */

export interface SamplerSettings {
	/** Temperature T > 0. T → 0 approaches greedy decoding. */
	temperature: number;
	/** Keep the k most likely tokens; 0 disables. */
	topK: number;
	/** Nucleus: smallest set whose cumulative probability ≥ p; 1 disables. */
	topP: number;
	/** Keep tokens with p ≥ minP · p_max; 0 disables. */
	minP: number;
}

export const defaultSampler: SamplerSettings = { temperature: 1, topK: 0, topP: 1, minP: 0 };

export function softmax(logits: number[], temperature = 1): number[] {
	const t = Math.max(temperature, 1e-4);
	const scaled = logits.map((z) => z / t);
	const max = Math.max(...scaled);
	const exps = scaled.map((z) => Math.exp(z - max));
	const sum = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / sum);
}

/** Convert probabilities to logits (up to an additive constant). */
export function logitsFromProbs(probs: number[]): number[] {
	return probs.map((p) => Math.log(Math.max(p, 1e-12)));
}

export interface SampledDistribution {
	/** Probabilities after temperature, before truncation. */
	tempered: number[];
	/** Which tokens survive truncation. */
	kept: boolean[];
	/** Final renormalised distribution (0 for dropped tokens). */
	final: number[];
}

export function applySampler(logits: number[], s: SamplerSettings): SampledDistribution {
	const tempered = softmax(logits, s.temperature);
	const order = tempered.map((_, i) => i).sort((a, b) => tempered[b] - tempered[a]);
	const kept = tempered.map(() => true);

	if (s.topK > 0) order.slice(s.topK).forEach((i) => (kept[i] = false));

	if (s.topP < 1) {
		let cum = 0;
		for (const i of order) {
			if (cum >= s.topP) kept[i] = false;
			cum += tempered[i];
		}
	}

	if (s.minP > 0) {
		const threshold = s.minP * tempered[order[0]];
		tempered.forEach((p, i) => {
			if (p < threshold) kept[i] = false;
		});
	}

	// The most likely token always survives, whatever the settings.
	kept[order[0]] = true;

	const mass = tempered.reduce((a, p, i) => a + (kept[i] ? p : 0), 0);
	const final = tempered.map((p, i) => (kept[i] ? p / mass : 0));
	return { tempered, kept, final };
}

/** Shannon entropy in bits. */
export function entropyBits(probs: number[]): number {
	return -probs.reduce((a, p) => (p > 0 ? a + p * Math.log2(p) : a), 0);
}
