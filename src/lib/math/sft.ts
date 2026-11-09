/**
 * SFT objective and the simulation used by chapter 2.
 *
 * Objective (docs/research/sft.md): cross-entropy on next tokens, masked to the assistant span:
 *   L = -(1 / Σ m_t) Σ m_t log p(x_t | x_<t)
 * Masked positions carry label -100 in HF/TRL and contribute nothing.
 */

export type Reduction = 'mean' | 'sum';

/** Masked negative log-likelihood. `probs[t]` is the probability the model gave the true token t. */
export function maskedNLL(probs: number[], mask: boolean[], reduction: Reduction = 'mean'): number {
	let sum = 0;
	let n = 0;
	probs.forEach((p, i) => {
		if (!mask[i]) return;
		sum += -Math.log(Math.max(p, 1e-12));
		n += 1;
	});
	if (n === 0) return 0;
	return reduction === 'mean' ? sum / n : sum;
}

const logit = (p: number) => Math.log(p / (1 - p));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/**
 * Simulated learning: the probability of a target token moves from its base-model value p0 toward
 * a fitted value p1 along the logit, with progress 1 - exp(-step / tau).
 */
export function progress(step: number, tau: number): number {
	return 1 - Math.exp(-Math.max(step, 0) / tau);
}

export function interpolateTokenProb(p0: number, p1: number, alpha: number): number {
	const c = (p: number) => Math.min(Math.max(p, 1e-6), 1 - 1e-6);
	return sigmoid((1 - alpha) * logit(c(p0)) + alpha * logit(c(p1)));
}

/** Geometric interpolation between two distributions (linear in log-space, renormalised). */
export function interpolateDistribution(p0: number[], p1: number[], alpha: number): number[] {
	const logs = p0.map(
		(p, i) => (1 - alpha) * Math.log(Math.max(p, 1e-9)) + alpha * Math.log(Math.max(p1[i], 1e-9))
	);
	const max = Math.max(...logs);
	const e = logs.map((l) => Math.exp(l - max));
	const z = e.reduce((a, b) => a + b, 0);
	return e.map((x) => x / z);
}

export interface PackedBin {
	items: { id: number; length: number }[];
	used: number;
}

/**
 * Best-fit decreasing packing (TRL `packing_strategy="bfd"`): sort by length (desc), put each sequence
 * in the fullest bin that still has room, else open a new bin. Sequences longer than the capacity
 * are truncated to it (as TRL does).
 */
export function bfdPack(lengths: number[], capacity: number): PackedBin[] {
	const items = lengths
		.map((length, id) => ({ id, length: Math.min(length, capacity) }))
		.sort((a, b) => b.length - a.length);
	const bins: PackedBin[] = [];
	for (const it of items) {
		let best: PackedBin | undefined;
		for (const b of bins) {
			if (capacity - b.used >= it.length && (!best || b.used > best.used)) best = b;
		}
		if (!best) {
			best = { items: [], used: 0 };
			bins.push(best);
		}
		best.items.push(it);
		best.used += it.length;
	}
	return bins;
}

/** Fraction of token slots that hold real tokens. */
export function utilization(rowsUsed: number[], rowCapacity: number): number {
	const used = rowsUsed.reduce((a, b) => a + b, 0);
	return used / (rowsUsed.length * rowCapacity);
}
