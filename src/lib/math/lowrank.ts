/**
 * Toy LoRA training for chapter 3: fit ΔW = B·A (n×r · r×n) to a target update T by plain
 * gradient descent on ½‖BA − T‖²_F, starting from B = 0 (as in LoRA), so ΔW = 0 at step 0.
 * Matrices are row-major Float64Arrays of length n·n. The α/r scale is folded into the learning rate.
 */
import { mulberry32 } from './rng.js';

export function gaussian(rnd: () => number): () => number {
	return () => Math.sqrt(-2 * Math.log(rnd() || 1e-12)) * Math.cos(2 * Math.PI * rnd());
}

/** k orthonormal vectors of length n (Gram–Schmidt on seeded Gaussian vectors). */
export function orthonormal(n: number, k: number, seed: number): number[][] {
	const g = gaussian(mulberry32(seed));
	const out: number[][] = [];
	while (out.length < k) {
		const v = Array.from({ length: n }, g);
		for (const u of out) {
			const d = u.reduce((a, x, i) => a + x * v[i], 0);
			for (let i = 0; i < n; i++) v[i] -= d * u[i];
		}
		const norm = Math.hypot(...v);
		if (norm > 1e-8) out.push(v.map((x) => x / norm));
	}
	return out;
}

export interface Target {
	n: number;
	T: Float64Array;
	U: number[][];
	V: number[][];
	sigmas: number[];
}

/** T = Σ σ_i u_i v_iᵀ with known singular vectors, so we can measure what was learned. */
export function lowRankTarget(n: number, sigmas: number[], seed: number): Target {
	const U = orthonormal(n, sigmas.length, seed);
	const V = orthonormal(n, sigmas.length, seed + 1000);
	const T = new Float64Array(n * n);
	sigmas.forEach((s, k) => {
		for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) T[i * n + j] += s * U[k][i] * V[k][j];
	});
	return { n, T, U, V, sigmas };
}

export const frob2 = (M: ArrayLike<number>) => {
	let s = 0;
	for (let i = 0; i < M.length; i++) s += M[i] * M[i];
	return s;
};

export interface Trajectory {
	/** ΔW after each step; snapshots[0] is the initial (zero) update. */
	snapshots: Float32Array[];
	/** Relative error ‖T − ΔW‖² / ‖T‖² after each step. */
	relError: number[];
}

export function trainFactorization(
	tg: Target,
	r: number,
	steps: number,
	lr: number,
	seed = 1,
	initStd = 0.05
): Trajectory {
	const { n, T } = tg;
	const g = gaussian(mulberry32(seed));
	const B = new Float64Array(n * r); // n×r, zero init
	const A = Float64Array.from({ length: r * n }, () => initStd * g()); // r×n
	const P = new Float64Array(n * n);
	const E = new Float64Array(n * n);
	const gB = new Float64Array(n * r);
	const gA = new Float64Array(r * n);
	const tNorm = frob2(T);
	const snapshots: Float32Array[] = [];
	const relError: number[] = [];

	const product = () => {
		P.fill(0);
		for (let i = 0; i < n; i++)
			for (let k = 0; k < r; k++) {
				const b = B[i * r + k];
				if (b === 0) continue;
				for (let j = 0; j < n; j++) P[i * n + j] += b * A[k * n + j];
			}
		for (let i = 0; i < n * n; i++) E[i] = P[i] - T[i];
		snapshots.push(Float32Array.from(P));
		relError.push(frob2(E) / tNorm);
	};

	product();
	for (let s = 0; s < steps; s++) {
		// ∂L/∂B = E Aᵀ, ∂L/∂A = Bᵀ E
		gB.fill(0);
		gA.fill(0);
		for (let i = 0; i < n; i++)
			for (let k = 0; k < r; k++) {
				let acc = 0;
				for (let j = 0; j < n; j++) acc += E[i * n + j] * A[k * n + j];
				gB[i * r + k] = acc;
			}
		for (let k = 0; k < r; k++)
			for (let j = 0; j < n; j++) {
				let acc = 0;
				for (let i = 0; i < n; i++) acc += B[i * r + k] * E[i * n + j];
				gA[k * n + j] = acc;
			}
		for (let i = 0; i < B.length; i++) B[i] -= lr * gB[i];
		for (let i = 0; i < A.length; i++) A[i] -= lr * gA[i];
		product();
	}
	return { snapshots, relError };
}

/** u_kᵀ ΔW v_k / σ_k: how much of target direction k has been learned (1 = all of it). */
export function learnedFractions(tg: Target, dW: ArrayLike<number>): number[] {
	const { n, U, V, sigmas } = tg;
	return sigmas.map((s, k) => {
		let acc = 0;
		for (let i = 0; i < n; i++) {
			let row = 0;
			for (let j = 0; j < n; j++) row += dW[i * n + j] * V[k][j];
			acc += U[k][i] * row;
		}
		return acc / s;
	});
}

/** Best achievable relative error at rank r (Eckart–Young): energy of the dropped singular values. */
export function bestRankError(sigmas: number[], r: number): number {
	const tot = sigmas.reduce((a, s) => a + s * s, 0);
	return sigmas.slice(r).reduce((a, s) => a + s * s, 0) / tot;
}
