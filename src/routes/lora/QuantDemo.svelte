<script lang="ts">
	import { INT4_LEVELS, mse, NF4_LEVELS, quantBits, quantizeBlock } from '#lib/math/lora.js';
	import { gaussian } from '#lib/math/lowrank.js';
	import { mulberry32 } from '#lib/math/rng.js';

	const BLOCK = 64;
	let seed = $state(11);
	let scheme: 'nf4' | 'int4' = $state('nf4');
	let doubleQuant = $state(true);

	const block = $derived.by(() => {
		const g = gaussian(mulberry32(seed));
		return Array.from({ length: BLOCK }, g);
	});
	const absmax = $derived(Math.max(...block.map(Math.abs)));
	const levels = $derived(scheme === 'nf4' ? NF4_LEVELS : INT4_LEVELS);
	const q = $derived(quantizeBlock(block, levels));

	// Average error over many Gaussian blocks: a fixed property of each code book.
	const avg = (() => {
		const g = gaussian(mulberry32(99));
		let nf = 0;
		let i4 = 0;
		const B = 400;
		for (let b = 0; b < B; b++) {
			const w = Array.from({ length: BLOCK }, g);
			nf += mse(w, quantizeBlock(w, NF4_LEVELS));
			i4 += mse(w, quantizeBlock(w, INT4_LEVELS));
		}
		return { nf4: nf / B, int4: i4 / B };
	})();

	const X = (v: number) => 30 + ((v / absmax + 1) / 2) * 540;
	// Stack original values into a dot histogram so overlapping weights stay visible.
	const dots = $derived.by(() => {
		const count: Record<number, number> = {};
		return block.map((v, i) => {
			const bin = Math.round(X(v) / 7);
			const k = count[bin] ?? 0;
			count[bin] = k + 1;
			return { i, x: X(v), y: 52 - k * 7, xq: X(q[i]) };
		});
	});
	const bits = $derived(quantBits(doubleQuant));
</script>

<figure class="card quant">
	<div class="head">
		<div class="group" role="group" aria-label="Code book">
			<button class="btn" aria-pressed={scheme === 'nf4'} onclick={() => (scheme = 'nf4')}
				>NF4 (16 levels)</button
			>
			<button class="btn" aria-pressed={scheme === 'int4'} onclick={() => (scheme = 'int4')}
				>uniform int4</button
			>
			<button class="btn" onclick={() => (seed += 1)}>New block ↻</button>
		</div>
		<label class="toggle">
			<input type="checkbox" bind:checked={doubleQuant} />
			double quantisation
		</label>
	</div>

	<svg viewBox="0 0 600 170" role="img" aria-label="A block of 64 weights snapped to 4-bit levels">
		{#each dots as d (d.i)}
			<line x1={d.x} y1={d.y} x2={d.xq} y2="140" class="link" />
		{/each}
		{#each levels as l, i (i)}
			<line x1={X(l * absmax)} x2={X(l * absmax)} y1="96" y2="150" class="level" />
		{/each}
		{#each dots as d (d.i)}
			<circle cx={d.x} cy={d.y} r="3" class="orig" />
			<circle cx={d.xq} cy="140" r="3.4" class="snapped" />
		{/each}
		<text x="30" y="166" class="ax">−absmax</text>
		<text x="300" y="166" class="ax" text-anchor="middle">0</text>
		<text x="570" y="166" class="ax" text-anchor="end">+absmax</text>
		<text x="596" y="56" class="ax" text-anchor="end">bf16 weights</text>
		<text x="596" y="128" class="ax" text-anchor="end">4-bit codes</text>
	</svg>

	<dl class="stats">
		<div>
			<dt>error on this block (MSE)</dt>
			<dd>{mse(block, q).toFixed(4)}</dd>
		</div>
		<div>
			<dt>average over 400 blocks</dt>
			<dd>
				NF4 {avg.nf4.toFixed(4)} · int4 {avg.int4.toFixed(4)}
			</dd>
		</div>
		<div>
			<dt>storage per weight</dt>
			<dd>{bits.toFixed(3)} bits</dd>
		</div>
	</dl>
	<figcaption>
		One block of 64 weights drawn from a normal distribution, as pretrained weights roughly are.
		Each block is divided by its absolute maximum, so the largest weight lands exactly on ±1, and
		every value snaps to the nearest of 16 codes. NF4 puts its codes at quantiles of a normal
		distribution, packed densely where most weights live, with an exact zero. Uniform int4 spaces
		them evenly. The per-block absmax is a 32-bit constant, 0.5 extra bits per weight; double
		quantisation stores those constants in 8 bits (blocks of 256) and brings the overhead to about
		0.127 bits.
	</figcaption>
</figure>

<style>
	.quant {
		padding: 16px;
		margin: 1.5em 0;
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 10px;
		align-items: center;
	}
	.group {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.toggle {
		font-size: 13px;
		display: flex;
		gap: 6px;
		align-items: center;
		color: var(--ink-2);
	}
	svg {
		width: 100%;
		height: auto;
		margin-top: 8px;
		font-family: var(--font);
	}
	.link {
		stroke: var(--ink-3);
		stroke-opacity: 0.3;
	}
	.level {
		stroke: var(--c-lora);
		stroke-opacity: 0.55;
	}
	.orig {
		fill: var(--c-base);
	}
	.snapped {
		fill: var(--c-lora);
	}
	.ax {
		font-size: 11px;
		font-family: var(--mono);
		fill: var(--ink-3);
	}
	.stats {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 8px;
		margin: 6px 0 0;
	}
	@media (max-width: 640px) {
		.stats {
			grid-template-columns: 1fr;
		}
	}
	.stats div {
		background: var(--surface-2);
		border-radius: 8px;
		padding: 8px 10px;
	}
	dt {
		font-size: 11.5px;
		color: var(--ink-3);
	}
	dd {
		margin: 2px 0 0;
		font-family: var(--mono);
		font-size: 14px;
		font-weight: 600;
	}
	figcaption {
		margin-top: 10px;
		font-size: 13px;
		color: var(--ink-3);
	}
</style>
