<script lang="ts">
	import { mulberry32 } from '#lib/math/rng.js';
	import { bfdPack, utilization } from '#lib/math/sft.js';

	const CAP = 2048;
	let seed = $state(3);
	let packed = $state(true);

	// 12 conversations with a long-tailed length distribution (most short, a few long).
	const lengths = $derived.by(() => {
		const r = mulberry32(seed);
		return Array.from({ length: 12 }, () => Math.round(80 + 1700 * r() ** 2.2));
	});
	const longest = $derived(Math.max(...lengths));
	const bins = $derived(bfdPack(lengths, CAP));
	const rows = $derived(
		packed
			? bins.map((b) => ({ items: b.items, cap: CAP }))
			: lengths.map((length, id) => ({ items: [{ id, length }], cap: longest }))
	);
	const util = $derived(
		packed
			? utilization(
					bins.map((b) => b.used),
					CAP
				)
			: utilization(lengths, longest)
	);
	const slots = $derived(packed ? bins.length * CAP : lengths.length * longest);
	const hue = (id: number) => `hsl(${(id * 47) % 360} 55% 55%)`;
</script>

<figure class="card packing">
	<div class="head">
		<div class="group" role="group" aria-label="Batching strategy">
			<button class="btn" aria-pressed={!packed} onclick={() => (packed = false)}
				>Pad to longest</button
			>
			<button class="btn" aria-pressed={packed} onclick={() => (packed = true)}
				>Pack (best-fit decreasing)</button
			>
			<button class="btn" onclick={() => (seed += 1)}>New batch ↻</button>
		</div>
		<p class="stat">
			<strong>{(util * 100).toFixed(0)}%</strong> of {slots.toLocaleString('en-US')} token slots hold
			real tokens
		</p>
	</div>
	<div class="rows">
		{#each rows as row, r (r)}
			<div class="row">
				{#each row.items as it (it.id)}
					<span
						class="seg"
						style:width={`${(100 * it.length) / row.cap}%`}
						style:background={hue(it.id)}
						title={`example ${it.id + 1}: ${it.length} tokens`}
					></span>
				{/each}
			</div>
		{/each}
	</div>
	<figcaption>
		Twelve chat examples. Padding gives each one its own row, filled with padding up to the longest.
		Packing concatenates them into {CAP}-token rows. With packing, attention has to be masked per
		example (or position ids reset, "padding-free") so one conversation can't attend to another.
		Grey = wasted compute.
	</figcaption>
</figure>

<style>
	.packing {
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
	.stat {
		margin: 0;
		font-size: 13px;
		color: var(--ink-2);
	}
	.rows {
		display: grid;
		gap: 3px;
		margin-top: 12px;
	}
	.row {
		display: flex;
		height: 14px;
		background: repeating-linear-gradient(
			45deg,
			var(--surface-2),
			var(--surface-2) 4px,
			var(--line) 4px,
			var(--line) 6px
		);
		border-radius: 3px;
		overflow: hidden;
	}
	.seg {
		height: 100%;
		border-right: 1px solid var(--surface);
		transition: width 0.3s;
	}
	figcaption {
		margin-top: 10px;
		font-size: 13px;
		color: var(--ink-3);
	}
</style>
