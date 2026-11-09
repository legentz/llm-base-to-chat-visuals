<script lang="ts">
	import { showToken } from '#lib/tokens.js';

	/**
	 * Horizontal bars for a next-token distribution. `values` are already-processed probabilities;
	 * `kept[i] = false` greys a token out (removed by truncation); `highlight` outlines the sampled one.
	 */
	let {
		tokens,
		values,
		kept = undefined,
		highlight = -1,
		color = 'var(--ink)',
		max = 1
	}: {
		tokens: string[];
		values: number[];
		kept?: boolean[];
		highlight?: number;
		color?: string;
		max?: number;
	} = $props();

	const pct = (v: number) => (100 * v) / max;
</script>

<ul class="bars" style:--bc={color}>
	{#each tokens as t, i (i)}
		<li class:dropped={kept && !kept[i]} class:hl={i === highlight}>
			<span class="tok">{showToken(t)}</span>
			<span class="track"><span class="fill" style:width={`${pct(values[i])}%`}></span></span>
			<span class="val">{(values[i] * 100).toFixed(1)}%</span>
		</li>
	{/each}
</ul>

<style>
	.bars {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 3px;
	}
	li {
		display: grid;
		grid-template-columns: minmax(70px, 110px) 1fr 52px;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		padding: 1px 4px;
		border-radius: 4px;
		border: 1px solid transparent;
	}
	li.hl {
		border-color: var(--bc);
	}
	.tok {
		font-family: var(--mono);
		white-space: pre;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.track {
		height: 10px;
		background: var(--surface-2);
		border-radius: 3px;
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
		background: var(--bc);
		border-radius: 3px;
		transition: width 0.3s ease;
	}
	.val {
		font-family: var(--mono);
		text-align: right;
		color: var(--ink-2);
	}
	.dropped {
		opacity: 0.35;
	}
	.dropped .fill {
		background: var(--c-masked);
	}
	@media (prefers-reduced-motion: reduce) {
		.fill {
			transition: none;
		}
	}
</style>
