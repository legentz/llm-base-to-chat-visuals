<script lang="ts">
	/**
	 * Diverging n×n heatmap (row-major values). Sign picks the colour token, |v| / max the opacity,
	 * so zero is the card background in both themes.
	 */
	let {
		values,
		n,
		max,
		label,
		pos = 'var(--c-lora)',
		neg = 'var(--c-base)',
		size = 168
	}: {
		values: ArrayLike<number>;
		n: number;
		max: number;
		label: string;
		pos?: string;
		neg?: string;
		size?: number;
	} = $props();

	const cells = $derived(Array.from({ length: n * n }, (_, i) => i));
</script>

<figure class="heat">
	<svg viewBox={`0 0 ${n} ${n}`} width={size} height={size} role="img" aria-label={label}>
		<rect width={n} height={n} class="frame" />
		{#each cells as i (i)}
			{@const v = values[i]}
			<rect
				x={i % n}
				y={Math.floor(i / n)}
				width="1.02"
				height="1.02"
				fill={v >= 0 ? pos : neg}
				fill-opacity={Math.min(1, Math.abs(v) / max)}
			/>
		{/each}
	</svg>
	<figcaption>{label}</figcaption>
</figure>

<style>
	.heat {
		margin: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		min-width: 0;
	}
	svg {
		max-width: 100%;
		height: auto;
		shape-rendering: crispEdges;
		border: 1px solid var(--line);
		border-radius: 4px;
	}
	.frame {
		fill: var(--surface);
	}
	figcaption {
		font-size: 12px;
		color: var(--ink-2);
		text-align: center;
	}
</style>
