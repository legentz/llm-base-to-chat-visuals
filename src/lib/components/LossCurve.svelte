<script lang="ts" module>
	/** Small line chart with a moving "current step" marker. Series share the x axis (steps). */
	export interface Series {
		name: string;
		color: string;
		values: number[];
		dashed?: boolean;
	}
</script>

<script lang="ts">
	import { line, scaleLinear } from 'd3';

	let {
		series,
		step,
		yLabel = 'loss',
		xLabel = 'training step',
		yMax = undefined,
		height = 180
	}: {
		series: Series[];
		step: number;
		yLabel?: string;
		xLabel?: string;
		yMax?: number;
		height?: number;
	} = $props();

	const W = 520;
	const m = { t: 10, r: 12, b: 30, l: 40 };
	const n = $derived(Math.max(...series.map((s) => s.values.length)) - 1);
	const top = $derived(yMax ?? Math.max(...series.flatMap((s) => s.values)) * 1.08);
	const x = $derived(
		scaleLinear()
			.domain([0, n])
			.range([m.l, W - m.r])
	);
	const y = $derived(
		scaleLinear()
			.domain([0, top])
			.range([height - m.b, m.t])
			.nice()
	);
	const path = (v: number[]) =>
		line<number>()
			.x((_, i) => x(i))
			.y((d) => y(d))(v) ?? '';
	const clamped = $derived(Math.min(step, n));
</script>

<svg viewBox={`0 0 ${W} ${height}`} class="curve" role="img" aria-label={`${yLabel} by ${xLabel}`}>
	{#each y.ticks(4) as tk (tk)}
		<line x1={m.l} x2={W - m.r} y1={y(tk)} y2={y(tk)} class="grid" />
		<text x={m.l - 6} y={y(tk) + 4} class="tick" text-anchor="end">{tk}</text>
	{/each}
	<text x={(m.l + W - m.r) / 2} y={height - 4} class="axis" text-anchor="middle">{xLabel}</text>
	<text x="10" y={m.t + 4} class="axis" transform={`rotate(-90 10 ${m.t + 4})`} text-anchor="end"
		>{yLabel}</text
	>
	{#each series as s (s.name)}
		<path d={path(s.values)} class="ghost" style:stroke={s.color} />
		<path
			d={path(s.values.slice(0, clamped + 1))}
			class="line"
			class:dashed={s.dashed}
			style:stroke={s.color}
		/>
		<circle cx={x(clamped)} cy={y(s.values[clamped])} r="4" style:fill={s.color} />
	{/each}
	<line x1={x(clamped)} x2={x(clamped)} y1={m.t} y2={height - m.b} class="cursor" />
</svg>
<ul class="legend">
	{#each series as s (s.name)}
		<li>
			<span class="sw" class:dashed={s.dashed} style:border-color={s.color}></span>{s.name}
			<span class="mono">{s.values[clamped].toFixed(3)}</span>
		</li>
	{/each}
</ul>

<style>
	.curve {
		width: 100%;
		height: auto;
		display: block;
	}
	.grid {
		stroke: var(--line);
	}
	.tick,
	.axis {
		font-size: 11px;
		fill: var(--ink-3);
	}
	.ghost {
		fill: none;
		stroke-width: 1.5;
		opacity: 0.18;
	}
	.line {
		fill: none;
		stroke-width: 2.2;
	}
	.dashed {
		stroke-dasharray: 5 4;
	}
	.cursor {
		stroke: var(--ink-3);
		stroke-dasharray: 2 3;
	}
	.legend {
		list-style: none;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		margin: 4px 0 0;
		padding: 0;
		font-size: 12px;
		color: var(--ink-2);
	}
	.legend li {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.sw {
		width: 16px;
		border-top: 2.5px solid;
	}
	.sw.dashed {
		border-top-style: dashed;
	}
</style>
