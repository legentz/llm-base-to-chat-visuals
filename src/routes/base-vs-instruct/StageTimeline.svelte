<script lang="ts">
	import { scaleLinear, scaleLog } from 'd3';

	/** Token budget per training stage, from the Olmo 3 paper and the SmolLM3 blog (see research notes). */
	const families = [
		{
			name: 'Olmo 3 · 7B',
			stages: [
				{ label: 'Pretraining', tokens: 5.93e12, kind: 'pre' },
				{ label: 'Mid-training', tokens: 100e9, kind: 'mid' },
				{ label: 'Long context', tokens: 50e9, kind: 'long' }
			]
		},
		{
			name: 'SmolLM3 · 3B',
			stages: [
				{ label: 'Pretraining (3 stages)', tokens: 11.2e12, kind: 'pre' },
				{ label: 'Long context', tokens: 100e9, kind: 'long' },
				{ label: 'Reasoning mid-training', tokens: 140e9, kind: 'mid' }
			]
		}
	];
	const colour: Record<string, string> = {
		pre: 'var(--c-base)',
		mid: 'var(--c-lora)',
		long: 'var(--c-ref)'
	};

	let log = $state(false);
	const max = 11.2e12;
	const x = $derived(
		log
			? scaleLog().domain([1e10, max]).range([0, 100])
			: scaleLinear().domain([0, max]).range([0, 100])
	);
	const fmt = (t: number) => (t >= 1e12 ? `${(t / 1e12).toFixed(2)}T` : `${Math.round(t / 1e9)}B`);
</script>

<figure class="card timeline">
	<div class="head">
		<p class="eyebrow">Tokens per training stage, before any post-training</p>
		<label><input type="checkbox" bind:checked={log} /> log scale</label>
	</div>
	{#each families as f (f.name)}
		<div class="fam">
			<strong>{f.name}</strong>
			{#each f.stages as s (s.label)}
				<div class="row">
					<span class="lbl">{s.label}</span>
					<span class="track"
						><span
							class="bar"
							style:width={`${Math.max(x(s.tokens), 0.4)}%`}
							style:background={colour[s.kind]}
						></span></span
					>
					<span class="val mono">{fmt(s.tokens)}</span>
				</div>
			{/each}
		</div>
	{/each}
	<figcaption>
		On a linear scale the later stages are barely visible. They're about 1–3% of the tokens, but
		they shape what a released “base” checkpoint looks like.
	</figcaption>
</figure>

<style>
	.timeline {
		padding: 16px;
		margin: 1.5em 0;
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 12px;
		font-size: 13px;
	}
	.head .eyebrow {
		margin: 0;
	}
	.fam {
		margin-top: 12px;
	}
	.row {
		display: grid;
		grid-template-columns: 170px 1fr 60px;
		gap: 8px;
		align-items: center;
		font-size: 13px;
		margin: 4px 0;
	}
	@media (max-width: 560px) {
		.row {
			grid-template-columns: 110px 1fr 52px;
		}
	}
	.lbl {
		color: var(--ink-2);
	}
	.track {
		height: 12px;
		background: var(--surface-2);
		border-radius: 3px;
	}
	.bar {
		display: block;
		height: 100%;
		border-radius: 3px;
		transition: width 0.4s ease;
	}
	.val {
		text-align: right;
		font-size: 12px;
	}
	figcaption {
		margin-top: 10px;
		font-size: 13px;
		color: var(--ink-3);
	}
</style>
