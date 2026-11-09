<script lang="ts">
	import ProbBars from '#lib/components/ProbBars.svelte';
	import Slider from '#lib/components/Slider.svelte';
	import {
		applySampler,
		defaultSampler,
		entropyBits,
		logitsFromProbs
	} from '#lib/math/sampling.js';

	// Worked example from docs/research/base-vs-instruct.md (top-10 next-token probabilities, T = 1).
	const tokens = [
		' Paris',
		' a',
		' the',
		' located',
		' known',
		' one',
		' home',
		' famous',
		' not',
		' called'
	];
	const logits = logitsFromProbs([0.42, 0.21, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]);

	let s = $state({ ...defaultSampler });
	const out = $derived(applySampler(logits, s));
	const keptN = $derived(out.kept.filter(Boolean).length);
</script>

<div class="lab card">
	<p class="eyebrow">Sampler lab · next token after “The capital of France is”</p>
	<div class="grid">
		<div>
			<p class="sub">After temperature</p>
			<ProbBars {tokens} values={out.tempered} color="var(--c-base)" />
		</div>
		<div>
			<p class="sub">After truncation + renormalising ({keptN} kept)</p>
			<ProbBars {tokens} values={out.final} kept={out.kept} color="var(--c-instruct)" />
		</div>
	</div>
	<div class="sliders">
		<Slider label="Temperature" bind:value={s.temperature} min={0.05} max={2} />
		<Slider
			label="Top-k"
			bind:value={s.topK}
			min={0}
			max={10}
			step={1}
			format={(v) => (v ? String(v) : 'off')}
		/>
		<Slider
			label="Top-p"
			bind:value={s.topP}
			min={0.05}
			max={1}
			format={(v) => (v >= 1 ? 'off' : v.toFixed(2))}
		/>
		<Slider
			label="Min-p"
			bind:value={s.minP}
			min={0}
			max={0.5}
			format={(v) => (v ? v.toFixed(2) : 'off')}
		/>
	</div>
	<p class="foot">
		Entropy {entropyBits(out.final).toFixed(2)} bits · try top-p 0.90 (keeps 6) or min-p 0.10 (keeps 5)
		at T = 1.
	</p>
</div>

<style>
	.lab {
		padding: 16px;
		margin: 1.5em 0;
	}
	.lab .eyebrow {
		margin: 0 0 8px;
	}
	.grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}
	@media (max-width: 640px) {
		.grid {
			grid-template-columns: 1fr;
		}
	}
	.sub {
		margin: 0 0 4px;
		font-size: 13px;
		color: var(--ink-2);
	}
	.sliders {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 20px;
		margin-top: 12px;
	}
	.foot {
		margin: 10px 0 0;
		font-size: 12px;
		color: var(--ink-3);
	}
</style>
