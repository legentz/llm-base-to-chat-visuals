<script lang="ts">
	import Slider from '#lib/components/Slider.svelte';
	import StepControls from '#lib/components/StepControls.svelte';
	import { promptCases } from '#lib/data/generations.js';
	import { defaultSampler, type SamplerSettings } from '#lib/math/sampling.js';
	import ModelColumn from './ModelColumn.svelte';

	let caseId = $state(promptCases[0].id);
	let step = $state(0);
	let sampler: SamplerSettings = $state({ ...defaultSampler, temperature: 0.7 });

	const pc = $derived(promptCases.find((c) => c.id === caseId)!);
	const maxStep = $derived(Math.max(pc.base.steps.length, pc.instruct.steps.length));

	function choose(id: string) {
		caseId = id;
		step = 0;
	}
</script>

<div class="compare">
	<div class="toolbar">
		<div class="prompts" role="group" aria-label="Prompt">
			{#each promptCases as c (c.id)}
				<button class="btn" aria-pressed={c.id === caseId} onclick={() => choose(c.id)}
					>{c.label}</button
				>
			{/each}
		</div>
		<StepControls bind:step max={maxStep} label="Token" />
	</div>

	<p class="prompt">
		<span class="eyebrow">User prompt</span>
		<code>{pc.userText.replaceAll('\n', '↵')}</code>
	</p>

	<div class="cols">
		<ModelColumn
			gen={pc.base}
			{step}
			{sampler}
			title="Base model (pretrained only)"
			color="var(--c-base)"
		/>
		<ModelColumn
			gen={pc.instruct}
			{step}
			{sampler}
			title="Post-trained chat model"
			color="var(--c-instruct)"
		/>
	</div>

	<div class="sampler card">
		<p class="eyebrow">Sampler: reshapes the distribution at the current step</p>
		<div class="sliders">
			<Slider label="Temperature" bind:value={sampler.temperature} min={0.05} max={2} />
			<Slider
				label="Top-k"
				bind:value={sampler.topK}
				min={0}
				max={8}
				step={1}
				format={(v) => (v === 0 ? 'off' : String(v))}
			/>
			<Slider
				label="Top-p"
				bind:value={sampler.topP}
				min={0.05}
				max={1}
				format={(v) => (v >= 1 ? 'off' : v.toFixed(2))}
			/>
			<Slider
				label="Min-p"
				bind:value={sampler.minP}
				min={0}
				max={0.5}
				format={(v) => (v === 0 ? 'off' : v.toFixed(2))}
			/>
		</div>
	</div>

	<p class="lesson">{pc.lesson}</p>
	<p class="illustrative">
		Illustrative simulation: token paths and probabilities are hand-written to match documented
		behaviour of base vs post-trained Qwen3 checkpoints. They aren't recorded outputs. The path is
		fixed; the sliders show how sampling would reshape each step.
	</p>
</div>

<style>
	.compare {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.toolbar {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
		justify-content: space-between;
		align-items: center;
	}
	.toolbar :global(.controls) {
		flex: 1;
		min-width: 280px;
	}
	.prompts {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.prompt {
		margin: 0;
		display: flex;
		gap: 10px;
		align-items: baseline;
		flex-wrap: wrap;
	}
	.prompt code {
		font-size: 15px;
		padding: 4px 10px;
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 6px;
	}
	.cols {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}
	@media (max-width: 820px) {
		.cols {
			grid-template-columns: 1fr;
		}
	}
	.sampler {
		padding: 12px 16px;
	}
	.sampler .eyebrow {
		margin: 0 0 8px;
	}
	.sliders {
		display: flex;
		flex-wrap: wrap;
		gap: 16px 24px;
	}
	.lesson {
		margin: 0;
		font-size: 1.05rem;
		color: var(--ink);
		max-width: 80ch;
	}
	.illustrative {
		margin: 0;
	}
</style>
