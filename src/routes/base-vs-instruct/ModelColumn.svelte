<script lang="ts">
	import ProbBars from '#lib/components/ProbBars.svelte';
	import TokenChip from '#lib/components/TokenChip.svelte';
	import type { Generation } from '#lib/data/generations.js';
	import { applySampler, logitsFromProbs, type SamplerSettings } from '#lib/math/sampling.js';

	let {
		gen,
		step,
		sampler,
		title,
		color
	}: { gen: Generation; step: number; sampler: SamplerSettings; title: string; color: string } =
		$props();

	const shown = $derived(Math.min(step, gen.steps.length));
	const output = $derived(gen.steps.slice(0, shown).map((s) => s.cands[s.pick].token));
	const done = $derived(step >= gen.steps.length);
	const current = $derived(done ? undefined : gen.steps[step]);
	const dist = $derived(
		current ? applySampler(logitsFromProbs(current.cands.map((c) => c.p)), sampler) : undefined
	);
	const pickedDropped = $derived(dist && current ? !dist.kept[current.pick] : false);
</script>

<div class="col card" style:--mc={color}>
	<header>
		<span class="dot"></span>
		<div>
			<strong>{title}</strong>
			<a class="repo mono" href={`https://huggingface.co/${gen.repo}`} rel="external noopener"
				>{gen.repo}</a
			>
		</div>
	</header>

	<p class="eyebrow">What the model receives</p>
	<div class="stream">
		{#each gen.input as t, i (i)}<TokenChip token={t} />{/each}{#each output as t, i (i)}<TokenChip
				token={t}
				variant={i === shown - 1 ? 'current' : 'output'}
				{color}
			/>{/each}
		{#if !done}<span class="caret" aria-hidden="true"></span>{/if}
	</div>

	<div class="status" aria-live="polite">
		{#if done && gen.stop.kind === 'eos'}
			<span class="badge stop">■ Stopped: emitted <code>{gen.stop.token}</code></span>
		{:else if done && gen.stop.kind === 'max_new_tokens'}
			<span class="badge cut">✂ Still going. Cut off at max_new_tokens = {gen.stop.limit}</span>
		{:else}
			<span class="muted">Generating token {step + 1}…</span>
		{/if}
	</div>

	<p class="eyebrow">Next-token distribution (top candidates)</p>
	{#if current && dist}
		<ProbBars
			tokens={current.cands.map((c) => c.token)}
			values={dist.final}
			kept={dist.kept}
			highlight={current.pick}
			{color}
		/>
		{#if pickedDropped}
			<p class="warn">
				With these sampler settings the token on the recorded path could never be sampled.
			</p>
		{/if}
	{:else}
		<p class="muted">Generation finished.</p>
	{/if}
</div>

<style>
	.col {
		padding: 16px;
		display: flex;
		flex-direction: column;
		gap: 8px;
		border-top: 4px solid var(--mc);
		min-width: 0;
	}
	header {
		display: flex;
		gap: 10px;
		align-items: center;
	}
	header div {
		display: flex;
		flex-direction: column;
	}
	.dot {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--mc);
		flex: none;
	}
	.repo {
		font-size: 12px;
		color: var(--ink-3);
	}
	.eyebrow {
		margin: 6px 0 0;
	}
	.stream {
		min-height: 120px;
		padding: 8px;
		background: var(--bg);
		border: 1px solid var(--line);
		border-radius: 8px;
		line-height: 1.9;
	}
	.caret {
		display: inline-block;
		width: 8px;
		height: 16px;
		vertical-align: middle;
		background: var(--mc);
		animation: blink 1s steps(2) infinite;
	}
	@keyframes blink {
		50% {
			opacity: 0;
		}
	}
	.status {
		min-height: 26px;
	}
	.badge {
		display: inline-block;
		padding: 2px 8px;
		border-radius: 6px;
		font-size: 13px;
	}
	.stop {
		background: color-mix(in srgb, var(--c-instruct) 16%, transparent);
		color: var(--c-instruct);
	}
	.cut {
		background: color-mix(in srgb, var(--c-rejected) 14%, transparent);
		color: var(--c-rejected);
	}
	.muted {
		color: var(--ink-3);
		font-size: 13px;
		margin: 0;
	}
	.warn {
		margin: 4px 0 0;
		font-size: 12px;
		color: var(--c-rejected);
	}
</style>
