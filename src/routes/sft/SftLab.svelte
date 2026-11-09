<script lang="ts">
	import LossCurve from '#lib/components/LossCurve.svelte';
	import ProbBars from '#lib/components/ProbBars.svelte';
	import StepControls from '#lib/components/StepControls.svelte';
	import { conversation, probes, targetFor, TRAIN_STEPS } from '#lib/data/sft.js';
	import { mulberry32 } from '#lib/math/rng.js';
	import {
		interpolateDistribution,
		interpolateTokenProb,
		maskedNLL,
		progress
	} from '#lib/math/sft.js';
	import { renderChat, type TemplateId } from '#lib/templates.js';
	import { isSpecial, showToken } from '#lib/tokens.js';

	let template: TemplateId = $state('chatml');
	let thinking = $state(false);
	let inference = $state(false);
	let assistantOnly = $state(true);
	let step = $state(0);

	const tokens = $derived(
		renderChat(conversation, {
			template,
			addGenerationPrompt: inference,
			enableThinking: thinking && template === 'chatml'
		})
	);
	// Position 0 has no prediction (nothing before it), so it's never in the loss.
	const mask = $derived(tokens.map((tk, i) => i > 0 && (assistantOnly ? tk.trainable : true)));
	const targets = $derived(tokens.map((tk, i) => targetFor(tk, i)));

	const probAt = (s: number) =>
		targets.map((tg) => interpolateTokenProb(tg.p0, tg.p1, progress(s, tg.tau)));
	const probsNow = $derived(probAt(step));
	const nTrain = $derived(mask.filter(Boolean).length);

	// Curves: loss on this (memorisable) example, plus a held-out curve that plateaus well above it
	// and drifts up late (3 epochs → mild overfitting). Seeded noise: same curve on every render.
	const curves = $derived.by(() => {
		const rnd = mulberry32(7);
		const train: number[] = [];
		const val: number[] = [];
		for (let s = 0; s <= TRAIN_STEPS; s++) {
			const base = maskedNLL(probAt(s), mask);
			const noise = (rnd() - 0.5) * 0.12 * Math.exp(-s / 250);
			train.push(Math.max(0.01, base + noise));
			val.push(0.85 + 0.62 * base + 0.0012 * Math.max(0, s - 200) + noise * 0.5);
		}
		return { train, val };
	});

	const probeList = $derived(probes(template));
	const alphaProbe = $derived(progress(step, 110));

	function heat(p: number): string {
		// −log p mapped to [0, 1]: 0 (p≈1, learned) … 1 (p ≤ 0.005, surprising).
		const h = Math.min(1, -Math.log(Math.max(p, 1e-9)) / 5.3);
		return `color-mix(in srgb, var(--c-rejected) ${Math.round(h * 100)}%, var(--c-chosen))`;
	}
	const roleColor: Record<string, string> = {
		system: 'var(--c-ref)',
		user: 'var(--c-base)',
		assistant: 'var(--c-instruct)',
		template: 'var(--ink-3)'
	};
</script>

<div class="lab">
	<div class="toolbar">
		<div class="group" role="group" aria-label="Template">
			<span class="eyebrow">Template</span>
			<button class="btn" aria-pressed={template === 'chatml'} onclick={() => (template = 'chatml')}
				>ChatML · Qwen3</button
			>
			<button class="btn" aria-pressed={template === 'llama3'} onclick={() => (template = 'llama3')}
				>Llama 3</button
			>
		</div>
		<div class="group" role="group" aria-label="View">
			<span class="eyebrow">View</span>
			<button class="btn" aria-pressed={!inference} onclick={() => (inference = false)}
				>Training example</button
			>
			<button class="btn" aria-pressed={inference} onclick={() => (inference = true)}
				>Inference prompt</button
			>
		</div>
		<label class="toggle" class:disabled={template !== 'chatml'}>
			<input type="checkbox" bind:checked={thinking} disabled={template !== 'chatml'} />
			enable_thinking
		</label>
		<label class="toggle" class:disabled={inference}>
			<input type="checkbox" bind:checked={assistantOnly} disabled={inference} />
			assistant_only_loss
		</label>
	</div>

	<div class="card tokens">
		<div class="tokhead">
			<p class="eyebrow">
				{inference ? 'What the model sees at inference' : 'One training example, as tokens'}
			</p>
			{#if !inference}
				<p class="count">
					<strong>{nTrain}</strong> of {tokens.length} tokens contribute to the loss
				</p>
			{/if}
		</div>
		<div class="stream">
			{#each tokens as tk, i (i)}
				{@const on = !inference && mask[i]}
				<span
					class="tok"
					class:special={isSpecial(tk.text)}
					class:masked={!inference && !mask[i]}
					style:--rc={roleColor[tk.role]}
					style:--heat={on ? heat(probsNow[i]) : 'transparent'}
					title={on
						? `p = ${probsNow[i].toFixed(3)} · loss = ${(-Math.log(probsNow[i])).toFixed(2)}`
						: inference
							? tk.role
							: 'label = -100 (ignored by the loss)'}>{showToken(tk.text)}</span
				>
			{/each}
		</div>
		<ul class="legend">
			<li><span class="sw" style:background="var(--c-ref)"></span>system</li>
			<li><span class="sw" style:background="var(--c-base)"></span>user</li>
			<li><span class="sw" style:background="var(--c-instruct)"></span>assistant</li>
			<li><span class="sw" style:background="var(--ink-3)"></span>template scaffolding</li>
			{#if !inference}
				<li>
					<span class="grad"></span>per-token loss: high → low · greyed = masked (label −100)
				</li>
			{/if}
		</ul>
	</div>

	{#if !inference}
		<div class="train">
			<div class="card pane">
				<StepControls bind:step max={TRAIN_STEPS} interval={40} label="Step" />
				<LossCurve
					series={[
						{ name: 'loss on this example', color: 'var(--c-instruct)', values: curves.train },
						{ name: 'held-out loss', color: 'var(--c-lora)', values: curves.val, dashed: true }
					]}
					{step}
				/>
			</div>
			<div class="card pane probes">
				{#each probeList as pr (pr.id)}
					{@const d = interpolateDistribution(pr.p0, pr.p1, alphaProbe)}
					<div>
						<p class="eyebrow">{pr.label}</p>
						<ProbBars tokens={pr.tokens} values={d} color="var(--c-instruct)" />
					</div>
				{/each}
			</div>
		</div>
	{:else}
		<p class="note">
			At inference the assistant turn is empty: the prompt ends with the assistant header
			{#if template === 'chatml' && !thinking}
				and, with <code>enable_thinking=False</code>, a pre-filled empty
				<code>&lt;think&gt;&lt;/think&gt;</code> block{/if}. The model's first sampled token is the
			start of its reply, which is exactly the position SFT trained.
		</p>
	{/if}

	<p class="illustrative">
		Illustrative simulation: the templates are rendered exactly like the published ones, with
		word-level token boundaries instead of real BPE pieces. Probabilities and curves are synthetic
		but follow the documented pattern: special tokens start near zero and the facts are already
		known.
	</p>
</div>

<style>
	.lab {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.toolbar {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 20px;
		align-items: center;
	}
	.group {
		display: flex;
		align-items: center;
		gap: 6px;
		flex-wrap: wrap;
	}
	.group .eyebrow {
		margin-right: 4px;
	}
	.toggle {
		font-family: var(--mono);
		font-size: 13px;
		display: flex;
		gap: 6px;
		align-items: center;
	}
	.toggle.disabled {
		opacity: 0.4;
	}
	.tokens {
		padding: 14px 16px;
	}
	.tokhead {
		display: flex;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 8px;
	}
	.tokhead p {
		margin: 0;
	}
	.count {
		font-size: 13px;
		color: var(--ink-2);
	}
	.stream {
		margin: 10px 0;
		line-height: 2.1;
	}
	.tok {
		display: inline-block;
		margin: 1px;
		padding: 0 4px;
		border-radius: 4px;
		font-family: var(--mono);
		font-size: 13px;
		white-space: pre;
		color: var(--ink);
		border-bottom: 3px solid var(--rc);
		background: color-mix(in srgb, var(--heat) 30%, transparent);
		transition: background 0.2s;
	}
	.tok.special {
		color: var(--c-ref);
		font-weight: 500;
	}
	.tok.masked {
		opacity: 0.4;
	}
	.legend {
		list-style: none;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		margin: 0;
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
		width: 14px;
		height: 3px;
	}
	.grad {
		width: 40px;
		height: 10px;
		border-radius: 3px;
		background: linear-gradient(
			90deg,
			color-mix(in srgb, var(--c-rejected) 30%, transparent),
			color-mix(in srgb, var(--c-chosen) 30%, transparent)
		);
	}
	.train {
		display: grid;
		grid-template-columns: 1.2fr 1fr;
		gap: 16px;
	}
	@media (max-width: 820px) {
		.train {
			grid-template-columns: 1fr;
		}
	}
	.pane {
		padding: 14px 16px;
		display: flex;
		flex-direction: column;
		gap: 10px;
		min-width: 0;
	}
	.probes .eyebrow {
		margin: 0 0 6px;
	}
	.note {
		margin: 0;
		color: var(--ink-2);
	}
	.illustrative {
		margin: 0;
	}
</style>
