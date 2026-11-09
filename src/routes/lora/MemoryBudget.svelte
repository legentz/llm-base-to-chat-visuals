<script lang="ts">
	import Slider from '#lib/components/Slider.svelte';
	import {
		presetModules,
		qwen3_8b,
		totalBytes,
		trainableParams,
		trainingMemory,
		type Method
	} from '#lib/math/lora.js';

	const RANKS = [4, 8, 16, 32, 64, 128, 256];
	let rIdx = $state(2);
	let master = $state(true);
	const r = $derived(RANKS[rIdx]);
	const adapter = $derived(trainableParams(qwen3_8b, r, presetModules['all-linear']));

	const methods: { id: Method; label: string }[] = [
		{ id: 'full', label: 'Full fine-tune (bf16 + AdamW)' },
		{ id: 'lora', label: 'LoRA (bf16 base)' },
		{ id: 'qlora', label: 'QLoRA (NF4 base)' }
	];
	const parts = [
		{ key: 'weights', label: 'weights', color: 'var(--c-base)' },
		{ key: 'grads', label: 'gradients', color: 'var(--c-ref)' },
		{ key: 'optimizer', label: 'Adam m, v (fp32)', color: 'var(--c-reward)' },
		{ key: 'master', label: 'fp32 master copy', color: 'var(--c-masked)' }
	] as const;
	const rows = $derived(
		methods.map((m) => {
			const b = trainingMemory(m.id, qwen3_8b, adapter, m.id === 'full' && master);
			return { ...m, b, total: totalBytes(b) };
		})
	);
	const MAX = 140e9;
	const gpus = [
		{ gb: 24, name: '24 GB' },
		{ gb: 48, name: '48 GB' },
		{ gb: 80, name: '80 GB' }
	];
	const gb = (x: number) => (x / 1e9).toFixed(x < 10e9 ? 1 : 0);
</script>

<figure class="card budget">
	<div class="head">
		<Slider
			label="LoRA rank (all-linear)"
			bind:value={rIdx}
			min={0}
			max={RANKS.length - 1}
			step={1}
			format={(i) => `r = ${RANKS[i]}`}
		/>
		<label class="toggle">
			<input type="checkbox" bind:checked={master} />
			full FT keeps an fp32 master copy
		</label>
	</div>

	<div class="chart">
		{#each gpus as g (g.gb)}
			<div class="gpu" style:left={`${(100 * g.gb * 1e9) / MAX}%`}>
				<span>{g.name}</span>
			</div>
		{/each}
		{#each rows as row (row.id)}
			<div class="row">
				<div class="rlabel">
					<span>{row.label}</span><strong>{gb(row.total)} GB</strong>
				</div>
				<div class="bar">
					{#each parts as p (p.key)}
						{@const v = row.b[p.key]}
						{#if v > 0}
							<span
								class="seg"
								style:width={`${(100 * v) / MAX}%`}
								style:background={p.color}
								title={`${p.label}: ${gb(v)} GB`}
							></span>
						{/if}
					{/each}
				</div>
			</div>
		{/each}
	</div>
	<ul class="legend">
		{#each parts as p (p.key)}
			<li><span class="sw" style:background={p.color}></span>{p.label}</li>
		{/each}
	</ul>
	<figcaption>
		Qwen3-8B (8.19B parameters), memory for weights, gradients and optimizer state only. LoRA
		adapters ({(adapter / 1e6).toFixed(1)}M parameters at r = {r}) are trained in fp32: 4 + 4 + 8
		bytes each. QLoRA stores the 6.95B block-linear weights in NF4 (≈4.127 bits with double
		quantisation) and keeps embeddings and the output head in bf16. Activations come on top and grow
		with batch size × sequence length; gradient checkpointing trades them for recompute.
	</figcaption>
</figure>

<style>
	.budget {
		padding: 16px;
		margin: 1.5em 0;
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 24px;
		align-items: flex-end;
	}
	.head :global(.slider) {
		max-width: 260px;
	}
	.toggle {
		font-size: 13px;
		display: flex;
		gap: 6px;
		align-items: center;
		color: var(--ink-2);
	}
	.chart {
		position: relative;
		margin-top: 30px;
		display: grid;
		gap: 12px;
	}
	.gpu {
		position: absolute;
		top: -18px;
		bottom: 0;
		border-left: 1px dashed var(--ink-3);
		pointer-events: none;
	}
	.gpu span {
		position: absolute;
		top: 0;
		left: 4px;
		font-size: 11px;
		font-family: var(--mono);
		color: var(--ink-3);
		white-space: nowrap;
	}
	.rlabel {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		font-size: 13px;
		margin-bottom: 3px;
		position: relative;
		background: var(--surface);
		padding-right: 2px;
		width: fit-content;
		min-width: min(100%, 300px);
	}
	.rlabel strong {
		font-family: var(--mono);
	}
	.bar {
		display: flex;
		height: 18px;
		background: var(--surface-2);
		border-radius: 4px;
		overflow: hidden;
	}
	.seg {
		height: 100%;
		transition: width 0.3s;
	}
	.legend {
		list-style: none;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		margin: 12px 0 0;
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
		width: 12px;
		height: 12px;
		border-radius: 3px;
	}
	figcaption {
		margin-top: 10px;
		font-size: 13px;
		color: var(--ink-3);
	}
</style>
