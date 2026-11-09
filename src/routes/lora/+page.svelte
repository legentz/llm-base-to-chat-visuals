<script lang="ts">
	import ChapterShell from '#lib/components/ChapterShell.svelte';
	import Cite from '#lib/components/Cite.svelte';
	import Tex from '#lib/components/Tex.svelte';
	import {
		presetModules,
		qwen3_8b,
		scaling,
		totalParams,
		trainableParams,
		type TargetPreset
	} from '#lib/math/lora.js';
	import LoraLab from './LoraLab.svelte';
	import MemoryBudget from './MemoryBudget.svelte';
	import QuantDemo from './QuantDemo.svelte';

	const total = totalParams(qwen3_8b);
	const presetRows: { id: TargetPreset; label: string }[] = [
		{ id: 'qv', label: 'q_proj, v_proj' },
		{ id: 'attn', label: 'q, k, v, o' },
		{ id: 'all-linear', label: 'all-linear (attention + MLP)' }
	];
	const scaleRanks = [4, 16, 64, 256];
</script>

<ChapterShell slug="lora">
	{#snippet hero()}
		<LoraLab />
	{/snippet}

	<h2>Why not just fine-tune every weight?</h2>
	<p>
		Full fine-tuning keeps a gradient and two Adam moments for every parameter. In bf16 with fp32
		optimizer state that is about 12 bytes per parameter, 16 with an fp32 master copy of the
		weights: over 130 GB for an 8B model before a single activation is stored. Each fine-tune is
		also a full 16 GB checkpoint. LoRA's observation is that the <em>update</em> a fine-tune makes
		to a weight matrix tends to have low "intrinsic rank", so it can be written as the product of
		two thin matrices <Cite id="lora" />:
	</p>
	<Tex
		display
		expr={String.raw`h = W_0 x + \frac{\alpha}{r}\,B A x, \qquad B \in \mathbb{R}^{d_\text{out}\times r},\; A \in \mathbb{R}^{r\times d_\text{in}},\; r \ll \min(d_\text{in}, d_\text{out})`}
	/>
	<p>
		<Tex expr="W_0" /> is frozen. Only <Tex expr="A" /> and <Tex expr="B" /> are trained, which is
		<Tex expr={String.raw`r\,(d_\text{in}+d_\text{out})`} /> parameters per matrix instead of
		<Tex expr={String.raw`d_\text{in}\, d_\text{out}`} />. For one 4096 × 4096 attention projection
		at <Tex expr="r = 16" /> that is 131k instead of 16.8M, 0.8%. Move the rank slider in the demo: the
		thin orange strips are drawn to scale.
	</p>

	<h3>Starting at zero, ending with no trace</h3>
	<p>
		<Tex expr="A" /> starts random and <Tex expr="B" /> starts at zero, so <Tex expr="BA = 0" /> and the
		adapted model is <em>exactly</em> the base model at step 0. The paper initialised
		<Tex expr="A" /> from a Gaussian <Cite id="lora" />; PEFT's default is Kaiming-uniform <Cite
			id="peft-lora"
		/>. Press play on the toy: at step 0 the middle heatmap is blank. After training, the update can
		be merged into the weights, <Tex expr={String.raw`W = W_0 + \tfrac{\alpha}{r}BA`} />, so
		inference costs nothing extra. In PEFT that is <code>merge_and_unload()</code>
		<Cite id="peft-lora" />.
	</p>
	<p>
		The toy also shows two things that hold beyond it. A rank-<Tex expr="r" /> update can only hold
		<Tex expr="r" /> directions: the error stops at the dashed line, which is the energy of the singular
		values it has no room for. And gradient descent from a small start picks up the strongest directions
		first, one after another, a classic result for linear networks <Cite id="saxe-linear" />.
	</p>

	<h2>Rank, alpha and the scale factor</h2>
	<p>
		The update is multiplied by <Tex expr={String.raw`\gamma = \alpha / r`} />. The original paper
		fixed <Tex expr={String.raw`\alpha`} /> to the first rank it tried and didn't tune it, so that changing
		<Tex expr="r" /> wouldn't require re-tuning the learning rate <Cite id="lora" />. PEFT's
		defaults are <code>r=8</code>, <code>lora_alpha=8</code>
		<Cite id="peft-lora" />. rsLoRA argues that
		<Tex expr={String.raw`\alpha/r`} /> shrinks updates too much at high rank, so higher ranks learn too
		slowly, and proposes <Tex expr={String.raw`\gamma = \alpha/\sqrt{r}`} /> (<code
			>use_rslora=True</code
		>) <Cite id="rslora" />.
	</p>
	<table>
		<thead>
			<tr
				><th>r</th>
				{#each scaleRanks as r (r)}<th>{r}</th>{/each}
			</tr>
		</thead>
		<tbody>
			<tr
				><td>α / r (α = 16)</td>
				{#each scaleRanks as r (r)}<td>{scaling(16, r).toFixed(3)}</td>{/each}
			</tr>
			<tr
				><td>α / √r (α = 16)</td>
				{#each scaleRanks as r (r)}<td>{scaling(16, r, 'rslora').toFixed(3)}</td>{/each}
			</tr>
		</tbody>
	</table>
	<p>
		The practical lesson is that the scale convention decides how the learning rate transfers across
		ranks. Thinking Machines' 2025 sweep found that with <Tex expr={String.raw`\alpha = 32`} /> and the
		<Tex expr={String.raw`1/r`} /> convention the best learning rate changed by less than 2× between ranks
		4 and 512, and was about 10× the full fine-tuning optimum (roughly 15× on very short runs)
		<Cite id="lora-without-regret" />. Rank itself matters less than you might expect: QLoRA found
		that once all linear layers are adapted, the choice of <Tex expr="r" /> was unrelated to final quality
		(LLaMA 7B on Alpaca) <Cite id="qlora" />.
	</p>

	<h2>Which matrices to adapt</h2>
	<p>
		The 2021 paper adapted only the query and value projections <Cite id="lora" />. That turned out
		to leave quality on the table: QLoRA found that LoRA on all linear layers of each block was
		needed to match full fine-tuning <Cite id="qlora" />, and Thinking Machines found that
		attention-only LoRA clearly underperforms MLP-only LoRA at a similar parameter count (rank 256
		on attention, 0.25B, vs rank 128 on the MLPs, 0.24B), while adding attention on top of the MLPs
		gave no further gain <Cite id="lora-without-regret" />. PEFT's
		<code>target_modules="all-linear"</code>
		selects every linear layer except the output head <Cite id="peft-lora" />. On Qwen3-8B at <Tex
			expr="r = 16"
		/>:
	</p>
	<table>
		<thead><tr><th>target_modules</th><th>trainable</th><th>share of 8.19B</th></tr></thead>
		<tbody>
			{#each presetRows as row (row.id)}
				{@const n = trainableParams(qwen3_8b, 16, presetModules[row.id])}
				<tr
					><td><code>{row.label}</code></td><td>{(n / 1e6).toFixed(1)}M</td><td
						>{((100 * n) / total).toFixed(2)}%</td
					></tr
				>
			{/each}
		</tbody>
	</table>

	<h2>QLoRA: shrink the frozen part too</h2>
	<p>
		Once the optimizer state is gone, the frozen bf16 weights are what fills the GPU. QLoRA keeps
		them in 4 bits and backpropagates through them into the adapters (bf16 in the paper): weights
		are dequantised on the fly for each matrix multiply, and gradients flow only into <Tex
			expr="A"
		/> and <Tex expr="B" />. With it a 65B model could be fine-tuned on one 48 GB GPU instead of
		more than 780 GB <Cite id="qlora" />. Three pieces make it work:
	</p>
	<ul>
		<li>
			<strong>4-bit NormalFloat (NF4)</strong>: 16 code values at quantiles of a normal
			distribution, which suits normally distributed weights better than evenly spaced integers.
		</li>
		<li>
			<strong>Double quantisation</strong>: the per-block scaling constants are themselves
			quantised, cutting their overhead from 0.5 to about 0.127 bits per weight.
		</li>
		<li>
			<strong>Paged optimizers</strong>: optimizer state is moved to CPU memory during spikes (long
			sequences with gradient checkpointing) instead of crashing with out-of-memory <Cite
				id="qlora"
			/>.
		</li>
	</ul>
	<QuantDemo />
	<MemoryBudget />
	<p>
		A few caveats keep these bars honest. They leave out activations, which for long reasoning
		traces can be the largest term. The bars assume fp32 adapters with fp32 Adam state, a
		conservative choice. Common recipes also upcast the unquantised parts: PEFT's
		<code>prepare_model_for_kbit_training</code> casts every remaining 16-bit parameter (embeddings,
		output head, norms) to fp32, which adds about 2.5 GB on Qwen3-8B <Cite id="peft-kbit" />. And
		QLoRA trades memory for speed: every matrix multiply first dequantises the 4-bit weights.
	</p>

	<h2>Does LoRA match full fine-tuning?</h2>
	<p>
		Hu et al. reported parity on the benchmarks of the time, with GPT-3's checkpoint shrinking from
		350 GB to 35 MB <Cite id="lora" />. The most careful recent study is Thinking Machines' “LoRA
		Without Regret” (2025) <Cite id="lora-without-regret" />:
	</p>
	<ul>
		<li>
			On small and medium instruction and reasoning datasets, LoRA performs the same as full
			fine-tuning. Once the dataset holds more information than the adapter can store, it becomes
			less sample-efficient and falls behind.
		</li>
		<li>
			For reinforcement learning, even <strong>rank 1</strong> matches full fine-tuning: a policy-gradient
			run absorbs very few bits per episode, far below what a rank-1 adapter can hold.
		</li>
		<li>LoRA is less tolerant of large batch sizes, and raising the rank doesn't fix that.</li>
	</ul>
	<p>
		An earlier study framed the trade-off as “LoRA learns less and forgets less”: on code and math,
		both continued pretraining and instruction tuning, LoRA learned the new domain less well than
		full fine-tuning but kept more of the base model's other skills <Cite id="lora-learns-less" />.
		So LoRA is a natural fit for the RL stages of the next chapters (RLHF, RLVR), and a riskier one
		for very large SFT mixes.
	</p>

	<h2>Variants</h2>
	<p>
		<strong>DoRA</strong> splits each weight into a magnitude vector and a direction, trains the
		magnitude directly and applies LoRA to the direction only. It reports consistent gains over LoRA
		at the same rank, and can still be merged, so it adds nothing at inference <Cite id="dora" />.
		PEFT supports it with <code>use_dora=True</code> and notes extra training overhead <Cite
			id="peft-lora"
		/>. <strong>rsLoRA</strong> only changes the scale factor <Cite id="rslora" />.
	</p>

	<h2>LoRA on Hugging Face</h2>
	<p>
		A LoRA repo usually ships only the adapter: a small <code>adapter_model.safetensors</code> and
		an
		<code>adapter_config.json</code> with <code>r</code>, <code>lora_alpha</code> and
		<code>target_modules</code>. Its model card names the base model in <code>base_model</code>, and
		the Hub lists it as an <em>adapter</em> in the base model's family tree, next to fine-tunes,
		quantisations and merges <Cite id="hf-model-cards" />. A LoRA that has been merged is just a
		full checkpoint, so it is published as an ordinary fine-tune. The same mechanics come back
		later: a LoRA delta is a task vector, the raw material of model merging (chapter 9), and
		abliteration (chapter 8) is another low-rank edit to the weights.
	</p>
</ChapterShell>

<style>
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 14px;
		display: block;
		overflow-x: auto;
		margin: 1em 0;
	}
	th,
	td {
		text-align: left;
		padding: 6px 8px;
		border-bottom: 1px solid var(--line);
		vertical-align: top;
		font-variant-numeric: tabular-nums;
	}
</style>
