<script lang="ts">
	import ChapterShell from '#lib/components/ChapterShell.svelte';
	import Cite from '#lib/components/Cite.svelte';
	import Tex from '#lib/components/Tex.svelte';
	import PackingDemo from './PackingDemo.svelte';
	import SftLab from './SftLab.svelte';
</script>

<ChapterShell slug="sft">
	{#snippet hero()}
		<SftLab />
	{/snippet}

	<h2>Same loss, different data</h2>
	<p>
		Supervised fine-tuning (SFT) uses exactly the pretraining objective, next-token cross-entropy.
		Two things change: <strong>the data</strong> are conversations written the way we want the model
		to behave, and <strong>a mask</strong> restricts the loss to the assistant's tokens. Given a
		prompt <Tex expr="c" /> (system + user turns, template included) and a demonstration
		<Tex expr="y" />, with <Tex expr="m_t = 1" /> only inside the assistant turn:
	</p>
	<Tex
		display
		expr={String.raw`\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{\sum_t m_t}\sum_{t} m_t \,\log p_\theta(x_t \mid x_{<t}), \qquad x = (c, y)`}
	/>
	<p>
		In Hugging Face code the masked positions get the label <code>-100</code>, which PyTorch's
		cross-entropy ignores. TRL exposes this as <code>assistant_only_loss=True</code>. It relies on
		<code>{'{% generation %}'}</code> markers in the chat template to know where assistant text
		starts and ends <Cite id="trl-sft" />. Turn the switch off in the demo: now the model would also
		be trained to <em>write the user's messages</em>, which is not what you want from an assistant.
	</p>

	<h2>Chat templates: the format is just tokens</h2>
	<p>
		A chat model doesn't receive a list of messages. It receives one token sequence produced by a
		<strong>chat template</strong>, a Jinja snippet shipped with the tokenizer and applied by
		<code>tokenizer.apply_chat_template()</code>
		<Cite id="hf-chat-templates" />. Qwen3, SmolLM3 and Olmo 3 use the ChatML layout (<code
			>&lt;|im_start|&gt;role … &lt;|im_end|&gt;</code
		>). Llama 3 uses header tokens and ends each turn with <code>&lt;|eot_id|&gt;</code>. At
		inference,
		<code>add_generation_prompt=True</code> appends the assistant header, so the model's next token
		starts the reply. Switch the demo to <em>Inference prompt</em> to see it.
	</p>
	<p>
		Qwen3 adds a <strong>thinking switch</strong>. With <code>enable_thinking=True</code> the reply
		begins with a reasoning trace inside <code>&lt;think&gt; … &lt;/think&gt;</code>. With
		<code>False</code> the template pre-fills an empty think block, so one checkpoint serves both
		modes. The Qwen3 report describes training both behaviours into one model, which it calls
		“thinking mode fusion” <Cite id="qwen3" />.
	</p>

	<h3>Learning to stop</h3>
	<p>
		The end-of-turn token (<code>&lt;|im_end|&gt;</code> or <code>&lt;|eot_id|&gt;</code>) is
		<em>inside</em> the trainable span. A base model gives it almost no probability, since it has
		probably never seen it (look at the “does it stop?” bars at step 0). SFT pushes it close to 1,
		and that alone fixes the never-stopping behaviour from chapter 1. One known gotcha: Qwen3-Base
		ships with <code>&lt;|endoftext|&gt;</code> as its EOS, so when you SFT a Qwen base on ChatML
		the EOS has to be set to <code>&lt;|im_end|&gt;</code>, or generation won't stop where training
		taught it to <Cite id="trl-sft" />.
	</p>

	<h2>Packing: don't pay for padding</h2>
	<p>
		Conversations vary wildly in length. Padding each one to the longest in the batch wastes most of
		the compute on padding tokens. Modern recipes <strong>pack</strong> several examples into each
		fixed-length row and mask attention so examples can't see each other. TRL's default
		<code>bfd</code> strategy is best-fit decreasing <Cite id="trl-sft" />. SmolLM3 trains SFT with
		BFD packing <Cite id="smollm3" />, and Olmo 3 packs SFT documents into 32K-token sequences
		<Cite id="olmo3" />.
	</p>
	<PackingDemo />
	<p>
		A related detail: how you <em>average</em> the loss. With a mean over tokens per batch, a token
		in a short example weighs more than one in a long example. Tülu 3 switched to a
		<strong>sum</strong> over tokens so every token counts equally across gradient accumulation and
		devices, and re-tuned the learning rate to match <Cite id="tulu3" />.
	</p>

	<h2>How much data does it take?</h2>
	<p>
		Opinions here have swung. InstructGPT's SFT step used roughly 13k labeler-written prompts
		<Cite id="instructgpt" />, and FLAN showed that instruction-formatted versions of existing NLP
		datasets give zero-shot gains <Cite id="flan" />. LIMA fine-tuned a 65B model on only 1,000
		carefully chosen examples and proposed the <strong>superficial alignment hypothesis</strong>:
		the knowledge comes from pretraining, and alignment mostly selects a style and format
		<Cite id="lima" />.
	</p>
	<p>
		Later open recipes qualify that. Tülu 3 found that math, coding and instruction-following skills
		keep improving with more, better SFT data <Cite id="tulu3" />. Reasoning models are SFT'd on
		long thinking traces at a scale LIMA never imagined: the Olmo 3 7B Think SFT set is about 45B
		tokens <Cite id="olmo3" />. A fair summary is that <em>style</em> is cheap and
		<em>skills</em> are not.
	</p>
	<table>
		<thead><tr><th>Recipe</th><th>SFT data</th><th>Notes</th></tr></thead>
		<tbody>
			<tr
				><td>InstructGPT <Cite id="instructgpt" /></td><td>~13k prompts</td><td
					>16 epochs; step 1 of SFT → reward model → PPO</td
				></tr
			>
			<tr><td>LIMA <Cite id="lima" /></td><td>1,000 examples</td><td>15 epochs on LLaMA-65B</td></tr
			>
			<tr
				><td>Tülu 3 <Cite id="tulu3" /></td><td>large curated mix</td><td
					>2 epochs, LR 5e-6 (8B), sum loss; followed by DPO and RLVR</td
				></tr
			>
			<tr><td>SmolLM2 <Cite id="smollm2" /></td><td>SmolTalk</td><td>2 epochs</td></tr>
			<tr
				><td>SmolLM3 <Cite id="smollm3" /></td><td>1.8B tokens</td><td
					>4 epochs, BFD packing, then preference tuning and merging</td
				></tr
			>
			<tr
				><td>Olmo 3 Think 7B <Cite id="olmo3" /></td><td>~45B tokens</td><td
					>2 epochs, packed 32K sequences</td
				></tr
			>
		</tbody>
	</table>

	<h2>Where SFT stops</h2>
	<p>
		SFT teaches the model to <em>imitate</em> demonstrations, one token at a time. It never sees a
		bad answer and never learns <em>which of two good answers people prefer</em>. It also inherits
		the limits of whoever wrote the demonstrations. The next stages fix this with comparisons
		(reward models and RLHF, DPO) and with rewards a program can check (RLVR). Fine-tuning also
		doesn't have to touch every weight: LoRA, next, trains a small low-rank update instead.
	</p>
</ChapterShell>

<style>
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 14px;
		display: block;
		overflow-x: auto;
	}
	th,
	td {
		text-align: left;
		padding: 6px 8px;
		border-bottom: 1px solid var(--line);
		vertical-align: top;
	}
</style>
