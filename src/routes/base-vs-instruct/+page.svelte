<script lang="ts">
	import ChapterShell from '#lib/components/ChapterShell.svelte';
	import Cite from '#lib/components/Cite.svelte';
	import Tex from '#lib/components/Tex.svelte';
	import GenerationCompare from './GenerationCompare.svelte';
	import SamplerLab from './SamplerLab.svelte';
	import StageTimeline from './StageTimeline.svelte';
</script>

<ChapterShell slug="base-vs-instruct">
	{#snippet hero()}
		<GenerationCompare />
	{/snippet}

	<h2>What a base model actually learned</h2>
	<p>
		A <strong>base model</strong> is a causal language model trained on trillions of tokens of web text,
		code and books to do one thing: predict the next token. Its whole training signal is the average negative
		log-likelihood of each token given the ones before it:
	</p>
	<Tex
		display
		expr={String.raw`\mathcal{L}_{\text{PT}}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})`}
	/>
	<p>
		Nothing in that objective says <em>answer the user</em>. As the Hugging Face documentation puts
		it, every causal LM, chat-trained or not, just continues a sequence of tokens
		<Cite id="hf-chat-templates" />. So a prompt is read as the <em>beginning of a document</em>,
		and the model writes whatever that kind of document usually says next. InstructGPT framed this
		as a mismatch: “predict the next token on a webpage” isn't the same objective as “follow the
		user's instructions helpfully and safely” <Cite id="instructgpt" />.
	</p>

	<h2>Why it doesn't answer, and why it doesn't stop</h2>
	<p>
		Play the three prompts above. With a bare question or instruction, the base model drifts into
		the most common continuation of such text on the web: a list of quiz questions, a worksheet of
		exercises. The third prompt dresses the question up as a <code>Q:</code>/<code>A:</code> page,
		and now the base model <strong>does</strong> answer. The knowledge was already there.
		Alignment-via-prompting work such as URIAL shows that a carefully written prefix can get a base
		model surprisingly close to a chat model <Cite id="urial" />.
	</p>
	<p>
		Look at how each column ends, though. The post-trained model emits its end-of-turn token
		<code>&lt;|im_end|&gt;</code> and stops. The base model has no notion of a “turn”. It keeps
		writing the next Q/A pair until it hits the <code>max_new_tokens</code> limit (or emits a
		document separator like <code>&lt;|endoftext|&gt;</code>). In the Qwen3 release the two
		checkpoints are even configured with different end-of-sequence tokens:
		<code>&lt;|endoftext|&gt;</code> for the base and <code>&lt;|im_end|&gt;</code> for the
		post-trained model <Cite id="qwen3-8b-card" />.
	</p>
	<p>
		Notice also <em>what the chat model receives</em>: your message wrapped in a chat template (<code
			>&lt;|im_start|&gt;user … &lt;|im_end|&gt;&lt;|im_start|&gt;assistant</code
		>, plus an empty
		<code>&lt;think&gt;&lt;/think&gt;</code> block when Qwen3's thinking mode is off). Teaching the
		model that format is the job of <strong>supervised fine-tuning</strong>, the next chapter.
	</p>

	<h2>Modern “base” models are not raw</h2>
	<p>
		The GPT-3-era picture of a base model as pure web text is out of date. Current recipes add a
		short <strong>mid-training</strong> stage after the main pretraining run: a few percent of the
		compute on up-sampled high-quality, math, code, reasoning and instruction-like data, with the
		learning rate annealed toward zero and the context window extended. Llama 3 anneals on
		high-quality data at the end of pretraining <Cite id="llama3" />. OLMo 2 names the stage
		explicitly and spends 5–10% of training FLOPs on it <Cite id="olmo2" />. Qwen3 runs a dedicated
		reasoning stage and then a long-context stage <Cite id="qwen3" />. SmolLM3 adds a 140B-token
		reasoning mid-training phase <Cite id="smollm3" />.
	</p>
	<StageTimeline />
	<p>
		One finding from Olmo 3 is worth knowing. Putting the chat special tokens (<code
			>&lt;|im_start|&gt;</code
		>, <code>&lt;|im_end|&gt;</code>) into mid-training data made the base model emit them at
		inference, and GSM8K fell from 49.4 to 0. The same data in a plain-text template did not have
		that effect, so the team keeps special tokens out until SFT
		<Cite id="olmo3" />. The practical consequence: 2025-era base checkpoints have already seen
		plenty of question-and-answer text and will sometimes answer. The demo above shows the typical
		behaviour, not a guarantee.
	</p>

	<h2>Sampling: the same model, different behaviour</h2>
	<p>
		The model outputs a distribution. A <em>sampler</em> decides which token to emit. Temperature
		rescales the logits <Tex expr="z_i" /> before the softmax:
	</p>
	<Tex display expr={String.raw`p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}`} />
	<p>
		Then truncation removes the unlikely tail. <strong>Top-k</strong> keeps the <Tex expr="k" /> most
		likely tokens. <strong>Top-p</strong> (nucleus) keeps the smallest set whose cumulative
		probability reaches <Tex expr="p" />
		<Cite id="nucleus" />. <strong>Min-p</strong> keeps tokens with <Tex
			expr={String.raw`p_i \ge p_{\text{base}}\cdot p_{\max}`}
		/>, so the cut-off adapts to how confident the model is <Cite id="minp" />. The kept set is
		renormalised and sampled.
	</p>
	<SamplerLab />
	<p>
		Model publishers recommend sampler settings alongside their weights. For Qwen3-8B these are T =
		0.6, top-p 0.95, top-k 20 in thinking mode and T = 0.7, top-p 0.8, top-k 20 otherwise, with an
		explicit warning against greedy decoding in thinking mode <Cite id="qwen3-8b-card" />.
	</p>

	<h2>What post-training has to add</h2>
	<ul>
		<li><strong>A format</strong>: turns, roles, a stop token (SFT on chat-templated data).</li>
		<li>
			<strong>Preferences</strong>: which of two valid answers people prefer (reward models + RLHF,
			or DPO).
		</li>
		<li>
			<strong>Verifiable skills</strong>: math and code that a checker can grade, often with long
			reasoning (RLVR).
		</li>
		<li>
			<strong>Behaviour boundaries</strong>: when to refuse. Some community variants try to undo
			exactly this (abliteration).
		</li>
	</ul>

	<h3>Try real pairs yourself</h3>
	<table>
		<thead><tr><th>Family</th><th>Base</th><th>Post-trained</th></tr></thead>
		<tbody>
			<tr
				><td>Qwen3</td><td><code>Qwen/Qwen3-8B-Base</code></td><td><code>Qwen/Qwen3-8B</code></td
				></tr
			>
			<tr
				><td>Olmo 3</td><td><code>allenai/Olmo-3-1025-7B</code></td><td
					><code>allenai/Olmo-3-7B-Instruct</code>, <code>…-Think</code></td
				></tr
			>
			<tr
				><td>SmolLM3</td><td><code>HuggingFaceTB/SmolLM3-3B-Base</code></td><td
					><code>HuggingFaceTB/SmolLM3-3B</code></td
				></tr
			>
			<tr
				><td>Llama 3.1</td><td><code>meta-llama/Llama-3.1-8B</code></td><td
					><code>meta-llama/Llama-3.1-8B-Instruct</code></td
				></tr
			>
		</tbody>
	</table>
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
	td code {
		font-size: 12px;
	}
</style>
