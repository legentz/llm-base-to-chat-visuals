<script lang="ts">
	import { chapters, chapterHref, stageLabel } from '#lib/chapters.js';
</script>

<svelte:head>
	<title>Post-Training Explainer: from base model to chat model</title>
	<meta
		name="description"
		content="Interactive, simulated walkthrough of how base LLMs become instruct/chat models: SFT, LoRA, RLHF, DPO, RLVR, distillation, abliteration and merging."
	/>
</svelte:head>

<section class="intro">
	<p class="eyebrow">Interactive explainer</p>
	<h1>How a base LLM becomes a chat model</h1>
	<p class="lede">
		A pretrained model only knows how to continue text. Everything that makes it answer, follow
		instructions, reason, or refuse comes from <strong>post-training</strong>. Every technique here
		is a hands-on simulation you can poke at.
	</p>
</section>

<ol class="toc">
	{#each chapters as c (c.slug)}
		<li class="card" style:--stage={`var(--stage-${c.stage})`}>
			<span class="eyebrow">{String(c.n).padStart(2, '0')} · {stageLabel[c.stage]}</span>
			{#if c.ready}
				<a href={chapterHref(c.slug)}><h2>{c.title}</h2></a>
			{:else}
				<h2 class="soon">{c.title} <small>coming soon</small></h2>
			{/if}
			<p>{c.blurb}</p>
		</li>
	{/each}
</ol>

<style>
	.intro {
		max-width: var(--maxw);
		margin: 0 auto;
		padding: 56px 16px 24px;
	}
	h1 {
		font-size: clamp(2rem, 5vw, 3.2rem);
		margin: 8px 0 12px;
		max-width: 18ch;
	}
	.lede {
		font-size: 1.2rem;
		color: var(--ink-2);
		max-width: 60ch;
	}
	.toc {
		list-style: none;
		max-width: var(--maxw);
		margin: 0 auto;
		padding: 0 16px;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
		gap: 16px;
	}
	.toc li {
		padding: 16px 18px;
		border-top: 4px solid var(--stage);
	}
	.toc h2 {
		font-size: 1.15rem;
		margin: 6px 0;
	}
	.toc a {
		color: var(--ink);
		text-decoration: none;
	}
	.toc a:hover h2 {
		text-decoration: underline;
	}
	.toc p {
		margin: 0;
		color: var(--ink-2);
		font-size: 14px;
	}
	.soon {
		color: var(--ink-3);
	}
	.soon small {
		font-size: 11px;
		font-weight: 500;
	}
</style>
