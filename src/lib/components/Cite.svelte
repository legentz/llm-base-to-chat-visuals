<script lang="ts">
	import { resolve } from '$app/paths';
	import { referenceById } from '#lib/data/references.js';

	let { id }: { id: string } = $props();
	const ref = $derived(referenceById.get(id));
</script>

{#if ref}
	<a class="cite" href={resolve('references/') + '#' + ref.id} title={ref.title}>({ref.cite})</a>
{:else}
	<span class="cite missing">(missing ref: {id})</span>
{/if}

<style>
	.cite {
		font-size: 0.85em;
		text-decoration: none;
		white-space: nowrap;
	}
	.cite:hover {
		text-decoration: underline;
	}
	.missing {
		color: var(--c-rejected);
	}
</style>
