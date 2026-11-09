<script lang="ts">
	import { isSpecial, showToken } from '#lib/tokens.js';

	/** One token. `variant` sets its role: prompt input, generated output, or the one just produced. */
	let {
		token,
		variant = 'input',
		color = 'var(--ink-3)'
	}: {
		token: string;
		variant?: 'input' | 'output' | 'current' | 'masked';
		color?: string;
	} = $props();
</script>

<span class="chip {variant}" class:special={isSpecial(token)} style:--tc={color}
	>{showToken(token)}</span
>

<style>
	.chip {
		display: inline-block;
		margin: 2px 1px;
		padding: 1px 4px;
		border-radius: 4px;
		font-family: var(--mono);
		font-size: 13px;
		line-height: 1.5;
		white-space: pre;
		background: var(--surface-2);
		color: var(--ink-2);
		border: 1px solid transparent;
	}
	.output {
		background: color-mix(in srgb, var(--tc) 14%, transparent);
		color: var(--ink);
	}
	.current {
		background: var(--tc);
		color: #fff;
		animation: pop 0.25s ease-out;
	}
	.masked {
		opacity: 0.45;
	}
	.special {
		border-color: color-mix(in srgb, var(--c-ref) 60%, transparent);
		color: var(--c-ref);
		font-weight: 500;
	}
	.current.special {
		color: #fff;
		background: var(--c-ref);
	}
	@keyframes pop {
		from {
			transform: scale(0.85);
			opacity: 0.4;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.current {
			animation: none;
		}
	}
</style>
