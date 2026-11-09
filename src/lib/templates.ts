/**
 * Minimal re-implementation of two real chat templates, for display. Token boundaries are
 * word-level (a real BPE tokenizer splits differently), but the special tokens and the layout are
 * the ones in the published templates (see docs/research/sft.md):
 * - ChatML as used by Qwen3 (`Qwen/Qwen3-8B` tokenizer_config, eos = <|im_end|>)
 * - Llama 3.1 (`<|start_header_id|>role<|end_header_id|>\n\n … <|eot_id|>`), whose template always
 *   opens with a system turn carrying "Cutting Knowledge Date / Today Date" lines.
 */

export type Role = 'system' | 'user' | 'assistant';
export interface Message {
	role: Role;
	content: string;
	/** Optional reasoning trace (Qwen3 thinking mode). */
	reasoning?: string;
}
export type TemplateId = 'chatml' | 'llama3';

export interface RenderedToken {
	text: string;
	/** Which message the token belongs to; 'template' marks the scaffolding around messages. */
	role: Role | 'template';
	/** True when the token is part of an assistant turn's trainable span (content + end-of-turn). */
	trainable: boolean;
}

export interface RenderOptions {
	template: TemplateId;
	/** Inference: append the assistant header so the model writes the reply (and drop the reply). */
	addGenerationPrompt: boolean;
	/**
	 * Qwen3 only, and only affects the generation prompt: false pre-fills an empty think block.
	 * In a rendered training turn the final assistant message always gets a think block (empty if
	 * the message has no reasoning), as in the real template.
	 */
	enableThinking: boolean;
}

export function splitWords(text: string): string[] {
	return text.match(/\n+|[^\s\w]|\s?\w+/g) ?? [];
}

const t = (text: string, role: RenderedToken['role'], trainable = false): RenderedToken => ({
	text,
	role,
	trainable
});

export const eosToken: Record<TemplateId, string> = { chatml: '<|im_end|>', llama3: '<|eot_id|>' };

function header(template: TemplateId, role: Role): RenderedToken[] {
	return template === 'chatml'
		? [t('<|im_start|>', 'template'), t(role, 'template'), t('\n', 'template')]
		: [
				t('<|start_header_id|>', 'template'),
				t(role, 'template'),
				t('<|end_header_id|>', 'template'),
				t('\n\n', 'template')
			];
}

function thinkBlock(m: Message, o: RenderOptions, trainable: boolean): RenderedToken[] {
	if (o.template !== 'chatml') return [];
	// Qwen3 template, final assistant turn: '<think>\n' + reasoning + '\n</think>\n\n' + content,
	// independent of enable_thinking. With no reasoning: the empty block used for non-thinking data.
	const r = m.reasoning ? splitWords(m.reasoning) : [];
	return [
		t('<think>', 'assistant', trainable),
		t('\n', 'assistant', trainable),
		...r.map((w) => t(w, 'assistant', trainable)),
		t('\n', 'assistant', trainable),
		t('</think>', 'assistant', trainable),
		t('\n\n', 'assistant', trainable)
	];
}

export function renderChat(messages: Message[], o: RenderOptions): RenderedToken[] {
	const out: RenderedToken[] = [];
	if (o.template === 'llama3') out.push(t('<|begin_of_text|>', 'template'));
	const msgs = o.addGenerationPrompt
		? messages.filter((m) => m.role !== 'assistant')
		: [...messages];

	// Llama 3.1 always emits a system turn that starts with its knowledge-cutoff preamble
	// (date_string defaults to "26 Jul 2024" when the caller doesn't pass one).
	if (o.template === 'llama3' && !msgs.some((m) => m.role === 'system')) {
		msgs.unshift({ role: 'system', content: '' });
	}

	for (const m of msgs) {
		out.push(...header(o.template, m.role));
		if (o.template === 'llama3' && m.role === 'system') {
			out.push(
				...splitWords('Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n').map(
					(w) => t(w, 'template')
				)
			);
		}
		const trainable = m.role === 'assistant';
		if (m.role === 'assistant') out.push(...thinkBlock(m, o, true));
		out.push(...splitWords(m.content).map((w) => t(w, m.role, trainable)));
		out.push(t(eosToken[o.template], m.role === 'assistant' ? 'assistant' : 'template', trainable));
		if (o.template === 'chatml') out.push(t('\n', 'template'));
	}

	if (o.addGenerationPrompt) {
		out.push(...header(o.template, 'assistant'));
		// Qwen3 with enable_thinking=False pre-fills an empty think block in the prompt.
		if (o.template === 'chatml' && !o.enableThinking) {
			out.push(
				t('<think>', 'template'),
				t('\n\n', 'template'),
				t('</think>', 'template'),
				t('\n\n', 'template')
			);
		}
	}
	return out;
}
