import { describe, expect, it } from 'vitest';
import { eosToken, renderChat, type Message } from './templates';

const conv: Message[] = [
	{ role: 'system', content: 'You are a helpful assistant.' },
	{ role: 'user', content: 'What is the capital of France?' },
	{ role: 'assistant', content: 'The capital of France is Paris.', reasoning: 'Easy: Paris.' }
];
const text = (ts: { text: string }[]) => ts.map((x) => x.text).join('');

describe('chat templates', () => {
	it('renders ChatML exactly like the Qwen3 template (non-thinking training example)', () => {
		const out = renderChat(conv, {
			template: 'chatml',
			addGenerationPrompt: false,
			enableThinking: false
		});
		expect(text(out)).toBe(
			'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' +
				'<|im_start|>user\nWhat is the capital of France?<|im_end|>\n' +
				'<|im_start|>assistant\n<think>\n\n</think>\n\nThe capital of France is Paris.<|im_end|>\n'
		);
	});

	it('wraps the reasoning trace in <think> … </think> when thinking is on', () => {
		const out = renderChat(conv, {
			template: 'chatml',
			addGenerationPrompt: false,
			enableThinking: true
		});
		expect(text(out)).toContain(
			'<|im_start|>assistant\n<think>\nEasy: Paris.\n</think>\n\nThe capital'
		);
	});

	it('appends the generation prompt with an empty think block when thinking is off', () => {
		const out = renderChat(conv, {
			template: 'chatml',
			addGenerationPrompt: true,
			enableThinking: false
		});
		expect(text(out).endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n')).toBe(true);
		expect(text(out)).not.toContain('Paris');
	});

	it('renders Llama 3 headers and <|eot_id|>', () => {
		const out = renderChat(conv, {
			template: 'llama3',
			addGenerationPrompt: false,
			enableThinking: false
		});
		expect(
			text(out).startsWith('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n')
		).toBe(true);
		expect(text(out).endsWith('Paris.<|eot_id|>')).toBe(true);
	});

	it('marks only the assistant span (content + end-of-turn token) as trainable', () => {
		for (const template of ['chatml', 'llama3'] as const) {
			const out = renderChat(conv, { template, addGenerationPrompt: false, enableThinking: true });
			const trainable = out.filter((x) => x.trainable);
			expect(trainable.every((x) => x.role === 'assistant')).toBe(true);
			expect(trainable.at(-1)!.text).toBe(eosToken[template]);
			expect(out.filter((x) => x.role === 'user').some((x) => x.trainable)).toBe(false);
		}
	});
});
