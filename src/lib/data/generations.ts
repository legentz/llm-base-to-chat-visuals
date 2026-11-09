/**
 * ILLUSTRATIVE generations for chapter 1 (Base vs Instruct). These are NOT recorded outputs.
 * They're hand-written to mirror documented behaviour (docs/research/base-vs-instruct.md):
 * a base model continues the document and does not stop, while an instruct model answers in the
 * chat format and emits its end-of-turn token.
 *
 * Step DSL: 'tok:p|tok:p*|…' lists the top candidates for the next token with probabilities at
 * T = 1. The token marked `*` is the one that was "sampled" (default: the first). Tokens are written
 * as the tokenizer sees them, with leading spaces; '\n' is a newline.
 */

export interface Candidate {
	token: string;
	p: number;
}
export interface GenStep {
	cands: Candidate[];
	pick: number;
}
export interface Generation {
	model: 'base' | 'instruct';
	repo: string;
	/** Exactly what the model is fed, split into display tokens. Special tokens are kept intact. */
	input: string[];
	steps: GenStep[];
	stop: { kind: 'eos'; token: string } | { kind: 'max_new_tokens'; limit: number };
}
export interface PromptCase {
	id: string;
	label: string;
	userText: string;
	lesson: string;
	base: Generation;
	instruct: Generation;
}

function parseStep(spec: string): GenStep {
	let pick = 0;
	// Split on '|' but not inside special tokens such as <|im_end|>.
	const cands = spec.split(/(?<!<)\|(?!>)/).map((part, i) => {
		const starred = part.endsWith('*');
		const body = starred ? part.slice(0, -1) : part;
		const at = body.lastIndexOf(':');
		if (starred) pick = i;
		return { token: body.slice(0, at).replaceAll('\\n', '\n'), p: Number(body.slice(at + 1)) };
	});
	return { cands, pick };
}

const steps = (...specs: string[]) => specs.map(parseStep);

/** Qwen3 chat template, non-thinking mode (enable_thinking=False injects an empty think block). */
function qwenChat(user: string): string[] {
	return [
		'<|im_start|>',
		'user',
		'\n',
		...splitWords(user),
		'<|im_end|>',
		'\n',
		'<|im_start|>',
		'assistant',
		'\n',
		'<think>',
		'\n\n',
		'</think>',
		'\n\n'
	];
}

/** Rough word-level split for display (a real BPE tokenizer splits differently). */
function splitWords(text: string): string[] {
	return text.match(/\n|[^\s\w]|\s?\w+/g) ?? [];
}

const BASE_REPO = 'Qwen/Qwen3-8B-Base';
const INSTRUCT_REPO = 'Qwen/Qwen3-8B';
const EOS_CHAT = { kind: 'eos', token: '<|im_end|>' } as const;
const MAX = { kind: 'max_new_tokens', limit: 24 } as const;

const capitalInstruct: Generation = {
	model: 'instruct',
	repo: INSTRUCT_REPO,
	input: qwenChat('What is the capital of France?'),
	steps: steps(
		'The:0.78|Paris:0.15|France:0.03|It:0.01',
		' capital:0.97| city:0.01',
		' of:0.99',
		' France:0.99',
		' is:0.99| has:0.003',
		' **:0.64| Paris:0.35',
		'Paris:0.99',
		'**:0.99',
		'.:0.86|,:0.11',
		'<|im_end|>:0.58| It:0.21|\n\n:0.1| Paris:0.07'
	),
	stop: EOS_CHAT
};

export const promptCases: PromptCase[] = [
	{
		id: 'capital',
		label: 'A plain question',
		userText: 'What is the capital of France?',
		lesson:
			'The base model treats the question as the start of a document, and on the web, lists of questions are often followed by more questions. The instruct model answers and then emits <|im_end|>.',
		base: {
			model: 'base',
			repo: BASE_REPO,
			input: splitWords('What is the capital of France?'),
			steps: steps(
				' What:0.26|\n:0.22| The:0.11| A:0.08| Paris:0.07| Answer:0.05| (:0.04| Is:0.03',
				' is:0.71| are:0.08|’s:0.06| was:0.05| city:0.02',
				' the:0.83| capital:0.04| a:0.03',
				' capital:0.62| largest:0.09| population:0.07| official:0.05| currency:0.05',
				' of:0.93| city:0.03',
				' Germany:0.19| Italy:0.14| Spain:0.12| England:0.08| the:0.07| France:0.05| Canada:0.04',
				'?:0.88|?\n:0.06',
				' What:0.58|\n:0.17| The:0.05| Which:0.04',
				' is:0.9| are:0.03',
				' the:0.86| a:0.02',
				' capital:0.79| largest:0.05',
				' of:0.95',
				' Italy:0.27| Spain:0.18| England:0.09| Russia:0.06| Japan:0.05',
				'?:0.9',
				' What:0.41|\n:0.33*| The:0.04',
				'\n:0.3| What:0.2| Answer:0.1| A:0.08',
				'Answer:0.21|What:0.19|Q:0.1|1:0.07|The:0.07',
				'::0.74|s:0.15| key:0.04',
				' Paris:0.48| The:0.12| 1:0.08| A:0.07',
				',:0.32*|.:0.2|\n:0.18| (:0.05',
				' Berlin:0.61| Rome:0.17| Madrid:0.06',
				',:0.44| and:0.21|\n:0.15',
				' Rome:0.72| Madrid:0.11',
				'\n:0.4|.:0.3|,:0.15'
			),
			stop: MAX
		},
		instruct: capitalInstruct
	},
	{
		id: 'haiku',
		label: 'An instruction',
		userText: 'Write a haiku about the sea.',
		lesson:
			'An instruction reads like a line from a worksheet, so the base model writes the next exercise. The instruct model carries it out.',
		base: {
			model: 'base',
			repo: BASE_REPO,
			input: splitWords('Write a haiku about the sea.'),
			steps: steps(
				'\n:0.34| Write:0.14| The:0.07| Use:0.06| It:0.05| (:0.04',
				'\n:0.41|Write:0.16|The:0.1|-:0.05',
				'Write:0.29|The:0.12|Haiku:0.08|Answer:0.06|Sea:0.05',
				' a:0.81| another:0.07| the:0.04',
				' haiku:0.73| poem:0.09| limerick:0.05',
				' about:0.91| that:0.03',
				' the:0.56| a:0.14| your:0.06',
				' moon:0.17| mountains:0.12| sky:0.11| wind:0.08| forest:0.07| sun:0.06',
				'.:0.83|,:0.05|\n:0.05',
				'\n\n:0.52| Write:0.2|\n:0.15',
				'Write:0.44|The:0.09|Now:0.06',
				' a:0.88| an:0.03',
				' haiku:0.55| poem:0.12| sonnet:0.09',
				' about:0.9',
				' the:0.5| your:0.12| a:0.1',
				' stars:0.1| wind:0.14*| city:0.1| rain:0.09| desert:0.07',
				'.:0.86',
				'\n\n:0.55|\n:0.2| Write:0.1',
				'Write:0.38|Now:0.08|Then:0.06',
				' a:0.84',
				' haiku:0.5| short:0.11| poem:0.1',
				' about:0.88| using:0.04',
				' a:0.31| the:0.3| your:0.1',
				' tree:0.11| friend:0.09| season:0.08| flower:0.08'
			),
			stop: MAX
		},
		instruct: {
			model: 'instruct',
			repo: INSTRUCT_REPO,
			input: qwenChat('Write a haiku about the sea.'),
			steps: steps(
				'Waves:0.31|Salt:0.18*|Endless:0.12|Blue:0.09|Wh:0.06|Moon:0.05',
				' wind:0.38| spray:0.21| air:0.14| waves:0.09',
				' on:0.29| and:0.2| across:0.11| over:0.1|,:0.08',
				' the:0.62| weathered:0.06| cold:0.05',
				' shore:0.44| dunes:0.14| rocks:0.12| sand:0.1',
				'\n:0.71| —:0.12|,:0.1',
				'Waves:0.33|The:0.2|Gulls:0.1|Tides:0.08',
				' whisper:0.21| fold:0.14| erase:0.12*| pull:0.1| carry:0.09',
				' what:0.41| the:0.22| our:0.08',
				' the:0.67| we:0.1',
				' tide:0.52| sand:0.13| day:0.08',
				' wrote:0.58| left:0.14| has:0.1',
				'\n:0.88',
				'G:0.31|The:0.22|Salt:0.1|A:0.07',
				'ulls:0.97',
				' cry:0.32| circle:0.18| drift:0.11| stitch:0.09*',
				' sky:0.44| the:0.3| through:0.06',
				' to:0.88| and:0.07',
				' sea:0.9| shore:0.05',
				'<|im_end|>:0.82|.:0.1'
			),
			stop: EOS_CHAT
		}
	},
	{
		id: 'qa',
		label: 'A Q/A-formatted page',
		userText: 'Q: What is the capital of France?\nA:',
		lesson:
			'Make the prompt look like a Q&A page and the base model does answer. That knowledge was there all along. It still never stops, though: it writes the next Q/A pair. SFT bakes the format and the stopping in.',
		base: {
			model: 'base',
			repo: BASE_REPO,
			input: splitWords('Q: What is the capital of France?\nA:'),
			steps: steps(
				' Paris:0.71| The:0.12| paris:0.04| France:0.02',
				'\n:0.52|.:0.3|,:0.06',
				'\n:0.38|Q:0.34| :0.06',
				'Q:0.62|What:0.07|The:0.05',
				'::0.93',
				' What:0.71| Which:0.08| Who:0.05',
				' is:0.9',
				' the:0.88',
				' capital:0.72| largest:0.08| population:0.05',
				' of:0.95',
				' Germany:0.21| Italy:0.17| Spain:0.13| England:0.08',
				'?:0.93',
				'\n:0.92',
				'A:0.97',
				'::0.98',
				' Berlin:0.85| The:0.07',
				'\n:0.55|.:0.32',
				'\n:0.4|Q:0.38',
				'Q:0.66',
				'::0.94',
				' What:0.69| Who:0.06',
				' is:0.88',
				' the:0.86',
				' capital:0.7'
			),
			stop: MAX
		},
		instruct: {
			...capitalInstruct,
			input: qwenChat('Q: What is the capital of France?\nA:'),
			steps: steps(
				'Paris:0.47|The:0.4| Paris:0.06',
				'.:0.62|<|im_end|>:0.18|\n\n:0.1',
				'<|im_end|>:0.74| It:0.12|\n\n:0.09'
			)
		}
	}
];
