import { describe, expect, it } from 'vitest';
import { promptCases } from './generations';

describe('illustrative generations', () => {
	for (const c of promptCases) {
		for (const g of [c.base, c.instruct]) {
			it(`${c.id}/${g.model}: candidates are well-formed`, () => {
				for (const s of g.steps) {
					const mass = s.cands.reduce((a, x) => a + x.p, 0);
					expect(mass).toBeLessThanOrEqual(1.0001);
					expect(s.pick).toBeLessThan(s.cands.length);
					for (const x of s.cands) {
						expect(x.token.length).toBeGreaterThan(0);
						expect(x.p).toBeGreaterThan(0);
					}
				}
			});
		}
		it(`${c.id}: instruct ends on its EOS token, base runs to the token limit`, () => {
			const last = c.instruct.steps.at(-1)!;
			expect(c.instruct.stop.kind).toBe('eos');
			expect(last.cands[last.pick].token).toBe('<|im_end|>');
			expect(c.base.stop).toEqual({ kind: 'max_new_tokens', limit: c.base.steps.length });
		});
	}
});
