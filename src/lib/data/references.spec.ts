import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { references, referenceById } from './references';

function svelteFiles(dir: string): string[] {
	return readdirSync(dir).flatMap((f) => {
		const p = join(dir, f);
		if (statSync(p).isDirectory()) return svelteFiles(p);
		return p.endsWith('.svelte') ? [p] : [];
	});
}

describe('references', () => {
	it('has unique ids and https urls', () => {
		expect(referenceById.size).toBe(references.length);
		for (const r of references) expect(r.url).toMatch(/^https:\/\//);
	});

	it('every <Cite id="…"> in the source points to a known reference', () => {
		const missing: string[] = [];
		for (const file of svelteFiles('src')) {
			const text = readFileSync(file, 'utf8');
			for (const m of text.matchAll(/<Cite\s+id="([^"]+)"/g)) {
				if (!referenceById.has(m[1])) missing.push(`${file}: ${m[1]}`);
			}
		}
		expect(missing).toEqual([]);
	});
});
