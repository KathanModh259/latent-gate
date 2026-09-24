---
name: token-optimizer
description: Read large logs, JSON/API responses, data exports and long documents through LatentGate's token optimizer instead of reading them raw. Use when a file you only need to understand (not edit) is likely over ~2,000 tokens — log files, stack-trace dumps, pretty-printed JSON, CSV/NDJSON exports, long docs or RAG chunks — or when the user asks to save tokens/context.
---

# Token optimizer

The `latent-gate` MCP server returns a compact version of text with the
facts intact. It runs locally in ~1ms per file, needs no Ollama or API key,
and gives the same output for the same input.

## Use it for files you need to *understand*

Call `read_file_optimized` instead of reading the file directly when it is
large and you only need its content:

- **Logs**: runs of repeated lines are folded to the first line, the last line,
  and the range of each varying field (`id=5001…5398`). Often 90%+ smaller.
- **Pretty-printed JSON**: minified losslessly (values and spellings untouched). ~30% smaller.
- **Long docs / RAG chunks**: pass `question` and `level: "aggressive"` or a
  `max_tokens` budget to keep only the sentences relevant to what you need.

Code blocks, URLs and quoted strings are always returned byte-for-byte.

## Don't use it for files you will edit

The optimized text is not byte-identical to the file (JSON is minified, runs
are folded). Read files you intend to modify with the normal file-reading tool,
so your edits match the real content.

## Levels

| level | what it does | facts kept |
|-------|--------------|------------|
| `lossless` | whitespace, JSON minify, exact-duplicate folding | all |
| `balanced` (default) | + log-run folding, filler removal | all except folded log values (ranges kept) |
| `aggressive` | + keeps the most relevant sentences (~50%) | most; the actual request is never dropped |

`optimize_text` does the same for text you already have, and `count_tokens`
measures size. Report the savings (`original_tokens` → `optimized_tokens`)
when it helps the user see what was saved.
