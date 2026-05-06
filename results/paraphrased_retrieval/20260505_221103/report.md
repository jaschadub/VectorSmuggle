# Paraphrased-Query Retrieval --- `20260505_221103`

## Setup

- Embedding model: `text-embedding-3-large`
- Paraphrase generator: `gpt-4o-mini`
- Corpus chunks: 68
- Paraphrases per chunk: 3
- Total queries: 204
- Noise level (where applicable): 0.01

## Retrieval quality: clean corpus vs obfuscated corpus

Each query is a paraphrase of one corpus chunk. The query is
embedded **without** obfuscation, then matched against an index
of the (possibly obfuscated) corpus. recall@k is the fraction
of queries whose source chunk appears in the top-k results.
MRR is mean reciprocal rank. ``margin`` is the mean cosine
score difference between the top hit and the second hit ---
smaller margin means the attack noise has eroded the signal.

| Technique | recall@1 | recall@3 | recall@5 | MRR | mean correct | mean top-1 | mean margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.828 | 0.990 | 0.995 | 0.902 | 0.5797 | 0.5872 | 0.0868 |
| noise | 0.804 | 0.985 | 0.995 | 0.888 | 0.5065 | 0.5134 | 0.0758 |
| rotation | 0.828 | 0.990 | 0.995 | 0.902 | 0.5798 | 0.5872 | 0.0869 |
| scaling | 0.828 | 0.990 | 0.995 | 0.902 | 0.5797 | 0.5872 | 0.0868 |
| offset | 0.745 | 0.892 | 0.961 | 0.832 | 0.2976 | 0.3154 | 0.0438 |

## How to read this table

The clean baseline establishes the upper bound on retrieval quality
for this corpus + embedding model + paraphrase generator. Every
obfuscation row should be compared against it. A row whose
recall@k is significantly below clean indicates that ordinary
RAG quality monitoring would catch the attack as a regression in
retrieval quality. A row whose recall@k matches clean within
noise is undetectable through that monitoring layer --- the
attack lives inside the embedding without breaking the user-facing
retrieval contract.

Caveats: paraphrases are LLM-generated and so depend on the
paraphrase model's quality; the clean recall is therefore not
expected to be 1.0. Differences relative to clean are the
interpretable signal, not the absolute numbers.