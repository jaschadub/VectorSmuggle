# VectorSmuggle Preprint-Rigor Results — `20260505_144209`

- Model: `text-embedding-3-large`
- Embedding dim: 3072
- Detector train/test: 27/41

## Detection vs. Quantization vs. Retrieval Utility

| Technique | cos(pre-q) | cos(post-q) | quant BER | IF AUC | IF TPR@1%FPR | OCSVM AUC | recall@5 clean | recall@5 obf | recall drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noise_low | 0.964 | 0.964 | 0.000 | 0.638 | 0.098 | 0.796 | 1.000 | 1.000 | +0.000 |
| noise_med | 0.874 | 0.874 | 0.000 | 0.864 | 0.390 | 1.000 | 1.000 | 1.000 | +0.000 |
| noise_high | 0.336 | 0.336 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 |
| rotation | 1.000 | 1.000 | 0.000 | 0.498 | 0.000 | 0.501 | 1.000 | 1.000 | +0.000 |
| scaling | 1.000 | 1.000 | 0.000 | 0.404 | 0.024 | 0.289 | 1.000 | 1.000 | +0.000 |
| offset | 0.527 | 0.527 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 |

## How to read this table

- **cos(pre/post-q)**: cosine similarity of obfuscated to original
  before and after int8 round-trip quantization. A large gap means
  the technique's hidden bits do not survive quantization.
- **quant BER**: fraction of sign-bits that flip during int8
  quantization. >0.05 means the channel is too noisy for the
  technique's payload to survive a real vector store.
- **IF AUC / OCSVM AUC**: detection ROC area for two standard
  anomaly detectors trained on held-out clean embeddings.
  AUC near 0.5 = invisible; AUC near 1.0 = trivially detected.
- **TPR@1%FPR**: a defender willing to flag 1% of legitimate
  traffic catches this fraction of obfuscated vectors.
- **recall@5 clean → obf**: drop in legitimate-user retrieval
  quality after obfuscation. Anything below clean is detectable
  via routine RAG quality monitoring.

## Known Limitations of This Run

- The retrieval-utility benchmark uses self-queries (each test
  embedding is the query for itself). This makes recall@5 = 1.0
  trivially. A faithful retrieval test requires *paraphrased*
  queries — embed the paraphrase, then check whether the original
  document is still in top-k. Adding this is the highest-priority
  follow-up for a publication-grade run.
- Quantization is modeled as per-dimension scalar int8, matching
  Qdrant defaults. Product quantization (FAISS IVF-PQ) and
  binary quantization are far more lossy and not yet covered.
- BER uses sign-bits as the channel; a real bit-channel analysis
  would require defining where the steganographic payload lives
  in the embedding (e.g., LSBs of specific dimensions).