# Architecture

VectorSmuggle is a research framework for studying steganographic exfiltration through vector embedding stores. The codebase is organized as a pipeline: documents are loaded, chunked, embedded, optionally perturbed by one or more steganographic techniques, and stored in a vector backend; queries traverse the same path in reverse.

This document describes the modules that make up that pipeline and the boundaries between them. It does not duplicate the research narrative — for the threat model, technique catalog, and empirical results, see the preprint at <https://doi.org/10.5281/zenodo.20058256>.

## Pipeline

```
documents -> loaders -> preprocessors -> embedder -> obfuscator -> backend
                                                                    |
                                                                    v
                                                                 queries -> recovery
```

The pipeline is intentionally one-way at the storage boundary. Once embeddings are written to a vector backend, the only way to recover them is through the backend's query interface — there is no separate side channel. This matches how production RAG systems are deployed and is what makes the bit-channel survival results in the paper meaningful.

## Modules

### `loaders/`

Reads documents from disk and produces LangChain `Document` objects. `DocumentLoaderFactory` dispatches on file extension; the supported formats (PDF, DOCX, XLSX, CSV, JSON, XML, HTML, EML, MBOX, SQLite, plain text, Markdown) are listed in `DocumentConfig.supported_formats`. `preprocessors.py` provides optional content sanitization and PII detection that runs after load and before chunking.

### `steganography/`

Implements the four base perturbation techniques and two architectural variants studied in the paper:

- `obfuscation.EmbeddingObfuscator` — noise, rotation, scaling, offset, and combinations. Each method takes a `(n, d)` numpy array and returns the perturbed array; rotation also returns the rotation matrix needed for recovery.
- `fragmentation.MultiModelFragmenter` — splits a payload across embeddings produced by different models so no single model's output reveals the full payload.
- `timing.TimedExfiltrator` — paces ingestion to mimic legitimate user activity.
- `decoys.DecoyGenerator` — interleaves cover documents with payload-bearing chunks.

The obfuscator is deterministic given a seed; this is how the paper's measurements are reproducible.

### `vector_backends/`

Thin adapters that present a uniform `add` / `query` / `dump` interface over heterogeneous vector stores:

- `faiss_backend.py` — `FaissFlatBackend`, `FaissHNSWBackend`, `FaissPQBackend` (PQ8 and PQ16 sub-quantizer variants).
- `chroma_backend.py` — Chroma in HNSW mode.
- `qdrant_backend.py` — Qdrant with float32 storage and with int8 scalar quantization.

The cross-backend study in §5.6 of the paper runs the same payload through every backend and measures bit-channel survival end-to-end.

### `evasion/`

Operational-stealth modules that surround the embedding pipeline rather than modify the embeddings themselves: traffic mimicry, behavioral camouflage, network evasion, OPSEC cleanup, and detection-avoidance heuristics. These are demonstrative — the empirical study does not depend on them, and the threat model in the paper assumes the adversary has already obtained the access these modules are designed to maintain.

### `analysis/`

Detection and forensic tooling used both as defenders inside the empirical study and as standalone analysis utilities:

- `detectors/isolation_forest_detector.py`, `detectors/one_class_svm_detector.py` — fit on a clean baseline subset and score new embeddings; used for the AUC measurements throughout §5 of the paper.
- `detection_signatures.py` — KS-test, entropy, and norm-shift statistical signatures.
- `risk_assessment.py`, `forensic_tools.py` — higher-level reporting wrappers.

### `query/`

Reconstruction and recovery utilities. `advanced_engine.py` implements multi-strategy search (semantic + keyword + cross-reference); `context_reconstruction.py` rebuilds document structure from chunks; `recovery_tools.py` performs end-to-end de-obfuscation given the obfuscator's seed and matrices.

### `scripts/`

Each script is a self-contained experiment driver; they are the entry points for reproducing the paper. The most important ones are described in [usage.md](usage.md).

## Configuration

All modules accept a single `Config` object from `config.py`. The config is populated from environment variables (see [configuration.md](configuration.md)) and validated at startup. Modules never read environment variables directly — they take typed dataclasses, which makes them easy to test in isolation.

## Determinism

The empirical study relies on bit-for-bit reproducibility. The seed flows from `Config.random_seed` into Python's `random`, NumPy's default RNG, and the obfuscator's per-technique RNG. The detectors are also seeded. As long as the embedding model is identical, two runs of `scripts/empirical_study.py` produce identical CSV output.

## Extension points

Three places are designed to be extended:

1. **New steganographic technique** — add a method to `EmbeddingObfuscator` or a new module in `steganography/` and register it in the technique map. The empirical scripts pick up new techniques automatically if they follow the existing signature.
2. **New vector backend** — implement the `BackendBase` interface in `vector_backends/base.py`. The cross-backend study consumes any subclass.
3. **New detector** — implement the detector protocol (`fit`, `score`, `decide`) and add it to `analysis/detectors/__init__.py`. The empirical scripts and adaptive-attacker driver loop over the registered detectors.
