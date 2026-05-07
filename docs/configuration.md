# Configuration

VectorSmuggle reads its runtime configuration from environment variables. The canonical source is `config.py`, which defines a set of typed dataclasses (`OpenAIConfig`, `VectorStoreConfig`, `DocumentConfig`, `SteganographyConfig`, etc.) and reads each field from the corresponding environment variable. A complete `.env.example` ships with the repository — the easiest way to get started is to copy it and fill in the secrets:

```bash
cp .env.example .env
$EDITOR .env
```

The remainder of this document explains the variables you are most likely to change. For the exhaustive list, read `config.py` directly — every dataclass field maps to an environment variable.

## Embedding provider

`OPENAI_API_KEY` is required if you embed through OpenAI. `OPENAI_EMBEDDING_MODEL` selects the model; the paper's headline numbers use `text-embedding-3-large`, while `.env.example` ships with `text-embedding-3-small` because it is cheaper for development. `OPENAI_LLM_MODEL` selects the chat model used by the query engine for answer synthesis (separate from the embedding model).

For local embedding, install a model into Ollama (`ollama pull nomic-embed-text`) and run `scripts/multi_model_study.py`, which talks to the local daemon at `localhost:11434`. The cross-model study in the paper uses four local Ollama models in addition to OpenAI.

## Vector backend

`VECTOR_DB` selects the backend: `faiss` (default), `qdrant`, or `pinecone`. Each has its own settings:

- **FAISS**: `FAISS_INDEX_PATH` controls where the index is persisted on disk. FAISS is process-local and is the simplest backend for reproducing single-machine experiments.
- **Qdrant**: `QDRANT_URL` points at the server (`http://localhost:6333` for a local docker container). The cross-backend study uses Qdrant with both float32 storage and int8 scalar quantization to measure how scalar quantization affects bit-channel survival.
- **Pinecone**: `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` are required. Pinecone is a hosted backend — running large studies against it incurs costs.

`COLLECTION_NAME` and `INDEX_NAME` apply to all backends and default to `rag-exfil-poc`.

For the cross-backend study (`scripts/cross_backend_study.py`), the script ignores `VECTOR_DB` and instantiates every backend in turn from `vector_backends/` directly.

## Document processing

`CHUNK_SIZE` and `CHUNK_OVERLAP` control text splitting; the paper's results use `512` and `50`, which are also the defaults. `CHUNKING_STRATEGY` accepts `auto`, `fixed`, or `semantic` — `auto` selects between the others based on document type.

`ENABLE_PREPROCESSING` toggles the entire preprocessing pipeline. `SANITIZE_CONTENT` and `DETECT_SENSITIVE_DATA` are independent: sanitization rewrites detected PII before embedding, while detection only flags it in the metadata. For the paper the corpus is synthetic PII, so detection is on but sanitization is off — the whole point is that the embeddings carry the PII.

The format-specific toggles (`OFFICE_EXTRACT_TABLES`, `CSV_DELIMITER`, `JSON_FLATTEN_NESTED`, `EMAIL_INCLUDE_ATTACHMENTS`, `DATABASE_QUERY`) only matter for non-text formats.

## Steganography

The base perturbation techniques each have a single intensity parameter:

- `STEGO_NOISE_LEVEL` — Gaussian noise standard deviation as a fraction of embedding L2 norm. `0.005` is "low," `0.01` is the default operating point used throughout the paper, `0.05` is "high."
- `STEGO_ROTATION_ANGLE` — angle in radians for the planar rotation generator. The rotation sweep in §5.5 sweeps this from `0.001` to `π`.
- `STEGO_SCALING_FACTOR` — multiplicative factor applied to vector norms. `0.95` reduces norms by 5%.
- `STEGO_OFFSET_RANGE` — magnitude of the constant offset added to every vector.

`STEGO_FRAGMENT_SIZE` and `STEGO_INTERLEAVE_RATIO` control multi-model fragmentation; `STEGO_DECOY_RATIO` controls how many cover chunks are interleaved with payload chunks. The temporal evasion settings (`STEGO_BASE_DELAY`, `STEGO_DELAY_VARIANCE`, `STEGO_BATCH_SIZE`, `STEGO_MAX_BATCHES_PER_HOUR`, `STEGO_BUSINESS_HOURS_ONLY`) only apply when the timing module is enabled.

`STEGO_ENABLED` is a master switch: with it set to `false`, the embedder runs as a normal RAG ingester with no perturbation, which is how the clean-baseline embeddings for the detectors are generated.

## Logging

`LOG_LEVEL` is `INFO` by default. Set it to `DEBUG` to trace per-chunk processing — useful when a script silently drops content. `LOG_FORMAT` is a Python `logging` format string, not a JSON/text toggle.

## Reproducing paper measurements

The paper's measurements are produced by setting:

```bash
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
CHUNK_SIZE=512
CHUNK_OVERLAP=50
STEGO_NOISE_LEVEL=0.01
STEGO_ROTATION_ANGLE=0.1
STEGO_SCALING_FACTOR=0.95
STEGO_OFFSET_RANGE=0.05
```

and running the empirical scripts in `scripts/`. See [usage.md](usage.md) for the full reproduction workflow.

## Validation

`Config.validate()` is called once at startup. It checks that required keys are present (e.g. `OPENAI_API_KEY` when the embedding provider is OpenAI) and that numeric ranges are sane (`CHUNK_OVERLAP < CHUNK_SIZE`, `STEGO_NOISE_LEVEL` non-negative, etc.). A misconfigured run fails at startup rather than partway through embedding.
