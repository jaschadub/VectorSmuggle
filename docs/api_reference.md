# API Reference

This document covers the public APIs that you are most likely to call when using VectorSmuggle as a library. It is not a generated reference — for full type signatures, read the source. The modules below are stable in the sense that the empirical scripts in `scripts/` depend on them; private modules and helper functions are not documented here.

## `config`

`config.Config` is the single configuration object. Construct it with no arguments to read from environment variables (and `.env`):

```python
from config import Config

cfg = Config()
cfg.validate()
```

`Config` is a dataclass containing nested config objects: `cfg.openai`, `cfg.vector_store`, `cfg.document`, `cfg.steganography`, `cfg.evasion`, `cfg.query`. Each nested object is itself a dataclass with named fields — see `config.py` for the full list.

## `loaders`

### `DocumentLoaderFactory`

```python
from loaders.document_factory import DocumentLoaderFactory

factory = DocumentLoaderFactory()
documents = factory.load_documents(["path/to/file1.pdf", "path/to/file2.csv"])
```

Returns a list of `langchain_core.documents.Document` objects. Dispatch is by file extension. Unsupported formats produce a warning and are skipped, not raised.

### `ContentPreprocessor`

```python
from loaders.preprocessors import ContentPreprocessor

pre = ContentPreprocessor(sanitize=False, detect_pii=True)
processed = pre.preprocess_documents(documents)
```

Adds `metadata["pii_detected"]` to documents containing detected PII patterns. With `sanitize=True`, also rewrites detected patterns inline.

## `steganography`

### `EmbeddingObfuscator`

```python
from steganography.obfuscation import EmbeddingObfuscator

obf = EmbeddingObfuscator(seed=42, noise_level=0.01)

noised = obf.inject_noise(embeddings)
rotated, rot_matrix = obf.apply_rotation(embeddings)
scaled = obf.apply_scaling(embeddings)
offset = obf.apply_offset(embeddings)

result = obf.obfuscate(embeddings, techniques=["noise", "rotation", "scaling"])
recovered = obf.deobfuscate(result)
```

`obfuscate` returns a dict containing the perturbed embeddings and the metadata needed by `deobfuscate` (the rotation matrix, scaling factor, and seed). Recovery is lossy for noise and offset (these have no inverse) and lossless for rotation and scaling.

### `MultiModelFragmenter`

```python
from steganography.fragmentation import MultiModelFragmenter

frag = MultiModelFragmenter(models=["text-embedding-3-small", "text-embedding-3-large"])
fragmented = frag.fragment(payload_text, strategy="round_robin")
recovered = frag.reassemble(fragmented)
```

## `vector_backends`

All backends implement the same interface defined in `vector_backends/base.BackendBase`:

```python
class BackendBase(Protocol):
    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None: ...
    def query(self, query_vec: np.ndarray, k: int) -> list[QueryResult]: ...
    def dump(self) -> np.ndarray: ...
```

Concrete classes: `FaissFlatBackend`, `FaissHNSWBackend`, `FaissPQBackend(m=8 | 16)`, `ChromaBackend`, `QdrantBackend(quantization=None | "int8")`. The cross-backend study iterates over all of them and confirms bit-channel survival end-to-end.

## `analysis.detectors`

### `IsolationForestDetector`, `OneClassSVMDetector`

```python
from analysis.detectors import IsolationForestDetector, OneClassSVMDetector
from analysis.detectors.isolation_forest_detector import evaluate

iforest = IsolationForestDetector(n_estimators=200, contamination=0.05).fit(clean_train)
metrics = evaluate(iforest, clean_test, obfuscated_test)
# metrics: {"auc": ..., "f1": ..., "tpr_at_1fpr": ..., "tpr_at_5fpr": ...}
```

Both detectors expose the same trio: `fit(clean: np.ndarray)`, `score(x: np.ndarray) -> np.ndarray`, `decide(x: np.ndarray) -> np.ndarray[bool]`. The `evaluate` helper from `isolation_forest_detector` works on either detector — it depends only on the public interface.

## `query`

### `AdvancedQueryEngine`

```python
from query.advanced_engine import AdvancedQueryEngine

engine = AdvancedQueryEngine(vector_store=backend, llm=llm, embeddings=embedder)
results = engine.multi_strategy_search("quarterly revenue", k=10)
```

Combines semantic search, keyword filtering, and cross-reference expansion. Returns a list of `Document` objects with similarity scores in the metadata.

### `ContextReconstructor`, `DataRecoveryTools`

`ContextReconstructor` rebuilds document structure from a set of retrieved chunks. `DataRecoveryTools.reconstruct_original_data(chunks, context, references)` is the end-to-end recovery entry point used by the empirical scripts to verify round-trip fidelity.

## Errors

Each module defines a single exception class for its public errors: `ConfigurationError`, `SteganographyError`, `BackendError`, `QueryError`. Internal errors are not wrapped — numpy and sklearn exceptions surface as-is.

## Logging

All modules use the standard `logging` library and write to the `vectorsmuggle.<module>` logger hierarchy. Setting `LOG_LEVEL=DEBUG` traces per-chunk processing.
