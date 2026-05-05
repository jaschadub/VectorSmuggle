#!/usr/bin/env python3
"""Empirical study for VectorSmuggle research paper.

Measures across all steganographic techniques:
  1. Fidelity     — cosine similarity preservation vs. original embeddings
  2. Capacity     — bits/dimension proxied by reconstruction error vs. noise floor
  3. Detection    — KS-test, entropy, and norm-shift statistical signatures
  4. Recovery     — accuracy of de-obfuscation
  5. Performance  — wall-clock time per technique

Output: results/empirical/<timestamp>/{summary.json, per_technique.csv, plots/}
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from loaders.document_factory import DocumentLoaderFactory  # noqa: E402
from steganography.obfuscation import EmbeddingObfuscator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("empirical")


@dataclass
class Measurement:
    technique: str
    n_vectors: int
    dim: int
    elapsed_ms: float
    cosine_sim_mean: float
    cosine_sim_std: float
    cosine_sim_p05: float
    norm_ratio_mean: float
    norm_ratio_std: float
    mse_vs_original: float
    ks_statistic: float
    ks_pvalue: float
    entropy_original: float
    entropy_obfuscated: float
    pairwise_corr: float
    recovery_cosine: float | None


def load_corpus(sample_dir: Path) -> list[str]:
    """Load and chunk all sample documents into a list of text chunks."""
    factory = DocumentLoaderFactory()
    files = sorted(p for p in sample_dir.iterdir() if p.is_file() and p.name != "README.md")
    log.info("Loading %d documents from %s", len(files), sample_dir)
    docs = factory.load_documents([str(f) for f in files])
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks if c.page_content.strip()]


def shannon_entropy(arr: np.ndarray, bins: int = 50) -> float:
    """Approximate Shannon entropy of a flattened embedding array."""
    flat = arr.flatten()
    hist, _ = np.histogram(flat, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)) / bins)


def measure(
    technique: str,
    original: np.ndarray,
    obfuscated: np.ndarray,
    elapsed_ms: float,
    recovered: np.ndarray | None = None,
) -> Measurement:
    """Compute the full measurement battery for a technique."""
    n, d = original.shape

    cos_per_vec = np.array(
        [np.dot(o, b) / (np.linalg.norm(o) * np.linalg.norm(b))
         for o, b in zip(original, obfuscated, strict=True)]
    )
    norm_ratio = np.linalg.norm(obfuscated, axis=1) / np.linalg.norm(original, axis=1)
    mse = float(np.mean((original - obfuscated) ** 2))

    ks_stat, ks_p = stats.ks_2samp(original.flatten(), obfuscated.flatten())

    orig_sim = cosine_similarity(original)
    obf_sim = cosine_similarity(obfuscated)
    pairwise_corr = float(np.corrcoef(orig_sim.flatten(), obf_sim.flatten())[0, 1])

    recovery_cos = None
    if recovered is not None:
        rec_per_vec = np.array(
            [np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r))
             for o, r in zip(original, recovered, strict=True)]
        )
        recovery_cos = float(np.mean(rec_per_vec))

    return Measurement(
        technique=technique,
        n_vectors=n,
        dim=d,
        elapsed_ms=elapsed_ms,
        cosine_sim_mean=float(cos_per_vec.mean()),
        cosine_sim_std=float(cos_per_vec.std()),
        cosine_sim_p05=float(np.percentile(cos_per_vec, 5)),
        norm_ratio_mean=float(norm_ratio.mean()),
        norm_ratio_std=float(norm_ratio.std()),
        mse_vs_original=mse,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        entropy_original=shannon_entropy(original),
        entropy_obfuscated=shannon_entropy(obfuscated),
        pairwise_corr=pairwise_corr,
        recovery_cosine=recovery_cos,
    )


def run_single_technique(
    technique: str,
    embeddings: np.ndarray,
    obfuscator: EmbeddingObfuscator,
) -> Measurement:
    """Apply one technique end-to-end and measure."""
    log.info("Running technique: %s", technique)
    t0 = time.perf_counter()

    if technique == "noise":
        obf = obfuscator.inject_noise(embeddings)
        recovered = None
    elif technique == "rotation":
        obf, rot_matrix = obfuscator.apply_rotation(embeddings)
        recovered = obf @ rot_matrix
    elif technique == "scaling":
        obf = obfuscator.apply_scaling(embeddings)
        recovered = obf / obfuscator.scaling_factor
    elif technique == "offset":
        obf = obfuscator.apply_offset(embeddings)
        recovered = None
    elif technique == "noise+rotation":
        noisy = obfuscator.inject_noise(embeddings)
        obf, rot_matrix = obfuscator.apply_rotation(noisy)
        recovered = obf @ rot_matrix
    elif technique == "noise+rotation+scaling":
        noisy = obfuscator.inject_noise(embeddings)
        rotated, rot_matrix = obfuscator.apply_rotation(noisy)
        obf = obfuscator.apply_scaling(rotated)
        recovered = (obf / obfuscator.scaling_factor) @ rot_matrix
    elif technique == "all":
        result = obfuscator.obfuscate(
            embeddings,
            techniques=["noise", "rotation", "scaling", "offset"],
        )
        obf = result["embeddings"]
        recovered = obfuscator.deobfuscate(result)
    else:
        raise ValueError(f"Unknown technique: {technique}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return measure(technique, embeddings, obf, elapsed_ms, recovered)


def main() -> int:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set")
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "empirical" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # Load corpus
    sample_dir = PROJECT_ROOT / "sample_docs"
    texts = load_corpus(sample_dir)
    log.info("Loaded %d text chunks", len(texts))

    # Generate baseline embeddings
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    log.info("Embedding %d chunks with %s", len(texts), model)
    embedder = OpenAIEmbeddings(model=model)
    t0 = time.perf_counter()
    raw = embedder.embed_documents(texts)
    embed_time = time.perf_counter() - t0
    embeddings = np.asarray(raw, dtype=np.float64)
    log.info(
        "Generated %d-dim embeddings in %.2fs (%.1f ms/chunk)",
        embeddings.shape[1], embed_time, embed_time * 1000 / len(texts),
    )

    # Save raw embeddings for reproducibility
    np.save(out_dir / "embeddings_baseline.npy", embeddings)

    # Sweep techniques. Keep params close to defaults so paper figures
    # reflect realistic operating points.
    configs = [
        ("noise_low",  {"noise_level": 0.005}),
        ("noise_med",  {"noise_level": 0.01}),
        ("noise_high", {"noise_level": 0.05}),
        ("default",    {}),
    ]

    technique_set = ["noise", "rotation", "scaling", "offset",
                     "noise+rotation", "noise+rotation+scaling", "all"]

    measurements: list[Measurement] = []
    for cfg_name, params in configs:
        log.info("--- Configuration: %s (%s)", cfg_name, params)
        obf = EmbeddingObfuscator(seed=42, **params)
        for tech in technique_set:
            m = run_single_technique(tech, embeddings, obf)
            m.technique = f"{cfg_name}::{tech}"
            measurements.append(m)
            log.info(
                "  %-40s cos=%.4f  pair_corr=%.4f  mse=%.6f  rec=%s  %.1fms",
                m.technique, m.cosine_sim_mean, m.pairwise_corr, m.mse_vs_original,
                f"{m.recovery_cosine:.4f}" if m.recovery_cosine is not None else "n/a",
                m.elapsed_ms,
            )

    # Persist results
    summary = {
        "timestamp": timestamp,
        "model": model,
        "n_chunks": len(texts),
        "embed_dim": int(embeddings.shape[1]),
        "embed_time_seconds": embed_time,
        "configurations": [c[0] for c in configs],
        "techniques": technique_set,
        "measurements": [asdict(m) for m in measurements],
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "per_technique.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(measurements[0]).keys()))
        w.writeheader()
        for m in measurements:
            w.writerow(asdict(m))

    log.info("Wrote %s", out_dir / "summary.json")
    log.info("Wrote %s", csv_path)
    log.info("Done. %d measurements collected.", len(measurements))
    return 0


if __name__ == "__main__":
    sys.exit(main())
