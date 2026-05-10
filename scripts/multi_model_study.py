#!/usr/bin/env python3
# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0

"""Cross-model empirical study: do the rotation/noise/scaling/offset
findings generalize across embedding architectures?

Embeds the standard sample corpus through each available local Ollama
model, applies the four base steganographic techniques, and runs the
detection battery (Isolation Forest + One-Class SVM) on each model's
embeddings independently.

Output: results/multi_model/<timestamp>/{summary.json, per_model.csv, report.md}
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from analysis.detectors import IsolationForestDetector, OneClassSVMDetector  # noqa: E402
from analysis.detectors.isolation_forest_detector import evaluate as evaluate_iforest  # noqa: E402
from loaders.document_factory import DocumentLoaderFactory  # noqa: E402
from steganography.obfuscation import EmbeddingObfuscator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multi_model")


MODELS = [
    "nomic-embed-text:latest",
    "embeddinggemma:300m",
    "snowflake-arctic-embed:335m",
    "mxbai-embed-large:335m",
]

TECHNIQUES = ["noise", "rotation", "scaling", "offset"]


@dataclass
class Row:
    model: str
    dim: int
    technique: str
    cosine_sim_mean: float
    pairwise_corr: float
    iforest_auc: float
    iforest_tpr_at_5fpr: float
    ocsvm_auc: float
    ocsvm_tpr_at_5fpr: float


def load_corpus(sample_dir: Path) -> list[str]:
    factory = DocumentLoaderFactory()
    files = sorted(p for p in sample_dir.iterdir() if p.is_file() and p.name != "README.md")
    log.info("Loading %d documents from %s", len(files), sample_dir)
    docs = factory.load_documents([str(f) for f in files])
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks if c.page_content.strip()]


def embed_with_ollama(model: str, texts: list[str]) -> np.ndarray:
    """Embed every chunk via the local Ollama daemon. One request per chunk
    keeps the implementation simple and matches what a typical RAG ingester
    would do."""
    out = []
    for i, t in enumerate(texts):
        r = ollama.embeddings(model=model, prompt=t)
        out.append(r["embedding"])
        if (i + 1) % 25 == 0:
            log.info("  %s: embedded %d/%d", model, i + 1, len(texts))
    return np.asarray(out, dtype=np.float64)


def run_technique(tech: str, embeddings: np.ndarray, obf: EmbeddingObfuscator) -> np.ndarray:
    if tech == "noise":
        return obf.inject_noise(embeddings)
    if tech == "rotation":
        rot, _ = obf.apply_rotation(embeddings)
        return rot
    if tech == "scaling":
        return obf.apply_scaling(embeddings)
    if tech == "offset":
        return obf.apply_offset(embeddings)
    raise ValueError(tech)


def cosine_pair(a: np.ndarray, b: np.ndarray) -> float:
    num = np.einsum("ij,ij->i", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(num / den))


def pairwise_corr(orig: np.ndarray, obf: np.ndarray) -> float:
    o = orig @ orig.T
    p = obf @ obf.T
    on = np.linalg.norm(orig, axis=1, keepdims=True)
    pn = np.linalg.norm(obf, axis=1, keepdims=True)
    o = o / (on @ on.T + 1e-12)
    p = p / (pn @ pn.T + 1e-12)
    return float(np.corrcoef(o.flatten(), p.flatten())[0, 1])


def detection_aucs(clean_train: np.ndarray, clean_test: np.ndarray, obf: np.ndarray):
    iforest = IsolationForestDetector(n_estimators=200, contamination=0.05).fit(clean_train)
    ocsvm = OneClassSVMDetector(nu=0.05).fit(clean_train)
    if_eval = evaluate_iforest(iforest, clean_test, obf)
    oc_eval = evaluate_iforest(ocsvm, clean_test, obf)
    return if_eval, oc_eval


def study_one_model(model: str, texts: list[str]) -> tuple[list[Row], dict]:
    t0 = time.perf_counter()
    embeddings = embed_with_ollama(model, texts)
    embed_time = time.perf_counter() - t0
    n, d = embeddings.shape
    log.info("%s: %d x %d in %.1fs", model, n, d, embed_time)

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    half = n // 2
    train_idx, test_idx = idx[:half], idx[half:]
    clean_train = embeddings[train_idx]
    clean_test = embeddings[test_idx]

    obfuscator = EmbeddingObfuscator(seed=42, noise_level=0.01)
    rows: list[Row] = []
    for tech in TECHNIQUES:
        obf_full = run_technique(tech, embeddings, obfuscator)
        cos = cosine_pair(embeddings, obf_full)
        pcorr = pairwise_corr(embeddings, obf_full)
        if_e, oc_e = detection_aucs(clean_train, clean_test, obf_full[test_idx])
        rows.append(Row(
            model=model, dim=d, technique=tech,
            cosine_sim_mean=cos, pairwise_corr=pcorr,
            iforest_auc=if_e["auc"], iforest_tpr_at_5fpr=if_e["tpr_at_5fpr"],
            ocsvm_auc=oc_e["auc"], ocsvm_tpr_at_5fpr=oc_e["tpr_at_5fpr"],
        ))
        log.info(
            "  %s/%-8s cos=%.4f pair=%.4f if_auc=%.3f oc_auc=%.3f",
            model, tech, cos, pcorr, if_e["auc"], oc_e["auc"],
        )

    meta = {"model": model, "n": n, "dim": d, "embed_seconds": embed_time}
    return rows, meta


def write_report(out_dir: Path, all_rows: list[Row], metas: list[dict]) -> None:
    by_model: dict[str, list[Row]] = {}
    for r in all_rows:
        by_model.setdefault(r.model, []).append(r)

    lines: list[str] = []
    lines.append("# Multi-model empirical study\n")
    lines.append("Same corpus, same techniques, four independent embedding models. ")
    lines.append("Reported numbers are mean cosine to original, pairwise-cosine correlation, ")
    lines.append("and detection AUCs (Isolation Forest, One-Class SVM) trained on a clean ")
    lines.append("half-corpus and evaluated on the held-out half.\n\n")

    lines.append("## Per-model corpus statistics\n\n")
    lines.append("| Model | Dim | Embed time (s) |\n|---|---|---|\n")
    for m in metas:
        lines.append(f"| `{m['model']}` | {m['dim']} | {m['embed_seconds']:.1f} |\n")

    lines.append("\n## Detection AUC by model and technique\n\n")
    lines.append("AUC = 0.5 means the detector is doing no better than random; AUC = 1.0 means perfect.\n\n")
    lines.append("| Model | Technique | cos | pair_corr | IF AUC | IF TPR@5%FPR | OC-SVM AUC | OC-SVM TPR@5%FPR |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for model in MODELS:
        for r in by_model.get(model, []):
            lines.append(
                f"| `{r.model}` | {r.technique} | {r.cosine_sim_mean:.4f} | {r.pairwise_corr:.4f} | "
                f"{r.iforest_auc:.3f} | {r.iforest_tpr_at_5fpr:.3f} | "
                f"{r.ocsvm_auc:.3f} | {r.ocsvm_tpr_at_5fpr:.3f} |\n"
            )

    lines.append("\n## Cross-model summary\n\n")
    for tech in TECHNIQUES:
        if_aucs = [r.iforest_auc for r in all_rows if r.technique == tech]
        oc_aucs = [r.ocsvm_auc for r in all_rows if r.technique == tech]
        lines.append(
            f"- **{tech}**: IF AUC range {min(if_aucs):.3f}--{max(if_aucs):.3f}, "
            f"OC-SVM AUC range {min(oc_aucs):.3f}--{max(oc_aucs):.3f} across "
            f"{len(if_aucs)} models.\n"
        )

    (out_dir / "report.md").write_text("".join(lines))


def main() -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "multi_model" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", out_dir)

    sample_dir = PROJECT_ROOT / "sample_docs"
    texts = load_corpus(sample_dir)
    log.info("Loaded %d chunks", len(texts))

    all_rows: list[Row] = []
    metas: list[dict] = []
    for model in MODELS:
        try:
            rows, meta = study_one_model(model, texts)
        except Exception as e:
            log.error("Model %s failed: %s", model, e)
            continue
        all_rows.extend(rows)
        metas.append(meta)

    summary = {
        "timestamp": timestamp,
        "models": metas,
        "techniques": TECHNIQUES,
        "n_chunks": len(texts),
        "rows": [asdict(r) for r in all_rows],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = out_dir / "per_model.csv"
    with csv_path.open("w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(asdict(r))

    write_report(out_dir, all_rows, metas)
    log.info("Wrote %s", out_dir / "summary.json")
    log.info("Wrote %s", csv_path)
    log.info("Wrote %s", out_dir / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
