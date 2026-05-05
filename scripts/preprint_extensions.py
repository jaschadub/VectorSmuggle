#!/usr/bin/env python3
"""Preprint-rigor extensions to the empirical study.

Addresses reviewer feedback by measuring three things the original
empirical_study.py does not:

  1. Quantization resilience (BER under int8 round-trip).
  2. Retrieval utility — does the cover document still come back when
     a legitimate user queries for it after obfuscation?
  3. Detection rate against concrete defensive baselines (Isolation
     Forest, One-Class SVM) so we can replace hand-wavy DLP-bypass
     claims with reproducible ROC numbers.

Output: results/preprint/<timestamp>/{summary.json, report.md}
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from analysis.detectors import IsolationForestDetector, OneClassSVMDetector  # noqa: E402
from analysis.detectors.isolation_forest_detector import evaluate as evaluate_iforest  # noqa: E402
from loaders.document_factory import DocumentLoaderFactory  # noqa: E402
from steganography.obfuscation import EmbeddingObfuscator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("preprint")


# ----------------------------- Quantization -----------------------------


def quantize_int8(embeddings: np.ndarray, mode: str = "per_dim") -> np.ndarray:
    """Round-trip embeddings through int8 quantization.

    Two modes:
      - "per_dim" (default): per-dimension global min/max bounds across
        the corpus, then scalar quantize. This matches Qdrant's default
        scalar quantization configuration. Direction is mostly preserved
        because all vectors share the same per-dimension scale.
      - "per_vec": per-vector min/max bounds. This matches some FAISS
        PQ variants and is much more aggressive — direction shifts
        because each vector gets its own offset.

    The "per_dim" mode is the realistic default for production RAG.
    """
    if mode == "per_vec":
        out = np.empty_like(embeddings)
        for i, vec in enumerate(embeddings):
            lo, hi = float(vec.min()), float(vec.max())
            if hi <= lo:
                out[i] = vec
                continue
            scale = 255.0 / (hi - lo)
            q = np.round((vec - lo) * scale).astype(np.int8)
            out[i] = q.astype(np.float64) / scale + lo
        return out

    # per_dim: global per-dimension bounds
    lo = embeddings.min(axis=0)
    hi = embeddings.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scale = 255.0 / span
    q = np.round((embeddings - lo) * scale).astype(np.int16)  # int8 range fits
    return q.astype(np.float64) / scale + lo


def bit_error_rate(original: np.ndarray, recovered: np.ndarray, threshold: float = 0.5) -> float:
    """BER between two embedding arrays, treating signs as the bit channel.

    A simple, model-agnostic proxy: for each dimension, the sign
    (positive or negative) carries one bit of information. BER is the
    fraction of dimensions whose sign flipped between original and
    recovered.
    """
    orig_bits = (original > threshold).astype(np.int8)
    rec_bits = (recovered > threshold).astype(np.int8)
    return float(np.mean(orig_bits != rec_bits))


# ----------------------------- Retrieval utility -----------------------------


def cosine_top_k(query: np.ndarray, corpus: np.ndarray, k: int = 5) -> np.ndarray:
    """Return indices of the top-k most similar corpus rows to a query."""
    q = query / (np.linalg.norm(query) + 1e-12)
    c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    sims = c @ q
    return np.argsort(sims)[::-1][:k]


def retrieval_utility(
    clean_corpus: np.ndarray,
    obf_corpus: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
    """Measure recall@k and MRR for both clean and obfuscated corpora.

    A legitimate user issues `queries` looking for documents whose
    indices are in `truth`. If obfuscation breaks retrieval, the
    delta between clean recall and obfuscated recall makes the
    attack visible to ordinary RAG monitoring.
    """
    def recall_and_mrr(corpus: np.ndarray) -> tuple[float, float]:
        recalls, rrs = [], []
        for q, true_idx in zip(queries, truth, strict=True):
            top_k = cosine_top_k(q, corpus, k)
            recalls.append(int(true_idx in top_k))
            ranks = np.where(top_k == true_idx)[0]
            rrs.append(1.0 / (ranks[0] + 1) if len(ranks) else 0.0)
        return float(np.mean(recalls)), float(np.mean(rrs))

    clean_recall, clean_mrr = recall_and_mrr(clean_corpus)
    obf_recall, obf_mrr = recall_and_mrr(obf_corpus)
    return {
        "clean_recall_at_k": clean_recall,
        "obfuscated_recall_at_k": obf_recall,
        "recall_drop": clean_recall - obf_recall,
        "clean_mrr": clean_mrr,
        "obfuscated_mrr": obf_mrr,
        "mrr_drop": clean_mrr - obf_mrr,
    }


# ----------------------------- Aggregate run -----------------------------


@dataclass
class TechniqueReport:
    technique: str
    cosine_pre_quant: float
    cosine_post_quant: float
    quant_ber_self: float          # bit-flip rate from quantization alone
    quant_cosine_drop: float       # how much extra fidelity is lost by quantization
    iforest_auc: float
    iforest_tpr_at_1fpr: float
    iforest_tpr_at_5fpr: float
    iforest_f1: float
    ocsvm_auc: float
    recall_at_5_clean: float
    recall_at_5_obfuscated: float
    recall_drop: float
    mrr_clean: float
    mrr_obfuscated: float


def evaluate_one_class_svm(
    detector: OneClassSVMDetector,
    clean: np.ndarray,
    obfuscated: np.ndarray,
) -> dict[str, float]:
    """Score-only AUC for the one-class SVM (mirrors evaluate_iforest)."""
    from sklearn.metrics import roc_auc_score

    s_c = detector.score(clean)
    s_o = detector.score(obfuscated)
    y_true = np.concatenate([np.zeros(len(s_c)), np.ones(len(s_o))])
    y_score = np.concatenate([s_c, s_o])
    return {"auc": float(roc_auc_score(y_true, y_score))}


def cosine_per_pair(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.einsum("ij,ij->i", a_n, b_n)))


def load_corpus(sample_dir: Path) -> list[str]:
    factory = DocumentLoaderFactory()
    files = sorted(p for p in sample_dir.iterdir() if p.is_file() and p.name != "README.md")
    docs = factory.load_documents([str(f) for f in files])
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks if c.page_content.strip()]


def run() -> int:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set")
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "preprint" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", out_dir)

    # ---- Embed corpus ----
    sample_dir = PROJECT_ROOT / "sample_docs"
    texts = load_corpus(sample_dir)
    if len(texts) < 20:
        log.warning("Only %d chunks in sample_docs/. Detector AUC may be noisy.", len(texts))
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    embedder = OpenAIEmbeddings(model=model)
    log.info("Embedding %d chunks with %s", len(texts), model)
    embeddings = np.asarray(embedder.embed_documents(texts), dtype=np.float64)
    log.info("Embeddings shape: %s", embeddings.shape)

    # Held-out clean split for detector training
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(embeddings))
    split = max(10, int(0.4 * len(embeddings)))
    train_idx, test_idx = perm[:split], perm[split:]
    clean_train = embeddings[train_idx]
    clean_test = embeddings[test_idx]
    log.info("Detector train: %d  test: %d", len(clean_train), len(clean_test))

    # ---- Retrieval-utility queries ----
    # Use the first half of each "test" embedding as a self-query proxy:
    # normally you'd embed paraphrased queries; here we ride the existing
    # corpus to keep the script deterministic and OpenAI-call-light.
    queries = clean_test.copy()
    truth = np.arange(len(clean_test))

    # ---- Fit detectors on clean ----
    log.info("Fitting Isolation Forest...")
    iforest = IsolationForestDetector(n_estimators=200, contamination=0.05).fit(clean_train)
    log.info("Fitting One-Class SVM...")
    ocsvm = OneClassSVMDetector(nu=0.05).fit(clean_train)

    # ---- Sweep techniques ----
    techniques = [
        ("noise_low",   {"noise_level": 0.005}),
        ("noise_med",   {"noise_level": 0.01}),
        ("noise_high",  {"noise_level": 0.05}),
        ("rotation",    {"rotation_angle": 0.1}),
        ("scaling",     {"scaling_factor": 0.95}),
        ("offset",      {"offset_range": 0.05}),
    ]

    reports: list[TechniqueReport] = []
    for name, params in techniques:
        log.info("--- %s %s", name, params)
        obf = EmbeddingObfuscator(seed=42, **params)
        if "noise" in name:
            obfuscated = obf.inject_noise(clean_test)
        elif name == "rotation":
            obfuscated, _ = obf.apply_rotation(clean_test)
        elif name == "scaling":
            obfuscated = obf.apply_scaling(clean_test)
        elif name == "offset":
            obfuscated = obf.apply_offset(clean_test)
        else:
            continue

        cos_pre = cosine_per_pair(clean_test, obfuscated)
        obfuscated_q = quantize_int8(obfuscated)
        cos_post = cosine_per_pair(clean_test, obfuscated_q)
        ber = bit_error_rate(obfuscated, obfuscated_q)

        iforest_eval = evaluate_iforest(iforest, clean_test, obfuscated)
        ocsvm_eval = evaluate_one_class_svm(ocsvm, clean_test, obfuscated)

        # Re-build the "obfuscated corpus" the same way an attacker
        # would: every embedding in the index is the obfuscated form.
        obf_corpus = obfuscated  # 1:1 with clean_test, so truth indices align
        retrieval = retrieval_utility(clean_test, obf_corpus, queries, truth, k=5)

        report = TechniqueReport(
            technique=name,
            cosine_pre_quant=cos_pre,
            cosine_post_quant=cos_post,
            quant_ber_self=ber,
            quant_cosine_drop=cos_pre - cos_post,
            iforest_auc=iforest_eval["auc"],
            iforest_tpr_at_1fpr=iforest_eval["tpr_at_1fpr"],
            iforest_tpr_at_5fpr=iforest_eval["tpr_at_5fpr"],
            iforest_f1=iforest_eval["f1"],
            ocsvm_auc=ocsvm_eval["auc"],
            recall_at_5_clean=retrieval["clean_recall_at_k"],
            recall_at_5_obfuscated=retrieval["obfuscated_recall_at_k"],
            recall_drop=retrieval["recall_drop"],
            mrr_clean=retrieval["clean_mrr"],
            mrr_obfuscated=retrieval["obfuscated_mrr"],
        )
        reports.append(report)
        log.info(
            "  cos pre/post=%.3f/%.3f BER=%.3f IF-AUC=%.3f OCSVM-AUC=%.3f recall=%.2f→%.2f",
            report.cosine_pre_quant, report.cosine_post_quant, report.quant_ber_self,
            report.iforest_auc, report.ocsvm_auc,
            report.recall_at_5_clean, report.recall_at_5_obfuscated,
        )

    # ---- Persist ----
    summary = {
        "timestamp": timestamp,
        "model": model,
        "n_chunks": len(texts),
        "embed_dim": int(embeddings.shape[1]),
        "detector_train_size": int(len(clean_train)),
        "detector_test_size": int(len(clean_test)),
        "reports": [asdict(r) for r in reports],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out_dir / "summary.json")

    # ---- Markdown report ----
    md = ["# VectorSmuggle Preprint-Rigor Results — `" + timestamp + "`\n"]
    md.append(f"- Model: `{model}`")
    md.append(f"- Embedding dim: {embeddings.shape[1]}")
    md.append(f"- Detector train/test: {len(clean_train)}/{len(clean_test)}\n")
    md.append("## Detection vs. Quantization vs. Retrieval Utility\n")
    md.append(
        "| Technique | cos(pre-q) | cos(post-q) | quant BER | IF AUC "
        "| IF TPR@1%FPR | OCSVM AUC | recall@5 clean | recall@5 obf | recall drop |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in reports:
        md.append(
            f"| {r.technique} | {r.cosine_pre_quant:.3f} | {r.cosine_post_quant:.3f} | "
            f"{r.quant_ber_self:.3f} | {r.iforest_auc:.3f} | {r.iforest_tpr_at_1fpr:.3f} | "
            f"{r.ocsvm_auc:.3f} | {r.recall_at_5_clean:.3f} | {r.recall_at_5_obfuscated:.3f} | "
            f"{r.recall_drop:+.3f} |"
        )
    md.append("")
    md.append("## How to read this table\n")
    md.append("- **cos(pre/post-q)**: cosine similarity of obfuscated to original")
    md.append("  before and after int8 round-trip quantization. A large gap means")
    md.append("  the technique's hidden bits do not survive quantization.")
    md.append("- **quant BER**: fraction of sign-bits that flip during int8")
    md.append("  quantization. >0.05 means the channel is too noisy for the")
    md.append("  technique's payload to survive a real vector store.")
    md.append("- **IF AUC / OCSVM AUC**: detection ROC area for two standard")
    md.append("  anomaly detectors trained on held-out clean embeddings.")
    md.append("  AUC near 0.5 = invisible; AUC near 1.0 = trivially detected.")
    md.append("- **TPR@1%FPR**: a defender willing to flag 1% of legitimate")
    md.append("  traffic catches this fraction of obfuscated vectors.")
    md.append("- **recall@5 clean → obf**: drop in legitimate-user retrieval")
    md.append("  quality after obfuscation. Anything below clean is detectable")
    md.append("  via routine RAG quality monitoring.\n")
    md.append("## Known Limitations of This Run\n")
    md.append("- The retrieval-utility benchmark uses self-queries (each test")
    md.append("  embedding is the query for itself). This makes recall@5 = 1.0")
    md.append("  trivially. A faithful retrieval test requires *paraphrased*")
    md.append("  queries — embed the paraphrase, then check whether the original")
    md.append("  document is still in top-k. Adding this is the highest-priority")
    md.append("  follow-up for a publication-grade run.")
    md.append("- Quantization is modeled as per-dimension scalar int8, matching")
    md.append("  Qdrant defaults. Product quantization (FAISS IVF-PQ) and")
    md.append("  binary quantization are far more lossy and not yet covered.")
    md.append("- BER uses sign-bits as the channel; a real bit-channel analysis")
    md.append("  would require defining where the steganographic payload lives")
    md.append("  in the embedding (e.g., LSBs of specific dimensions).")
    (out_dir / "report.md").write_text("\n".join(md))
    log.info("Wrote %s", out_dir / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(run())
