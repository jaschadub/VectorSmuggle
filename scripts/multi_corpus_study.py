#!/usr/bin/env python3
# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Cross-corpus empirical study: do the rotation/noise/scaling/offset
findings generalize beyond the small synthetic sample corpus?

The companion ``multi_model_study.py`` varied the embedding model on a
fixed corpus. A reviewer-defensible answer to "show this at scale" is
the dual: hold the model fixed, vary the corpus. We embed three corpora
through one local Ollama embedding model, apply the same four base
steganographic techniques, and run the same off-the-shelf detector
battery (Isolation Forest + One-Class SVM) on each corpus
independently.

Corpora:

  - ``sample``: the existing 68-chunk synthetic corpus from
    ``sample_docs/`` (the baseline used in the rest of the paper).
  - ``nfcorpus``: BEIR NFCorpus, ~3.6k medical-domain documents.
  - ``quora``: a ``--quora-n`` subset of BEIR Quora question pairs
    (web Q&A breadth; defaults to 10,000). Substituted for MS MARCO
    because the BEIR-distribution MS MARCO zip is ~1 GB to extract a
    few thousand passages; Quora is the same kind of web-scraped open
    corpus at ~16 MB and gives the same "show this on a real public
    corpus" cross-domain replication.

The BEIR corpora are fetched from their primary distribution at
``public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/`` as zip
files containing uncompressed ``corpus.jsonl``; nothing beyond stdlib
is needed to parse them. Raw chunks and embeddings are cached under
``data/corpora/`` so re-runs only pay the steganography + detection
cost.

Output: ``results/multi_corpus/<timestamp>/{summary.json, per_corpus.csv, report.md}``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
import urllib.request
import zipfile
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
log = logging.getLogger("multi_corpus")


TECHNIQUES = ["noise", "rotation", "scaling", "offset"]

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
NFCORPUS_URL = f"{BEIR_BASE_URL}/nfcorpus.zip"
QUORA_URL = f"{BEIR_BASE_URL}/quora.zip"


@dataclass
class Row:
    corpus: str
    n_chunks: int
    model: str
    dim: int
    technique: str
    cosine_sim_mean: float
    pairwise_corr: float
    iforest_auc: float
    iforest_tpr_at_5fpr: float
    ocsvm_auc: float
    ocsvm_tpr_at_5fpr: float


# --- corpus loaders --------------------------------------------------------


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)


def _chunk_strings(texts: list[str]) -> list[str]:
    spl = _splitter()
    out: list[str] = []
    for t in texts:
        for c in spl.split_text(t):
            if c.strip():
                out.append(c)
    return out


def load_sample_chunks(project_root: Path) -> list[str]:
    factory = DocumentLoaderFactory()
    sample_dir = project_root / "sample_docs"
    files = sorted(p for p in sample_dir.iterdir() if p.is_file() and p.name != "README.md")
    log.info("sample: loading %d files from %s", len(files), sample_dir)
    docs = factory.load_documents([str(f) for f in files])
    chunks = _splitter().split_documents(docs)
    return [c.page_content for c in chunks if c.page_content.strip()]


def _download_beir_zip(url: str, cache_dir: Path, name: str) -> Path:
    """Download a BEIR primary-distribution zip if not already cached."""
    zip_path = cache_dir / f"{name}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", url, zip_path)
    req = urllib.request.Request(url, headers={"User-Agent": "VectorSmuggle/research"})
    with urllib.request.urlopen(req, timeout=120) as resp, zip_path.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    log.info("downloaded %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / (1 << 20))
    return zip_path


def _read_beir_corpus_jsonl(zip_path: Path, max_docs: int | None) -> list[dict]:
    """Stream the ``corpus.jsonl`` member out of a BEIR zip without
    extracting the whole archive. Stops after ``max_docs`` records."""
    out: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        try:
            member = next(n for n in zf.namelist() if n.endswith("/corpus.jsonl") or n == "corpus.jsonl")
        except StopIteration as e:
            raise RuntimeError(f"corpus.jsonl not found in {zip_path}") from e
        log.info("reading %s/%s (max_docs=%s)", zip_path.name, member,
                 max_docs if max_docs is not None else "all")
        with zf.open(member) as f, io.TextIOWrapper(f, encoding="utf-8") as tf:
            for raw in tf:
                line = raw.strip()
                if not line:
                    continue
                out.append(json.loads(line))
                if max_docs is not None and len(out) >= max_docs:
                    break
    return out


def load_nfcorpus_chunks(cache_dir: Path) -> list[str]:
    zip_path = _download_beir_zip(NFCORPUS_URL, cache_dir, "nfcorpus")
    objs = _read_beir_corpus_jsonl(zip_path, max_docs=None)
    log.info("nfcorpus: %d documents", len(objs))
    texts = [(o.get("title", "") + "\n" + o.get("text", "")).strip() for o in objs]
    texts = [t for t in texts if t]
    return _chunk_strings(texts)


def load_quora_chunks(cache_dir: Path, n_docs: int) -> list[str]:
    zip_path = _download_beir_zip(QUORA_URL, cache_dir, "quora")
    objs = _read_beir_corpus_jsonl(zip_path, max_docs=n_docs)
    log.info("quora: %d documents", len(objs))
    texts = [o.get("text", "").strip() for o in objs]
    texts = [t for t in texts if t]
    return _chunk_strings(texts)


# --- embedding -------------------------------------------------------------


def embed_chunks(model: str, chunks: list[str], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(chunks):
            log.info("using cached embeddings %s (%d x %d)", cache_path.name, *arr.shape)
            return arr
        log.warning("cache size mismatch (%d != %d); re-embedding", arr.shape[0], len(chunks))
    log.info("embedding %d chunks via %s", len(chunks), model)
    out: list[list[float]] = []
    t0 = time.perf_counter()
    for i, t in enumerate(chunks):
        r = ollama.embeddings(model=model, prompt=t)
        out.append(r["embedding"])
        if (i + 1) % 500 == 0 or i + 1 == len(chunks):
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            log.info("  %d/%d (%.1f chunk/s)", i + 1, len(chunks), rate)
    arr = np.asarray(out, dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


# --- steganography + detection (mirroring multi_model_study) ---------------


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
    return float(np.mean(num / (den + 1e-12)))


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


def study_one_corpus(
    corpus_name: str,
    chunks: list[str],
    model: str,
    embed_cache: Path,
    seed: int,
) -> tuple[list[Row], dict]:
    t0 = time.perf_counter()
    embeddings = embed_chunks(model, chunks, embed_cache)
    embed_time = time.perf_counter() - t0
    n, d = embeddings.shape
    log.info("%s: %d x %d embeddings in %.1fs", corpus_name, n, d, embed_time)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    half = n // 2
    train_idx, test_idx = idx[:half], idx[half:]
    clean_train = embeddings[train_idx]
    clean_test = embeddings[test_idx]

    obfuscator = EmbeddingObfuscator(seed=seed, noise_level=0.01)
    rows: list[Row] = []
    for tech in TECHNIQUES:
        obf_full = run_technique(tech, embeddings, obfuscator)
        cos = cosine_pair(embeddings, obf_full)
        pcorr = pairwise_corr(embeddings, obf_full)
        if_e, oc_e = detection_aucs(clean_train, clean_test, obf_full[test_idx])
        rows.append(Row(
            corpus=corpus_name,
            n_chunks=n,
            model=model,
            dim=d,
            technique=tech,
            cosine_sim_mean=cos,
            pairwise_corr=pcorr,
            iforest_auc=if_e["auc"],
            iforest_tpr_at_5fpr=if_e["tpr_at_5fpr"],
            ocsvm_auc=oc_e["auc"],
            ocsvm_tpr_at_5fpr=oc_e["tpr_at_5fpr"],
        ))
        log.info(
            "  %s/%-8s cos=%.4f pair=%.4f if_auc=%.3f oc_auc=%.3f",
            corpus_name, tech, cos, pcorr, if_e["auc"], oc_e["auc"],
        )

    meta = {
        "corpus": corpus_name,
        "n_chunks": n,
        "dim": d,
        "embed_seconds": embed_time,
        "model": model,
    }
    return rows, meta


# --- report ----------------------------------------------------------------


def write_report(out_dir: Path, all_rows: list[Row], metas: list[dict], model: str) -> None:
    by_corpus: dict[str, list[Row]] = {}
    for r in all_rows:
        by_corpus.setdefault(r.corpus, []).append(r)

    lines: list[str] = []
    lines.append("# Cross-corpus empirical study\n\n")
    lines.append(
        f"Same model (`{model}`), same techniques, three independent "
        "corpora. Reported numbers are mean cosine to original, "
        "pairwise-cosine correlation, and detection AUCs (Isolation "
        "Forest, One-Class SVM) trained on a clean half-corpus and "
        "evaluated on the held-out half. The cross-corpus point: if a "
        "row's AUC stays at chance (~0.5) across all three corpora, the "
        "rotation/noise/scaling/offset finding is not an artifact of "
        "the small synthetic sample corpus.\n\n"
    )

    lines.append("## Per-corpus statistics\n\n")
    lines.append("| Corpus | n chunks | Dim | Embed time (s) |\n|---|---|---|---|\n")
    for m in metas:
        lines.append(
            f"| `{m['corpus']}` | {m['n_chunks']} | {m['dim']} | "
            f"{m['embed_seconds']:.1f} |\n"
        )

    lines.append("\n## Detection AUC by corpus and technique\n\n")
    lines.append(
        "AUC = 0.5 means the detector is doing no better than random; "
        "AUC = 1.0 means perfect.\n\n"
    )
    lines.append(
        "| Corpus | Technique | cos | pair_corr | IF AUC | IF TPR@5%FPR "
        "| OC-SVM AUC | OC-SVM TPR@5%FPR |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for corpus_name in sorted(by_corpus):
        for r in by_corpus[corpus_name]:
            lines.append(
                f"| `{r.corpus}` | {r.technique} | {r.cosine_sim_mean:.4f} "
                f"| {r.pairwise_corr:.4f} | {r.iforest_auc:.3f} "
                f"| {r.iforest_tpr_at_5fpr:.3f} | {r.ocsvm_auc:.3f} "
                f"| {r.ocsvm_tpr_at_5fpr:.3f} |\n"
            )

    lines.append("\n## Cross-corpus summary\n\n")
    for tech in TECHNIQUES:
        if_aucs = [r.iforest_auc for r in all_rows if r.technique == tech]
        oc_aucs = [r.ocsvm_auc for r in all_rows if r.technique == tech]
        if not if_aucs:
            continue
        lines.append(
            f"- **{tech}**: IF AUC range "
            f"{min(if_aucs):.3f}--{max(if_aucs):.3f}, "
            f"OC-SVM AUC range {min(oc_aucs):.3f}--{max(oc_aucs):.3f} "
            f"across {len(if_aucs)} corpora.\n"
        )

    (out_dir / "report.md").write_text("".join(lines))


# --- driver ----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--corpora", default="sample,nfcorpus,quora",
        help="Comma-separated subset of {sample, nfcorpus, quora}.",
    )
    parser.add_argument(
        "--model", default="nomic-embed-text:latest",
        help="Local Ollama embedding model name.",
    )
    parser.add_argument(
        "--quora-n", type=int, default=10000, dest="quora_n",
        help="Number of Quora question-pair documents to load.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "corpora",
        dest="cache_dir",
        help="Directory for cached raw corpora and embeddings.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "multi_corpus" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)

    requested = [c.strip() for c in args.corpora.split(",") if c.strip()]
    valid = {"sample", "nfcorpus", "quora"}
    bad = set(requested) - valid
    if bad:
        log.error("unknown corpora: %s (valid: %s)", bad, valid)
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace(":", "_").replace("/", "_")

    all_rows: list[Row] = []
    metas: list[dict] = []
    for corpus_name in requested:
        try:
            if corpus_name == "sample":
                chunks = load_sample_chunks(PROJECT_ROOT)
            elif corpus_name == "nfcorpus":
                chunks = load_nfcorpus_chunks(args.cache_dir)
            elif corpus_name == "quora":
                chunks = load_quora_chunks(args.cache_dir, args.quora_n)
            else:
                continue
            log.info("%s: %d chunks ready", corpus_name, len(chunks))

            embed_cache = args.cache_dir / f"emb_{corpus_name}_{model_safe}.npy"
            rows, meta = study_one_corpus(corpus_name, chunks, args.model, embed_cache, args.seed)
        except Exception as e:
            log.error("corpus %s failed: %s", corpus_name, e, exc_info=True)
            continue
        all_rows.extend(rows)
        metas.append(meta)

    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "corpora": metas,
        "techniques": TECHNIQUES,
        "quora_n_requested": args.quora_n,
        "seed": args.seed,
        "rows": [asdict(r) for r in all_rows],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = out_dir / "per_corpus.csv"
    with csv_path.open("w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(asdict(r))

    write_report(out_dir, all_rows, metas, args.model)
    log.info("wrote %s", out_dir / "summary.json")
    log.info("wrote %s", csv_path)
    log.info("wrote %s", out_dir / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
