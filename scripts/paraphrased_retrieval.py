#!/usr/bin/env python3
# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Paraphrased-query retrieval benchmark.

Addresses the reviewer feedback that the existing self-query retrieval
test was trivial (each test embedding queried itself, yielding
recall=1.0 by construction). The right benchmark is:

  - corpus is obfuscated (the steganographer has perturbed it)
  - queries are paraphrases of the original source content
    (a legitimate user is searching for it as they would in normal RAG)
  - measure recall@k, MRR, and the cosine margin between the correct
    result and the nearest incorrect one

If obfuscation degrades these metrics relative to the clean corpus,
ordinary RAG quality monitoring catches the attack. If they don't
degrade, the attack is invisible to that monitoring layer too --- which
is the empirically interesting case.

Workflow:

  1. Load source chunks from sample_docs/ (real OpenAI embeddings).
  2. Generate K=3 paraphrased queries per chunk via gpt-4o-mini, cached
     under results/paraphrases/<corpus_hash>.json so subsequent runs do
     not re-pay for paraphrase generation.
  3. Embed both source chunks (the corpus) and the paraphrased queries.
  4. For each technique (clean baseline + obfuscation variants),
     index the (possibly obfuscated) corpus into a FAISS-flat backend
     and run the paraphrase queries.
  5. Report recall@1/3/5, MRR, and cosine margin for clean vs each
     obfuscation technique.

Output: results/paraphrased_retrieval/<timestamp>/{summary.json, report.md}
"""

from __future__ import annotations

import argparse
import hashlib
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

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from loaders.document_factory import DocumentLoaderFactory  # noqa: E402
from steganography.obfuscation import EmbeddingObfuscator  # noqa: E402
from vector_backends import FaissFlatBackend  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("paraphrased_retrieval")


PARAPHRASE_PROMPT = (
    "You are a paraphrase generator for a research benchmark. Given a "
    "source passage, produce {n} distinct paraphrased queries that a "
    "legitimate user might issue to retrieve this passage from a search "
    "system. Vary phrasing and word choice but preserve meaning. Do not "
    "copy phrases verbatim from the source. Return one paraphrase per line "
    "with no numbering, no bullets, and no extra commentary.\n\n"
    "Source:\n{source}\n\n"
    "{n} paraphrased queries:"
)


# --- corpus + paraphrases ---------------------------------------------------


def load_corpus(sample_dir: Path) -> list[str]:
    factory = DocumentLoaderFactory()
    files = sorted(p for p in sample_dir.iterdir() if p.is_file() and p.name != "README.md")
    docs = factory.load_documents([str(f) for f in files])
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks if c.page_content.strip()]


def corpus_hash(chunks: list[str], n_paraphrases: int, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(str(n_paraphrases).encode())
    for c in chunks:
        h.update(c.encode())
    return h.hexdigest()[:16]


def generate_paraphrases(
    chunks: list[str],
    n_paraphrases: int,
    cache_path: Path,
    paraphrase_model: str = "gpt-4o-mini",
) -> list[list[str]]:
    """Generate (or load cached) paraphrases for each chunk.

    Returns a list of length len(chunks); element i is a list of
    n_paraphrases paraphrased queries for chunks[i].
    """
    if cache_path.exists():
        log.info("loading cached paraphrases from %s", cache_path)
        cached = json.loads(cache_path.read_text())
        if cached["n_chunks"] == len(chunks) and cached["n_paraphrases"] == n_paraphrases:
            return cached["paraphrases"]
        log.warning("cache shape mismatch; regenerating")

    log.info("generating %d paraphrases for each of %d chunks via %s", n_paraphrases, len(chunks), paraphrase_model)
    llm = ChatOpenAI(model=paraphrase_model, temperature=0.0)
    paraphrases: list[list[str]] = []
    for i, chunk in enumerate(chunks):
        prompt = PARAPHRASE_PROMPT.format(n=n_paraphrases, source=chunk)
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        lines = [line.strip(" -*0123456789.").strip() for line in text.splitlines() if line.strip()]
        # Filter empty / overly short lines, then truncate to n_paraphrases
        usable = [line for line in lines if len(line) > 10][:n_paraphrases]
        if len(usable) < n_paraphrases:
            log.warning("chunk %d produced only %d/%d paraphrases", i, len(usable), n_paraphrases)
            # Pad with the source itself rather than failing the run
            while len(usable) < n_paraphrases:
                usable.append(chunk)
        paraphrases.append(usable)
        if (i + 1) % 10 == 0:
            log.info("  paraphrased %d/%d", i + 1, len(chunks))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "n_chunks": len(chunks),
        "n_paraphrases": n_paraphrases,
        "paraphrase_model": paraphrase_model,
        "paraphrases": paraphrases,
    }, indent=2))
    log.info("wrote paraphrase cache: %s", cache_path)
    return paraphrases


# --- retrieval metrics ------------------------------------------------------


@dataclass
class RetrievalRow:
    technique: str
    n_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    mean_correct_score: float
    mean_top1_score: float
    mean_margin: float


def measure_retrieval(
    corpus_vectors: np.ndarray,
    query_vectors: np.ndarray,
    truth: list[int],
) -> RetrievalRow | None:
    """Index corpus_vectors in FAISS-flat, run query_vectors, score
    against truth labels (which corpus index each query should retrieve)."""
    n_queries = len(truth)
    backend = FaissFlatBackend()
    backend.open(dim=corpus_vectors.shape[1])
    try:
        ids = [str(i) for i in range(corpus_vectors.shape[0])]
        backend.insert_arrays(ids, corpus_vectors.astype(np.float32))

        recall1 = recall3 = recall5 = 0
        mrr_sum = 0.0
        correct_scores: list[float] = []
        top1_scores: list[float] = []
        margins: list[float] = []
        for q, expected in zip(query_vectors, truth, strict=True):
            hits = backend.search(q.astype(np.float32), k=5)
            ranks = [int(h.id) for h in hits]
            scores = [h.score for h in hits]
            if expected in ranks:
                pos = ranks.index(expected)
                if pos == 0:
                    recall1 += 1
                if pos < 3:
                    recall3 += 1
                if pos < 5:
                    recall5 += 1
                mrr_sum += 1.0 / (pos + 1)
                correct_scores.append(scores[pos])
            else:
                correct_scores.append(0.0)
            if scores:
                top1_scores.append(scores[0])
                margin = scores[0] - (scores[1] if len(scores) > 1 else 0.0)
                margins.append(margin)

        return RetrievalRow(
            technique="",  # filled in by caller
            n_queries=n_queries,
            recall_at_1=recall1 / n_queries,
            recall_at_3=recall3 / n_queries,
            recall_at_5=recall5 / n_queries,
            mrr=mrr_sum / n_queries,
            mean_correct_score=float(np.mean(correct_scores)),
            mean_top1_score=float(np.mean(top1_scores)),
            mean_margin=float(np.mean(margins)),
        )
    finally:
        backend.close()


# --- driver -----------------------------------------------------------------


def apply_technique(
    technique: str,
    vectors: np.ndarray,
    obf: EmbeddingObfuscator,
) -> np.ndarray:
    if technique == "clean":
        return vectors.copy()
    if technique == "noise":
        return obf.inject_noise(vectors)
    if technique == "rotation":
        rotated, _ = obf.apply_rotation(vectors)
        return rotated
    if technique == "scaling":
        return obf.apply_scaling(vectors)
    if technique == "offset":
        return obf.apply_offset(vectors)
    raise ValueError(f"unknown technique: {technique}")


def run(args: argparse.Namespace) -> int:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set")
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "paraphrased_retrieval" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)

    sample_dir = PROJECT_ROOT / "sample_docs"
    chunks = load_corpus(sample_dir)
    log.info("loaded %d source chunks", len(chunks))

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    log.info("embedding %d chunks with %s", len(chunks), embedding_model)
    embedder = OpenAIEmbeddings(model=embedding_model)
    corpus_vectors = np.asarray(embedder.embed_documents(chunks), dtype=np.float64)
    log.info("corpus shape: %s", corpus_vectors.shape)

    cache_key = corpus_hash(chunks, args.n_paraphrases, args.paraphrase_model)
    cache_path = PROJECT_ROOT / "results" / "paraphrases" / f"{cache_key}.json"
    paraphrases = generate_paraphrases(chunks, args.n_paraphrases, cache_path, args.paraphrase_model)

    flat_paraphrases: list[str] = []
    truth: list[int] = []
    for chunk_idx, ps in enumerate(paraphrases):
        for p in ps:
            flat_paraphrases.append(p)
            truth.append(chunk_idx)
    log.info("embedding %d paraphrased queries", len(flat_paraphrases))
    query_vectors = np.asarray(embedder.embed_documents(flat_paraphrases), dtype=np.float64)

    obf = EmbeddingObfuscator(noise_level=args.noise, seed=args.seed)
    techniques = ["clean", "noise", "rotation", "scaling", "offset"]

    rows: list[RetrievalRow] = []
    for technique in techniques:
        log.info("--- technique: %s ---", technique)
        obfuscated_corpus = apply_technique(technique, corpus_vectors, obf)
        result = measure_retrieval(obfuscated_corpus, query_vectors, truth)
        if result is None:
            continue
        result = RetrievalRow(
            technique=technique,
            n_queries=result.n_queries,
            recall_at_1=result.recall_at_1,
            recall_at_3=result.recall_at_3,
            recall_at_5=result.recall_at_5,
            mrr=result.mrr,
            mean_correct_score=result.mean_correct_score,
            mean_top1_score=result.mean_top1_score,
            mean_margin=result.mean_margin,
        )
        rows.append(result)
        log.info(
            "%-10s recall@1=%.3f @3=%.3f @5=%.3f MRR=%.3f margin=%.4f",
            technique, result.recall_at_1, result.recall_at_3, result.recall_at_5,
            result.mrr, result.mean_margin,
        )

    summary = {
        "timestamp": timestamp,
        "embedding_model": embedding_model,
        "paraphrase_model": args.paraphrase_model,
        "n_chunks": len(chunks),
        "n_paraphrases_per_chunk": args.n_paraphrases,
        "n_queries_total": len(flat_paraphrases),
        "noise_level": args.noise,
        "rows": [asdict(r) for r in rows],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out_dir, summary, rows)
    log.info("done.")
    return 0


def write_report(out_dir: Path, summary: dict, rows: list[RetrievalRow]) -> None:
    md = [f"# Paraphrased-Query Retrieval --- `{summary['timestamp']}`\n"]
    md.append("## Setup\n")
    md.append(f"- Embedding model: `{summary['embedding_model']}`")
    md.append(f"- Paraphrase generator: `{summary['paraphrase_model']}`")
    md.append(f"- Corpus chunks: {summary['n_chunks']}")
    md.append(f"- Paraphrases per chunk: {summary['n_paraphrases_per_chunk']}")
    md.append(f"- Total queries: {summary['n_queries_total']}")
    md.append(f"- Noise level (where applicable): {summary['noise_level']}\n")

    md.append("## Retrieval quality: clean corpus vs obfuscated corpus\n")
    md.append("Each query is a paraphrase of one corpus chunk. The query is")
    md.append("embedded **without** obfuscation, then matched against an index")
    md.append("of the (possibly obfuscated) corpus. recall@k is the fraction")
    md.append("of queries whose source chunk appears in the top-k results.")
    md.append("MRR is mean reciprocal rank. ``margin`` is the mean cosine")
    md.append("score difference between the top hit and the second hit ---")
    md.append("smaller margin means the attack noise has eroded the signal.\n")

    md.append("| Technique | recall@1 | recall@3 | recall@5 | MRR | mean correct | mean top-1 | mean margin |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r.technique} | {r.recall_at_1:.3f} | {r.recall_at_3:.3f} | {r.recall_at_5:.3f} | "
            f"{r.mrr:.3f} | {r.mean_correct_score:.4f} | {r.mean_top1_score:.4f} | {r.mean_margin:.4f} |"
        )
    md.append("")

    md.append("## How to read this table\n")
    md.append("The clean baseline establishes the upper bound on retrieval quality")
    md.append("for this corpus + embedding model + paraphrase generator. Every")
    md.append("obfuscation row should be compared against it. A row whose")
    md.append("recall@k is significantly below clean indicates that ordinary")
    md.append("RAG quality monitoring would catch the attack as a regression in")
    md.append("retrieval quality. A row whose recall@k matches clean within")
    md.append("noise is undetectable through that monitoring layer --- the")
    md.append("attack lives inside the embedding without breaking the user-facing")
    md.append("retrieval contract.\n")
    md.append("Caveats: paraphrases are LLM-generated and so depend on the")
    md.append("paraphrase model's quality; the clean recall is therefore not")
    md.append("expected to be 1.0. Differences relative to clean are the")
    md.append("interpretable signal, not the absolute numbers.")
    (out_dir / "report.md").write_text("\n".join(md))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n-paraphrases", type=int, default=3, dest="n_paraphrases")
    parser.add_argument("--paraphrase-model", default="gpt-4o-mini", dest="paraphrase_model")
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
