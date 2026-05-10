#!/usr/bin/env python3
# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0

"""Generate publication-style report and plots from empirical_study.py output.

Reads the latest results/empirical/<timestamp>/summary.json and emits:
  - report.md         (Markdown table + interpretation, paper-ready)
  - plots/fidelity.png        cosine similarity vs technique
  - plots/detection.png       KS statistic & pairwise correlation
  - plots/performance.png     wall-clock per technique

Usage:
    python scripts/empirical_report.py [run_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def latest_run() -> Path:
    runs = sorted((PROJECT_ROOT / "results" / "empirical").glob("[0-9]*"))
    if not runs:
        sys.exit("no runs in results/empirical/")
    return runs[-1]


def write_report(run: Path, summary: dict) -> None:
    rows = summary["measurements"]

    md = []
    md.append(f"# VectorSmuggle Empirical Results — `{summary['timestamp']}`\n")
    md.append("## Setup\n")
    md.append(f"- Embedding model: `{summary['model']}`")
    md.append(f"- Embedding dimension: {summary['embed_dim']}")
    md.append(f"- Document chunks: {summary['n_chunks']}")
    md.append(f"- Baseline embedding wall time: {summary['embed_time_seconds']:.2f}s")
    md.append(f"- Configurations: {', '.join(summary['configurations'])}")
    md.append(f"- Techniques: {', '.join(summary['techniques'])}")
    md.append("")

    md.append("## Headline Table\n")
    md.append("| Config | Technique | cos(orig, obf) | pair-corr | MSE | recovery cos | time (ms) |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for m in rows:
        cfg, tech = m["technique"].split("::", 1)
        rec = f"{m['recovery_cosine']:.4f}" if m["recovery_cosine"] is not None else "—"
        md.append(
            f"| {cfg} | {tech} | {m['cosine_sim_mean']:.4f} | "
            f"{m['pairwise_corr']:.4f} | {m['mse_vs_original']:.6f} | "
            f"{rec} | {m['elapsed_ms']:.1f} |"
        )
    md.append("")

    md.append("## Detection Signatures\n")
    md.append("| Config | Technique | KS stat | KS p-value | norm-ratio mean ± std | entropy Δ |")
    md.append("|---|---|---:|---:|---:|---:|")
    for m in rows:
        cfg, tech = m["technique"].split("::", 1)
        ent_delta = m["entropy_obfuscated"] - m["entropy_original"]
        md.append(
            f"| {cfg} | {tech} | {m['ks_statistic']:.4f} | {m['ks_pvalue']:.2e} | "
            f"{m['norm_ratio_mean']:.4f} ± {m['norm_ratio_std']:.4f} | {ent_delta:+.4f} |"
        )
    md.append("")

    md.append("## Notes for Interpretation\n")
    md.append("- **cos(orig, obf)** is the per-vector cosine similarity between the original")
    md.append("  embedding and the obfuscated one. Values close to 1.0 mean the technique")
    md.append("  did not meaningfully shift direction; values below ~0.7 indicate strong")
    md.append("  semantic perturbation that will likely degrade retrieval.")
    md.append("- **pair-corr** is the correlation between pairwise cosine-similarity")
    md.append("  matrices before and after obfuscation. This is the operationally")
    md.append("  important fidelity measure: high pair-corr means similar items remain")
    md.append("  similar after obfuscation, even if absolute directions shift.")
    md.append("- **KS stat** is the two-sample Kolmogorov-Smirnov statistic comparing")
    md.append("  the flattened distribution of original vs. obfuscated values. Higher KS")
    md.append("  with p<0.05 indicates a detectable distributional fingerprint.")
    md.append("- **recovery cos** is the cosine similarity between original and the")
    md.append("  attempt to invert the obfuscation. Rotation and scaling are exactly")
    md.append("  invertible given the metadata; noise and offset are not.")
    md.append("")

    (run / "report.md").write_text("\n".join(md))


def plot_fidelity(run: Path, summary: dict) -> None:
    rows = summary["measurements"]
    techs = sorted({m["technique"].split("::", 1)[1] for m in rows})
    cfgs = summary["configurations"]

    cos = {(cfg, tech): None for cfg in cfgs for tech in techs}
    pair = {(cfg, tech): None for cfg in cfgs for tech in techs}
    for m in rows:
        cfg, tech = m["technique"].split("::", 1)
        cos[(cfg, tech)] = m["cosine_sim_mean"]
        pair[(cfg, tech)] = m["pairwise_corr"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(techs))
    width = 0.18

    for i, cfg in enumerate(cfgs):
        offset = (i - (len(cfgs) - 1) / 2) * width
        ax1.bar(x + offset, [cos[(cfg, t)] for t in techs], width, label=cfg)
        ax2.bar(x + offset, [pair[(cfg, t)] for t in techs], width, label=cfg)

    for ax, title, ylabel in [
        (ax1, "Per-vector cosine similarity (orig vs. obfuscated)", "cos(orig, obf)"),
        (ax2, "Pairwise similarity matrix correlation", "corr"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(techs, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.7, ls="--", color="gray", alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(run / "plots" / "fidelity.png", dpi=150)
    plt.close(fig)


def plot_detection(run: Path, summary: dict) -> None:
    rows = summary["measurements"]
    techs = sorted({m["technique"].split("::", 1)[1] for m in rows})
    cfgs = summary["configurations"]

    ks = {(cfg, tech): None for cfg in cfgs for tech in techs}
    for m in rows:
        cfg, tech = m["technique"].split("::", 1)
        ks[(cfg, tech)] = m["ks_statistic"]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(techs))
    width = 0.18

    for i, cfg in enumerate(cfgs):
        offset = (i - (len(cfgs) - 1) / 2) * width
        ax.bar(x + offset, [ks[(cfg, t)] for t in techs], width, label=cfg)

    ax.set_xticks(x)
    ax.set_xticklabels(techs, rotation=30, ha="right")
    ax.set_title("KS statistic — distributional shift vs. baseline")
    ax.set_ylabel("KS statistic (higher → more detectable)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "plots" / "detection.png", dpi=150)
    plt.close(fig)


def plot_performance(run: Path, summary: dict) -> None:
    rows = summary["measurements"]
    techs = sorted({m["technique"].split("::", 1)[1] for m in rows})

    # Average wall-clock across configs
    avg_ms = {t: np.mean([m["elapsed_ms"] for m in rows
                          if m["technique"].endswith("::" + t)]) for t in techs}

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(techs, [avg_ms[t] for t in techs])
    ax.set_xticklabels(techs, rotation=30, ha="right")
    ax.set_title("Wall-clock time per technique (mean across configs)")
    ax.set_ylabel("Time (ms)")
    ax.grid(axis="y", alpha=0.3)
    for b, t in zip(bars, techs, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{avg_ms[t]:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "plots" / "performance.png", dpi=150)
    plt.close(fig)


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    print(f"Reading {run}")
    summary = json.loads((run / "summary.json").read_text())
    (run / "plots").mkdir(exist_ok=True)
    write_report(run, summary)
    plot_fidelity(run, summary)
    plot_detection(run, summary)
    plot_performance(run, summary)
    print(f"Wrote {run / 'report.md'}")
    print(f"Plots in {run / 'plots'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
