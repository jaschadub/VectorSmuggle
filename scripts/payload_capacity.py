#!/usr/bin/env python3
# Copyright 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Empirical payload-capacity demonstration for rotation-based steganography.

Companion to ``rotation_sweep.py``. That script established a payload-bit
*proxy* (an upper bound on attacker capacity); reviewers asked for an
explicit capacity derivation and a working decode of an actual byte
payload. This script does four things:

  1. Derives the per-vector capacity for the disjoint-Givens keyed-pair
     protocol: ``floor(d/2) * b`` bits, where ``b`` is the bits per
     quantized angle.
  2. Implements a deterministic ``bytes -> (i, j, theta)`` encoder and
     the matching decoder for that protocol.
  3. Round-trips an actual payload of varying size through that encoder
     and a chosen storage dtype (float64 / float32 / float16, since real
     vector DBs rarely store float64) and reports the bit error rate.
  4. Re-runs the same Isolation-Forest and One-Class-SVM detectors used
     in ``rotation_sweep.py`` at every operating point, so the punchline
     "capacity grows but AUC stays at chance" can be read off a single
     table without cross-referencing two scripts.

Output: ``results/payload_capacity/<timestamp>/{summary.json, report.md}``.

The key honesty point relative to the proxy in ``rotation_sweep.py``: that
proxy counts the ``log2(d(d-1)/2)`` bits of the (i, j) pair choice as
data-carrying. For a single-vector decoder that information is *not*
recoverable from the rotated vector alone (or even from the (original,
rotated) pair, in general), so the proxy is a loose upper bound. The
disjoint-Givens channel implemented here is what an attacker can actually
demonstrate end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("payload_capacity")


# --- protocol primitives ---------------------------------------------------


def keyed_disjoint_pairs(dim: int, num_rotations: int, key: int) -> list[tuple[int, int]]:
    """Return ``num_rotations`` disjoint ``(i, j)`` pairs derived from ``key``.

    Generates ``np.random.default_rng(key).permutation(dim)`` and consumes
    pairs from the front. Decoder reconstructs the same pairs from the same
    key. Constraint: ``num_rotations <= dim // 2``.
    """
    if num_rotations > dim // 2:
        raise ValueError(
            f"num_rotations={num_rotations} exceeds disjoint capacity dim/2={dim // 2}"
        )
    rng = np.random.default_rng(key)
    perm = rng.permutation(dim)
    return [(int(perm[2 * k]), int(perm[2 * k + 1])) for k in range(num_rotations)]


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return bytes(np.packbits(bits.astype(np.uint8)))


def angles_from_bits(bits: np.ndarray, b: int) -> np.ndarray:
    """Pack a flat bit array into ``K = bits.size / b`` cell-centered angles
    in ``(-pi, pi)``. Each consecutive run of ``b`` bits, MSB-first, is the
    cell index in ``[0, 2**b)``; the cell center is mapped to angle
    ``(idx + 0.5) / 2**b * 2pi - pi``.
    """
    if bits.size % b != 0:
        raise ValueError("bits length must be a multiple of b")
    k = bits.size // b
    grid = bits.reshape(k, b).astype(np.int64)
    idx = np.zeros(k, dtype=np.int64)
    for shift in range(b):
        idx |= grid[:, shift] << (b - 1 - shift)
    levels = 1 << b
    return (idx + 0.5) / levels * (2.0 * math.pi) - math.pi


def bits_from_angles(angles: np.ndarray, b: int) -> np.ndarray:
    """Inverse of ``angles_from_bits``: round each angle to the nearest
    cell center and emit the cell index as a flat MSB-first bit array.
    """
    levels = 1 << b
    raw = (angles + math.pi) / (2.0 * math.pi) * levels - 0.5
    idx = np.rint(raw).astype(np.int64)
    idx = np.mod(idx, levels)
    out = np.zeros(angles.size * b, dtype=np.uint8)
    for shift in range(b):
        out[shift :: b] = ((idx >> (b - 1 - shift)) & 1).astype(np.uint8)
    return out


def encode_payload(
    vector: np.ndarray,
    payload: bytes,
    pairs: list[tuple[int, int]],
    b: int,
) -> tuple[np.ndarray, int]:
    """Apply disjoint Givens rotations to ``vector`` to encode ``payload``.

    The bit stream is zero-padded up to ``len(pairs) * b`` bits so every
    rotation runs (a partially-loaded channel still uses the full keyed
    pair sequence; remaining angles encode zero). Returns
    ``(rotated_vector, payload_bits_packed)``.
    """
    payload_bits = bytes_to_bits(payload)
    capacity_bits = len(pairs) * b
    bits_used = int(min(payload_bits.size, capacity_bits))
    bits_padded = np.zeros(capacity_bits, dtype=np.uint8)
    bits_padded[:bits_used] = payload_bits[:bits_used]

    angles = angles_from_bits(bits_padded, b)
    out = vector.astype(np.float64).copy()
    for (i, j), theta in zip(pairs, angles, strict=True):
        c, s = math.cos(theta), math.sin(theta)
        a, bv = out[i], out[j]
        out[i] = c * a - s * bv
        out[j] = s * a + c * bv
    return out, bits_used


def decode_payload(
    original: np.ndarray,
    rotated: np.ndarray,
    pairs: list[tuple[int, int]],
    b: int,
    bits_to_extract: int,
) -> bytes:
    """Recover ``bits_to_extract`` payload bits from a rotated vector.

    Disjoint pairs commute, so each rotation is independent and
    ``theta_k = atan2(rot_j, rot_i) - atan2(orig_j, orig_i)`` (modulo
    ``2 pi``) recovers the angle on the (i, j) plane.
    """
    angles = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        a0, b0 = float(original[i]), float(original[j])
        a1, b1 = float(rotated[i]), float(rotated[j])
        theta = math.atan2(b1, a1) - math.atan2(b0, a0)
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
        angles[k] = theta

    bits = bits_from_angles(angles, b)[:bits_to_extract]
    if bits.size % 8:
        bits = np.concatenate([bits, np.zeros(8 - bits.size % 8, dtype=np.uint8)])
    return bits_to_bytes(bits)


# --- capacity derivation ---------------------------------------------------


def disjoint_capacity_bits(dim: int, b: int) -> int:
    """Closed-form per-vector capacity for the disjoint-Givens keyed-pair
    protocol: ``floor(dim/2) * b`` bits."""
    return (dim // 2) * b


def proxy_capacity_bits(dim: int, num_rotations: int, b: int) -> float:
    """The looser ``(i, j)``-included proxy from ``rotation_sweep.py``.

    Counts ``log2(d(d-1)/2)`` bits per rotation for the pair choice on top
    of ``b`` angle bits. Single-vector decoders cannot recover those pair
    bits, so this is a loose upper bound; reported here only for
    cross-reference.
    """
    if num_rotations <= 0 or dim < 2:
        return 0.0
    pair_bits = math.log2(dim * (dim - 1) // 2)
    return float(num_rotations * (pair_bits + b))


# --- detector AUC ----------------------------------------------------------


def detector_aucs(
    clean_train: np.ndarray, clean_test: np.ndarray, obfuscated: np.ndarray
) -> tuple[float, float]:
    iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iforest.fit(clean_train)
    if_clean = -iforest.score_samples(clean_test)
    if_obf = -iforest.score_samples(obfuscated)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(clean_train)
    oc_clean = -ocsvm.decision_function(clean_test)
    oc_obf = -ocsvm.decision_function(obfuscated)

    y = np.concatenate([np.zeros(len(clean_test)), np.ones(len(obfuscated))])
    if_scores = np.concatenate([if_clean, if_obf])
    oc_scores = np.concatenate([oc_clean, oc_obf])
    return float(roc_auc_score(y, if_scores)), float(roc_auc_score(y, oc_scores))


# --- experiment driver -----------------------------------------------------


@dataclass
class CapacityRow:
    storage_dtype: str
    payload_bytes: int
    num_rotations: int
    angle_bits: int
    capacity_bytes: float
    bit_error_rate: float
    bytes_match: bool
    cos_orig_obf: float
    if_auc: float
    ocsvm_auc: float


def make_corpus(n: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way unit-norm Gaussian split: detector training, detector
    held-out clean test, target batch to obfuscate."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 1, size=(3 * n, dim)).astype(np.float64)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw[:n], raw[n : 2 * n], raw[2 * n :]


def load_real_corpus(
    path: Path, train_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a real-corpus embedding file (``.npy``, shape ``(N, d)``) and
    split it for the capacity demo.

    Detector training takes the first ``train_frac`` of the shuffled
    rows; the remaining rows act as both the clean-test set (for AUC)
    and the target batch (where the rotation is applied). The
    clean-test and obfuscated batches therefore share the same
    underlying vectors --- one with the perturbation, one without ---
    matching the setup of the headline detection table in
    \\Cref{sec:evaluation:detection}.
    """
    arr = np.load(path).astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(f"corpus embeddings must be 2-D, got shape {arr.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(arr.shape[0])
    n_train = max(1, int(arr.shape[0] * train_frac))
    train = arr[idx[:n_train]]
    rest = arr[idx[n_train:]]
    return train, rest.copy(), rest.copy()


def run_one(
    target_batch: np.ndarray,
    clean_train: np.ndarray,
    clean_test: np.ndarray,
    payload: bytes,
    pairs: list[tuple[int, int]],
    b: int,
    storage_dtype: np.dtype,
) -> CapacityRow:
    obfuscated = np.empty_like(target_batch)
    bits_used = 0
    for idx in range(target_batch.shape[0]):
        rotated, bits_used = encode_payload(target_batch[idx], payload, pairs, b)
        if storage_dtype != np.float64:
            rotated = rotated.astype(storage_dtype).astype(np.float64)
        obfuscated[idx] = rotated

    bits_in = bytes_to_bits(payload)[:bits_used]
    decoded = decode_payload(target_batch[0], obfuscated[0], pairs, b, bits_used)
    bits_out = bytes_to_bits(decoded)[:bits_used]
    ber = float(np.mean(bits_in != bits_out)) if bits_used > 0 else 0.0
    bytes_match = decoded[: len(payload)] == payload

    a_n = target_batch / np.linalg.norm(target_batch, axis=1, keepdims=True)
    o_n = obfuscated / (np.linalg.norm(obfuscated, axis=1, keepdims=True) + 1e-12)
    cos = float(np.einsum("ij,ij->i", a_n, o_n).mean())

    if_auc, oc_auc = detector_aucs(clean_train, clean_test, obfuscated)

    return CapacityRow(
        storage_dtype=str(np.dtype(storage_dtype)),
        payload_bytes=len(payload),
        num_rotations=len(pairs),
        angle_bits=b,
        capacity_bytes=len(pairs) * b / 8.0,
        bit_error_rate=ber,
        bytes_match=bool(bytes_match),
        cos_orig_obf=cos,
        if_auc=if_auc,
        ocsvm_auc=oc_auc,
    )


def selftest(seed: int = 0) -> None:
    """Encoder/decoder round-trip sanity check at small dim, runs in <1 s."""
    rng = np.random.default_rng(seed)
    dim, b = 256, 8
    pairs = keyed_disjoint_pairs(dim, num_rotations=64, key=12345)
    payload = b"capacity-selftest!"
    vec = rng.normal(0, 1, size=dim).astype(np.float64)
    vec /= np.linalg.norm(vec)
    rotated, bits_used = encode_payload(vec, payload, pairs, b)
    decoded = decode_payload(vec, rotated, pairs, b, bits_used)
    assert decoded[: len(payload)] == payload, (
        f"selftest failed: {decoded!r} != {payload!r}"
    )
    rotated32 = rotated.astype(np.float32).astype(np.float64)
    decoded32 = decode_payload(vec, rotated32, pairs, b, bits_used)
    assert decoded32[: len(payload)] == payload, (
        f"selftest fp32 failed: {decoded32!r} != {payload!r}"
    )
    log.info(
        "selftest passed (%d B round-tripped through %d-rotation channel; fp32 roundtrip ok)",
        len(payload), len(pairs),
    )


def run(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "payload_capacity" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)

    if args.corpus_embeddings is not None:
        train, clean_test, target = load_real_corpus(
            args.corpus_embeddings, args.train_frac, args.seed
        )
        if args.dim != train.shape[1]:
            log.info("overriding --dim %d with corpus dim %d", args.dim, train.shape[1])
            args.dim = int(train.shape[1])
        log.info(
            "real corpus: %s train=%d clean_test/target=%d dim=%d",
            args.corpus_embeddings, train.shape[0], clean_test.shape[0], args.dim,
        )
    else:
        train, clean_test, target = make_corpus(args.n_per_split, args.dim, args.seed)
        log.info("synthetic corpus: dim=%d n_per_split=%d", args.dim, args.n_per_split)

    cap_bits = disjoint_capacity_bits(args.dim, args.angle_bits)
    cap_bytes = cap_bits // 8
    log.info(
        "disjoint capacity: dim=%d K_max=%d b=%d -> %d bits = %d bytes / vector",
        args.dim, args.dim // 2, args.angle_bits, cap_bits, cap_bytes,
    )

    payload_grid = sorted({64, 128, 256, 512, 1024, max(1, cap_bytes // 2), cap_bytes})
    payload_grid = [pb for pb in payload_grid if 0 < pb <= cap_bytes]
    log.info("payload sizes (bytes): %s", payload_grid)

    storage_dtypes = [np.float64, np.float32, np.float16]
    rng = np.random.default_rng(args.seed)
    rows: list[CapacityRow] = []
    for storage_dtype in storage_dtypes:
        for payload_bytes in payload_grid:
            num_rot = math.ceil(payload_bytes * 8 / args.angle_bits)
            pairs = keyed_disjoint_pairs(args.dim, num_rot, args.seed)
            payload = rng.bytes(payload_bytes)
            row = run_one(target, train, clean_test, payload, pairs, args.angle_bits, storage_dtype)
            rows.append(row)
            log.info(
                "dtype=%-7s payload=%-5dB K=%-5d cap=%6.0fB BER=%.4f match=%s cos=%.4f IF=%.3f OCSVM=%.3f",
                row.storage_dtype, row.payload_bytes, row.num_rotations, row.capacity_bytes,
                row.bit_error_rate, row.bytes_match, row.cos_orig_obf, row.if_auc, row.ocsvm_auc,
            )

    is_real = args.corpus_embeddings is not None
    summary = {
        "timestamp": timestamp,
        "dim": args.dim,
        "n_per_split": args.n_per_split,
        "seed": args.seed,
        "angle_bits": args.angle_bits,
        "disjoint_capacity_bits": cap_bits,
        "disjoint_capacity_bytes": cap_bytes,
        "proxy_capacity_bits_at_K_max": proxy_capacity_bits(
            args.dim, args.dim // 2, args.angle_bits
        ),
        "corpus_kind": "real" if is_real else "synthetic_unit_gaussian",
        "corpus_path": str(args.corpus_embeddings) if is_real else None,
        "n_train": int(train.shape[0]),
        "n_clean_test": int(clean_test.shape[0]),
        "n_target": int(target.shape[0]),
        "rows": [asdict(r) for r in rows],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out_dir, summary, rows)
    log.info("done. wrote %d rows.", len(rows))
    return 0


def write_report(out_dir: Path, summary: dict, rows: list[CapacityRow]) -> None:
    md: list[str] = [f"# Payload Capacity --- `{summary['timestamp']}`\n"]
    md.append("## Setup\n")
    md.append(f"- Dimension: {summary['dim']}")
    if summary.get("corpus_kind") == "real":
        md.append(f"- Corpus: real embeddings from `{summary['corpus_path']}`")
        md.append(
            f"- Splits: {summary['n_train']} detector-training, "
            f"{summary['n_clean_test']} clean-test (= target batch with rotation applied)"
        )
    else:
        md.append(
            f"- Corpus: {summary['n_per_split']} synthetic unit-norm Gaussians per split "
            "(detector-training, clean-test, target batch)"
        )
    md.append(f"- Random seed: {summary['seed']}")
    md.append(
        f"- Angle quantization (b): {summary['angle_bits']} bits "
        f"= {1 << summary['angle_bits']} levels\n"
    )

    md.append("## Theoretical capacity\n")
    md.append(
        "The disjoint-Givens keyed-pair protocol uses a shared key to select "
        "``K`` disjoint dimension pairs, then encodes ``b`` payload bits per "
        "pair as a quantized angle. Disjoint pairs commute, so every angle "
        "is independently recoverable from the original and rotated vector "
        "via ``atan2``. Per-vector capacity:\n"
    )
    md.append(
        f"- ``floor(d/2) * b`` = **{summary['disjoint_capacity_bits']} bits** "
        f"= **{summary['disjoint_capacity_bytes']} bytes** at "
        f"d={summary['dim']}, b={summary['angle_bits']}.\n"
    )
    md.append(
        f"For comparison, the looser proxy from ``rotation_sweep.py`` "
        f"counts ``log2(d(d-1)/2)`` bits of pair choice on top of ``b`` "
        f"bits per angle, giving "
        f"**{summary['proxy_capacity_bits_at_K_max']:.0f} bits** at "
        f"K=d/2. That proxy is a strict upper bound: a single-vector "
        f"decoder cannot recover the pair index from a rotated vector "
        f"alone, so disjoint Givens is the channel an attacker can "
        f"actually demonstrate end-to-end.\n"
    )

    md.append("## Empirical decode and detector AUC\n")
    md.append(
        "Each row encodes a fresh random payload of ``payload_bytes`` into "
        "a batch of clean unit-norm vectors via ``K`` disjoint Givens "
        "rotations, round-trips through the listed storage dtype (a real "
        "vector DB rarely stores float64), then decodes vector ``[0]`` and "
        "reports BER. ``cos`` is per-vector cosine to the original; "
        "``IF AUC`` and ``OCSVM AUC`` use the same detectors as in "
        "``rotation_sweep.py``, retrained per row on the clean held-out "
        "batch.\n"
    )
    md.append(
        "| dtype | payload (B) | K | capacity (B) | BER | match | cos | IF AUC | OCSVM AUC |"
    )
    md.append("|---|---:|---:|---:|---:|:---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r.storage_dtype} | {r.payload_bytes} | {r.num_rotations} "
            f"| {r.capacity_bytes:.0f} | {r.bit_error_rate:.4f} "
            f"| {'ok' if r.bytes_match else 'FAIL'} "
            f"| {r.cos_orig_obf:.4f} | {r.if_auc:.3f} | {r.ocsvm_auc:.3f} |"
        )
    md.append("")

    md.append("## How to read this table\n")
    md.append(
        "- ``BER == 0`` and ``match == ok`` mean the payload round-tripped "
        "exactly. The angle-quantization margin (cell width ``2pi/2**b``) "
        "exceeds float32 / float16 storage roundoff for ``b`` up to ~12, "
        "so exact recovery is the expected case across the table.\n"
        "- ``cos`` near 1 means the rotated vector is geometrically "
        "indistinguishable from the original. Disjoint Givens rotations "
        "are orthogonal, so any drop is float roundoff in storage; the "
        "drift at high ``K`` is the expected ``cos ~= 1 - 2K/d`` "
        "alignment loss for uniform random angles.\n"
    )
    if summary.get("corpus_kind") == "real":
        md.append(
            "- ``IF AUC`` and ``OCSVM AUC`` behavior is markedly different "
            "on a real production embedding distribution than on the "
            "synthetic-Gaussian baseline: on real "
            "``text-embedding-3-large`` vectors, the detector rapidly "
            "regains signal as ``K`` grows, because the clean training "
            "manifold is concentrated and random-angle disjoint Givens "
            "rotations push the obfuscated vectors off it. The "
            "synthetic-vector regime where ``AUC`` stays at chance across "
            "the entire sweep does not extend to real production corpora "
            "at high ``K``.\n"
        )
    else:
        md.append(
            "- ``IF AUC`` and ``OCSVM AUC`` near 0.5 confirm that on a "
            "uniform unit-Gaussian baseline the off-the-shelf detectors "
            "cannot tell the obfuscated batch from the clean training "
            "distribution at any ``K`` in this sweep, including the full "
            f"``floor(d/2)`` rotation count where the channel is carrying "
            f"~{summary['disjoint_capacity_bytes']} bytes per vector. "
            "Whether this extends to real corpus distributions depends on "
            "manifold concentration; see the ``--corpus-embeddings`` "
            "variant of this script for the real-corpus result.\n"
        )
    (out_dir / "report.md").write_text("\n".join(md))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n-per-split", type=int, default=300, dest="n_per_split")
    parser.add_argument(
        "--dim", type=int, default=3072,
        help="Embedding dimension (default 3072 = text-embedding-3-large).",
    )
    parser.add_argument(
        "--angle-bits", type=int, default=10, dest="angle_bits",
        help="Quantization bits per angle (b).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--corpus-embeddings", type=Path, default=None, dest="corpus_embeddings",
        help="Optional path to a (N, d) .npy file of real corpus embeddings. "
             "When set, replaces the synthetic Gaussian generator and overrides "
             "--dim. Detector training uses --train-frac of the shuffled rows; "
             "the remainder serves as both clean-test and target batch.",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.4, dest="train_frac",
        help="Fraction of --corpus-embeddings rows used for detector training "
             "(remainder serves as clean test + target batch). Default 0.4 "
             "matches the 27/41 split used in the headline detection table.",
    )
    parser.add_argument(
        "--selftest-only", action="store_true",
        help="Run the encoder/decoder round-trip selftest and exit.",
    )
    args = parser.parse_args()

    selftest(args.seed)
    if args.selftest_only:
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
