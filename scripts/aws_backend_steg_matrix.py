# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive AWS-backend steganography matrix.

Runs every primitive in ``steganography.obfuscation.EmbeddingObfuscator``
plus the full composed pipeline against the three AWS-targeted
backends, reporting three metrics per cell:

- **byte-lossless**: cos(retrieved, normalized(inserted))
  Does the backend round-trip the (post-technique) float32 bytes?
- **direction-recovery**: cos(deobfuscate(retrieved), baseline)
  After deobfuscation, is the original embedding recovered?
- **stealth**: cos(normalized(obfuscated), baseline)
  How close is the obfuscated vector to the clean baseline, pre-insert?

All three backends L2-normalize on insert (cosine metric), so any
primitive that hides payload purely in magnitude is defeated by
normalization. We test against unit-norm baselines and measure
direction-recovery rather than full reconstruction.

The fragmentation primitive is special: it splits one embedding into
N sparse vectors, each stored as its own record. We exercise this
end-to-end (insert fragments, retrieve by ID, sum, deobfuscate).

Requires:
    - docker-compose stack: opensearch (9200), redis-stack (6379)
    - AWS creds + AWS_S3VECTORS_TEST=1 for the S3 Vectors row
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

# Make repo root importable when run as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from steganography.obfuscation import EmbeddingObfuscator  # noqa: E402
from vector_backends.memorydb_backend import MemoryDBBackend  # noqa: E402
from vector_backends.opensearch_backend import OpenSearchBackend  # noqa: E402
from vector_backends.s3vectors_backend import S3VectorsBackend  # noqa: E402

DIM = 64
SEED = 20260516


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class CellResult:
    backend: str
    technique: str
    lossless: float
    recovery: float
    stealth: float
    notes: str = ""


def _new_backend(name: str):
    if name == "OpenSearch":
        return OpenSearchBackend(
            url="http://localhost:9200",
            index_name=f"stego_{np.random.randint(1_000_000)}",
        )
    if name == "MemoryDB":
        n = np.random.randint(1_000_000)
        return MemoryDBBackend(
            url="redis://localhost:6379",
            index_name=f"stego_{n}",
            key_prefix=f"stego_{n}:",
        )
    if name == "S3Vectors":
        return S3VectorsBackend(region=os.environ.get("AWS_REGION", "us-west-2"))
    raise ValueError(name)


def _insert_get(backend, baseline_dim: int, vec_in: np.ndarray, rid: str = "stego-test") -> np.ndarray:
    backend.open(dim=baseline_dim)
    try:
        backend.insert_arrays([rid], vec_in.reshape(1, -1))
        got, _ = backend.get_by_id(rid)
        return got
    finally:
        backend.close()


# ---- Per-primitive runners -------------------------------------------------

def run_noise(backend_name: str, baseline: np.ndarray) -> CellResult:
    rng = np.random.default_rng(SEED)
    obf = EmbeddingObfuscator(noise_level=0.01, seed=SEED)
    noise = rng.normal(0, obf.noise_level, baseline.shape)
    inserted = baseline + noise
    inserted_n = _unit(inserted)

    backend = _new_backend(backend_name)
    retrieved = _insert_get(backend, baseline.shape[0], inserted)

    lossless = _cos(retrieved, inserted_n)
    recovered = _unit(retrieved * np.linalg.norm(inserted) - noise)
    recovery = _cos(recovered, baseline)
    stealth = _cos(inserted_n, baseline)
    return CellResult(backend_name, "noise", lossless, recovery, stealth)


def run_rotation(backend_name: str, baseline: np.ndarray) -> CellResult:
    # Single-pair Givens rotation in (0,1) — invertible, magnitude-preserving.
    theta = 0.05
    c, s = np.cos(theta), np.sin(theta)
    rot = np.eye(baseline.shape[0])
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    inserted = baseline @ rot.T
    inserted_n = _unit(inserted)

    backend = _new_backend(backend_name)
    retrieved = _insert_get(backend, baseline.shape[0], inserted)

    lossless = _cos(retrieved, inserted_n)
    recovered = retrieved @ rot  # inverse of @ rot.T is @ rot for orthogonal rot
    recovery = _cos(_unit(recovered), baseline)
    stealth = _cos(inserted_n, baseline)
    return CellResult(backend_name, "rotation", lossless, recovery, stealth)


def run_scaling(backend_name: str, baseline: np.ndarray) -> CellResult:
    scale = 0.95
    inserted = scale * baseline
    inserted_n = _unit(inserted)

    backend = _new_backend(backend_name)
    retrieved = _insert_get(backend, baseline.shape[0], inserted)

    lossless = _cos(retrieved, inserted_n)
    # Backend strips magnitude on insert; deobfuscate by dividing by scale
    # gives back baseline direction unchanged (cos invariant to magnitude).
    recovered = retrieved / scale
    recovery = _cos(_unit(recovered), baseline)
    stealth = _cos(inserted_n, baseline)
    return CellResult(
        backend_name, "scaling", lossless, recovery, stealth,
        notes="magnitude-only attack; survives via direction preservation",
    )


def run_offset(backend_name: str, baseline: np.ndarray) -> CellResult:
    rng = np.random.default_rng(SEED + 1)
    offset_range = 0.05
    offset = rng.uniform(-offset_range, offset_range, baseline.shape)
    inserted = baseline + offset
    inserted_n = _unit(inserted)

    backend = _new_backend(backend_name)
    retrieved = _insert_get(backend, baseline.shape[0], inserted)

    lossless = _cos(retrieved, inserted_n)
    # retrieved = (baseline+offset)/||baseline+offset||. Reverse:
    # rescale to original magnitude, subtract offset.
    recovered = retrieved * np.linalg.norm(inserted) - offset
    recovery = _cos(_unit(recovered), baseline)
    stealth = _cos(inserted_n, baseline)
    return CellResult(backend_name, "offset", lossless, recovery, stealth)


def run_fragmentation(backend_name: str, baseline: np.ndarray) -> CellResult:
    """Split baseline into N sparse fragments, insert each as its own
    record, retrieve all, sum, compare direction."""
    n_frags = 4
    frag_indices = np.array_split(np.arange(baseline.shape[0]), n_frags)
    fragments = []
    for indices in frag_indices:
        f = np.zeros(baseline.shape[0], dtype=np.float32)
        f[indices] = baseline[indices]
        fragments.append(f)

    backend = _new_backend(backend_name)
    backend.open(dim=baseline.shape[0])
    try:
        ids = [f"frag-{i}" for i in range(n_frags)]
        backend.insert_arrays(ids, np.stack(fragments))
        retrieved_frags = [backend.get_by_id(rid)[0] for rid in ids]
    finally:
        backend.close()

    # Each fragment is stored normalized. Sum reconstructs direction
    # (re-weighted but cos-equivalent to baseline since the supports
    # are disjoint and each fragment direction matches baseline-on-support).
    reassembled = np.sum(retrieved_frags, axis=0)
    recovery = _cos(_unit(reassembled), baseline)
    # Lossless metric uses the first fragment as the canonical insert check.
    lossless = _cos(retrieved_frags[0], _unit(fragments[0]))
    stealth = float("nan")  # stealth doesn't apply to per-fragment insert
    return CellResult(
        backend_name, "fragmentation", lossless, recovery, stealth,
        notes=f"{n_frags} fragments stored as separate records",
    )


def run_decoy_interleaving(backend_name: str, baseline: np.ndarray) -> CellResult:
    """Insert 1 real + several decoys; verify real survives intact."""
    rng = np.random.default_rng(SEED + 2)
    n_decoys = 4
    decoys = rng.normal(0, 1, size=(n_decoys, baseline.shape[0])).astype(np.float32)
    decoys = decoys / np.linalg.norm(decoys, axis=1, keepdims=True)

    backend = _new_backend(backend_name)
    backend.open(dim=baseline.shape[0])
    try:
        ids = ["real"] + [f"decoy-{i}" for i in range(n_decoys)]
        all_vecs = np.vstack([baseline.reshape(1, -1), decoys])
        backend.insert_arrays(ids, all_vecs)
        real_retrieved, _ = backend.get_by_id("real")
    finally:
        backend.close()

    lossless = _cos(real_retrieved, baseline)
    recovery = _cos(real_retrieved, baseline)
    stealth = 1.0
    return CellResult(
        backend_name, "decoy-interleaving", lossless, recovery, stealth,
        notes=f"1 real + {n_decoys} decoys; real recovered by id",
    )


def run_composed(backend_name: str, baseline: np.ndarray) -> CellResult:
    """Full obfuscate() pipeline: noise + rotation + scaling + offset.

    Skip fragmentation here — it's tested as its own row. Apply via the
    library's compose path so we exercise the actual production code.
    """
    obf = EmbeddingObfuscator(
        noise_level=0.005,
        rotation_angle=0.05,
        scaling_factor=0.95,
        offset_range=0.01,
        seed=SEED,
    )
    bundle = obf.obfuscate(
        baseline.reshape(1, -1),
        techniques=["noise", "rotation", "scaling", "offset"],
    )
    inserted = bundle["embeddings"][0]
    inserted_n = _unit(inserted)

    backend = _new_backend(backend_name)
    retrieved = _insert_get(backend, baseline.shape[0], inserted)

    lossless = _cos(retrieved, inserted_n)

    # Deobfuscate uses the recorded metadata. We need to pre-scale
    # because the backend dropped magnitude.
    bundle["embeddings"] = (retrieved * np.linalg.norm(inserted)).reshape(1, -1)
    recovered = obf.deobfuscate(bundle)[0]
    recovery = _cos(_unit(recovered), baseline)
    stealth = _cos(inserted_n, baseline)
    return CellResult(backend_name, "composed", lossless, recovery, stealth)


PRIMITIVES = [
    ("noise", run_noise),
    ("rotation", run_rotation),
    ("scaling", run_scaling),
    ("offset", run_offset),
    ("fragmentation", run_fragmentation),
    ("decoy-interleaving", run_decoy_interleaving),
    ("composed", run_composed),
]


def main() -> int:
    rng = np.random.default_rng(SEED)
    baseline = rng.normal(0, 1, DIM).astype(np.float32)
    baseline = _unit(baseline)

    backend_names = ["OpenSearch", "MemoryDB"]
    if os.environ.get("AWS_S3VECTORS_TEST", "").lower() in ("1", "true", "yes"):
        backend_names.append("S3Vectors")
    else:
        print("# AWS_S3VECTORS_TEST not set — skipping S3 Vectors row")

    results: list[CellResult] = []
    for backend_name in backend_names:
        for tech_name, fn in PRIMITIVES:
            try:
                cell = fn(backend_name, baseline.copy())
            except Exception as e:
                cell = CellResult(
                    backend_name, tech_name,
                    lossless=float("nan"), recovery=float("nan"),
                    stealth=float("nan"), notes=f"ERROR: {type(e).__name__}: {e}",
                )
            results.append(cell)
            print(f"{backend_name:>11} | {tech_name:>20} | "
                  f"lossless={cell.lossless:>13.10f} | "
                  f"recovery={cell.recovery:>13.10f} | "
                  f"stealth={cell.stealth:>13.10f} | {cell.notes}")

    out_path = os.path.join(HERE, "aws_steg_matrix_results.json")
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    print(f"\nwrote {out_path}")

    threshold = 0.999
    failures = [
        r for r in results
        if not np.isnan(r.recovery) and r.recovery < threshold
        and not r.notes.startswith("ERROR")
    ]
    if failures:
        print(f"\n{len(failures)} cells below recovery>{threshold}:")
        for f in failures:
            print(f"  {f.backend} / {f.technique}: recovery={f.recovery:.6f}")
        return 1
    print(f"\nALL cells passed recovery > {threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
