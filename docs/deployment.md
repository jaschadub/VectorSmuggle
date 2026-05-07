# Deployment

VectorSmuggle is a research framework, not a production service — there is no long-running server to deploy. This document covers the two deployment-adjacent scenarios that come up in practice: running the experiments inside Docker for reproducibility, and bringing up the local vector-database services that the cross-backend study depends on.

## Running the experiments in Docker

The repository ships with a `Dockerfile` that pins Python 3.11 and the exact dependency versions used to produce the paper's measurements. Build and run:

```bash
docker build -t vectorsmuggle:latest .
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v "$PWD/results:/app/results" \
  vectorsmuggle:latest \
  python scripts/empirical_study.py
```

The volume mount is what brings the experiment's output back out of the container. Without it, results disappear when the container exits.

For local-embedding experiments, expose the host's Ollama daemon to the container:

```bash
docker run --rm \
  --network host \
  -v "$PWD/results:/app/results" \
  vectorsmuggle:latest \
  python scripts/multi_model_study.py
```

`--network host` is the simplest way to give the container access to the Ollama daemon at `localhost:11434`. On macOS use `--add-host=host.docker.internal:host-gateway` and set the Ollama URL accordingly.

## Local vector-database services

The cross-backend study (`scripts/cross_backend_study.py`) instantiates Qdrant and Chroma alongside in-process FAISS variants. The `docker-compose.yml` in `test_vector_dbs_docker/` brings up both services on their default ports:

```bash
cd test_vector_dbs_docker
docker compose up -d
```

This starts Qdrant on `localhost:6333` and Chroma on `localhost:8000`. Verify both are healthy before running the study:

```bash
curl -s http://localhost:6333/collections
curl -s http://localhost:8000/api/v1/heartbeat
```

Tear them down when finished:

```bash
docker compose down
```

The study creates and destroys collections per run, so leftover state between runs is not a concern.

## Pinecone

If you want to run the empirical study against Pinecone, set `VECTOR_DB=pinecone`, `PINECONE_API_KEY`, and `PINECONE_ENVIRONMENT`, then provision an index with the right dimension for your embedding model (`3072` for `text-embedding-3-large`). Pinecone is a hosted backend; large studies incur cost. The paper's reported Pinecone numbers were produced against a free-tier index.

## Resource expectations

The full empirical study against the 68-chunk corpus completes in under five minutes on a laptop with 8 GB RAM. The cross-backend study takes longer (≈10 minutes) because it embeds the corpus once and then writes/reads through every backend. The multi-model study takes ≈40 seconds per Ollama model after model warm-up. None of the experiments need a GPU.

## CI

The repository's CI runs the test suite and the quick-start demo on every push. It does not run the full empirical study because that would require an OpenAI API key in CI; the empirical numbers are reproduced manually and committed alongside the paper releases.
