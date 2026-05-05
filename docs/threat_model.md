# VectorSmuggle Threat Model

This document specifies the threat model under which VectorSmuggle's
steganographic techniques are evaluated. It exists because reviewers
correctly object that "if the attacker can read raw vectors, why not
just exfiltrate the source corpus directly?"

The short answer: in two of the three threat models we consider, the
source corpus is *not* available to the attacker, while the vector
store *is*.

## Adversary Capabilities Spectrum

| Capability | Insider Backup | Compromised DB Creds | Query-Only |
|---|:---:|:---:|:---:|
| Read source documents | ✗ | ✗ | ✗ |
| Write to source corpus pipeline | ✓ | ✗ | ✗ |
| Read raw vectors from store | ✓ | ✓ | ✗ |
| Issue search queries | ✓ | ✓ | ✓ |
| DLP scans source documents | ✓ | n/a | n/a |
| DLP scans vector store | ✗ | ✗ | ✗ |

## Threat Model A: Insider with Backup Access (Primary)

**Scenario.** A privileged engineer in an organization that uses RAG
for internal knowledge retrieval. The engineer can ingest documents
into the pipeline (the legitimate authoring path) and has read access
to vector-store backups (reasonable for an SRE or ML platform
engineer). The engineer cannot exfiltrate raw documents — they pass
through DLP, are watermarked, or are kept in a data room the engineer
does not have direct access to.

**Goal.** Smuggle sensitive source content out via the vector backup.
Once outside, the attacker reverses the obfuscation using metadata
they wrote into the embedding pipeline at ingestion time.

**Why steganography?** Without obfuscation, raw vectors of sensitive
documents look identical to vectors of any other document — but the
data is essentially the same content embedded with a known model, so a
defender with the model and a few suspicions can run nearest-neighbor
queries against a public corpus and discover what each vector
"sounds like." Obfuscation with attacker-controlled secrets prevents
the defender from doing this, even if the backup is intercepted.

**This is the model under which capacity, fidelity, and detection
metrics in the paper are reported.**

## Threat Model B: Compromised Vector-DB Credentials (Secondary)

**Scenario.** External attacker who has stolen vector-database
credentials (e.g., an exposed API key) but does not have access to the
source documents — they were either deleted, stored elsewhere, or
encrypted at rest with keys the attacker does not hold.

**Why steganography rather than direct dump?** Three reasons.

1. **Egress shaping.** A direct `qdrant export` is a single large
   transfer that trips egress monitoring. Steganographic exfiltration
   piggybacks on routine query traffic patterns that already exist on
   the wire.
2. **Targeted recovery.** The attacker only wants specific high-value
   documents, not the whole corpus. Steganographic markers let them
   pinpoint the right vectors instead of pulling gigabytes.
3. **Plausible deniability.** A vector store full of innocuous-
   looking embeddings is unremarkable. A defender who notices the
   credential theft sees "they read embeddings, but everything looks
   normal" — they don't know which queries succeeded.

This is a weaker model than A and is supplementary in the paper.

## Threat Model C: Query-Only (Out of Scope)

**Scenario.** Attacker has only similarity-search query access — no
ability to read raw vectors. They observe ranked results and try to
recover hidden bits from rank patterns.

**Status.** This model is theoretically interesting but is not what
this codebase implements. Recovering steganographic payloads from rank
data alone is a search-based oracle attack with much lower bandwidth
than the techniques here. We mention it for completeness; we do not
claim our techniques work under this model.

## Why This Matters for the Empirical Results

- All "recovery" results assume the attacker can read raw vectors and
  has access to the obfuscation metadata they wrote at ingestion.
- The defensive baselines (Isolation Forest, One-Class SVM in
  `analysis/detectors/`) assume a defender with a held-out batch of
  trusted clean embeddings — typical for an enterprise that ingests
  in batches and can flag a baseline period before deployment.
- The retrieval-utility metric (recall@k of cover documents) assumes
  the system continues to serve legitimate user queries — i.e., the
  attack is detectable to the extent it harms downstream RAG quality.

## What the Paper Should NOT Claim

- That steganography is the *only* way an attacker with raw-vector
  access can exfiltrate data. It isn't — direct dump is faster.
- That the techniques work against an adversary who only has query
  access. They don't.
- That a defender who does not maintain a clean baseline can detect
  these techniques. The detector baselines we report assume one.
