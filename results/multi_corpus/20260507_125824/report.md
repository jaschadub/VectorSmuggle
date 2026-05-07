# Cross-corpus empirical study

Same model (`nomic-embed-text:latest`), same techniques, three independent corpora. Reported numbers are mean cosine to original, pairwise-cosine correlation, and detection AUCs (Isolation Forest, One-Class SVM) trained on a clean half-corpus and evaluated on the held-out half. The cross-corpus point: if a row's AUC stays at chance (~0.5) across all three corpora, the rotation/noise/scaling/offset finding is not an artifact of the small synthetic sample corpus.

## Per-corpus statistics

| Corpus | n chunks | Dim | Embed time (s) |
|---|---|---|---|
| `sample` | 68 | 768 | 0.0 |
| `nfcorpus` | 16763 | 768 | 857.5 |
| `quora` | 10002 | 768 | 263.5 |

## Detection AUC by corpus and technique

AUC = 0.5 means the detector is doing no better than random; AUC = 1.0 means perfect.

| Corpus | Technique | cos | pair_corr | IF AUC | IF TPR@5%FPR | OC-SVM AUC | OC-SVM TPR@5%FPR |
|---|---|---|---|---|---|---|---|
| `nfcorpus` | noise | 0.9999 | 1.0000 | 0.501 | 0.050 | 0.501 | 0.051 |
| `nfcorpus` | rotation | 1.0000 | 1.0000 | 0.500 | 0.050 | 0.500 | 0.050 |
| `nfcorpus` | scaling | 1.0000 | 1.0000 | 0.398 | 0.051 | 0.362 | 0.079 |
| `nfcorpus` | offset | 0.9991 | 0.9997 | 0.504 | 0.051 | 0.507 | 0.052 |
| `quora` | noise | 0.9999 | 1.0000 | 0.501 | 0.046 | 0.501 | 0.051 |
| `quora` | rotation | 1.0000 | 1.0000 | 0.500 | 0.051 | 0.500 | 0.050 |
| `quora` | scaling | 1.0000 | 1.0000 | 0.365 | 0.071 | 0.325 | 0.121 |
| `quora` | offset | 0.9993 | 0.9997 | 0.505 | 0.052 | 0.508 | 0.052 |
| `sample` | noise | 0.9999 | 1.0000 | 0.506 | 0.088 | 0.500 | 0.059 |
| `sample` | rotation | 1.0000 | 1.0000 | 0.497 | 0.029 | 0.496 | 0.059 |
| `sample` | scaling | 1.0000 | 1.0000 | 0.464 | 0.059 | 0.396 | 0.059 |
| `sample` | offset | 0.9992 | 0.9999 | 0.504 | 0.029 | 0.522 | 0.059 |

## Cross-corpus summary

- **noise**: IF AUC range 0.501--0.506, OC-SVM AUC range 0.500--0.501 across 3 corpora.
- **rotation**: IF AUC range 0.497--0.500, OC-SVM AUC range 0.496--0.500 across 3 corpora.
- **scaling**: IF AUC range 0.365--0.464, OC-SVM AUC range 0.325--0.396 across 3 corpora.
- **offset**: IF AUC range 0.504--0.505, OC-SVM AUC range 0.507--0.522 across 3 corpora.
