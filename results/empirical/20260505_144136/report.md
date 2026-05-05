# VectorSmuggle Empirical Results — `20260505_144136`

## Setup

- Embedding model: `text-embedding-3-large`
- Embedding dimension: 3072
- Document chunks: 68
- Baseline embedding wall time: 1.77s
- Configurations: noise_low, noise_med, noise_high, default
- Techniques: noise, rotation, scaling, offset, noise+rotation, noise+rotation+scaling, all

## Headline Table

| Config | Technique | cos(orig, obf) | pair-corr | MSE | recovery cos | time (ms) |
|---|---|---:|---:|---:|---:|---:|
| noise_low | noise | 0.9637 | 0.9983 | 0.000025 | — | 3.5 |
| noise_low | rotation | 1.0000 | 1.0000 | 0.000000 | 1.0000 | 570.6 |
| noise_low | scaling | 1.0000 | 1.0000 | 0.000003 | 1.0000 | 1.0 |
| noise_low | offset | 0.5289 | 0.7347 | 0.000832 | — | 1.6 |
| noise_low | noise+rotation | 0.9637 | 0.9981 | 0.000025 | 0.9637 | 568.1 |
| noise_low | noise+rotation+scaling | 0.9637 | 0.9982 | 0.000027 | 0.9637 | 655.7 |
| noise_low | all | 0.5042 | 0.6868 | 0.000857 | 0.5043 | 756.0 |
| noise_med | noise | 0.8749 | 0.9830 | 0.000100 | — | 4.1 |
| noise_med | rotation | 1.0000 | 1.0000 | 0.000000 | 1.0000 | 681.0 |
| noise_med | scaling | 1.0000 | 1.0000 | 0.000003 | 1.0000 | 1.3 |
| noise_med | offset | 0.5289 | 0.7347 | 0.000832 | — | 1.5 |
| noise_med | noise+rotation | 0.8749 | 0.9816 | 0.000100 | 0.8749 | 563.2 |
| noise_med | noise+rotation+scaling | 0.8746 | 0.9820 | 0.000099 | 0.8746 | 572.2 |
| noise_med | all | 0.4897 | 0.6736 | 0.000924 | 0.4898 | 634.0 |
| noise_high | noise | 0.3418 | 0.5711 | 0.002501 | — | 3.5 |
| noise_high | rotation | 1.0000 | 1.0000 | 0.000000 | 1.0000 | 696.1 |
| noise_high | scaling | 1.0000 | 1.0000 | 0.000003 | 1.0000 | 1.2 |
| noise_high | offset | 0.5289 | 0.7347 | 0.000832 | — | 1.5 |
| noise_high | noise+rotation | 0.3432 | 0.5541 | 0.002510 | 0.3432 | 752.4 |
| noise_high | noise+rotation+scaling | 0.3380 | 0.5592 | 0.002402 | 0.3380 | 693.6 |
| noise_high | all | 0.2925 | 0.5178 | 0.003111 | 0.2925 | 750.2 |
| default | noise | 0.8749 | 0.9830 | 0.000100 | — | 3.4 |
| default | rotation | 1.0000 | 1.0000 | 0.000000 | 1.0000 | 609.3 |
| default | scaling | 1.0000 | 1.0000 | 0.000003 | 1.0000 | 1.1 |
| default | offset | 0.5289 | 0.7347 | 0.000832 | — | 1.6 |
| default | noise+rotation | 0.8749 | 0.9816 | 0.000100 | 0.8749 | 755.4 |
| default | noise+rotation+scaling | 0.8746 | 0.9820 | 0.000099 | 0.8746 | 637.3 |
| default | all | 0.4897 | 0.6736 | 0.000924 | 0.4898 | 698.8 |

## Detection Signatures

| Config | Technique | KS stat | KS p-value | norm-ratio mean ± std | entropy Δ |
|---|---|---:|---:|---:|---:|
| noise_low | noise | 0.0121 | 1.05e-13 | 1.0385 ± 0.0042 | +0.5991 |
| noise_low | rotation | 0.0001 | 1.00e+00 | 1.0000 ± 0.0000 | -0.0000 |
| noise_low | scaling | 0.0130 | 7.42e-16 | 0.9525 ± 0.0907 | -0.4617 |
| noise_low | offset | 0.1866 | 0.00e+00 | 1.8843 ± 0.0178 | +5.1976 |
| noise_low | noise+rotation | 0.0122 | 7.48e-14 | 1.0393 ± 0.0049 | +0.5493 |
| noise_low | noise+rotation+scaling | 0.0048 | 1.65e-02 | 1.0113 ± 0.1011 | +0.2112 |
| noise_low | all | 0.1877 | 0.00e+00 | 1.8777 ± 0.0597 | +4.8635 |
| noise_med | noise | 0.0393 | 2.41e-140 | 1.1447 ± 0.0077 | +1.5587 |
| noise_med | rotation | 0.0001 | 1.00e+00 | 1.0000 ± 0.0000 | -0.0000 |
| noise_med | scaling | 0.0130 | 7.42e-16 | 0.9525 ± 0.0907 | -0.4617 |
| noise_med | offset | 0.1866 | 0.00e+00 | 1.8843 ± 0.0178 | +5.1976 |
| noise_med | noise+rotation | 0.0397 | 2.49e-143 | 1.1464 ± 0.0093 | +1.6922 |
| noise_med | noise+rotation+scaling | 0.0313 | 2.71e-89 | 1.1137 ± 0.1114 | +1.1157 |
| noise_med | all | 0.1900 | 0.00e+00 | 1.9319 ± 0.0693 | +5.3111 |
| noise_high | noise | 0.2464 | 0.00e+00 | 2.9490 ± 0.0309 | +8.8804 |
| noise_high | rotation | 0.0001 | 1.00e+00 | 1.0000 ± 0.0000 | -0.0000 |
| noise_high | scaling | 0.0130 | 7.42e-16 | 0.9525 ± 0.0907 | -0.4617 |
| noise_high | offset | 0.1866 | 0.00e+00 | 1.8843 ± 0.0178 | +5.1976 |
| noise_high | noise+rotation | 0.2468 | 0.00e+00 | 2.9562 ± 0.0372 | +8.6012 |
| noise_high | noise+rotation+scaling | 0.2396 | 0.00e+00 | 2.8688 ± 0.2934 | +8.4276 |
| noise_high | all | 0.2629 | 0.00e+00 | 3.2219 ± 0.2576 | +9.3122 |
| default | noise | 0.0393 | 2.41e-140 | 1.1447 ± 0.0077 | +1.5587 |
| default | rotation | 0.0001 | 1.00e+00 | 1.0000 ± 0.0000 | -0.0000 |
| default | scaling | 0.0130 | 7.42e-16 | 0.9525 ± 0.0907 | -0.4617 |
| default | offset | 0.1866 | 0.00e+00 | 1.8843 ± 0.0178 | +5.1976 |
| default | noise+rotation | 0.0397 | 2.49e-143 | 1.1464 ± 0.0093 | +1.6922 |
| default | noise+rotation+scaling | 0.0313 | 2.71e-89 | 1.1137 ± 0.1114 | +1.1157 |
| default | all | 0.1900 | 0.00e+00 | 1.9319 ± 0.0693 | +5.3111 |

## Notes for Interpretation

- **cos(orig, obf)** is the per-vector cosine similarity between the original
  embedding and the obfuscated one. Values close to 1.0 mean the technique
  did not meaningfully shift direction; values below ~0.7 indicate strong
  semantic perturbation that will likely degrade retrieval.
- **pair-corr** is the correlation between pairwise cosine-similarity
  matrices before and after obfuscation. This is the operationally
  important fidelity measure: high pair-corr means similar items remain
  similar after obfuscation, even if absolute directions shift.
- **KS stat** is the two-sample Kolmogorov-Smirnov statistic comparing
  the flattened distribution of original vs. obfuscated values. Higher KS
  with p<0.05 indicates a detectable distributional fingerprint.
- **recovery cos** is the cosine similarity between original and the
  attempt to invert the obfuscation. Rotation and scaling are exactly
  invertible given the metadata; noise and offset are not.
