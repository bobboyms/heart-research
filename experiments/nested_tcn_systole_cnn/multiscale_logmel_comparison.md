# Comparacao: multiscale com STFT vs log-mel

Experimentos comparados:

- `tcn_attn_weight2_platt_thr045_ignore__20260526_131751_962264`: baseline residual STFT.
- `tcn_attn_weight2_platt_thr045_ignore_multiscale_reuse_tcn`: multiscale STFT.
- `tcn_attn_weight2_platt_thr045_ignore_logmel_reuse_tcn`: residual log-mel 64 bins.
- `tcn_attn_weight2_platt_thr045_ignore_multiscale_logmel_reuse_tcn`: multiscale log-mel 64 bins.

Todos usam sístole, margem 50 ms, `--systole-threshold 0.45`, pooling attention, calibracao Platt e 5 folds por paciente.

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual STFT baseline | 90.52% | 84.15% | 82.92% | 68.72% | 97.12% | 86.01% | 76.40% | 675 | 20 | 56 | 123 | 6.96% |
| Multiscale STFT | 90.07% | 84.24% | 84.31% | 72.07% | 96.55% | 84.31% | 77.71% | 671 | 24 | 50 | 129 | 6.88% |
| Residual log-mel | 82.92% | 72.46% | 77.46% | 58.66% | 96.26% | 80.15% | 67.74% | 669 | 26 | 74 | 105 | 9.76% |
| Multiscale log-mel | 86.66% | 77.78% | 79.41% | 63.13% | 95.68% | 79.02% | 70.19% | 665 | 30 | 66 | 113 | 8.71% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual STFT baseline | 90.20% | 82.94% | 84.67% | 81.56% | 87.77% | 63.20% | 71.22% | 610 | 85 | 33 | 146 |
| Multiscale STFT | 90.26% | 83.82% | 85.84% | 79.89% | 91.80% | 71.50% | 75.46% | 638 | 57 | 36 | 143 |
| Residual log-mel | 82.78% | 71.28% | 76.63% | 70.95% | 82.30% | 50.80% | 59.21% | 572 | 123 | 52 | 127 |
| Multiscale log-mel | 85.83% | 74.92% | 81.15% | 75.98% | 86.33% | 58.87% | 66.34% | 600 | 95 | 43 | 136 |

## Leitura

O `multiscale` tambem ajuda quando a entrada e log-mel: melhora AUROC, AUPRC, balanced accuracy, sensibilidade, F1 e Brier em relacao ao log-mel residual.

Mesmo assim, o log-mel multiscale ainda fica abaixo do STFT multiscale e abaixo do baseline STFT residual no ponto calibrado `0.5`.

Conclusao: a melhor configuracao continua sendo `multiscale + STFT`. O log-mel multiscale e uma melhora sobre log-mel residual, mas nao deve substituir STFT.
