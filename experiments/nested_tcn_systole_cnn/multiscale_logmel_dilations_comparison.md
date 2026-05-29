# Comparacao: log-mel multiscale com dilatacoes extras

Experimentos comparados:

- `tcn_attn_weight2_platt_thr045_ignore_multiscale_reuse_tcn`: multiscale STFT com `--dilations 1,2,4,8`.
- `tcn_attn_weight2_platt_thr045_ignore_multiscale_logmel_reuse_tcn`: multiscale log-mel com `--dilations 1,2,4,8`.
- `tcn_attn_weight2_platt_thr045_ignore_multiscale_logmel_dilated32_reuse_tcn`: multiscale log-mel com `--dilations 1,2,4,8,16,32`.

Todos usam sístole, margem 50 ms, `--systole-threshold 0.45`, pooling attention, calibracao Platt e 5 folds por paciente.

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Multiscale STFT `1,2,4,8` | 90.07% | 84.24% | 84.31% | 72.07% | 96.55% | 84.31% | 77.71% | 671 | 24 | 50 | 129 | 6.88% |
| Multiscale log-mel `1,2,4,8` | 86.66% | 77.78% | 79.41% | 63.13% | 95.68% | 79.02% | 70.19% | 665 | 30 | 66 | 113 | 8.71% |
| Multiscale log-mel `1,2,4,8,16,32` | 87.33% | 77.77% | 81.50% | 67.60% | 95.40% | 79.08% | 72.89% | 663 | 32 | 58 | 121 | 8.50% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Multiscale STFT `1,2,4,8` | 90.26% | 83.82% | 85.84% | 79.89% | 91.80% | 71.50% | 75.46% | 638 | 57 | 36 | 143 |
| Multiscale log-mel `1,2,4,8` | 85.83% | 74.92% | 81.15% | 75.98% | 86.33% | 58.87% | 66.34% | 600 | 95 | 43 | 136 |
| Multiscale log-mel `1,2,4,8,16,32` | 86.49% | 76.91% | 82.28% | 77.65% | 86.91% | 60.43% | 67.97% | 604 | 91 | 40 | 139 |

## Leitura

Adicionar dilatacoes `16,32` melhorou o log-mel multiscale em balanced accuracy, sensibilidade, F1, Brier e threshold Youden. O AUPRC operacional ficou praticamente igual, mas o modelo recuperou 8 positivos a mais no ponto calibrado `0.5` em relacao ao log-mel multiscale anterior.

Mesmo assim, o melhor modelo geral continua sendo multiscale STFT. O log-mel com dilatacoes extras reduz parte da lacuna, mas ainda fica abaixo em AUPRC, F1, Brier e numero de verdadeiros positivos.

Conclusao: para log-mel, dilatacoes maiores ajudam. Para o pipeline principal, manter `multiscale + STFT + dilations 1,2,4,8` como melhor configuracao atual.
