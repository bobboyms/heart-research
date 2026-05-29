# Comparacao: margem sistolica 50 ms vs 100 ms

Experimentos comparados:

- `tcn_attn_weight2_platt_thr045_ignore__20260526_131751_962264`: sistole, margem 50 ms.
- `tcn_attn_weight2_platt_thr045_ignore_margin100_reuse_tcn`: sistole, margem 100 ms, TCN reaproveitado do baseline.

Ambos usam `--systole-threshold 0.45`, `--cnn-phase-mode systole`, calibracao Platt, pooling attention, 5 folds por paciente e os mesmos locais `AV PV TV MV`.

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sistole margem 50 ms | 90.52% | 84.15% | 82.92% | 68.72% | 97.12% | 86.01% | 76.40% | 675 | 20 | 56 | 123 | 6.96% |
| Sistole margem 100 ms | 91.03% | 80.82% | 83.19% | 69.83% | 96.55% | 83.89% | 76.22% | 671 | 24 | 54 | 125 | 7.60% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sistole margem 50 ms | 90.20% | 82.94% | 84.67% | 81.56% | 87.77% | 63.20% | 71.22% | 610 | 85 | 33 | 146 |
| Sistole margem 100 ms | 90.12% | 80.75% | 83.59% | 76.54% | 90.65% | 67.82% | 71.92% | 630 | 65 | 42 | 137 |

## Leitura

A margem de 100 ms melhora levemente AUROC, balanced accuracy e sensibilidade no ponto calibrado `0.5`, mas piora AUPRC, Brier, PPV e F1. Na avaliacao por threshold Youden, a margem de 100 ms melhora especificidade, PPV e F1, mas perde bastante sensibilidade e balanced accuracy.

Conclusao: margem 100 ms nao substitui o baseline de 50 ms como melhor configuracao geral. Se o objetivo for screening, a margem 50 ms continua mais forte porque preserva sensibilidade maior no threshold ajustado. Se o objetivo for reduzir falsos positivos, a margem 100 ms pode ser interessante.
