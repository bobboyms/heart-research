# Comparacao: baseline vs LTSRR na classe minoritaria

Experimentos comparados:

- `tcn_attn_weight2_platt_thr045_ignore__20260526_131751_962264`: baseline sistolico, margem 50 ms, sem LTSRR.
- `tcn_attn_weight2_platt_thr045_ignore_ltsrr_minority_reuse_tcn`: baseline sistolico, margem 50 ms, TCN reaproveitado, LTSRR aplicado apenas em `Present`.

Configuracao LTSRR:

- `--ltsrr-prob 1.0`
- `--ltsrr-k 4`
- `--ltsrr-frequency-ratio 0.25`
- `--ltsrr-minority-only`

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline sistole 50 ms | 90.52% | 84.15% | 82.92% | 68.72% | 97.12% | 86.01% | 76.40% | 675 | 20 | 56 | 123 | 6.96% |
| LTSRR minority only | 79.24% | 65.87% | 72.71% | 49.16% | 96.26% | 77.19% | 60.07% | 669 | 26 | 91 | 88 | 10.86% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline sistole 50 ms | 90.20% | 82.94% | 84.67% | 81.56% | 87.77% | 63.20% | 71.22% | 610 | 85 | 33 | 146 |
| LTSRR minority only | 76.69% | 63.16% | 72.15% | 53.07% | 91.22% | 60.90% | 56.72% | 634 | 61 | 84 | 95 |

## Leitura

O LTSRR forte na classe minoritaria piorou o desempenho. A queda mais importante foi em AUPRC, sensibilidade e F1, indicando que o modelo perdeu capacidade de rankear e recuperar pacientes `Present`.

Conclusao: com `prob=1.0`, `k=4` e `frequency_ratio=0.25`, a augmentacao ficou agressiva demais para este sinal sistolico. Ela reduziu falsos positivos no modo raw, mas aumentou muito falsos negativos, o que nao e desejavel para triagem de sopro.

Proximos testes mais plausiveis:

- reduzir `--ltsrr-prob` para `0.25` ou `0.5`;
- manter `--ltsrr-minority-only`;
- testar `--ltsrr-frequency-ratio 0.10` ou `0.15`;
- comparar com aumento de peso de perda para sopros fracos antes de insistir em augmentacao forte.
