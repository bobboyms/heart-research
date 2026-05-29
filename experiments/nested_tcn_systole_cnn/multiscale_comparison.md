# Comparacao: encoder residual vs multiscale

Experimentos comparados:

- `tcn_attn_weight2_platt_thr045_ignore__20260526_131751_962264`: baseline STFT com encoder `residual`.
- `tcn_attn_weight2_platt_thr045_ignore_multiscale_reuse_tcn`: mesmos parametros do baseline, TCN reaproveitado, `--encoder-block multiscale`.

Ambos usam STFT, sístole, margem 50 ms, `--systole-threshold 0.45`, pooling attention, calibracao Platt e 5 folds por paciente.

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual baseline | 90.52% | 84.15% | 82.92% | 68.72% | 97.12% | 86.01% | 76.40% | 675 | 20 | 56 | 123 | 6.96% |
| Multiscale | 90.07% | 84.24% | 84.31% | 72.07% | 96.55% | 84.31% | 77.71% | 671 | 24 | 50 | 129 | 6.88% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual baseline | 90.20% | 82.94% | 84.67% | 81.56% | 87.77% | 63.20% | 71.22% | 610 | 85 | 33 | 146 |
| Multiscale | 90.26% | 83.82% | 85.84% | 79.89% | 91.80% | 71.50% | 75.46% | 638 | 57 | 36 | 143 |

## Leitura

O `multiscale` melhorou o ponto operacional calibrado: maior balanced accuracy, sensibilidade, F1 e Brier levemente menor. Ele perdeu um pouco em AUROC e PPV contra o residual, mas a diferenca e pequena.

No threshold Youden por fold, o `multiscale` tambem melhora balanced accuracy, AUPRC, especificidade, PPV e F1, com pequena perda de sensibilidade.

Conclusao: `multiscale` passa a ser o melhor candidato geral ate aqui. Para uso operacional em `0.5`, ele recupera 6 positivos a mais que o residual baseline, ao custo de 4 falsos positivos adicionais.
