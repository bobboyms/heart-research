# Comparacao: focal loss no multiscale log-mel com dilatacoes extras

Data: 2026-05-27

## Experimentos comparados

Todos os modelos usam 874 pacientes OOF, 5 folds por paciente, locais
`AV/PV/TV/MV`, sistole, margem 50 ms, `--systole-threshold 0.45`,
pooling attention, calibracao Platt e o mesmo conjunto de checkpoints TCN por
fold.

| Experimento | Spectrograma | Encoder | Dilatacoes | Loss | Diretorio |
|---|---|---|---|---|---|
| Melhor atual | STFT | multiscale | `1,2,4,8` | BCE | `tcn_attn_weight2_platt_thr045_ignore_multiscale_reuse_tcn` |
| Log-mel dilated32 | log-mel | multiscale | `1,2,4,8,16,32` | BCE | `tcn_attn_weight2_platt_thr045_ignore_multiscale_logmel_dilated32_reuse_tcn` |
| Log-mel dilated32 focal | log-mel | multiscale | `1,2,4,8,16,32` | focal, gamma 2.0, alpha none | `tcn_attn_weight2_platt_thr045_ignore_multiscale_logmel_dilated32_focal_reuse_tcn` |

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Multiscale STFT BCE | 90.07% | 84.24% | 84.31% | 72.07% | 96.55% | 84.31% | 77.71% | 671 | 24 | 50 | 129 | 6.88% |
| Multiscale log-mel BCE `1,2,4,8,16,32` | 87.33% | 77.77% | 81.50% | 67.60% | 95.40% | 79.08% | 72.89% | 663 | 32 | 58 | 121 | 8.50% |
| Multiscale log-mel focal `1,2,4,8,16,32` | 84.47% | 72.86% | 78.86% | 61.45% | 96.26% | 80.88% | 69.84% | 669 | 26 | 69 | 110 | 9.17% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Multiscale STFT BCE | 90.26% | 83.82% | 85.84% | 79.89% | 91.80% | 71.50% | 75.46% | 638 | 57 | 36 | 143 |
| Multiscale log-mel BCE `1,2,4,8,16,32` | 86.49% | 76.91% | 82.28% | 77.65% | 86.91% | 60.43% | 67.97% | 604 | 91 | 40 | 139 |
| Multiscale log-mel focal `1,2,4,8,16,32` | 83.68% | 72.36% | 79.00% | 70.95% | 87.05% | 58.53% | 64.14% | 605 | 90 | 52 | 127 |

## Comparacao direta: log-mel BCE vs focal

No mesmo setup log-mel multiscale com dilatacoes `1,2,4,8,16,32`, a focal loss
piorou as metricas principais:

| Metrica | BCE | Focal | Diferenca |
|---|---:|---:|---:|
| AUROC | 87.33% | 84.47% | -2.86 pp |
| AUPRC | 77.77% | 72.86% | -4.90 pp |
| BA | 81.50% | 78.86% | -2.64 pp |
| Sensibilidade | 67.60% | 61.45% | -6.15 pp |
| F1 | 72.89% | 69.84% | -3.05 pp |
| Brier | 8.50% | 9.17% | +0.67 pp |
| TP | 121 | 110 | -11 |
| FN | 58 | 69 | +11 |
| FP | 32 | 26 | -6 |

## Leitura

A focal loss deixou o modelo mais conservador no ponto calibrado `0.5`: reduziu
falsos positivos, mas perdeu 11 verdadeiros positivos e aumentou os falsos
negativos. Para triagem de sopro, essa troca nao e boa, porque sensibilidade e
AUPRC cairam de forma relevante.

Conclusao: nao recomendo seguir com focal loss neste setup. O melhor modelo geral
continua sendo `multiscale + STFT + BCE`. Para log-mel, a melhor variante ainda
e `multiscale + dilations 1,2,4,8,16,32 + BCE`, mas ela permanece abaixo do STFT.
