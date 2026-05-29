# Comparacao de modos de fase da CNN vs paper

Data: 2026-05-26

## Experimentos comparados

Todos os experimentos locais usam 874 pacientes OOF, 5 folds, locais AV/PV/TV/MV,
calibracao Platt e o mesmo conjunto de checkpoints TCN por fold. A diferenca
principal e a fase cardiaca usada para gerar o STFT da CNN.

| Modelo local | Modo da CNN | Diretorio |
|---|---|---|
| Sistole | `systole` | `experiments/nested_tcn_systole_cnn/tcn_attn_weight2_platt_thr045_ignore__20260526_131751_962264` |
| Sistole+diastole | `both` | `experiments/nested_tcn_systole_cnn/tcn_attn_weight2_platt_thr045_ignore_both_phases_reuse_tcn` |
| Diastole | `diastole` | `experiments/nested_tcn_systole_cnn/tcn_attn_weight2_platt_thr045_ignore_diastole_only_reuse_tcn` |

## Resultado operacional calibrado @0.5

| Modelo | Acc | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sistole | 91.30% | 90.52% | 84.15% | 82.92% | 68.72% | 97.12% | 86.01% | 76.40% | 675 | 20 | 56 | 123 | 6.96% |
| Sistole+diastole | 90.05% | 88.44% | 80.05% | 80.88% | 65.36% | 96.40% | 82.39% | 72.90% | 670 | 25 | 62 | 117 | 8.04% |
| Diastole | 80.89% | 66.97% | 38.13% | 56.67% | 15.64% | 97.70% | 63.64% | 25.11% | 679 | 16 | 151 | 28 | 15.07% |

## Resultado com threshold Youden por fold

| Modelo | Acc | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sistole | 86.50% | 90.20% | 82.94% | 84.67% | 81.56% | 87.77% | 63.20% | 71.22% | 610 | 85 | 33 | 146 |
| Sistole+diastole | 86.73% | 87.96% | 78.56% | 83.15% | 77.09% | 89.21% | 64.79% | 70.41% | 620 | 75 | 41 | 138 |
| Diastole | 66.02% | 65.56% | 36.53% | 62.25% | 55.87% | 68.63% | 31.45% | 40.24% | 477 | 218 | 79 | 100 |

## Comparacao com paper

Paper: `papers/2026_10.1038_s41598-026-45276-9.md`.

O paper reporta para PhysioNet 2022: accuracy 98.60%, precision 98.26%,
recall 98.95%, F1 98.61% e specificity 98.30%. Ele nao reporta AUROC/AUPRC/Brier
nas tabelas principais.

| Modelo | Protocolo/dataset | Acc | Sens/Recall | Spec | PPV/Precision | F1 | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Paper 2026, PhysioNet 2022 | Hold-out/A-Test; paper declara split patient-level | 98.60% | 98.95% | 98.30% | 98.26% | 98.61% | NR | NR |
| Local sistole | 5-fold nested OOF por paciente | 91.30% | 68.72% | 97.12% | 86.01% | 76.40% | 90.52% | 84.15% |
| Local sistole+diastole | 5-fold nested OOF por paciente | 90.05% | 65.36% | 96.40% | 82.39% | 72.90% | 88.44% | 80.05% |
| Local diastole | 5-fold nested OOF por paciente | 80.89% | 15.64% | 97.70% | 63.64% | 25.11% | 66.97% | 38.13% |

## Conclusao

O melhor modelo local continua sendo a CNN treinada apenas com sistole. Adicionar
diastole degradou AUROC, AUPRC, balanced accuracy, F1 e Brier. Diastole isolada
ficou muito fraca, principalmente por baixa sensibilidade.

O paper reporta numeros muito superiores, mas a comparacao nao e direta porque
o protocolo, a segmentacao, a agregacao e o conjunto experimental reportado nao
sao identicos. Ainda assim, a diferenca de recall/F1 indica que o proximo baseline
util para testar localmente seria MFCC + SMOTE + RNN no mesmo protocolo nested
OOF por paciente.
