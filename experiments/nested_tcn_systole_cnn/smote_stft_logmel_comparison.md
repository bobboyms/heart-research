# Comparacao: SMOTE em STFT e log-mel

Experimentos comparados:

- `tcn_attn_weight2_platt_thr045_ignore__20260526_131751_962264`: baseline STFT, sem SMOTE.
- `tcn_attn_weight2_platt_thr045_ignore_smote_stft_reuse_tcn`: STFT com SMOTE.
- `tcn_attn_weight2_platt_thr045_ignore_logmel_reuse_tcn`: log-mel 64 bins, sem SMOTE.
- `tcn_attn_weight2_platt_thr045_ignore_smote_logmel_reuse_tcn`: log-mel 64 bins com SMOTE.

Configuracao SMOTE:

- `--smote-minority-augmentation`
- `--smote-k-neighbors 5`
- `--smote-target-ratio 1.0`

O SMOTE foi aplicado apenas no split `cnn_fit`, sem tocar em `cnn_tune` ou na validacao externa OOF. Foram gerados 6034 espectrogramas sinteticos no total em cada experimento, sempre para a classe minoritaria `Present`.

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STFT baseline | 90.52% | 84.15% | 82.92% | 68.72% | 97.12% | 86.01% | 76.40% | 675 | 20 | 56 | 123 | 6.96% |
| STFT + SMOTE | 89.04% | 81.55% | 81.86% | 67.60% | 96.12% | 81.76% | 74.01% | 668 | 27 | 58 | 121 | 7.67% |
| Log-mel baseline | 82.92% | 72.46% | 77.46% | 58.66% | 96.26% | 80.15% | 67.74% | 669 | 26 | 74 | 105 | 9.76% |
| Log-mel + SMOTE | 84.92% | 76.53% | 79.77% | 63.13% | 96.40% | 81.88% | 71.29% | 670 | 25 | 66 | 113 | 8.76% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STFT baseline | 90.20% | 82.94% | 84.67% | 81.56% | 87.77% | 63.20% | 71.22% | 610 | 85 | 33 | 146 |
| STFT + SMOTE | 88.85% | 80.92% | 83.45% | 76.54% | 90.36% | 67.16% | 71.54% | 628 | 67 | 42 | 137 |
| Log-mel baseline | 82.78% | 71.28% | 76.63% | 70.95% | 82.30% | 50.80% | 59.21% | 572 | 123 | 52 | 127 |
| Log-mel + SMOTE | 84.97% | 74.39% | 79.71% | 71.51% | 87.91% | 60.38% | 65.47% | 611 | 84 | 51 | 128 |

## Leitura

No STFT, o SMOTE piorou o ranking e o resultado operacional: AUROC, AUPRC, F1, PPV e Brier caem contra o baseline. No threshold Youden, ele reduz falsos positivos e aumenta PPV, mas perde sensibilidade e balanced accuracy.

No log-mel, o SMOTE ajudou em relacao ao proprio log-mel baseline: melhora AUROC, AUPRC, sensibilidade, F1 e Brier. Mesmo assim, log-mel + SMOTE ainda fica abaixo do STFT baseline.

Conclusao: o melhor modelo continua sendo STFT sem SMOTE. O SMOTE nao deve substituir o baseline atual. Se for usado, ele parece mais util para recuperar parte da perda do log-mel do que para melhorar STFT.
