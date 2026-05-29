# Comparacao: STFT baseline vs log-mel

Experimentos comparados:

- `tcn_attn_weight2_platt_thr045_ignore__20260526_131751_962264`: melhor baseline sistolico com STFT, margem 50 ms.
- `tcn_attn_weight2_platt_thr045_ignore_logmel_reuse_tcn`: mesmos parametros do baseline, TCN reaproveitado, `--spectrogram-type log-mel`, `--n-mels 64`.

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STFT baseline | 90.52% | 84.15% | 82.92% | 68.72% | 97.12% | 86.01% | 76.40% | 675 | 20 | 56 | 123 | 6.96% |
| Log-mel 64 bins | 82.92% | 72.46% | 77.46% | 58.66% | 96.26% | 80.15% | 67.74% | 669 | 26 | 74 | 105 | 9.76% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STFT baseline | 90.20% | 82.94% | 84.67% | 81.56% | 87.77% | 63.20% | 71.22% | 610 | 85 | 33 | 146 |
| Log-mel 64 bins | 82.78% | 71.28% | 76.63% | 70.95% | 82.30% | 50.80% | 59.21% | 572 | 123 | 52 | 127 |

## Leitura

O `log-mel` com 64 bins piorou o ranking e a classificacao paciente-level. A queda em AUPRC e F1 e grande, e o modelo calibrado recuperou menos pacientes `Present` que o baseline STFT.

Conclusao: manter STFT como representacao principal. O resultado sugere que, nesta arquitetura e com estes parametros, a compressao Mel remove informacao espectral util para sopro sistolico.

Se `log-mel` ainda for explorado, os proximos testes deveriam mudar tambem a configuracao da representacao, por exemplo `--n-mels 80` ou `128`, ou ajustar `high_hz`, em vez de trocar apenas o tipo de espectrograma mantendo o resto igual.
