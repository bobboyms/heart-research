# Comparacao: multiscale STFT com dilatacoes extras e BCE + AUC loss

Data: 2026-05-27

## Experimentos comparados

Todos os modelos usam 874 pacientes OOF, 5 folds por paciente, locais
`AV/PV/TV/MV`, sistole, margem 50 ms, `--systole-threshold 0.45`,
pooling attention, calibracao Platt e o mesmo conjunto de checkpoints TCN por
fold.

| Experimento | Spectrograma | Encoder | Dilatacoes | Loss | AUC loss weight | Diretorio |
|---|---|---|---|---|---:|---|
| Referencia atual | STFT | multiscale | `1,2,4,8` | BCE | 0.0 | `tcn_attn_weight2_platt_thr045_ignore_multiscale_reuse_tcn` |
| Focal anterior | STFT | multiscale | `1,2,4,8,16,32` | focal | 0.0 | `tcn_attn_weight2_platt_thr045_ignore_multiscale_stft_dilated32_focal_reuse_tcn` |
| Teste atual | STFT | multiscale | `1,2,4,8,16,32` | BCE | 0.1 | `tcn_attn_weight2_platt_thr045_ignore_multiscale_stft_dilated32_bce_auc01_reuse_tcn` |

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STFT multiscale BCE `1,2,4,8` | 90.07% | 84.24% | 84.31% | 72.07% | 96.55% | 84.31% | 77.71% | 671 | 24 | 50 | 129 | 6.88% |
| STFT multiscale focal `1,2,4,8,16,32` | 91.10% | 84.31% | 83.68% | 70.95% | 96.40% | 83.55% | 76.74% | 670 | 25 | 52 | 127 | 7.08% |
| STFT multiscale BCE + AUC 0.1 `1,2,4,8,16,32` | 90.71% | 83.20% | 84.93% | 73.74% | 96.12% | 83.02% | 78.11% | 668 | 27 | 47 | 132 | 7.19% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STFT multiscale BCE `1,2,4,8` | 90.26% | 83.82% | 85.84% | 79.89% | 91.80% | 71.50% | 75.46% | 638 | 57 | 36 | 143 |
| STFT multiscale focal `1,2,4,8,16,32` | 91.25% | 83.79% | 82.97% | 79.89% | 86.04% | 59.58% | 68.26% | 598 | 97 | 36 | 143 |
| STFT multiscale BCE + AUC 0.1 `1,2,4,8,16,32` | 90.77% | 83.27% | 85.70% | 79.89% | 91.51% | 70.79% | 75.07% | 636 | 59 | 36 | 143 |

## Leitura

O termo auxiliar `--auc-loss-weight 0.1` melhorou o ponto operacional calibrado
`0.5` em relacao a referencia STFT BCE:

| Metrica | STFT BCE referencia | STFT BCE + AUC 0.1 | Diferenca |
|---|---:|---:|---:|
| Balanced accuracy | 84.31% | 84.93% | +0.62 pp |
| Sensibilidade | 72.07% | 73.74% | +1.68 pp |
| F1 | 77.71% | 78.11% | +0.40 pp |
| TP | 129 | 132 | +3 |
| FN | 50 | 47 | -3 |
| FP | 24 | 27 | +3 |
| AUPRC | 84.24% | 83.20% | -1.04 pp |
| Brier | 6.88% | 7.19% | +0.31 pp |

No threshold Youden, o resultado ficou praticamente empatado com a referencia:
mesmos 143 verdadeiros positivos e 36 falsos negativos, mas com 2 falsos
positivos a mais.

Conclusao: este e um bom candidato se a prioridade for o ponto operacional
calibrado `0.5`, porque recupera mais pacientes `Present` e melhora F1/BA. A
referencia STFT BCE `1,2,4,8` ainda e mais limpa em AUPRC e Brier. Eu trataria
`STFT multiscale BCE + AUC 0.1 + dilations 1,2,4,8,16,32` como novo candidato
operacional, mas ainda nao como substituto definitivo sem testar tambem
`STFT multiscale BCE + dilations 1,2,4,8,16,32` sem AUC loss.
