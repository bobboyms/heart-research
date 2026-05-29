# Comparacao: multiscale STFT com dilatacoes extras e focal loss

Data: 2026-05-27

## Experimentos comparados

Todos os modelos usam 874 pacientes OOF, 5 folds por paciente, locais
`AV/PV/TV/MV`, sistole, margem 50 ms, `--systole-threshold 0.45`,
pooling attention, calibracao Platt e o mesmo conjunto de checkpoints TCN por
fold.

| Experimento | Spectrograma | Encoder | Dilatacoes | Loss | Diretorio |
|---|---|---|---|---|---|
| Melhor atual | STFT | multiscale | `1,2,4,8` | BCE | `tcn_attn_weight2_platt_thr045_ignore_multiscale_reuse_tcn` |
| Teste atual | STFT | multiscale | `1,2,4,8,16,32` | focal, gamma 2.0, alpha none | `tcn_attn_weight2_platt_thr045_ignore_multiscale_stft_dilated32_focal_reuse_tcn` |
| Controle log-mel focal | log-mel | multiscale | `1,2,4,8,16,32` | focal, gamma 2.0, alpha none | `tcn_attn_weight2_platt_thr045_ignore_multiscale_logmel_dilated32_focal_reuse_tcn` |

## Resultado operacional calibrado @0.5

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Multiscale STFT BCE `1,2,4,8` | 90.07% | 84.24% | 84.31% | 72.07% | 96.55% | 84.31% | 77.71% | 671 | 24 | 50 | 129 | 6.88% |
| Multiscale STFT focal `1,2,4,8,16,32` | 91.10% | 84.31% | 83.68% | 70.95% | 96.40% | 83.55% | 76.74% | 670 | 25 | 52 | 127 | 7.08% |
| Multiscale log-mel focal `1,2,4,8,16,32` | 84.47% | 72.86% | 78.86% | 61.45% | 96.26% | 80.88% | 69.84% | 669 | 26 | 69 | 110 | 9.17% |

## Threshold Youden por fold

| Experimento | AUROC | AUPRC | BA | Sens | Spec | PPV | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Multiscale STFT BCE `1,2,4,8` | 90.26% | 83.82% | 85.84% | 79.89% | 91.80% | 71.50% | 75.46% | 638 | 57 | 36 | 143 |
| Multiscale STFT focal `1,2,4,8,16,32` | 91.25% | 83.79% | 82.97% | 79.89% | 86.04% | 59.58% | 68.26% | 598 | 97 | 36 | 143 |
| Multiscale log-mel focal `1,2,4,8,16,32` | 83.68% | 72.36% | 79.00% | 70.95% | 87.05% | 58.53% | 64.14% | 605 | 90 | 52 | 127 |

## Leitura

O teste STFT focal com dilatacoes `1,2,4,8,16,32` melhorou levemente o ranking
global em relacao ao melhor atual: AUROC subiu de 90.07% para 91.10%, e AUPRC
ficou praticamente empatado, 84.24% para 84.31%.

Mesmo assim, ele nao melhorou o modelo operacional. No threshold calibrado `0.5`,
perdeu 2 verdadeiros positivos, ganhou 2 falsos negativos, reduziu F1 e piorou o
Brier. No threshold Youden, manteve os mesmos 143 verdadeiros positivos do STFT
BCE, mas gerou 40 falsos positivos a mais, derrubando PPV e F1.

Conclusao: o experimento e interessante porque mostra que STFT continua muito
superior ao log-mel com focal loss, mas ainda nao substitui o melhor modelo atual.
Manter `multiscale + STFT + BCE + dilations 1,2,4,8` como referencia principal.
O proximo teste justo, se quisermos separar o efeito das dilatacoes do efeito da
loss, e rodar `multiscale + STFT + dilations 1,2,4,8,16,32 + BCE`.
