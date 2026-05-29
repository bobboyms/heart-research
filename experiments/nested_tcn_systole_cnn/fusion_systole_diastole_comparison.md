# Fusão sístole + diástole (paciente-level OOF)

- Sístole run: `tcn_attn_weight2_platt_thr045_ignore_multiscale_stft_dilated32_focal_reuse_tcn`
- Diástole run: `tcn_attn_weight2_platt_thr045_ignore_diastole_only_reuse_tcn`
- N pacientes fundidos: 874  (positivos: 179)

## Resultados

| strategy                     |   auprc |   auroc |
|:-----------------------------|--------:|--------:|
| Sístole (calibrada)          |  0.8431 |  0.911  |
| Diástole (calibrada)         |  0.3813 |  0.6697 |
| Média simples                |  0.8103 |  0.8901 |
| Max                          |  0.8097 |  0.8804 |
| Média ponderada (w_sys=1.00) |  0.8431 |  0.911  |
| Stacker LR (calibradas, OOF) |  0.8378 |  0.9085 |
| Stacker LR (raw, OOF)        |  0.8302 |  0.9088 |

## Leitura

Comparar AUPRC das estratégias de fusão contra o baseline sístole (primeira linha).
Se nenhuma estratégia ultrapassar o baseline em mais de ~0.01, a diástole
não carrega sinal complementar suficiente para esta combinação de modelos.
