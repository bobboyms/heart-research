# Grupo F - MLP para sopro com features PV+TV TCN

Este experimento treina um modelo deep learning tabular para prever:

```text
Murmur = Present vs Absent
```

Entrada:

```text
feature extraction/Grupo B v2 features relativas por local com TCN predito/outputs_pv_tv/patient_relative_phase_features.csv
```

Esse CSV ja esta no nivel de paciente, usando apenas `PV` e `TV`, com features Grupo B v2 agregadas por:

```text
mean_*
max_*
```

## Pipeline

```text
audio PV/TV
=> TCN segmenta S1 / systole / S2 / diastole
=> Grupo B v2 extrai features relativas
=> agregacao mean/max por paciente
=> MLP supervisionado
=> probabilidade de Murmur Present
```

## Execucao

```bash
uv run "modeling/Grupo F MLP sopro PV TV TCN features/train_patient_mlp_murmur.py"
```

Com MPS:

```bash
uv run "modeling/Grupo F MLP sopro PV TV TCN features/train_patient_mlp_murmur.py" --device mps
```

Para um teste mais curto:

```bash
uv run "modeling/Grupo F MLP sopro PV TV TCN features/train_patient_mlp_murmur.py" \
  --epochs 40 \
  --patience 10
```

## Validacao

O script usa `StratifiedKFold` no nivel de paciente. Como cada linha ja e um paciente, nao ha multiplas gravacoes do mesmo paciente dentro do mesmo CSV.

Metricas salvas:

- AUROC
- AUPRC
- balanced accuracy
- sensitivity / recall de `Present`
- specificity
- precision
- F1
- matriz de confusao derivada das predicoes out-of-fold

## Saidas

As saidas ficam em:

```text
modeling/Grupo F MLP sopro PV TV TCN features/outputs/
```

Arquivos principais:

- `summary.md`: resumo do experimento e metricas.
- `feature_matrix.csv`: matriz final `patient_id` + target + features.
- `oof_predictions.csv`: predicoes out-of-fold por paciente.
- `fold_metrics.csv`: metricas por fold.
- `training_history.csv`: historico de treino.
- `precision_recall_oof.png`: curva precision-recall.
- `features_used.json`: lista de features usadas.
- `fold_*_best_model.pt`: checkpoints por fold.
- `final_model.pt`: modelo final treinado em todos os pacientes, com scaler e threshold.
