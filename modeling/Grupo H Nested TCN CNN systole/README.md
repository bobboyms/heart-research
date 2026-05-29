# Grupo H - Nested TCN + CNN sistole

Este experimento implementa a validacao mais rigorosa discutida:

```text
para cada fold de classificacao de sopro:
  treinar TCN somente nos pacientes de treino do fold
  predizer sistole nos pacientes de treino e validacao com esse TCN
  dividir os pacientes de treino em cnn_fit e cnn_tune
  treinar CNN somente em cnn_fit
  usar cnn_tune para early stopping, threshold e calibracao
  avaliar CNN nos pacientes de validacao do fold
```

Assim, o TCN de um fold nunca ve os `.tsv` dos pacientes usados para validar o classificador de sopro naquele fold.

## Execucao completa recomendada

```bash
caffeinate -dimsu uv run "modeling/Grupo H Nested TCN CNN systole/train_nested_tcn_systole_cnn.py" \
  --folds 5 \
  --locations AV PV TV MV \
  --tcn-epochs 10 \
  --tcn-batch-size 42 \
  --tcn-device mps \
  --tcn-pooling attention \
  --cnn-epochs 50 \
  --cnn-patience 8 \
  --cnn-batch-size 32 \
  --cnn-inner-val-size 0.15 \
  --cnn-device mps \
  --pooling attention \
  --calibration platt \
  --output-dir "modeling/Grupo H Nested TCN CNN systole/outputs_nested"
```

## Smoke test

```bash
uv run "modeling/Grupo H Nested TCN CNN systole/train_nested_tcn_systole_cnn.py" \
  --max-patients 30 \
  --folds 2 \
  --tcn-epochs 1 \
  --cnn-epochs 1 \
  --cnn-patience 1 \
  --tcn-batch-size 8 \
  --cnn-batch-size 8 \
  --cnn-inner-val-size 0.2 \
  --tcn-device cpu \
  --tcn-pooling attention \
  --cnn-device cpu \
  --tcn-val-size 0.1 \
  --tcn-test-size 0.1 \
  --output-dir "modeling/Grupo H Nested TCN CNN systole/outputs_smoke" \
  --no-progress
```

## Saidas

O diretorio de saida contem:

- `summary.md`: metricas finais paciente-level OOF.
- `patient_oof_predictions.csv`: probabilidades brutas e calibradas por paciente.
- `fold_metrics.csv`: metricas por fold.
- `training_history.csv`: historico da CNN por fold.
- `threshold_metrics_by_fold.md`: metricas por fold em varios thresholds.
- `threshold_metrics_by_fold_raw.csv`: thresholds para probabilidade bruta.
- `threshold_metrics_by_fold_calibrated.csv`: thresholds para probabilidade calibrada.
- `fold_*/tcn/best_model.pt`: TCN treinado apenas nos pacientes de treino daquele fold.
- `fold_*/predicted_tsvs/`: segmentacoes preditas pelo TCN daquele fold.
- `fold_*/cnn/fold_*_best_model.pt`: CNN treinada no fold.
- `fold_*/cnn_fit_patient_ids.txt`: pacientes usados para ajustar pesos da CNN.
- `fold_*/cnn_tune_patient_ids.txt`: pacientes usados para early stopping, threshold e calibracao da CNN.

## Observacao metodologica

Este protocolo e mais rigoroso que usar um TCN global pre-treinado no CirCor inteiro, porque remove o vazamento das segmentacoes `.tsv` dos pacientes de validacao do classificador.

A versao atual tambem evita usar os rotulos do fold externo para selecionar checkpoint da CNN, threshold ou calibracao Platt. Essas decisoes sao feitas em `cnn_tune`, um split interno criado apenas a partir dos pacientes de treino externo.

Ainda assim, o resultado continua sendo validacao interna no CirCor. A validacao externa continua sendo necessaria antes de interpretar o modelo como robusto fora deste dataset.
