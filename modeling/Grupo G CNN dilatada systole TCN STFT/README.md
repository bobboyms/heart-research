# Grupo G - CNN dilatada em STFT da sistole predita por TCN

Este experimento testa uma hipotese mais direta:

```text
o sopro no CirCor e majoritariamente sistolico
=> o modelo deve olhar principalmente para a sistole
```

Pipeline:

```text
audio PV/TV
=> TCN prediz S1 / systole / S2 / diastole
=> extrai apenas os trechos de systole
=> concatena systoles da gravacao
=> gera STFT log-magnitude
=> CNN 1D com convolucoes dilatadas no tempo, usando bins da STFT como canais
=> probabilidade de Murmur Present
=> agregacao por paciente usando max entre gravacoes
```

## Execucao

Smoke test:

```bash
uv run "modeling/Grupo G CNN dilatada systole TCN STFT/train_systole_stft_dilated_cnn.py" \
  --max-recordings 80 \
  --epochs 2 \
  --patience 1 \
  --batch-size 8 \
  --device cpu
```

Treino completo inicial:

```bash
caffeinate -dimsu uv run "modeling/Grupo G CNN dilatada systole TCN STFT/train_systole_stft_dilated_cnn.py" \
  --epochs 50 \
  --patience 8 \
  --batch-size 32 \
  --device cpu
```

## Saidas

As saidas ficam em:

```text
modeling/Grupo G CNN dilatada systole TCN STFT/outputs/
```

Arquivos principais:

- `summary.md`: resumo e metricas paciente-level.
- `recording_metadata.csv`: gravacoes usadas e estatisticas dos trechos sistolicos.
- `recording_oof_predictions.csv`: predicoes out-of-fold por gravacao.
- `patient_oof_predictions.csv`: predicoes out-of-fold por paciente.
- `fold_metrics.csv`: metricas por fold.
- `training_history.csv`: historico de treino.
- `precision_recall_oof.png`: curva precision-recall.
- `fold_*_best_model.pt`: checkpoints por fold.
