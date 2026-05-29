# Grupo F - TCN + attention segmentacao frame a frame

Segundo teste supervisionado para segmentacao frame a frame do CirCor. Ele reaproveita o pipeline do Grupo E, mas troca a cabeca do modelo para:

```text
log-mel + deltas
  -> TCN causal dilatada residual
  -> self-attention temporal causal local
  -> Conv1D 1x1 para 5 classes
```

Classes:

| Classe | Significado |
|---:|---|
| 0 | outro / nao anotado |
| 1 | S1 |
| 2 | sistole |
| 3 | S2 |
| 4 | diastole |

## Arquitetura padrao

- Entrada: `80 x n_frames`, com 40 log-mel bins + 40 deltas.
- TCN: 7 blocos residuais com convolucoes causais dilatadas.
- Dilatacoes: `1, 2, 4, 8, 16, 32, 64`.
- Canais internos: `96`.
- Kernel temporal: `5`.
- Self-attention: 1 bloco temporal causal local.
- Cabecas de attention: `4`.
- Janela de attention: `65` frames, cerca de 0.65 s com hop de 10 ms.
- Batches agrupados por duracao por padrao, para reduzir padding e custo da attention.
- Limite padrao de frames preenchidos por batch: `12000`, calculado como `maior_sequencia_do_batch * quantidade_de_gravacoes`.
- Treino por janelas curtas de 6 s com hop de 3 s.
- Loss padrao: `CE + Dice`.
- Pos-processamento temporal padrao nas metricas.
- Saida: `5 x n_frames`.

A attention e local para evitar custo quadratico em gravacoes longas. Cada frame atende apenas aos frames anteriores dentro da janela e ao frame atual.

O split e feito por grupo de paciente para evitar data leak: `Patient ID` e `Additional ID` sao unidos quando indicam o mesmo participante em campanhas diferentes.

## Execucao

Teste rapido:

```bash
uv run "modeling/Grupo F TCN attention segmentacao frame a frame/train_tcn_attention_frame_segmenter.py" --max-recordings 40 --epochs 1 --batch-size 4 --device cpu
```

Treino completo recomendado no MacBook Pro M3 com 18 GB:

```bash
uv run "modeling/Grupo F TCN attention segmentacao frame a frame/train_tcn_attention_frame_segmenter.py" --epochs 30 --batch-size 8 --device auto
```

Se MPS ficar instavel ou sem memoria, use CPU ou reduza batch/attention:

```bash
uv run "modeling/Grupo F TCN attention segmentacao frame a frame/train_tcn_attention_frame_segmenter.py" --epochs 30 --batch-size 4 --attention-window 65 --device cpu
```

Se um batch ficar muito lento no MPS, interrompa com `Ctrl+C` e rode:

```bash
uv run "modeling/Grupo F TCN attention segmentacao frame a frame/train_tcn_attention_frame_segmenter.py" --epochs 30 --batch-size 4 --max-frames-per-batch 8000 --attention-window 65 --device mps
```

## Saidas

As saidas ficam em:

```text
modeling/Grupo F TCN attention segmentacao frame a frame/outputs/
```

Principais arquivos:

- `best_model.pt`: checkpoint do melhor modelo por macro F1 de validacao.
- `summary.md`: configuracao, split, distribuicao dos rotulos e metricas finais.
- `metrics.json`: metricas completas.
- `training_history.csv`: metricas por epoca.
- `training_curves.png`: curvas de treino.
- `val_confusion_matrix.csv` / `test_confusion_matrix.csv`: matrizes de confusao.
- `val_confusion_matrix.png` / `test_confusion_matrix.png`: visualizacoes das matrizes.
- `split_manifest.json`: auditoria do split por paciente.

Depois de uma execucao interrompida, o script reutiliza por padrao `normalization.json` e `train_label_counts.json` se eles ja existirem. Isso evita reler e descomprimir todos os `.npz` do cache antes de voltar ao treino. Para forcar recomputacao, use `--no-reuse-stats` ou `--overwrite-cache`.

Para preparar cache e estatisticas sem iniciar o treinamento:

```bash
uv run "modeling/Grupo F TCN attention segmentacao frame a frame/train_tcn_attention_frame_segmenter.py" --prepare-only
```

## Ajustes de segmentacao

- `--train-window-seconds`: tamanho das janelas curtas usadas no treino. Padrao: `6`.
- `--train-window-hop-seconds`: hop entre janelas. Padrao: `3`.
- `--loss`: `ce`, `ce_dice`, `focal` ou `focal_dice`. Padrao: `ce_dice`.
- `--dice-weight`: peso do termo Dice quando usado. Padrao: `0.5`.
- `--postprocess`: aplica suavizacao temporal nas predicoes antes das metricas.
- `--median-filter-frames`: filtro mediano nos rotulos previstos. Padrao: `5`.
- `--min-segment-frames`: remove segmentos previstos muito curtos. Padrao: `3`.
- Para voltar ao comportamento antigo: `--train-window-seconds 0 --loss ce --no-postprocess`.
