# Grupo E - TCN segmentacao frame a frame

Este experimento treina uma Temporal Convolutional Network supervisionada para prever a fase cardiaca de cada frame do audio CirCor:

| Classe | Significado |
|---:|---|
| 0 | outro / nao anotado |
| 1 | S1 |
| 2 | sistole |
| 3 | S2 |
| 4 | diastole |

O script usa os `.wav` e `.tsv` locais em `circor-heart-sound-1.0.3/training_data/`. Cada frame recebe o rotulo do intervalo `.tsv` que contem o centro temporal do frame. O split e feito por grupo de paciente para evitar vazamento entre treino, validacao e teste: `Patient ID` e `Additional ID` sao unidos quando indicam o mesmo participante em campanhas diferentes.

## Execucao

Teste rapido:

```bash
uv run "modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py" --max-recordings 40 --epochs 1 --batch-size 4
```

Treino completo recomendado para o MacBook Pro M3 com 18 GB:

```bash
uv run "modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py" --epochs 30 --batch-size 8 --device auto
```

Treino curto para gerar um novo checkpoint com a versao nao causal e rotulagem por sobreposicao:

```bash
caffeinate -dimsu uv run "modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py" \
  --epochs 10 \
  --batch-size 42 \
  --device mps \
  --output-dir "modeling/Grupo E TCN segmentacao frame a frame/outputs_noncausal_overlap"
```

Por padrao, o treino usa janelas de 6 s com hop de 3 s, loss `CE + Dice`, TCN nao causal, rotulagem por maior sobreposicao entre frame e intervalo `.tsv`, e pos-processamento temporal nas metricas. Para voltar ao comportamento antigo mais proximo da primeira versao, use:

```bash
uv run "modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py" --causal --label-mode center --train-window-seconds 0 --loss ce --no-postprocess
```

Durante a execucao, o script mostra barras de progresso para preparacao das features, treino por batch e avaliacao por batch. Para logs mais limpos em execucoes automatizadas, desative com `--no-progress`.

Se MPS ficar instavel ou sem memoria, use CPU:

```bash
uv run "modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py" --epochs 30 --batch-size 8 --device cpu
```

## Inferencia em audio novo

Para gerar uma segmentacao `.tsv` predita a partir de um `.wav`:

```bash
uv run "modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py" \
  --checkpoint "modeling/Grupo E TCN segmentacao frame a frame/outputs_focal_dice/best_model.pt" \
  --predict-wav "circor-heart-sound-1.0.3/training_data/13918_AV.wav" \
  --predict-output "circor-heart-sound-1.0.3/training_data/13918_AV.predicted.tsv"
```

Para inferencia em um unico audio, CPU costuma ser mais rapido/estavel que MPS neste caminho:

```bash
uv run "modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py" \
  --device cpu \
  --checkpoint "modeling/Grupo E TCN segmentacao frame a frame/outputs_focal_dice/best_model.pt" \
  --predict-wav "circor-heart-sound-1.0.3/training_data/13918_AV.wav" \
  --predict-output "circor-heart-sound-1.0.3/training_data/13918_AV.predicted.tsv"
```

O arquivo `--predict-output` tem o formato compatível com os `.tsv` do CirCor:

```text
start_time    end_time    label
```

O script tambem escreve um CSV lateral com confianca por segmento, com sufixo `_segments_with_confidence.csv`. Para salvar probabilidades por frame:

```bash
uv run "modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py" \
  --checkpoint "modeling/Grupo E TCN segmentacao frame a frame/outputs_focal_dice/best_model.pt" \
  --predict-wav "circor-heart-sound-1.0.3/training_data/13918_AV.wav" \
  --predict-frame-output "modeling/Grupo E TCN segmentacao frame a frame/outputs/13918_AV_frames.csv"
```

## Saidas

As saidas ficam em:

```text
modeling/Grupo E TCN segmentacao frame a frame/outputs/
```

Principais arquivos:

- `best_model.pt`: checkpoint com pesos, normalizacao e configuracao.
- `summary.md`: resumo do split, configuracao e metricas.
- `metrics.json`: metricas completas de validacao e teste.
- `training_history.csv`: perda e metricas por epoca.
- `training_curves.png`: curvas de treino.
- `val_confusion_matrix.csv` / `test_confusion_matrix.csv`: matrizes de confusao.
- `val_confusion_matrix.png` / `test_confusion_matrix.png`: visualizacoes das matrizes.
- `cache/`: features log-mel por gravacao para acelerar novas execucoes.

## Metricas

O script reporta:

- acuracia frame-level;
- macro F1;
- weighted F1;
- balanced accuracy;
- mean IoU;
- precision, recall, F1, IoU e suporte por classe;
- matriz de confusao para validacao e teste.

## Ajustes de segmentacao

- `--train-window-seconds`: tamanho das janelas curtas usadas no treino. Padrao: `6`.
- `--train-window-hop-seconds`: hop entre janelas. Padrao: `3`.
- `--loss`: `ce`, `ce_dice`, `focal` ou `focal_dice`. Padrao: `ce_dice`.
- `--dice-weight`: peso do termo Dice quando usado. Padrao: `0.5`.
- `--causal` / `--no-causal`: escolhe TCN causal ou nao causal. Padrao: nao causal.
- `--label-mode`: `overlap` usa maior sobreposicao do frame com o intervalo `.tsv`; `center` usa apenas o centro temporal do frame.
- `--boundary-ignore-ms`: ignora frames muito perto das fronteiras anotadas durante o treino.
- `--postprocess`: aplica suavizacao temporal nas predicoes antes das metricas.
- `--median-filter-frames`: filtro mediano nos rotulos previstos. Padrao: `5`.
- `--min-segment-frames`: remove segmentos previstos muito curtos. Padrao: `3`.
