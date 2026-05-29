# Nested TCN + Systole CNN

Pacote reorganizado para o experimento que antes ficava concentrado em
`modeling/Grupo H Nested TCN CNN systole/train_nested_tcn_systole_cnn.py`.

## Execucao

Novo entrypoint:

```bash
uv run python -m nested_tcn_systole_cnn.train
```

O wrapper antigo continua funcionando:

```bash
uv run "modeling/Grupo H Nested TCN CNN systole/train_nested_tcn_systole_cnn.py"
```

Exemplo com peso extra para sistole no TCN:

```bash
uv run python -m nested_tcn_systole_cnn.train --tcn-systole-weight-multiplier 2.0
```

## Experimentos nomeados

Para rodar varios experimentos comparaveis, use `--run-name`:

```bash
uv run python -m nested_tcn_systole_cnn.train \
  --run-name tcn_weight_2 \
  --tcn-systole-weight-multiplier 2.0
```

Quando `--run-name` e usado sem `--output-dir`, o pipeline cria uma pasta unica em:

```text
experiments/nested_tcn_systole_cnn/<run_name>__YYYYMMDD_HHMMSS/
```

Cada experimento salva:

- `config.json`
- `metrics_summary.json`
- `patient_oof_predictions.csv`
- `fold_metrics.csv`
- `training_history.csv`
- `summary.md`

O registry consolidado fica em:

```text
experiments/nested_tcn_systole_cnn/registry.csv
experiments/nested_tcn_systole_cnn/registry.jsonl
```

O `mean_score` padrao e a media de sensibilidade, especificidade, precisao/PPV
e F1-score. Para dar mais peso a sensibilidade:

```bash
uv run python -m nested_tcn_systole_cnn.train \
  --run-name high_sensitivity \
  --score-weights sensitivity=2,specificity=1,precision=1,f1=1
```

## Dashboard estatico

Gere o HTML comparativo com:

```bash
uv run python -m nested_tcn_systole_cnn.dashboard
```

Isso cria:

```text
experiments/nested_tcn_systole_cnn/dashboard.html
```

O HTML ja contem os dados embutidos a partir do `registry.csv`, entao pode ser
aberto direto no navegador sem servidor local. Depois de novos treinos, rode o
comando do dashboard novamente para atualizar o HTML.

Por padrao, ao concluir cada fold o pipeline remove artefatos intermediarios
grandes daquele fold, como `tcn_dataset_train_patients/`, `predicted_tsvs/`,
`spectrogram_cache/` e `tcn/cache/`. Checkpoints, metricas, configuracoes,
plots e CSVs de resultado sao preservados.

Para manter esses artefatos durante uma depuracao:

```bash
uv run python -m nested_tcn_systole_cnn.train --no-cleanup-fold-artifacts
```

## Parametros aceitos

| Grupo | Parametro | Padrao | Valores/observacao |
|---|---|---:|---|
| Dados/saida | `--dataset-dir` | `circor-heart-sound-1.0.3` | Pasta local do dataset CirCor. |
| Dados/saida | `--output-dir` | automatico | Pasta de saida. Se omitido com `--run-name`, cria pasta unica em `--experiments-dir`. |
| Dados/saida | `--locations` | `AV PV TV MV` | Lista entre `AV`, `PV`, `TV`, `MV`. |
| Experimento | `--run-name` | `None` | Nome legivel do experimento; ativa saida nomeada no registry. |
| Experimento | `--experiments-dir` | `experiments/nested_tcn_systole_cnn` | Pasta do registry e dos experimentos nomeados. |
| Experimento | `--score-weights` | `sensitivity=1,specificity=1,precision=1,f1=1` | Pesos do `mean_score`. Chaves aceitas: `sensitivity`, `specificity`, `precision`, `f1`. |
| Validacao | `--folds` | `5` | Numero de folds por paciente. |
| Validacao | `--seed` | `42` | Semente aleatoria. |
| Validacao | `--max-patients` | `None` | Limita pacientes para smoke tests ou testes rapidos. |
| Cache/limpeza | `--force-retrain-tcn` | `False` | Retreina o TCN mesmo se houver checkpoint compativel. |
| Cache/limpeza | `--overwrite-cache` | `False` | Recria caches de espectrogramas/features. |
| Cache/limpeza | `--cleanup-fold-artifacts` / `--no-cleanup-fold-artifacts` | `True` | Remove artefatos grandes do fold apos coletar resultados. |
| Execucao | `--progress` / `--no-progress` | `True` | Mostra/oculta barras de progresso. |
| TCN | `--tcn-epochs` | `10` | Epocas do TCN por fold. |
| TCN | `--tcn-batch-size` | `42` | Batch size do TCN. |
| TCN | `--tcn-device` | `mps` | `auto`, `cpu`, `mps`. |
| TCN | `--tcn-val-size` | `0.15` | Fracao de validacao interna do TCN. |
| TCN | `--tcn-test-size` | `0.15` | Fracao de teste interno do TCN. |
| TCN | `--tcn-pooling` | `none` | `none`, `attention`. |
| TCN | `--tcn-boundary-ignore-ms` | `0.0` | Ignora frames proximos de fronteiras de fase. Deve ser >= 0. |
| TCN | `--tcn-systole-weight-multiplier` | `1.0` | Multiplicador do peso da classe sistole. Deve ser > 0. |
| TCN | `--tcn-target-mode` | `cardiac-phase` | `cardiac-phase`, `systole-binary`. |
| TCN | `--other-mode` / `--tcn-other-mode` | `keep` | `keep`, `ignore`. Define tratamento do label original 0. |
| CNN | `--cnn-epochs` | `50` | Epocas maximas da CNN. |
| CNN | `--cnn-patience` | `8` | Early stopping patience. |
| CNN | `--cnn-batch-size` | `32` | Batch size da CNN. |
| CNN | `--cnn-inner-val-size` | `0.15` | Split interno `cnn_tune`. Deve estar entre 0 e 0.5. |
| CNN | `--cnn-device` | `mps` | `auto`, `cpu`, `mps`. |
| CNN | `--pooling` | `attention` | `avg`, `attention`. |
| CNN | `--calibration` | `platt` | `none`, `platt`. |
| CNN | `--decision-threshold` | `0.5` | Threshold operacional para probabilidade calibrada. |
| CNN | `--weak-murmur-weight` | `1.0` | Peso de perda para `Present` com grading I/VI. Deve ser > 0. |
| CNN | `--moderate-murmur-weight` | `1.0` | Peso de perda para `Present` com grading II/VI. Deve ser > 0. |
| CNN | `--location-aware-calibration` | `False` | Calibra probabilidade paciente-level usando probabilidades por local. |
| CNN augmentation | `--ltsrr-prob` | `0.0` | Probabilidade de aplicar Local Time-Frequency Spectrum Random Replacement em cada espectrograma de treino da CNN. `0` desativa. |
| CNN augmentation | `--ltsrr-k` | `4` | Numero de segmentos temporais nao sobrepostos usados pelo LTSRR. |
| CNN augmentation | `--ltsrr-frequency-ratio` | `0.25` | Fracao dos bins de frequencia substituida dentro de cada segmento temporal do LTSRR. |
| CNN augmentation | `--ltsrr-minority-only` | `False` | Aplica LTSRR apenas em amostras positivas/`Present` durante o treino da CNN. |
| CNN augmentation | `--smote-minority-augmentation` | `False` | Gera espectrogramas sinteticos da classe minoritaria com SMOTE apenas no treino da CNN. Incompativel com `--patient-mil-attention`. |
| CNN augmentation | `--smote-k-neighbors` | `5` | Numero de vizinhos minoritarios considerados pelo SMOTE. |
| CNN augmentation | `--smote-target-ratio` | `1.0` | Razao alvo minoria/maioria apos SMOTE. `1.0` balanceia o split `cnn_fit`. |
| CNN loss | `--loss` | `bce` | Funcao de perda da CNN: `bce` ou `focal`. |
| CNN loss | `--focal-gamma` | `2.0` | Parametro de foco usado quando `--loss focal`. Deve ser >= 0. |
| CNN loss | `--focal-alpha` | `None` | Peso opcional da classe positiva quando `--loss focal`. Deve estar entre 0 e 1. |
| CNN loss | `--auc-loss-weight` | `0.0` | Peso do termo auxiliar Pairwise AUC Ranking Loss somado a BCE/Focal. `0` desativa. |
| CNN loss | `--auc-loss-margin` | `1.0` | Margem usada no termo Pairwise AUC Ranking Loss. Deve ser >= 0. |
| CNN loss | batches estratificados | automatico | Quando `--auc-loss-weight > 0`, o treino da CNN usa batches com exemplos positivos e negativos sempre que o split contem as duas classes. |
| CNN | `--lr` | `0.0008` | Learning rate. |
| CNN | `--weight-decay` | `0.0003` | Weight decay. |
| CNN | `--base-channels` | `16` | Canais base do encoder. |
| CNN | `--dropout` | `0.25` | Dropout da CNN. |
| CNN | `--dilations` | `1,2,4,8` | Dilatacoes temporais separadas por virgula. |
| CNN | `--encoder-block` | `residual` | `residual`, `multiscale`. |
| MIL | `--patient-mil-attention` | `False` | Treina fusao paciente-level por AV/PV/TV/MV. |
| MIL | `--mil-max-instances` | `8` | Opcao mantida por compatibilidade. Deve ser > 0. |
| MIL | `--mil-location-embedding-dim` | `4` | Opcao mantida por compatibilidade. Deve ser > 0. |
| MIL | `--mil-instance-loss-weight` | `0.25` | Peso auxiliar por local. Deve ser >= 0. |
| STFT/sistole | `--target-sample-rate` | `4000` | Sample rate alvo. |
| STFT/sistole | `--n-fft` | `128` | Janela FFT. |
| STFT/sistole | `--hop-length` | `32` | Hop da STFT. |
| STFT/sistole | `--low-hz` | `0.0` | Frequencia minima. Deve ser >= 0. |
| STFT/sistole | `--high-hz` | `1000.0` | Frequencia maxima. Deve ser > `--low-hz`. |
| Espectrograma CNN | `--spectrogram-type` | `stft` | Representacao de entrada da CNN: `stft` ou `log-mel`. |
| Espectrograma CNN | `--n-mels` | `64` | Numero de bins Mel quando `--spectrogram-type log-mel`. |
| STFT/fase | `--max-frames` | `256` | Numero maximo de frames no espectrograma. |
| STFT/fase | `--cnn-phase-mode` | `systole` | `systole`, `diastole`, `both`. Define se a CNN recebe apenas sistole, apenas diastole, ou sistole+diastole concatenadas. `diastole`/`both` exigem `--tcn-target-mode cardiac-phase`. |
| STFT/fase | `--min-systole-seconds` | `0.10` | Duracao minima do audio de fase selecionado antes da STFT. Mantido com este nome por compatibilidade. |
| STFT/sistole | `--systole-threshold` | `None` | Threshold de probabilidade do TCN; se omitido, usa argmax. |
| STFT/sistole | `--systole-margin-ms` | `0.0` | Expande segmentos selecionados em ms. Deve ser >= 0. |

Parametros do dashboard:

| Parametro | Padrao | Observacao |
|---|---:|---|
| `--experiments-dir` | `experiments/nested_tcn_systole_cnn` | Pasta onde esta o `registry.csv`. |
| `--output` | `<experiments-dir>/dashboard.html` | Caminho do HTML gerado. |

## Organizacao

- `train.py`: entrypoint de treinamento.
- `cli.py`: argumentos e validacoes.
- `pipeline.py`: orquestracao do treino nested por paciente.
- `artifacts.py`: limpeza de artefatos intermediarios por fold.
- `data.py`: preparacao e splits do dataset.
- `evaluation.py`: metricas OOF, colunas de predicao e resumo.
- `experiment_registry.py`: registry CSV/JSONL e resumo comparavel por run.
- `scoring.py`: calculo do score medio ponderado.
- `dashboard.py`: gerador de dashboard HTML estatico.
- `models/tcn_segmenter.py`: adaptador da arquitetura TCN.
- `models/systole_cnn.py`: adaptador da arquitetura CNN sistolica.

## Testes

```bash
uv run python -m unittest discover -s tests -p 'test_nested_tcn_systole_cnn*.py'
```
