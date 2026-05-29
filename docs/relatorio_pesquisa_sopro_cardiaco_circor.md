# Pesquisa preliminar: deteccao de sopro cardiaco em fonocardiogramas usando segmentacao sistolica e deep learning

## Resumo executivo

Este documento resume uma linha de pesquisa experimental para detectar `Murmur = Present` versus `Murmur = Absent` no dataset CirCor DigiScope Phonocardiogram Dataset v1.0.3.

A hipotese central e que, neste dataset, a maior parte da informacao acustica relevante para sopro esta concentrada na fase sistolica. Em vez de treinar um classificador diretamente sobre o audio completo, a abordagem proposta primeiro segmenta o audio em fases cardiacas, extrai apenas os trechos de `systole`, transforma esses trechos em uma representacao tempo-frequencia via STFT e entao treina uma CNN 1D com convolucoes dilatadas e atencao temporal para prever a probabilidade de sopro.

O melhor resultado atual, em validacao out-of-fold por paciente usando todos os locais de ausculta (`AV`, `PV`, `TV`, `MV`), foi:

| Modelo | AUROC | AUPRC | Sensibilidade | Especificidade | Precisao | F1 |
|---|---:|---:|---:|---:|---:|---:|
| CNN dilatada + atencao temporal, probabilidade bruta, threshold 0.5 | 0.910 | 0.845 | 0.888 | 0.722 | 0.454 | 0.601 |
| CNN dilatada + atencao temporal, threshold Youden por fold | 0.910 | 0.845 | 0.826 | 0.890 | 0.662 | 0.735 |
| CNN dilatada + atencao temporal + calibracao Platt, threshold 0.5 | 0.912 | 0.840 | 0.680 | 0.965 | 0.834 | 0.749 |

Esses resultados indicam que o modelo tem bom poder de ranking para separar pacientes com e sem sopro. A calibracao de probabilidade tornou o threshold `0.5` mais conservador e mais preciso, reduzindo falsos positivos de `190` para `24`, ao custo de reduzir a sensibilidade.

Este ainda nao e um modelo clinico pronto. Os resultados devem ser interpretados como prova de conceito interna no CirCor, exigindo validacao externa, analise de erro e calibracao adicional antes de qualquer uso pratico.

## Motivacao

Fonocardiogramas contem uma mistura de eventos acusticos: `S1`, sistole, `S2`, diastole, ruido, diferencas de contato do sensor, variacao de volume e variacao por local de ausculta. Um modelo treinado no audio inteiro pode aprender correlacoes espurias ou diluir o sinal do sopro em trechos pouco informativos.

No CirCor, os experimentos exploratorios indicaram que a assinatura mais forte de `Murmur = Present` aparece quando a sistole e comparada contra outras fases do ciclo cardiaco. Isso motivou uma arquitetura em duas etapas:

1. Segmentar automaticamente a fase cardiaca, especialmente a sistole.
2. Classificar o sopro usando apenas a informacao acustica dos trechos sistolicos.

Essa estrategia tambem e coerente com a distribuicao observada das anotacoes do dataset: a maioria dos casos `Murmur = Present` tem descricao sistolica.

## Dataset

Dataset local:

```text
circor-heart-sound-1.0.3/
```

Arquivos utilizados:

- `training_data/*.wav`: audios de fonocardiograma.
- `training_data/*.tsv`: segmentacoes de fase cardiaca usadas nos experimentos iniciais e no treinamento do segmentador.
- `training_data.csv`: metadados e rotulo `Murmur`.

Rotulo supervisionado:

```text
Murmur = Present versus Murmur = Absent
```

Os casos `Murmur = Unknown` foram excluidos dos experimentos iniciais de classificacao binaria.

Campos descritivos diretos do sopro, como `Murmur locations`, `Systolic murmur timing`, `Systolic murmur grading` e variaveis equivalentes, nao foram usados como features, pois poderiam vazar informacao do rotulo.

No melhor experimento deep learning com todos os locais:

| Item | Valor |
|---|---:|
| Gravacoes usadas | 2880 |
| Pacientes usados | 861 |
| Pacientes `Present` | 178 |
| Pacientes `Absent` | 683 |
| Locais incluidos | AV, PV, TV, MV |

A validacao foi feita em nivel paciente, ou seja, gravacoes do mesmo paciente nao foram divididas entre treino e validacao no mesmo fold.

## Preparacao do dataset e prevencao de data leak

Esta secao descreve especificamente como os dados sao preparados para o treino do modelo `Grupo G CNN dilatada systole TCN STFT` e quais medidas foram usadas para reduzir vazamento de informacao entre treino e validacao.

### 1. Selecionar apenas pacientes com rotulo binario

O arquivo `training_data.csv` e carregado usando `Patient ID` como string, para evitar problemas com IDs numericos.

O script filtra explicitamente:

```python
metadata = metadata.loc[metadata["Murmur"].isin(["Present", "Absent"])].copy()
```

Portanto, o treino binario usa apenas:

- `Murmur = Present`;
- `Murmur = Absent`.

Os pacientes `Murmur = Unknown` sao excluidos. Isso evita misturar uma classe incerta com os negativos ou positivos e evita ensinar o modelo com rotulos ambiguos.

### 2. Remover campos que vazam diretamente informacao do sopro

O modelo final nao usa campos descritivos do sopro como entrada. Campos como:

- `Murmur locations`;
- `Most audible location`;
- `Systolic murmur timing`;
- `Systolic murmur shape`;
- `Systolic murmur grading`;
- `Systolic murmur pitch`;
- `Systolic murmur quality`;
- variaveis diastolicas equivalentes.

nao entram no classificador. Esses campos sao anotacoes derivadas da presenca/caracteristica do sopro e poderiam tornar o problema artificialmente facil.

No `Grupo G`, o input efetivo do classificador e apenas:

```text
audio .wav -> sistole predita pelo TCN -> STFT log-magnitude
```

O rotulo `Murmur` e usado somente como target supervisionado.

### 3. Mapear gravacoes para pacientes e locais

Cada arquivo `.wav` tem nome no formato aproximado:

```text
<patient_id>_<location>.wav
```

Exemplo:

```text
13918_AV.wav
```

O script extrai:

```text
patient_id = 13918
location = AV
```

Cada gravacao vira um item com:

- `recording_id`;
- `patient_id`;
- `location`;
- caminho do `.wav`;
- rotulo `Murmur` do paciente.

No experimento final foram aceitos os locais:

```text
AV, PV, TV, MV
```

### 4. Nao usar o `.tsv` real no classificador final

Nos experimentos exploratorios, os `.tsv` reais foram usados para entender se a fase cardiaca era importante. No classificador final, o objetivo e simular um audio novo, onde nao existe `.tsv`.

Por isso, no `Grupo G`, o classificador nao corta a sistole a partir do `.tsv` real. Ele usa segmentos preditos pelo TCN:

```text
.wav -> TCN -> predicted.tsv -> extracao da sistole
```

O arquivo cacheado tem nome como:

```text
<recording_id>.predicted.tsv
```

e contem apenas:

```text
start_time, end_time, label
```

Esses segmentos sao preditos pelo modelo de segmentacao, nao copiados do `.tsv` manual durante o treino do classificador.

### 5. Extrair somente trechos de sistole

O TCN prediz labels de fase cardiaca:

```text
other, S1, systole, S2, diastole
```

O classificador usa apenas o label de sistole:

```python
LABEL_SYSTOLE = 2
```

Para cada gravacao, todos os intervalos preditos como sistole sao recortados do audio e concatenados. Se a gravacao nao tiver sistole suficiente, ela e descartada pelo criterio minimo:

```text
min_systole_seconds = 0.10
```

Isso evita alimentar o classificador com espectrogramas vazios ou quase vazios.

### 6. Preprocessamento do audio

O audio e preparado de forma deterministica:

1. Se o `.wav` tiver mais de um canal, os canais sao convertidos para mono por media.
2. Se o audio vier como inteiro, ele e normalizado pela faixa do dtype.
3. O sinal e centralizado removendo a media.
4. Os trechos de sistole sao reamostrados para `4000 Hz`.

Depois e gerada uma STFT log-magnitude:

| Parametro | Valor |
|---|---:|
| `target_sample_rate` | 4000 Hz |
| `n_fft` | 128 |
| `hop_length` | 32 |
| `high_hz` | 1000 Hz |
| `max_frames` | 256 |
| `freq_bins` | 33 |

A representacao final por gravacao tem formato:

```text
[33 freq_bins, 256 frames]
```

Se a STFT tiver mais de `256` frames, o centro e recortado. Se tiver menos, o espectrograma e preenchido com zeros ate `256` frames.

Esse passo nao usa rotulos nem estatisticas globais ajustadas no dataset inteiro; e uma transformacao deterministica do audio.

### 7. Cache sem ajuste estatistico global

O script pode cachear os espectrogramas para acelerar experimentos:

```text
spectrogram_cache/
```

O cache armazena:

- espectrograma da sistole;
- tempo total de sistole;
- numero de segmentos sistolicos.

Esse cache nao contem nenhuma normalizacao aprendida no dataset inteiro. A normalizacao estatistica usada pelo modelo e feita depois, dentro de cada fold, usando apenas as gravacoes de treino daquele fold.

### 8. Split por paciente, nao por gravacao

Este e o ponto principal contra data leak.

O modelo nao divide gravacoes aleatoriamente. Primeiro ele cria uma tabela unica por paciente:

```python
patient_table = meta.drop_duplicates("patient_id")[["patient_id", "target"]]
```

Depois separa os pacientes positivos e negativos:

```python
pos = patient_ids[y_patient == 1]
neg = patient_ids[y_patient == 0]
```

Os IDs de pacientes sao embaralhados com seed fixa e distribuidos nos folds de forma estratificada. Assim, cada fold recebe uma proporcao semelhante de `Present` e `Absent`.

Quando um paciente cai no fold de validacao, todas as gravacoes dele entram na validacao:

```python
val_mask = meta["patient_id"].isin(val_patient_ids)
val_idx = indices das gravacoes dos pacientes de validacao
train_idx = indices das demais gravacoes
```

Isso impede o vazamento mais comum em datasets de audio medico: uma gravacao do mesmo paciente no treino e outra gravacao do mesmo paciente na validacao.

Sem esse cuidado, o modelo poderia aprender caracteristicas do paciente, do sensor, do ambiente ou da aquisicao, em vez de aprender sopro.

### 9. Normalizacao calculada apenas no treino de cada fold

Antes de treinar cada fold, o script calcula media e desvio padrao apenas nos espectrogramas de treino:

```python
train_mean = specs[train_indices].mean()
train_std = specs[train_indices].std()
```

Depois aplica esses valores tanto no treino quanto na validacao:

```python
train_ds = (train_specs - train_mean) / train_std
val_ds = (val_specs - train_mean) / train_std
```

Isso evita data leak estatistico. A validacao nao contribui para a media nem para o desvio padrao usados na normalizacao.

### 10. Agregacao por paciente feita dentro do fold

O modelo produz probabilidade por gravacao. Para avaliar em nivel paciente, as probabilidades das gravacoes do mesmo paciente sao agregadas com `max`:

```text
patient_probability = max(probabilidades das gravacoes do paciente)
```

Essa agregacao e feita separadamente dentro dos indices do fold. Em outras palavras:

- pacientes de treino sao agregados usando apenas gravacoes de treino;
- pacientes de validacao sao agregados usando apenas gravacoes de validacao.

Isso evita misturar probabilidades de um mesmo paciente entre treino e validacao.

### 11. Threshold escolhido sem olhar os rotulos da validacao

O threshold ajustado por Youden e escolhido usando apenas pacientes do treino de cada fold:

```python
threshold = choose_threshold(train_patient["target"], train_patient["prob"])
```

Depois esse threshold e aplicado no fold de validacao.

Isso e importante porque escolher o melhor threshold diretamente no conjunto de validacao inflaria artificialmente as metricas.

### 12. Calibracao Platt aprendida apenas no treino do fold

A calibracao de probabilidade tambem e ajustada apenas no conjunto de treino de cada fold:

```python
calibrator = fit_platt_calibrator(train_patient["target"], train_patient["prob"])
```

Depois a transformacao calibrada e aplicada nos pacientes da validacao:

```python
prob_calibrated_val = apply_platt_calibrator(prob_val, calibrator)
```

Assim, a probabilidade calibrada OOF de um paciente nao usa o rotulo daquele paciente para ajustar a calibracao.

### 13. Protocolo mais rigoroso implementado: Grupo H nested

Depois dos experimentos iniciais com um TCN global fixo, foi implementado um protocolo mais rigoroso chamado:

```text
Grupo H Nested TCN CNN systole
```

Nesse protocolo, o TCN nao e treinado uma unica vez no dataset inteiro. Em vez disso, para cada fold do classificador de sopro, um novo TCN e treinado apenas nos pacientes de treino daquele fold:

```text
para cada fold do classificador:
    separar pacientes de treino e validacao
    treinar um TCN novo usando wav + tsv somente dos pacientes de treino
    usar esse TCN para predizer sistole nos pacientes de treino
    usar esse TCN para predizer sistole nos pacientes de validacao
    treinar a CNN de sopro somente com os pacientes de treino
    avaliar a CNN somente nos pacientes de validacao
```

Assim, o TCN do fold nunca ve os `.tsv` dos pacientes que serao usados para validar o classificador de sopro naquele fold.

Isso corrige o principal acoplamento metodologico dos experimentos anteriores: antes, o segmentador TCN era tratado como um modelo pre-treinado fixo no mesmo dominio CirCor. Ele nao usava o rotulo `Murmur`, portanto nao havia vazamento direto do alvo de sopro, mas ele podia ter visto a estrutura temporal anotada dos pacientes avaliados depois pelo classificador.

No Grupo H, esse acoplamento e reduzido porque cada TCN ve apenas as segmentacoes dos pacientes de treino do seu fold.

Os artefatos sao salvos por fold:

```text
outputs_nested/
  fold_1/
    train_patient_ids.txt
    val_patient_ids.txt
    tcn/
      best_model.pt
      summary.md
      metrics.json
    predicted_tsvs/
      <recording_id>.predicted.tsv
    cnn/
      fold_1_best_model.pt

  fold_2/
    ...
```

O arquivo final `patient_oof_predictions.csv` consolida as predicoes out-of-fold por paciente. O arquivo `threshold_metrics_by_fold.md` salva, para cada fold, as metricas em varios thresholds para probabilidade bruta e calibrada.

Essa versao ainda continua sendo validacao interna no CirCor. A validacao externa continua sendo necessaria para demonstrar robustez fora do dataset.

Outra limitacao: o treino da CNN usa o fold de validacao para early stopping via AUPRC paciente-level. Isso e comum em validacao cruzada, mas significa que as metricas OOF medem desempenho de selecao interna, nao substituem um conjunto de teste externo totalmente intocado.

### 14. Resumo das protecoes contra data leak

| Risco de vazamento | Medida usada |
|---|---|
| Mesmo paciente em treino e validacao | Split por `patient_id`, nao por gravacao |
| Campos que descrevem o sopro usados como feature | Campos de descricao do sopro nao entram no modelo |
| Casos `Unknown` contaminando classe negativa ou positiva | `Unknown` excluido do treino binario |
| Uso do `.tsv` real no classificador final | Classificador usa sistole predita pelo TCN |
| Estatisticas da validacao na normalizacao | Media/desvio calculados apenas no treino do fold |
| Threshold escolhido na validacao | Threshold Youden escolhido no treino do fold |
| Calibracao usando validacao | Platt scaling ajustado no treino do fold |
| Multiplas gravacoes do mesmo paciente | Agregacao e split feitos em nivel paciente |
| TCN vendo `.tsv` de pacientes de validacao do classificador | Grupo H treina um TCN separado por fold, usando apenas pacientes de treino |

## Evolucao dos experimentos

### Grupo A: features classicas globais

O primeiro baseline extraiu features acusticas do audio inteiro, sem segmentacao por fase cardiaca. Foram usadas estatisticas como duracao, RMS, pico, crest factor, zero-crossing rate, descritores espectrais, energia por bandas, MFCCs e log-mel.

Resultado exploratorio:

| Nivel | Melhor grupo | Percentual `Present` |
|---|---:|---:|
| Gravacao global | 1299 gravacoes | 29.6% |
| Paciente agregado | 350 pacientes | 31.7% |

Interpretacao: o audio inteiro contem algum sinal, mas a separacao e fraca. A mistura de fases cardiacas, volume, ruido e local de ausculta dilui a assinatura do sopro.

### Grupo B v2: features relativas por local

O Grupo B v2 removeu features absolutas de volume/energia e passou a usar features relativas entre fases. Exemplos:

- razoes `systole / S1+S2`;
- razoes `systole / diastole`;
- energia relativa de alta frequencia na sistole;
- deltas `systole - diastole`;
- deltas `systole - S1`;
- deltas `systole - S2`;
- MFCCs em forma de deltas.

Resultado exploratorio:

| Nivel | Melhor grupo | Percentual `Present` |
|---|---:|---:|
| Gravacao global | 301 gravacoes | 92.0% |
| AV | 58 gravacoes | 87.9% |
| PV | 84 gravacoes | 91.7% |
| TV | 67 gravacoes | 95.5% |
| MV | 79 gravacoes | 97.5% |
| Paciente agregado | 86 pacientes | 89.5% |

Interpretacao: este foi o primeiro resultado forte. A assinatura do sopro apareceu de forma mais clara quando a sistole foi comparada contra outras fases da mesma gravacao.

### Grupo B v2 com segmentacao TCN predita

O Grupo B v2 original depende dos `.tsv` reais. Para aproximar o uso em audio novo, foi treinado um TCN frame-level para predizer fases cardiacas diretamente do audio.

O TCN original e nao causal e preserva a dimensao temporal ate o final, pois precisa emitir uma classe por frame. Depois dos experimentos iniciais, foi adicionada uma opcao de atencao temporal contextual:

```bash
--pooling attention
```

No TCN, essa atencao nao e um pooling final que reduz a sequencia. Ela funciona como contexto global:

```text
TCN features [batch, channels, frames]
-> atencao aprende peso temporal por frame
-> gera contexto global ponderado
-> injeta esse contexto de volta em todos os frames
-> Conv1d final prediz classe por frame
```

Portanto, a saida continua frame-a-frame:

```text
[batch, 5 classes, frames]
```

Resultado do segmentador TCN no teste:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.7068 |
| Balanced accuracy | 0.7572 |
| Macro F1 | 0.7113 |
| Mean IoU | 0.5524 |
| F1 sistole | 0.7194 |
| Recall sistole | 0.7809 |
| F1 diastole | 0.7458 |

Quando o Grupo B v2 foi refeito usando segmentacoes preditas pelo TCN, os locais `PV` e `TV` mantiveram clusters muito enriquecidos em sopro. Isso sugeriu que o pipeline automatico podia preservar parte importante do sinal originalmente observado com `.tsv` reais.

### Grupo F: MLP com features Grupo B v2 PV+TV

Foi treinado um modelo supervisionado tabular usando features do Grupo B v2 extraidas com segmentacao TCN predita, filtrando apenas `PV` e `TV`, e agregando por paciente com `mean` e `max`.

| Item | Valor |
|---|---:|
| Pacientes | 740 |
| Present | 148 |
| Absent | 592 |
| Features | 382 |
| Modelo | MLP 256, 128, 64 |
| Validacao | 5 folds por paciente |

Resultado out-of-fold:

| Threshold | AUROC | AUPRC | Sensibilidade | Especificidade | Precisao | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.876 | 0.807 | 0.797 | 0.872 | 0.608 | 0.690 |
| Youden por fold | 0.876 | 0.807 | 0.791 | 0.885 | 0.632 | 0.703 |

Interpretacao: o MLP supervisionado confirmou que as features relativas baseadas na sistole carregavam informacao preditiva real, nao apenas estrutura de cluster.

### Grupo G: STFT da sistole + CNN dilatada

O experimento Grupo G substituiu as features manuais por uma representacao neural direta da sistole.

Pipeline:

```text
.wav
-> TCN frame-level segmenter
-> trechos preditos de systole
-> concatenacao dos trechos sistolicos da gravacao
-> STFT log-magnitude
-> CNN 1D com convolucoes dilatadas
-> agregacao por paciente usando max entre gravacoes
-> probabilidade de Murmur Present
```

Arquitetura base:

```text
Input: [batch, 33 freq_bins, 256 frames]

Conv1d(33 -> 16, kernel=3)
GroupNorm
GELU

DilatedBlock dilation=1
DilatedBlock dilation=2
DilatedBlock dilation=4
DilatedBlock dilation=8

Pooling temporal
Dropout
Linear(16 -> 1)
Sigmoid
```

Resultado usando todos os locais:

| Modelo | AUROC | AUPRC | Sensibilidade @0.5 | Especificidade @0.5 | F1 @0.5 |
|---|---:|---:|---:|---:|---:|
| CNN dilatada com average pooling | 0.894 | 0.815 | 0.843 | 0.722 | 0.579 |
| CNN dilatada com atencao temporal | 0.910 | 0.845 | 0.888 | 0.722 | 0.601 |

Interpretacao: a atencao temporal melhorou o poder de ranking e a deteccao de pacientes `Present`, sem alterar a entrada nem a segmentacao.

### Calibracao de probabilidade

Como o ranking do modelo era bom, mas o threshold fixo `0.5` gerava muitos falsos positivos, foi aplicada calibracao Platt por fold.

A calibracao foi aprendida apenas no conjunto de treino de cada fold e aplicada nos pacientes da validacao daquele fold. Isso evita calibrar a probabilidade usando o proprio rotulo OOF do paciente avaliado.

Comparacao no threshold `0.5`:

| Probabilidade | AUROC | AUPRC | Sensibilidade | Especificidade | Precisao | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bruta | 0.910 | 0.845 | 0.888 | 0.722 | 0.454 | 0.601 | 190 | 20 |
| Calibrada Platt | 0.912 | 0.840 | 0.680 | 0.965 | 0.834 | 0.749 | 24 | 57 |

Tambem houve melhora no Brier score:

| Probabilidade | Brier score |
|---|---:|
| Bruta | 0.1693 |
| Calibrada Platt | 0.0720 |

Interpretacao: a calibracao tornou a probabilidade muito mais conservadora e mais util no threshold `0.5`, reduzindo falsos positivos de forma substancial. O custo foi a reducao da sensibilidade.

### Grupo H: validacao nested TCN + CNN

O Grupo H foi criado para testar o pipeline de forma mais rigorosa contra vazamento de segmentacao.

Nos experimentos anteriores, a CNN usava sistoles preditas por um TCN global pre-treinado no CirCor. Esse TCN nao usava o rotulo de sopro, mas podia ter visto os `.tsv` dos mesmos pacientes depois avaliados pelo classificador.

No Grupo H, cada fold treina seu proprio TCN:

```text
Fold k
-> pacientes de treino do fold treinam TCN_k usando wav + tsv
-> TCN_k prediz sistole em treino e validacao
-> CNN_k treina apenas nos pacientes de treino usando sistole predita por TCN_k
-> CNN_k avalia apenas nos pacientes de validacao usando sistole predita por TCN_k
```

Com `5` folds, o experimento treina:

```text
5 TCNs
5 CNNs
```

O comando recomendado usa atencao contextual no TCN e atencao temporal na CNN:

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
  --cnn-device mps \
  --pooling attention \
  --calibration platt \
  --output-dir "modeling/Grupo H Nested TCN CNN systole/outputs_nested"
```

Esse experimento e mais caro computacionalmente, mas e metodologicamente mais limpo. Os resultados finais devem ser analisados a partir de:

```text
modeling/Grupo H Nested TCN CNN systole/outputs_nested/summary.md
modeling/Grupo H Nested TCN CNN systole/outputs_nested/patient_oof_predictions.csv
modeling/Grupo H Nested TCN CNN systole/outputs_nested/threshold_metrics_by_fold.md
```

## Achados principais

1. Features globais do audio inteiro foram insuficientes para uma boa separacao.
2. A comparacao da sistole contra outras fases cardiacas produziu o primeiro sinal forte.
3. A segmentacao automatica por TCN preservou informacao util o suficiente para substituir os `.tsv` reais em experimentos posteriores.
4. Modelos supervisionados em nivel paciente melhoraram a utilidade dos achados exploratorios.
5. A CNN dilatada usando apenas STFT da sistole atingiu desempenho semelhante ou superior ao MLP com features manuais.
6. A atencao temporal melhorou AUROC e AUPRC.
7. A calibracao Platt reduziu fortemente falsos positivos no threshold `0.5`, mas tornou o modelo menos sensivel.
8. Foi implementado um protocolo nested por paciente no Grupo H, no qual o TCN de cada fold nao ve os `.tsv` dos pacientes de validacao do classificador.
9. O TCN agora aceita atencao contextual opcional via `--pooling attention`, preservando a saida frame-a-frame.

## Interpretacao atual

O melhor modelo para ranking de risco e a CNN dilatada com atencao temporal sobre STFT da sistole:

```text
AUROC = 0.910
AUPRC = 0.845
```

Para uma configuracao equilibrada entre sensibilidade e especificidade, o threshold Youden por fold sobre a probabilidade bruta foi o melhor compromisso:

```text
Sensibilidade = 0.826
Especificidade = 0.890
Precisao = 0.662
F1 = 0.735
```

Para uma configuracao conservadora, com maior precisao e menos falsos positivos, a probabilidade calibrada em `0.5` foi superior:

```text
Sensibilidade = 0.680
Especificidade = 0.965
Precisao = 0.834
F1 = 0.749
```

Assim, o modelo pode ser ajustado para dois objetivos diferentes:

- triagem sensivel: usar threshold menor ou threshold Youden, aceitando mais falsos positivos;
- predicao conservadora: usar probabilidade calibrada com threshold `0.5`, aceitando menor sensibilidade.

## Limitacoes

Estes resultados ainda sao preliminares.

Limitacoes importantes:

- A validacao foi interna no CirCor, sem dataset externo.
- O TCN segmentador e o classificador foram desenvolvidos no mesmo dominio de dados. O Grupo H reduz o vazamento de segmentacao dentro da validacao cruzada, mas ainda nao substitui validacao externa.
- A calibracao Platt tambem precisa ser validada externamente.
- Casos `Murmur = Unknown` nao foram modelados como classe propria.
- Ainda nao ha analise detalhada por idade, qualidade de audio, campanha de coleta, local de ausculta e tipo de sopro.
- O modelo usa agregacao por paciente com `max`, que e simples e pode ser substituida por uma estrategia multi-instance mais robusta.
- A atencao temporal ainda nao foi explorada qualitativamente para verificar se esta focando em regioes acusticamente plausiveis.
- O resultado nao deve ser interpretado como evidencia de validade clinica.

## Proximos passos recomendados

1. Avaliar o modelo em um conjunto externo ou em um split temporal/institucional, se disponivel.
2. Fazer analise de erro dos falsos positivos e falsos negativos.
3. Visualizar a atencao temporal sobre a sistole e comparar com trechos onde o sopro e audivel.
4. Testar thresholds calibrados para objetivos especificos: alta sensibilidade, alta especificidade ou equilibrio.
5. Testar modelo multi-instance por ciclo sistolico em vez de concatenar todos os trechos sistolicos.
6. Incluir local de ausculta como entrada explicita do modelo.
7. Comparar contra um modelo treinado no audio inteiro para quantificar o ganho especifico de focar na sistole.
8. Repetir o experimento em `PV+TV` com atencao temporal e calibracao para comparar contra todos os locais.
9. Avaliar inclusao de `Murmur = Unknown` como classe separada ou como grupo de incerteza.
10. Executar e analisar o Grupo H completo com `5` folds para comparar contra o Grupo G com TCN global.
11. Criar um protocolo de inferencia final para audio novo: segmentacao TCN, extracao de sistole, STFT, predicao, calibracao e agregacao por paciente.
12. Testar um segmentador treinado em dataset externo, caso exista um dataset PCG com segmentacao de fases cardiacas.

## Arquivos principais

Segmentador TCN:

```text
modeling/Grupo E TCN segmentacao frame a frame/train_tcn_frame_segmenter.py
modeling/Grupo E TCN segmentacao frame a frame/outputs_noncausal_overlap/summary.md
```

O TCN aceita:

```text
--pooling none
--pooling attention
```

No modo `attention`, a atencao gera contexto temporal global e injeta esse contexto nos frames antes da classificacao frame-a-frame.

Features relativas com TCN:

```text
feature extraction/Grupo B v2 features relativas por local com TCN predito/
```

MLP supervisionado com features PV+TV:

```text
modeling/Grupo F MLP sopro PV TV TCN features/train_patient_mlp_murmur.py
modeling/Grupo F MLP sopro PV TV TCN features/outputs/summary.md
```

CNN dilatada com STFT da sistole:

```text
modeling/Grupo G CNN dilatada systole TCN STFT/train_systole_stft_dilated_cnn.py
modeling/Grupo G CNN dilatada systole TCN STFT/outputs_all_locations_attention_calibrated_mps/summary.md
```

Comparacoes salvas:

```text
modeling/Grupo G CNN dilatada systole TCN STFT/outputs_all_locations_attention_mps/comparacao_attention_vs_avg.md
modeling/Grupo G CNN dilatada systole TCN STFT/outputs_all_locations_attention_calibrated_mps/comparacao_calibracao_platt.md
modeling/Grupo G CNN dilatada systole TCN STFT/outputs_all_locations_attention_calibrated_mps/threshold_metrics_by_fold.md
```

Validacao nested:

```text
modeling/Grupo H Nested TCN CNN systole/train_nested_tcn_systole_cnn.py
modeling/Grupo H Nested TCN CNN systole/README.md
modeling/Grupo H Nested TCN CNN systole/outputs_nested/
```

## Conclusao

A direcao mais promissora ate agora e tratar a deteccao de sopro como um problema guiado por fase cardiaca: primeiro localizar a sistole, depois classificar o conteudo acustico dessa fase.

Os resultados sugerem que a combinacao `TCN segmentador -> STFT da sistole -> CNN dilatada -> atencao temporal -> calibracao` e uma arquitetura compacta, interpretavel em termos de pipeline e promissora para pesquisa adicional.

O ponto mais importante e que o modelo nao depende dos `.tsv` reais durante a inferencia. Para um novo audio, ele pode primeiro predizer as fases com o TCN, extrair a sistole e gerar a probabilidade de sopro em nivel paciente.

Antes de qualquer aplicacao clinica, a prioridade deve ser validacao externa, analise dos erros e definicao explicita do objetivo operacional: triagem sensivel ou predicao conservadora com alta especificidade.
