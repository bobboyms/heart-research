# Resumo dos experimentos de feature extraction

Este documento consolida os experimentos feitos ate agora para separar audios/pacientes com `Murmur = Present` e `Murmur = Absent` no dataset CirCor.

O objetivo inicial nao foi treinar um classificador final, mas testar quais grupos de features geram representacoes em que os casos com sopro ficam mais proximos entre si e mais distantes dos casos sem sopro.

## Dados usados

Dataset local:

```text
circor-heart-sound-1.0.3/
```

Arquivos usados:

- `training_data/*.wav`: audios de fonocardiograma;
- `training_data/*.tsv`: segmentacoes de fase cardiaca, quando o experimento usa fases;
- `training_data.csv`: metadata e rotulo `Murmur`.

Os campos de descricao direta do sopro nao foram usados como features:

- `Murmur locations`;
- `Most audible location`;
- `Systolic murmur timing`;
- `Systolic murmur shape`;
- `Systolic murmur grading`;
- `Systolic murmur pitch`;
- `Systolic murmur quality`;
- campos diastolicos equivalentes.

Esses campos poderiam vazar o rotulo.

Por padrao, os experimentos iniciais excluem `Murmur = Unknown`.

## Criterio de leitura dos resultados

Como estes experimentos sao exploratorios e nao supervisionados, a principal pergunta foi:

> Os clusters/projecoes ficam enriquecidos em `Murmur = Present`?

A taxa base de `Present` nos experimentos fica em torno de 20.5%. Portanto:

- um cluster com 20% `Present` nao separa nada;
- um cluster com 30% `Present` mostra sinal fraco;
- um cluster com 80% ou mais `Present` mostra um sinal forte e merece virar baseline supervisionado.

Importante: cluster enriquecido nao e classificador pronto. Ainda precisa validacao por paciente.

## Experimento 1 - Grupo A: features classicas por gravacao

Pasta:

```text
feature extraction/Grupo A features classicas por gravacao/
```

Script:

```text
extract_group_a_classical_features.py
```

Ideia:

Extrair features classicas do audio inteiro de cada gravacao, sem usar segmentacao por fase cardiaca.

Features usadas:

- duracao;
- RMS;
- pico absoluto;
- crest factor;
- zero-crossing rate;
- clipping aproximado;
- energia total;
- centroid, bandwidth, rolloff, flatness e entropia espectral;
- energia por bandas;
- MFCCs;
- delta MFCCs;
- log-mel.

Resultado principal:

| Nivel | Melhor grupo | % Present |
|---|---:|---:|
| Gravacao global | 1299 gravacoes | 29.6% |
| Paciente agregado | 350 pacientes | 31.7% |

Interpretacao:

O Grupo A captura algum sinal acustico, mas ele e fraco. O audio inteiro mistura S1, sistole, S2, diastole, ruido, local de ausculta e volume. Isso dilui o sinal do sopro.

Conclusao:

Grupo A e um baseline simples util, mas nao e o melhor caminho para detectar sopro.

## Experimento 2 - Grupo B v1: features segmentadas por fase cardiaca

Pasta:

```text
feature extraction/Grupo B features segmentadas por fase cardiaca/
```

Script:

```text
extract_group_b_features.py
```

Ideia:

Usar os `.tsv` para cortar cada audio em:

- `S1`;
- sistole;
- `S2`;
- diastole.

Depois calcular features acusticas dentro de cada fase.

Resultado principal:

| Nivel | Resultado |
|---|---|
| Gravacao global | clusters nao separaram bem `Present` e `Absent` |
| PCA2 | valores altos de `pca_2` tinham maior proporcao de `Present` |

Interpretacao:

O experimento mostrou que havia algum sinal associado a sopro, principalmente no eixo `pca_2`, mas os clusters ainda misturavam muito `Present` e `Absent`.

Problema identificado:

As features ainda continham energia absoluta, RMS, peak e `MFCC_1`, que podem refletir volume, ganho, contato do sensor ou qualidade da gravacao.

Conclusao:

Foi um bom experimento diagnostico, mas precisava de uma versao mais limpa com features relativas.

## Experimento 3 - Grupo B v2: features relativas por local

Pasta:

```text
feature extraction/Grupo B v2 features relativas por local/
```

Script:

```text
extract_relative_phase_features_by_location.py
```

Ideia:

Refazer o Grupo B focando em features relativas, para reduzir confundimento por volume.

Foram removidos:

- RMS;
- peak;
- energia absoluta;
- `MFCC_1`.

Foram mantidos:

- `systole / S1+S2`;
- `systole / diastole`;
- energia de alta frequencia na sistole / energia da sistole;
- deltas `systole - diastole`;
- deltas `systole - S1`;
- deltas `systole - S2`;
- MFCCs 2 a 13 apenas em forma de deltas;
- analise separada por local: `AV`, `PV`, `TV`, `MV`;
- agregacao por paciente com media e maximo entre locais.

Resultado principal:

| Nivel | Melhor grupo | % Present |
|---|---:|---:|
| Gravacao global | 301 gravacoes | 92.0% |
| AV | 58 gravacoes | 87.9% |
| PV | 84 gravacoes | 91.7% |
| TV | 67 gravacoes | 95.5% |
| MV | 79 gravacoes | 97.5% |
| Paciente agregado | 86 pacientes | 89.5% |

Interpretacao:

Este foi o primeiro experimento em que apareceu um grupo pequeno e fortemente enriquecido em `Murmur = Present`.

A melhora indica que a assinatura do sopro aparece melhor quando comparamos a sistole contra outras fases do mesmo ciclo/mesma gravacao, em vez de usar valores absolutos do audio inteiro.

Conclusao:

Grupo B v2 e o melhor experimento ate agora.

## Experimento 4 - Grupo C1: PANNs embeddings globais por gravacao

Pasta:

```text
feature extraction/Grupo C1 PANNs embeddings globais por gravacao/
```

Script:

```text
extract_panns_global_embeddings.py
```

Ideia:

Usar PANNs/Cnn14 pre-treinado em AudioSet como extrator congelado de embeddings do audio inteiro.

Processo:

- audio `.wav` inteiro;
- reamostragem para 32 kHz;
- janelas de 10 segundos com hop de 5 segundos;
- embedding PANNs por janela;
- pooling por gravacao com `mean`, `std` e `max`;
- agregacao por paciente com `mean` e `max`.

Resultado principal:

| Nivel | Melhor grupo | % Present |
|---|---:|---:|
| Gravacao global | 1212 gravacoes | 28.1% |
| Paciente agregado | 354 pacientes | 28.5% |

Interpretacao:

O PANNs global captura algum sinal, mas de forma fraca. O resultado ficou parecido ou inferior ao Grupo A.

Conclusao:

PANNs no audio inteiro nao superou as features manuais por fase.

## Experimento 5 - Grupo C2: PANNs embeddings por fase cardiaca

Pasta:

```text
feature extraction/Grupo C2 PANNs embeddings por fase cardiaca/
```

Script:

```text
extract_panns_phase_embeddings.py
```

Ideia:

Fazer o equivalente neural do Grupo B v2:

- cortar o audio em `S1`, sistole, `S2`, diastole usando `.tsv`;
- extrair embeddings PANNs de cada fase;
- comparar sistole contra as outras fases;
- criar deltas, deltas absolutos, distancias cosseno/L2 e razoes de norma;
- gerar PCA/UMAP por local;
- agregar por paciente.

Resultado principal:

| Nivel | Melhor grupo | % Present |
|---|---:|---:|
| Gravacao global | 192 gravacoes | 22.4% |
| PV | 28 gravacoes | 28.6% |
| MV | 79 gravacoes | 24.1% |
| Paciente agregado | 119 pacientes | 23.5% |

Observacao:

Em `TV`, apareceu um cluster com 100% `Present`, mas ele tinha apenas 1 gravacao. Isso nao deve ser interpretado como evidencia.

Interpretacao:

Segmentar PANNs por fase nao melhorou. O modelo provavelmente nao aprendeu representacoes adequadas para trechos curtos de fonocardiograma, ja que foi treinado em AudioSet, um dominio muito diferente.

Conclusao:

Grupo C2 nao foi promissor. Ficou pior que C1, Grupo A e Grupo B v2.

## Comparacao geral

| Experimento | Tipo | Melhor resultado por paciente |
|---|---|---:|
| Grupo A | features classicas globais | 31.7% `Present` |
| Grupo B v1 | features por fase com valores absolutos | sem separacao clara |
| Grupo B v2 | features relativas por fase/local | 89.5% `Present` |
| Grupo C1 | PANNs global | 28.5% `Present` |
| Grupo C2 | PANNs por fase | 23.5% `Present` |

## Melhor experimento ate o momento

O melhor experimento ate agora foi:

```text
Grupo B v2: features relativas por local
```

Motivo:

- produziu um cluster de 86 pacientes com 89.5% `Murmur = Present`;
- tambem produziu clusters por local fortemente enriquecidos em sopro;
- controlou melhor o efeito de volume ao remover RMS, peak, energia absoluta e `MFCC_1`;
- usou informacao clinicamente alinhada ao problema: diferencas entre sistole e outras fases cardiacas.

## Conclusao atual

A melhor direcao nao e usar o audio inteiro nem embeddings pre-treinados globais.

O sinal mais forte apareceu quando usamos features relativas por fase cardiaca, especialmente comparando a sistole com `S1`, `S2` e diastole.

Portanto, o proximo passo recomendado e treinar um baseline supervisionado usando as features do Grupo B v2, com validacao correta por paciente.

Modelos sugeridos:

- regressao logistica com regularizacao;
- SVM linear;
- gradient boosting.

Validacao obrigatoria:

- split por paciente;
- `Unknown` fora do treino inicial;
- metricas: sensibilidade, especificidade, balanced accuracy, AUPRC e matriz de confusao.

