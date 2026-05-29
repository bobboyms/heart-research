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

## Experimento 6 - Grupo B v3: contraste sistole menos diastole

Pasta:

```text
feature extraction/Grupo B v3 contraste sistole menos diastole/
```

Script:

```text
extract_systole_diastole_contrast_clusters.py
```

Ideia:

Testar diretamente a intuicao de usar a diastole como "ruido de fundo" da propria gravacao:

```text
contraste[f,t] = log(1 + |STFT(sistole)[f,t]|) - mediana_t(log(1 + |STFT(diastole)[f,t]|))
```

Depois resumir esse contraste por bandas e perfil de frequencia, gerar PCA/UMAP e rodar k-means.

Resultado principal:

| Nivel | Cluster enriquecido | % Present | Captura dos Present |
|---|---:|---:|---:|
| Gravacao global | 11 gravacoes | 100.0% | 1.8% |
| AV | 6 gravacoes | 100.0% | 3.9% |
| PV | 2 gravacoes | 100.0% | 1.4% |
| TV | 3 gravacoes | 100.0% | 2.1% |
| MV | 7 gravacoes | 85.7% | 3.5% |
| Paciente agregado | 7 pacientes | 100.0% | 3.9% |

Interpretacao:

O contraste puro separa um nucleo pequeno e extremamente limpo de sopros, principalmente casos acusticamente evidentes. Porem, como cluster nao supervisionado resumido, ele captura poucos `Present`.

Conclusao:

O Grupo B v3 confirma qualitativamente que a subtracao da diastole revela o excesso sistolico, mas nao supera o Grupo B v2 para separacao ampla. Ele e melhor visto como visualizacao demonstrativa do mecanismo `phase_contrast`, nao como o melhor conjunto de features exploratorias.

## Experimento 7 - Grupo B v3.1: contraste robusto por ciclo

Pasta:

```text
feature extraction/Grupo B v3.1 contraste robusto por ciclo/
```

Script:

```text
extract_robust_cycle_contrast_clusters.py
```

Ideia:

Incrementar o v3 para destacar mais pacientes com murmurio:

- normalizar o contraste pela variabilidade robusta da diastole (`MAD`);
- calcular features por ciclo sistolico;
- agregar por `mean`, `max`, `p90` e `top3_mean`;
- criar visoes separadas por banda (`low`, `mid`, `high`, `profile`, `all`);
- rodar sweep de k-means com `k = 2, 4, 6, 8, 10`.

Resultado principal paciente-level:

| Leitura | Pacientes no grupo | Present | % Present | Captura dos Present |
|---|---:|---:|---:|---:|
| low band, k=2, melhor cluster | 69 | 67 | 97.1% | 37.4% |
| mid band, k=6, clusters enriquecidos | 78 | 76 | 97.4% | 42.5% |
| low band, k=10, clusters enriquecidos | 109 | 95 | 87.2% | 53.1% |

Interpretacao:

O incremento funcionou. O v3 puro gerava um cluster perfeito mas pequeno demais (7/7 pacientes `Present`). O v3.1 preservou alta pureza e ampliou a captura, principalmente na banda baixa (`25-200 Hz`).

Conclusao:

O Grupo B v3.1 melhorou bastante o v3 puro e virou a referencia a bater para clusterizacao exploratoria ampla.

## Experimento 8 - Grupo B v3.2: murmur map realcado

Pasta:

```text
feature extraction/Grupo B v3.2 murmur map realcado/
```

Script:

```text
extract_enhanced_murmur_map_clusters.py
```

Ideia:

Realcar a regiao do murmurio antes de clusterizar:

- cortar 15% do inicio e do fim da sistole para reduzir vazamento de `S1` e `S2`;
- manter apenas contraste positivo `max(z, 0)`;
- suavizar levemente o mapa tempo-frequencia;
- aplicar threshold para remover ruido fraco;
- extrair features de persistencia temporal;
- usar STFT de maior resolucao na banda baixa;
- salvar um mapa compacto `16 x 32` da banda baixa e heatmaps comparativos.

Resultado principal paciente-level:

| Leitura | Pacientes no grupo | Present | % Present | Captura dos Present |
|---|---:|---:|---:|---:|
| low band, k=2, melhor cluster | 85 | 84 | 98.8% | 46.9% |
| low band, k=10, clusters enriquecidos | 101 | 98 | 97.0% | 54.7% |
| mid band, k=10, clusters enriquecidos | 102 | 93 | 91.2% | 52.0% |

Interpretacao:

O realce da regiao do murmurio melhorou o v3.1: o melhor grupo amplo ficou mais puro e capturou mais `Present`.

Conclusao:

O Grupo B v3.2 passa a ser o melhor experimento exploratorio de clusterizacao. A comparacao principal e:

| Experimento | Melhor grupo amplo | Present | % Present | Captura |
|---|---:|---:|---:|---:|
| Grupo B v3.1 | 109 pacientes | 95 | 87.2% | 53.1% |
| Grupo B v3.2 | 101 pacientes | 98 | 97.0% | 54.7% |

Otimização posterior:

Foi feito um sweep sobre as features paciente-level ja extraidas do v3.2, variando subconjunto de features, scaler, PCA e `k`. O melhor resultado amplo veio de `low_persistence` com `StandardScaler`:

| Configuracao | Pacientes selecionados | Present | % Present | Captura |
|---|---:|---:|---:|---:|
| v3.2 original, `low`, k=10 | 101 | 98 | 97.0% | 54.7% |
| v3.2 otimizado, `low_persistence`, PCA 50, k=10 | 105 | 102 | 97.1% | 57.0% |
| v3.2 otimizado, `low_persistence`, PCA 2, k=9 | 101 | 99 | 98.0% | 55.3% |

A leitura de cluster unico tambem melhorou: `low_persistence`, PCA 2, k=2 gerou 85 pacientes, 85 `Present`, 100.0% de pureza e 47.5% de captura.

O proximo passo natural e usar features combinadas B v2 + B v3.2 otimizado em baseline supervisionado simples.

## Experimento 9 - Grupo B v3.3: textura na banda baixa + separabilidade

Pasta:

```text
feature extraction/Grupo B v3.3 textura banda baixa e separabilidade/
```

Script:

```text
extract_lowband_texture_separability.py
```

Ideia (duas perguntas encadeadas):

1. Dentro da banda baixa (<=260 Hz), que extracoes alem de energia/persistencia (v3.2) descrevem o sopro? Adicionou-se o eixo de TEXTURA ruido-vs-tonal e forma espectral: flatness/entropia, tilt, sub-bandas finas (25-80/80-150/150-260 Hz), skew/kurtosis, Gini/esparsidade, flux, proxy de HNR e razao sistole/diastole.
2. Quanto um audio com sopro se afasta de um sem sopro? Camada explicita de SEPARABILIDADE: AUC/Cohen d por feature, Mahalanobis/Fisher/silhueta, e um score continuo de distancia de Mahalanobis ao centroide Absent.

O script nao recomputa o v3.2: le o CSV salvo, extrai so a textura nova e junta por `recording_id`. Roda a separabilidade em tres conjuntos (v3.2 / textura / combinado).

Resultado principal:

- A feature individual mais forte de todas e NOVA: `tex_gini_map_p90` (AUC 0.864, d=-1.53). Sopro = Gini baixo = energia espalhada/sustentada pela banda baixa; normal = concentrada/transiente. Bate a melhor feature do v3.2 (energia low-mid, AUC 0.834).
- A textura ADICIONA separacao multivariada, porem modesto: combinado Mahalanobis 3.02 vs v3.2 2.84; AUC dist-ao-Absent 0.851 vs 0.838 (nivel gravacao).
- O score `dist_to_absent` e clinicamente coerente: Absent 20.8 < I/VI 49.8 ~ II/VI 37.9 << III/VI 181.2. Reproduz o achado de que sopros suaves ficam colados no normal.

Ressalva: metricas multivariadas sao in-sample (sem CV); o `dist_to_absent_auc=1.0` no nivel paciente e artefato de p>>n (3152 features / ~850 pacientes) e deve ser ignorado. A AUC por feature e a leitura robusta.

Conclusao:

A "regiao de informacao" da banda baixa nao e so energia/persistencia: a textura (concentracao/Gini, variabilidade espectral) carrega o sinal mais forte por feature. A separacao Present x Absent agora tem metrica explicita e um score continuo de distancia-ao-normal alinhado ao grading. Proximo passo: levar o score (ou o subconjunto Gini+energia+persistencia+forma) a um baseline supervisionado com validacao por paciente.

## Comparacao geral

| Experimento | Tipo | Melhor resultado por paciente |
|---|---|---:|
| Grupo A | features classicas globais | 31.7% `Present` |
| Grupo B v1 | features por fase com valores absolutos | sem separacao clara |
| Grupo B v2 | features relativas por fase/local | 89.5% `Present` |
| Grupo B v3 | contraste puro sistole-diastole | 100.0% `Present`, mas so 7 pacientes |
| Grupo B v3.1 | contraste robusto por ciclo | 109 pacientes com 87.2% `Present` |
| Grupo B v3.2 | murmur map realcado | 101 pacientes com 97.0% `Present` |
| Grupo B v3.2 otimizado | persistencia em banda baixa | 105 pacientes com 97.1% `Present` |
| Grupo B v3.3 | textura banda baixa + separabilidade | Gini AUC 0.864 (feature mais forte); score dist-ao-Absent ordena por grading |
| Grupo C1 | PANNs global | 28.5% `Present` |
| Grupo C2 | PANNs por fase | 23.5% `Present` |

## Melhor experimento ate o momento

O melhor experimento exploratorio para clusterizacao ate agora foi:

```text
Grupo B v3.2: murmur map realcado
```

Motivo:

- depois da otimizacao, produziu na leitura `low_persistence`, PCA 50, k=10, clusters enriquecidos com 105 pacientes, 102 `Present` e 97.1% de pureza;
- capturou cerca de 57.0% dos pacientes `Present`, acima do Grupo B v3.1 e do v3.2 original;
- preservou uma leitura simples de alta pureza: `low_persistence`, PCA 2, k=9 teve 101 pacientes com 98.0% `Present`;
- melhorou o cluster unico: `low_persistence`, PCA 2, k=2 teve 85 pacientes com 100.0% `Present`;
- gerou heatmaps comparativos da regiao realcada do murmurio.

## Conclusao atual

A melhor direcao nao e usar o audio inteiro nem embeddings pre-treinados globais.

O sinal mais forte apareceu quando usamos features relativas por fase cardiaca, especialmente comparando a sistole com `S1`, `S2` e diastole.

O contraste puro sistole-diastole confirma o mecanismo acustico, mas como cluster manual ele detecta principalmente extremos. O contraste robusto por ciclo corrige isso em parte, e o murmur map realcado melhora mais a pureza/captura.

Portanto, o proximo passo recomendado e treinar um baseline supervisionado usando features combinadas do Grupo B v2 e do Grupo B v3.2 otimizado, com validacao correta por paciente.

Modelos sugeridos:

- regressao logistica com regularizacao;
- SVM linear;
- gradient boosting.

Validacao obrigatoria:

- split por paciente;
- `Unknown` fora do treino inicial;
- metricas: sensibilidade, especificidade, balanced accuracy, AUPRC e matriz de confusao.
