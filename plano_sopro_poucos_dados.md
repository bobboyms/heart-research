# Plano para detectar sopro cardiaco com poucos dados

## Objetivo

Queremos decidir como abordar a deteccao de sopro cardiaco no CirCor antes de treinar modelos mais complexos. A decisao inicial deve ser guiada por tres frentes:

1. Entender o rotulo e os vieses do dataset.
2. Extrair caracteristicas clinicamente interpretaveis e embeddings de audio.
3. Comparar baselines simples, com validacao correta por paciente, antes de fine-tuning profundo.

A tarefa inicial recomendada e binaria: `Murmur = Present` versus `Murmur = Absent`. Os casos `Unknown` devem ficar fora do primeiro treino e depois virar conjunto de analise de incerteza. O campo `Outcome` nao deve ser tratado como sinonimo de sopro, porque representa o diagnostico cardiologico global.

## Fontes lidas

Resumo consolidado:

- [papers/resumos_papers.md](papers/resumos_papers.md)

Papers escolhidos para leitura mais aprofundada:

- [PANNs: Large-Scale Pretrained Audio Neural Networks](papers/audio_feature_extraction_sota_2020_2026/2020_PANNs_large_scale_pretrained_audio_neural_networks.md)
- [BYOL for Audio](papers/audio_feature_extraction_sota_2020_2026/2021_BYOL_A_self_supervised_general_purpose_audio_representation.md)
- [Audio-MAE: Masked Autoencoders that Listen](papers/audio_feature_extraction_sota_2020_2026/2022_AudioMAE_masked_autoencoders_that_listen.md)
- [BEATs: Audio Pre-Training with Acoustic Tokenizers](papers/audio_feature_extraction_sota_2020_2026/2023_BEATs_audio_pretraining_with_acoustic_tokenizers.md)
- [EAT: Efficient Audio Transformer](papers/audio_feature_extraction_similarity_2024_2026/2024_EAT_self_supervised_pretraining_efficient_audio_transformer.md)
- [Temporal Pooling Strategies for Training-Free Anomalous Sound Detection](papers/audio_feature_extraction_similarity_2024_2026/2026_temporal_pooling_training_free_anomalous_sound_detection_embeddings.md)
- [Evaluation and Management of Heart Murmurs in Children](papers/clinical_heart_murmur_studies/2022_heart_murmurs_children_evaluation_management_aafp.md)
- [Auscultation While Standing](papers/clinical_heart_murmur_studies/2017_auscultation_while_standing_pathologic_murmur_children.md)
- [Dicionario de variaveis do CirCor](circor-heart-sound-1.0.3/DICIONARIO_DE_VARIAVEIS.md)

## O que os dados mostram agora

Levantamento local em `circor-heart-sound-1.0.3/training_data.csv`:

| Item | Valor |
|---|---:|
| Pacientes | 942 |
| Audios `.wav` | 3163 |
| Segmentacoes `.tsv` | 3163 |
| Duracao total aproximada | 20.09 horas |
| Duracao mediana por audio | 21.46 s |
| Frequencia observada nos exemplos | 4000 Hz |

Distribuicao de `Murmur`:

| Rotulo | Pacientes |
|---|---:|
| `Absent` | 695 |
| `Present` | 179 |
| `Unknown` | 68 |

Distribuicao de `Outcome`:

| Rotulo | Pacientes |
|---|---:|
| `Normal` | 486 |
| `Abnormal` | 456 |

Cruzamento critico:

| `Murmur` | `Outcome` | Pacientes |
|---|---|---:|
| `Absent` | `Normal` | 432 |
| `Absent` | `Abnormal` | 263 |
| `Present` | `Abnormal` | 150 |
| `Present` | `Normal` | 29 |
| `Unknown` | `Abnormal` | 43 |
| `Unknown` | `Normal` | 25 |

Conclusao: detectar sopro e detectar anormalidade cardiologica sao tarefas relacionadas, mas diferentes. Misturar as duas no inicio criaria ruido de rotulo.

Entradas de local de ausculta disponiveis em `Recording locations:`:

| Local | Entradas |
|---|---:|
| `MV` | 861 |
| `AV` | 800 |
| `PV` | 766 |
| `TV` | 732 |
| `Phc` | 4 |

Entre pacientes com `Murmur = Present`, o local mais audivel foi:

| Local | Pacientes |
|---|---:|
| `PV` | 62 |
| `TV` | 56 |
| `MV` | 42 |
| `AV` | 19 |

Caracteristicas do sopro nos positivos:

| Campo | Distribuicao principal |
|---|---|
| Timing sistolico | `Holosystolic`: 101; `Early-systolic`: 59; `Mid-systolic`: 17 |
| Shape sistolico | `Plateau`: 111; `Decrescendo`: 34; `Diamond`: 31 |
| Grading sistolico | `I/VI`: 104; `III/VI`: 46; `II/VI`: 28 |
| Pitch sistolico | `Low`: 87; `Medium`: 49; `High`: 42 |
| Quality sistolica | `Harsh`: 96; `Blowing`: 78; `Musical`: 4 |

As segmentacoes `.tsv` cobrem S1, sistole, S2 e diastole. Isso e importante porque a maior parte dos sopros positivos e sistolica. Portanto, caracteristicas por fase do ciclo cardiaco devem ser prioridade.

## Cuidados contra vazamento de informacao

Nao usar como features no modelo primario:

- `Murmur locations`
- `Most audible location`
- `Systolic murmur timing`
- `Systolic murmur shape`
- `Systolic murmur grading`
- `Systolic murmur pitch`
- `Systolic murmur quality`
- variaveis diastolicas equivalentes

Esses campos sao derivados da anotacao do sopro e vazam o rotulo. Eles devem ser usados para analise, estratificacao de erro e talvez tarefas secundarias depois que a deteccao de sopro estiver estabelecida.

Tambem precisamos dividir treino/teste por paciente, nao por audio. Audios do mesmo paciente em locais diferentes sao correlacionados. Se o mesmo paciente cair em treino e teste, a metrica fica artificialmente alta.

## Licoes dos papers

### Clinica

Os estudos clinicos reforcam que sopros inocentes e patologicos podem compartilhar sinais acusticos, e que a interpretacao depende de timing, local de maior intensidade, posicao corporal, grau, qualidade e achados associados. Para este dataset, nao temos posicao corporal dinamica como "em pe" versus decubito, mas temos:

- local de ausculta;
- segmentacao do ciclo cardiaco;
- anotacoes de timing, intensidade, pitch e qualidade para analise dos positivos;
- idade, sexo, peso e altura.

Isso sugere que o primeiro modelo deve ser de deteccao de sopro audivel, nao de decisao clinica final.

### Poucos dados

PANNs mostra que modelos pre-treinados em AudioSet transferem bem para tarefas com poucos exemplos. O ponto mais relevante para este projeto: com menos de 10 exemplos por classe, usar o modelo como extrator congelado pode superar fine-tuning completo; com mais dados, fine-tuning tende a melhorar.

BYOL-A mostra que aprendizado auto-supervisionado com augmentations de audio pode gerar representacoes uteis sem labels, usando log-mel, mixup de fundo e random resize crop. Isso e relevante se quisermos adaptar embeddings usando todos os audios CirCor, inclusive sem usar `Murmur`.

Audio-MAE, BEATs e EAT apontam para a mesma direcao pratica: representacoes modernas de audio sao aprendidas sobre espectrogramas com masking forte, patches e objetivos auto-supervisionados. Para nosso caso, a melhor primeira aposta nao e treinar um desses modelos do zero, mas usar checkpoints prontos como extratores, ou fazer adaptacao leve se houver capacidade computacional.

O paper de pooling temporal e especialmente importante: media temporal pode apagar desvios curtos. Sopro pode ser local no ciclo cardiaco, entao devemos comparar `mean pooling`, `max pooling`, estatisticas de media+desvio, pooling por fase cardiaca e pooling tipo RDP/GeM nos embeddings frame-level.

## Analises que devemos fazer antes de modelar

### 1. EDA tabular e clinica

Gerar tabelas e graficos:

- `Murmur` por `Age`, `Sex`, `Height`, `Weight`, `Campaign`.
- `Murmur` por quantidade de locais gravados.
- `Murmur` por `Outcome`, apenas para entender divergencias.
- Distribuicao de locais auscultados por paciente.
- Distribuicao de duracao dos audios por classe e local.
- Casos `Unknown`: comparar com `Present` e `Absent` para ver se sao ruidosos, intermediarios ou mal gravados.

Perguntas que essa etapa deve responder:

- Existe vies forte por campanha?
- Alguma faixa etaria domina os positivos?
- A ausencia de certos locais de ausculta esta correlacionada com o rotulo?
- `Unknown` parece mais perto de `Present` ou `Absent`?

### 2. Qualidade e pre-processamento do audio

Para cada `.wav`, calcular:

- duracao;
- RMS;
- pico absoluto;
- clipping;
- razao sinal/ruido aproximada;
- energia por bandas;
- espectrogramas amostrais por local e classe;
- consistencia entre `.hea`, `.wav` e `.tsv`.

Pre-processamento inicial recomendado:

- resample apenas se o extrator exigir, porque os audios locais estao em 4000 Hz e varios modelos prontos esperam 16000 Hz;
- filtro passa-faixa para features classicas, por exemplo 25-800 Hz ou 25-1000 Hz, mantendo uma versao sem filtro para embeddings pre-treinados;
- normalizacao por gravacao para reduzir variacao de ganho;
- segmentacao em ciclos usando `.tsv`.

### 3. Analise baseada em ciclo cardiaco

Como os `.tsv` marcam `S1`, sistole, `S2` e diastole, devemos extrair features por fase:

- duracao media e variancia de S1, sistole, S2 e diastole;
- energia total por fase;
- razao energia sistole / energia S1+S2;
- razao energia diastole / energia S1+S2;
- energia de alta frequencia na sistole;
- MFCC/log-mel por fase;
- entropia espectral por fase;
- variacao ciclo-a-ciclo.

Para sopro sistolico, a hipotese mais direta e: positivos devem apresentar energia/estrutura espectral adicional durante a sistole em relacao a S1/S2 e diastole.

### 4. Visualizacoes diagnosticas

Criar paineis por paciente:

- waveform;
- log-mel;
- anotacoes S1/sistole/S2/diastole sobrepostas;
- local de ausculta;
- rotulo `Murmur`;
- se positivo, local mais audivel e timing anotado.

Esses paineis devem ser feitos para:

- positivos tipicos;
- negativos tipicos;
- `Unknown`;
- falsos positivos e falsos negativos dos baselines.

## Features candidatas

### Grupo A: features classicas por gravacao

Baixo custo, interpretaveis e boas para baseline:

- duracao;
- RMS, peak, crest factor;
- zero-crossing rate;
- centroid, bandwidth, rolloff;
- spectral flatness;
- entropia espectral;
- energia por bandas, por exemplo 25-80, 80-200, 200-400, 400-800 Hz;
- MFCCs com media, desvio, percentis;
- delta MFCC;
- log-mel com estatisticas por tempo.

### Grupo B: features segmentadas por fase

Mais alinhadas ao problema:

- features do Grupo A calculadas separadamente em S1, sistole, S2 e diastole;
- diferencas sistole - diastole;
- razoes sistole / sons cardiacos;
- maximos locais durante sistole;
- proporcao de ciclos com pico anormal na sistole;
- estabilidade do padrao em ciclos repetidos.

### Grupo C: embeddings pre-treinados

Extrair embeddings congelados por janela e por gravacao:

- PANNs/CNN14 como baseline forte e relativamente simples.
- BEATs ou EAT como extratores modernos de representacao.
- Audio-MAE se checkpoint e pipeline estiverem faceis de usar.
- BYOL-A como alternativa leve ou ponto de partida para adaptacao auto-supervisionada.

Para poucos dados, a ordem recomendada e:

1. Extrator congelado + regressao logistica/SVM.
2. Extrator congelado + pooling melhor.
3. Fine-tuning parcial apenas se os baselines indicarem ganho potencial.

### Grupo D: agregacao por paciente

O rotulo e por paciente, mas os audios sao por local. Precisamos testar agregacoes:

- concatenar estatisticas por local (`AV`, `PV`, `TV`, `MV`);
- media dos embeddings entre locais;
- max pooling entre locais;
- atencao simples ou multiple instance learning;
- usar o maior score entre locais como score do paciente;
- modelo hierarquico: predicao por gravacao seguida de agregador por paciente.

Como o local mais audivel nos positivos se concentra em `PV`, `TV` e `MV`, a estrategia por local pode ser mais informativa do que misturar tudo cedo.

## Baselines recomendados

### Baseline 0: sanidade sem audio

Modelo com apenas metadados permitidos:

- `Age`, `Sex`, `Height`, `Weight`, `Pregnancy status`, `Campaign`, quantidade de locais gravados.

Objetivo: medir vies tabular. Se esse baseline for muito forte, precisamos investigar risco de confounding.

### Baseline 1: features classicas

Features globais por gravacao, agregadas por paciente. Modelos:

- regressao logistica com regularizacao;
- random forest;
- gradient boosting;
- SVM linear/RBF.

Esse baseline deve ser interpretavel e rapido.

### Baseline 2: features por fase cardiaca

Usar `.tsv` para extrair features por S1, sistole, S2 e diastole. Esse deve ser o baseline clinicamente mais importante.

Comparar:

- features globais vs segmentadas;
- somente sistole vs todas as fases;
- razoes sistole/S1/S2 vs features absolutas.

### Baseline 3: embeddings congelados

Extrair embeddings por janelas de audio e treinar apenas classificador leve:

- regressao logistica;
- SVM;
- kNN/prototipos por classe para inspecao de similaridade.

Comparar pooling:

- mean;
- max;
- mean+std;
- GeM;
- RDP ou aproximacao por pesos de desvio temporal;
- pooling separado por fase cardiaca.

### Baseline 4: hibrido

Combinar:

- features segmentadas interpretaveis;
- embeddings congelados;
- metadados permitidos.

Manter regularizacao forte para evitar overfitting.

### Baseline 5: fine-tuning controlado

So avancar para fine-tuning se os embeddings congelados forem promissores. Estrategia:

- congelar a maior parte do encoder;
- treinar cabeca pequena;
- liberar ultimas camadas se houver ganho consistente;
- usar augmentations conservadoras: ruido leve, ganho, time shift pequeno, SpecAugment leve, mixup no espectrograma.

Nao treinar modelo profundo do zero nesta fase.

## Validacao e metricas

Divisao:

- `GroupKFold` ou `StratifiedGroupKFold` por `Patient ID`;
- agrupar tambem `Additional ID` quando existir, para evitar que o mesmo sujeito de campanhas diferentes vaze entre folds;
- nenhum audio do mesmo paciente pode aparecer em treino e teste;
- `Unknown` fora do treino inicial.

Metricas:

- sensibilidade/recall para `Present`;
- especificidade para `Absent`;
- balanced accuracy;
- AUROC;
- AUPRC, por causa do desbalanceamento;
- F1 para `Present`;
- matriz de confusao por idade, campanha e local.

Para triagem, sensibilidade deve ser priorizada. Uma especificidade um pouco menor pode ser aceitavel se a aplicacao for selecionar quem precisa de avaliacao adicional. Se o objetivo for reduzir alarmes falsos em uso pratico, calibracao e especificidade passam a pesar mais.

## Criterios de decisao

Depois dos baselines, decidir assim:

| Resultado observado | Proximo passo |
|---|---|
| Features por fase vencem features globais | Investir em segmentacao/ciclo cardiaco e modelos por fase |
| Embeddings congelados vencem features classicas | Testar BEATs/EAT/PANNs com pooling melhor e classificador regularizado |
| Mean pooling perde para max/RDP/GeM | Manter pooling temporal como componente central |
| Modelo tabular sem audio e forte | Auditar vies de campanha/idade/local antes de confiar no audio |
| Erros concentrados em `Unknown` ou positivos grau I | Criar protocolo de incerteza em vez de forcar classificacao binaria |
| Ganho pequeno entre modelos | Preferir modelo simples, interpretavel e calibrado |
| Fine-tuning nao melhora validacao por paciente | Nao usar fine-tuning; manter extrator congelado |

## Riscos principais

- Poucos positivos: 179 pacientes `Present`.
- Desbalanceamento moderado: 695 `Absent` contra 179 `Present`.
- Rotulo ruidoso: `Unknown` existe e nao deve ser ignorado na analise.
- `Outcome` pode induzir conclusoes erradas se usado como alvo de sopro.
- Gravacoes por paciente nao sao independentes.
- Campos de caracterizacao do sopro vazam informacao do rotulo.
- Modelos pre-treinados em AudioSet podem ter dominio diferente de fonocardiograma; por isso embeddings congelados devem ser comparados com features especificas de PCG.

## Plano de execucao sugerido

1. Criar notebook/script de EDA tabular e salvar tabelas de distribuicao.
2. Criar script de inventario dos audios e segmentacoes.
3. Gerar paineis waveform/log-mel/segmentacao para amostras selecionadas.
4. Implementar features classicas globais por gravacao.
5. Implementar features por fase cardiaca usando `.tsv`.
6. Treinar Baseline 0, 1 e 2 com validacao por paciente.
7. Extrair embeddings congelados de pelo menos um modelo pronto, inicialmente PANNs/CNN14 ou BEATs.
8. Comparar pooling temporal e agregacao por paciente.
9. Fazer analise de erro por local, idade, campanha, grau e timing do sopro.
10. So entao decidir se vale fine-tuning, adaptacao auto-supervisionada ou um modelo hibrido final.

## Recomendacao inicial

A primeira abordagem tecnicamente defensavel e:

1. Excluir `Unknown` do treino inicial.
2. Fazer split estratificado por paciente.
3. Extrair features por fase cardiaca com base nos `.tsv`.
4. Treinar regressao logistica e gradient boosting como baseline.
5. Em paralelo, extrair embeddings congelados de um modelo pre-treinado e testar pooling por fase.
6. Comparar as abordagens com sensibilidade, especificidade, AUPRC e analise de erro.

Essa rota usa a informacao mais especifica do CirCor, respeita a limitacao de poucos dados e evita comecar por fine-tuning pesado antes de sabermos se o sinal acustico e os rotulos sustentam esse investimento.
