# Analise exploratoria TCN - fold_1

Esta pasta contem os scripts e resultados da analise exploratoria do TCN do `fold_1` em `modeling/Grupo H Nested TCN CNN systole/outputs_nested/fold_1/tcn`.

## Scripts

- `analyze_tcn_class_location_split_distribution.py`: distribuicao de frames por classe, local e split.
- `analyze_tcn_segment_durations.py`: duracao dos segmentos reais anotados nos `.tsv`.
- `analyze_tcn_systole_errors_by_duration_boundary.py`: cruzamento dos erros de sistole do TCN com duracao real e distancia ate a fronteira.

## 1. Distribuicao por classe, local e split

A distribuicao global de `systole` e estavel entre os splits:

| split | recordings | patients | systole frames | systole % |
|---|---:|---:|---:|---:|
| train | 1687 | 486 | 469212 | 12.32 |
| val | 361 | 107 | 108244 | 12.39 |
| test | 365 | 106 | 99967 | 12.44 |

Isso sugere que o comportamento do TCN no `fold_1` nao vem de um desbalanceamento global de sistole entre `train`, `val` e `test`.

Por local, ha diferencas relevantes. `AV` tem menor proporcao de sistole, enquanto `PV` e `TV` tendem a ter mais:

| split | AV % systole | PV % systole | TV % systole | MV % systole |
|---|---:|---:|---:|---:|
| train | 10.49 | 14.30 | 13.39 | 11.38 |
| val | 9.13 | 14.31 | 14.78 | 11.40 |
| test | 11.04 | 13.33 | 12.97 | 12.42 |

Tambem existem 2 gravacoes `Phc` no treino, mas sao muito poucas para explicar o comportamento global.

## 2. Duracao dos segmentos reais

As sistoles reais sao curtas em todos os splits. A mediana e `140 ms` em `train`, `val` e `test`.

| split | systole segments | median ms | p5 ms | p25 ms | p75 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| train | 34401 | 140.0 | 80.0 | 120.0 | 160.0 | 187.9 |
| val | 7891 | 140.0 | 98.5 | 120.0 | 160.0 | 200.0 |
| test | 7351 | 140.0 | 80.0 | 120.0 | 160.0 | 180.0 |

Percentual de sistoles curtas:

| split | <100 ms | <150 ms | <200 ms |
|---|---:|---:|---:|
| train | 13.24 | 68.88 | 97.23 |
| val | 10.99 | 71.42 | 96.71 |
| test | 13.13 | 68.39 | 98.34 |

Isso e importante porque o TCN usa frame de `25 ms` e hop de `10 ms`. Muitas sistoles tem poucos frames efetivos, entao erro de fronteira pode ter impacto grande.

Foram encontrados 11 segmentos com duracao zero nos `.tsv`; eles foram salvos em `outputs/fold_1/segment_durations/invalid_segment_durations.csv` e ignorados nos calculos.

## 3. Erros de sistole por duracao e fronteira

Esta analise roda o `best_model.pt` do TCN do `fold_1` nos caches de `val/test` e reproduz a inferencia em batch com padding, para bater com as matrizes originais do TCN.

Categorias:

- `detected_majority`: pelo menos 50% dos frames do segmento real de sistole foram preditos como `systole`.
- `missed_as_other_majority`: pelo menos 50% foram preditos como `other`.
- `missed_as_other_classes`: a maioria virou outra classe cardiaca.

Resumo por segmento:

| split | category | segments | median duration ms | mean recall % | mean other rate % |
|---|---|---:|---:|---:|---:|
| val | detected_majority | 6695 | 140.0 | 91.56 | 1.14 |
| val | missed_as_other_majority | 1156 | 137.7 | 3.15 | 96.02 |
| val | missed_as_other_classes | 40 | 160.0 | 23.58 | 21.55 |
| test | detected_majority | 5815 | 140.0 | 91.87 | 1.60 |
| test | missed_as_other_majority | 1412 | 131.0 | 2.50 | 96.72 |
| test | missed_as_other_classes | 124 | 127.1 | 11.45 | 8.12 |

O erro `systole -> other` nao e explicado apenas por sistoles curtas. As faixas curtas pioram, mas tambem ha segmentos longos, de `240-380 ms`, completamente classificados como `other`.

Erro por faixa de duracao:

| split | duration bin | frame recall % | systole -> other % |
|---|---|---:|---:|
| val | <80 ms | 57.17 | 39.87 |
| val | 80-99 ms | 76.79 | 19.88 |
| val | 100-119 ms | 77.16 | 16.89 |
| val | 120-149 ms | 81.51 | 11.37 |
| val | 150-199 ms | 76.79 | 16.06 |
| test | <80 ms | 60.92 | 32.27 |
| test | 80-99 ms | 68.60 | 26.77 |
| test | 100-119 ms | 69.97 | 23.81 |
| test | 120-149 ms | 76.60 | 16.96 |
| test | 150-199 ms | 74.81 | 17.80 |

Erro por distancia ate a borda:

| split | distancia ate borda | frame recall % | systole -> other % |
|---|---|---:|---:|
| val | 0-9 ms | 50.09 | 16.40 |
| val | 20-29 ms | 82.72 | 15.22 |
| val | 40-59 ms | 86.20 | 13.30 |
| test | 0-9 ms | 50.10 | 21.29 |
| test | 20-29 ms | 76.51 | 20.14 |
| test | 40-59 ms | 80.62 | 17.71 |

Ha um efeito forte de fronteira: nos primeiros/ultimos `0-9 ms` da sistole, o recall cai para cerca de `50%`. No miolo da sistole, especialmente `40-79 ms` da borda, o recall sobe para aproximadamente `80-86%`.

Erro por local:

| split | location | frame recall % | systole -> other % |
|---|---|---:|---:|
| val | AV | 62.13 | 30.39 |
| val | MV | 76.94 | 17.09 |
| val | PV | 82.45 | 10.50 |
| val | TV | 84.93 | 8.00 |
| test | AV | 60.70 | 32.19 |
| test | MV | 75.55 | 18.84 |
| test | PV | 76.22 | 14.28 |
| test | TV | 80.05 | 14.02 |

`AV` e claramente o local mais problemático para detectar sistole no `fold_1`.

## Conclusao

O comportamento do TCN parece vir de uma combinacao de:

1. segmentos sistolicos curtos, onde poucos frames representam a fase;
2. erro de fronteira, com recall muito baixo nos primeiros/ultimos milissegundos;
3. dificuldade especifica por local, principalmente em `AV`;
4. alguns segmentos longos completamente perdidos como `other`, indicando que nao e apenas um problema de duracao curta.

O proximo passo mais util e gerar exemplos visuais/auditivos dos piores casos, principalmente `AV` e segmentos longos classificados 100% como `other`.
