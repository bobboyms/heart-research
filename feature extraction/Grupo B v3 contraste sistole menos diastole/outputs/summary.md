# Grupo B v3 - contraste sistole menos diastole

## Objetivo

Gerar clusters para visualizar a separacao entre gravacoes/pacientes com `Murmur = Present` e `Murmur = Absent` usando a diastole como ruido de fundo da propria gravacao.

Feature central:

`contraste[f,t] = log(1 + |STFT(sistole)[f,t]|) - mediana_t(log(1 + |STFT(diastole)[f,t]|))`

Ou seja: para cada frequencia, subtrai-se da sistole a referencia acustica da diastole do mesmo microfone/paciente.

## Configuracao

- Target sample rate: 4000 Hz
- STFT: n_fft=128, hop_length=32
- Faixa analisada: 25.0-1000.0 Hz
- Profile bins: 24
- K-means clusters: 2
- UMAP gerado: sim

## Resumo

- Gravacoes analisadas: 3002
- Pacientes analisados: 874
- Features de contraste por gravacao: 144
- Pacientes agregados: 874
- Features agregadas por paciente: 290

## Murmur por local

| location   |   Absent |   Present |   All |
|:-----------|---------:|----------:|------:|
| AV         |      603 |       152 |   755 |
| MV         |      636 |       172 |   808 |
| PV         |      586 |       147 |   733 |
| TV         |      563 |       143 |   706 |
| All        |     2388 |       614 |  3002 |

## Metricas por visualizacao

| level              | location   |   rows |   patients |   present_rate |   best_cluster |   best_cluster_rows |   best_cluster_present_count |   best_cluster_present_rate |   best_cluster_present_capture |   best_cluster_enrichment |   cluster_0_rows |   cluster_0_present_rate |   cluster_1_rows |   cluster_1_present_rate |   silhouette_pca |   pca12_variance |
|:-------------------|:-----------|-------:|-----------:|---------------:|---------------:|--------------------:|-----------------------------:|----------------------------:|-------------------------------:|--------------------------:|-----------------:|-------------------------:|-----------------:|-------------------------:|-----------------:|-----------------:|
| recording_global   | all        |   3002 |        874 |       0.20453  |              1 |                  11 |                           11 |                    1        |                      0.0179153 |                   4.88925 |             2991 |                 0.201605 |               11 |                 1        |         0.940702 |         0.828908 |
| recording_location | AV         |    755 |        748 |       0.201325 |              1 |                   6 |                            6 |                    1        |                      0.0394737 |                   4.96711 |              749 |                 0.194927 |                6 |                 1        |         0.92032  |         0.827372 |
| recording_location | PV         |    733 |        731 |       0.200546 |              1 |                   2 |                            2 |                    1        |                      0.0136054 |                   4.98639 |              731 |                 0.198358 |                2 |                 1        |         0.950102 |         0.852766 |
| recording_location | TV         |    706 |        700 |       0.20255  |              1 |                   3 |                            3 |                    1        |                      0.020979  |                   4.93706 |              703 |                 0.199147 |                3 |                 1        |         0.941724 |         0.848189 |
| recording_location | MV         |    808 |        803 |       0.212871 |              1 |                   7 |                            6 |                    0.857143 |                      0.0348837 |                   4.02658 |              801 |                 0.207241 |                7 |                 0.857143 |         0.915598 |         0.824779 |
| patient_aggregated | mean_max   |    874 |        874 |       0.204805 |              1 |                   7 |                            7 |                    1        |                      0.0391061 |                   4.88268 |              867 |                 0.198385 |                7 |                 1        |         0.913347 |         0.818249 |

## Leitura rapida

- Compare `best_cluster_present_rate` contra `present_rate`: quanto maior o enriquecimento, mais o contraste conseguiu juntar sopros.
- `best_cluster_present_capture` mede quanto dos casos `Present` caem no cluster enriquecido; pureza alta com captura baixa indica um subgrupo claro, mas incompleto.
- A leitura mais importante e o nivel `patient_aggregated`, porque o rotulo `Murmur` e por paciente.

## Arquivos gerados

- `recording_contrast_features.csv`: features de contraste por gravacao.
- `recording_contrast_features_with_projection.csv`: PCA/k-means global por gravacao.
- `patient_contrast_features.csv`: agregacao por paciente usando media e maximo entre locais.
- `patient_contrast_features_with_projection.csv`: PCA/k-means por paciente.
- `*_pca_murmur.png` e `*_pca_cluster.png`: visualizacoes PCA.
- `*_umap_murmur.png` e `*_umap_cluster.png`: visualizacoes UMAP, se habilitado.
- `by_location/`: visualizacoes separadas por local de ausculta.
