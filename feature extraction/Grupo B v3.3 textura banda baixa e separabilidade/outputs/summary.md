# Grupo B v3.3 - textura banda baixa + separabilidade Present x Absent

## Objetivo

Adicionar (1) extracoes de TEXTURA na banda baixa (<=260 Hz) que faltavam no v3.2
(flatness/entropia, tilt, sub-bandas finas, skew/kurtosis, Gini, flux, HNR-proxy) e
(2) uma camada explicita de SEPARABILIDADE que mede o quanto um audio com sopro se
afasta de um sem sopro (AUC por feature, Mahalanobis/Fisher/silhueta, score dist-ao-Absent).

## Dados

- Gravacoes: 3002
- Pacientes: 874
- Features de textura novas: 103

## Leitura

- `auc` por feature = P(Present > Absent). 0.5 = inutil; >=0.8 = forte.
- `mahalanobis_centroids` = distancia normalizada entre os centroides Present/Absent.
- `fisher_trace_ratio` = scatter entre-grupos / dentro-do-grupo (maior = mais separavel).
- `silhouette_by_label` = coesao/separacao dos dois grupos no espaco de features (-1 a 1).
- `dist_to_absent_auc` = AUC do score continuo de distancia ao centroide Absent.

## Separabilidade multivariada por conjunto de features

| set          |   n_features |   mahalanobis_centroids |   fisher_trace_ratio |   silhouette_by_label |   dist_to_absent_auc | level     |
|:-------------|-------------:|------------------------:|---------------------:|----------------------:|---------------------:|:----------|
| v32_only     |         1050 |                 2.84021 |             1.00547  |              0.268322 |             0.838485 | recording |
| texture_only |          103 |                 2.50284 |             0.64921  |              0.161972 |             0.694778 | recording |
| combined     |         1153 |                 3.01928 |             0.972076 |              0.256861 |             0.851418 | recording |
| v32_only     |         3152 |                 5.9494  |             1.30012  |              0.294204 |             1        | patient   |
| texture_only |          309 |                 3.27861 |             0.84513  |              0.207752 |             0.821518 | patient   |
| combined     |         3461 |                 7.07671 |             1.25703  |              0.285516 |             1        | patient   |

## Top 20 features por AUC (combinado, nivel gravacao)

| feature                                        |      auc |   auc_oriented |   cohens_d |        mwu_p |   present_mean |   absent_mean |
|:-----------------------------------------------|---------:|---------------:|-----------:|-------------:|---------------:|--------------:|
| tex_gini_map_p90                               | 0.136208 |       0.863792 |  -1.52706  | 1.196e-170   |       0.637065 |     0.832279  |
| tex_gini_map_top3_mean                         | 0.150872 |       0.849128 |  -1.45181  | 2.49105e-157 |       0.656316 |     0.847793  |
| tex_gini_map_max                               | 0.151717 |       0.848283 |  -1.43748  | 1.40315e-156 |       0.687611 |     0.880551  |
| tex_freq_centroid_std                          | 0.152229 |       0.847771 |  -1.44795  | 3.98881e-156 |      18.7781   |    32.3303    |
| tex_frac_b25_80_std                            | 0.1563   |       0.8437   |  -1.45862  | 1.53944e-152 |       0.11953  |     0.213239  |
| tex_gini_map_mean                              | 0.156807 |       0.843193 |  -1.41057  | 4.28007e-152 |       0.538238 |     0.690256  |
| tex_gini_map_p50                               | 0.160068 |       0.839932 |  -1.41311  | 2.94832e-149 |       0.542848 |     0.704351  |
| enh_low_mid_80_200hz_energy_p50                | 0.833951 |       0.833951 |   0.996301 | 1.44196e-194 |       2.27716  |     0.049635  |
| enh_low_mid_80_200hz_active_fraction_p50       | 0.833872 |       0.833872 |   1.49087  | 1.57935e-194 |       0.485535 |     0.0330544 |
| enh_low_mid_80_200hz_max_p50                   | 0.833264 |       0.833264 |   1.23987  | 8.92581e-194 |       3.86534  |     0.342875  |
| enh_low_mid_80_200hz_top30_frame_mean_p50      | 0.833102 |       0.833102 |   1.04112  | 1.36947e-193 |       2.54338  |     0.0852247 |
| enh_low_mid_80_200hz_freq_centroid_mean        | 0.829976 |       0.829976 |   1.48005  | 9.19019e-141 |     102.952    |    48.7133    |
| enh_low_mid_80_200hz_frame_active_fraction_p50 | 0.829855 |       0.829855 |   1.61549  | 1.79614e-190 |       0.659395 |     0.1071    |
| enh_low_mid_80_200hz_longest_run_fraction_p50  | 0.829658 |       0.829658 |   1.61619  | 3.00988e-190 |       0.654808 |     0.104095  |
| enh_low_25_200hz_active_fraction_p50           | 0.82851  |       0.82851  |   1.47062  | 1.2374e-150  |       0.438894 |     0.0509033 |
| enh_low_25_200hz_freq_centroid_p50             | 0.828088 |       0.828088 |   1.41793  | 3.03495e-150 |      91.8977   |    31.87      |
| enh_low_25_200hz_energy_p50                    | 0.827305 |       0.827305 |   0.996069 | 1.5613e-149  |       1.85956  |     0.0739225 |
| enh_low_25_200hz_top30_frame_mean_p50          | 0.826182 |       0.826182 |   1.03537  | 1.59816e-148 |       2.09141  |     0.128034  |
| enh_low_25_200hz_freq_centroid_mean            | 0.825953 |       0.825953 |   1.41993  | 2.13855e-137 |      93.6936   |    48.2651    |
| tex_frac_b25_80_p90                            | 0.174278 |       0.825722 |  -1.35128  | 3.32957e-137 |       0.382819 |     0.623255  |
