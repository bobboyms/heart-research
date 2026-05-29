# Grupo B v3.3 — textura na banda baixa + separabilidade Present×Absent

## Pergunta

Duas, encadeadas:

1. Dentro da banda baixa (≤260 Hz), **que extrações de informação** descrevem o sopro além de energia/persistência (já cobertas pelo v3.2)?
2. **Quanto um áudio com sopro se afasta de um áudio sem sopro?** — uma métrica explícita de separação, que o v3.x exploratório nunca calculou (só media pureza de cluster).

## O que foi feito

- Sobre o mesmo **mapa de contraste sístole−diástole robusto** (z por bin, banda ≤260 Hz, n_fft=256), adicionei o eixo de **textura ruído-vs-tonal** e **forma espectral** que faltava: flatness/entropia espectral, tilt, sub-bandas finas (25–80 / 80–150 / 150–260 Hz), skew/kurtosis temporal e espectral, **Gini/esparsidade**, flux temporal, proxy de HNR (pico tonal) e razão sístole/diástole.
- Não recompus o v3.2: li o CSV já salvo, extraí só a textura nova dos `.wav` e juntei por `recording_id`.
- Camada de **separabilidade** em três conjuntos (v3.2 / textura nova / combinado): AUC + Cohen's d + Mann-Whitney por feature; Mahalanobis entre centroides + Fisher (traço) + silhueta por rótulo; e um **score contínuo de distância de Mahalanobis ao centroide Absent** ("o quanto o áudio se afasta do normal").

Dados: **3002 gravações** (614 Present, 2388 Absent), 103 features de textura novas.

## Resultado 1 — a textura nova é o melhor separador *individual*

Top features por AUC (nível gravação, oriented):

| feature | AUC | Cohen's d | eixo |
|---|---:|---:|---|
| `tex_gini_map_p90` | **0.864** | −1.53 | concentração (NOVO) |
| `tex_gini_map_top3_mean` | 0.849 | −1.45 | concentração (NOVO) |
| `tex_freq_centroid_std` | 0.848 | −1.45 | forma espectral (NOVO) |
| `tex_frac_b25_80_std` | 0.844 | −1.46 | sub-banda fina (NOVO) |
| `enh_low_mid_80_200hz_energy_p50` | 0.834 | +1.00 | energia v3.2 |
| `enh_low_25_200hz_active_fraction_p50` | 0.829 | +1.47 | persistência v3.2 |

**O Gini do murmur map é a feature individual mais forte de todas — e é nova.** O sinal do d negativo conta a história física: **sopro = Gini BAIXO = energia espalhada/sustentada pela banda baixa** (ruído de banda larga sustentado), enquanto o normal tem energia **concentrada/transiente** (S1/S2 pontuais). Confirma o achado do MEMORY ("o sopro preenche a sístole") por um ângulo novo: não é só *quanta* energia (v3.2), é o **quão espalhada/desconcentrada** ela está.

## Resultado 2 — a textura *adiciona* separação multivariada (modesto)

| conjunto | n_feat | Mahalanobis centroides | silhueta | AUC dist-ao-Absent |
|---|---:|---:|---:|---:|
| v3.2 só | 1050 | 2.84 | 0.268 | 0.838 |
| textura só | 103 | 2.50 | 0.162 | 0.695 |
| **combinado** | 1153 | **3.02** | 0.257 | **0.851** |

Combinar a textura aumenta a distância de Mahalanobis entre os centroides (2.84→3.02) e a AUC do score contínuo (0.838→0.851). Ganho real, porém **modesto** — coerente com a lição do MEMORY de que refinamentos hand-crafted somam pouco quando a representação base já é boa. A textura sozinha (103 features) já chega perto do v3.2 inteiro (1050 features) em Mahalanobis: é **informação densa**.

## Resultado 3 — o score "distância ao normal" é clinicamente coerente

Distância média de Mahalanobis ao centroide Absent (conjunto combinado, nível gravação):

| grupo | dist média |
|---|---:|
| Absent | 20.8 |
| Present I/VI | 49.8 |
| Present II/VI | 37.9 |
| Present III/VI | **181.2** |

O score **reproduz quantitativamente o achado central do MEMORY**: os III/VI (altos) ficam dramaticamente longe do normal (181), enquanto **I/VI e II/VI (suaves) ficam logo acima do Absent** (38–50 vs 20.8) — os sopros suaves vivem *coladas* na população normal, que é exatamente o que segura o teto. As 8 gravações mais distantes são quase todas III/VI (uma exceção: `84690_MV` I/VI, candidato a inspeção como outlier). É um **score contínuo de "anormalidade" diretamente interpretável** — responde "o quanto este áudio se afasta de um sem sopro" com um número.

## Ressalvas (honestidade metodológica)

- **As métricas multivariadas são in-sample, sem validação cruzada** — diagnóstico de *geometria/separabilidade*, não performance de classificador.
- O `dist_to_absent_auc = 1.0` no **nível paciente** é artefato de **p≫n** (3152 features para ~850 pacientes): com mais features que amostras a separação in-sample é trivial. **Ignorar esse 1.0.** O número confiável é o de **gravação** (3002 amostras), e mesmo ele é otimista por não ter CV.
- A AUC **por feature** é robusta (univariada) e é a leitura principal a levar adiante.

## Conclusão

1. **A "região de informação" da banda baixa não é só energia/persistência** — a *textura* (concentração/Gini, variabilidade espectral) carrega o sinal mais forte por feature, e é nova.
2. A separação Present×Absent agora tem **métrica explícita**: AUC por feature, Mahalanobis/silhueta global e um **score contínuo de distância-ao-Absent** que se alinha ao grading.
3. O ganho de combinar textura+v3.2 é real porém modesto; **o caminho de valor é levar o score `dist_to_absent` (ou o subconjunto Gini+energia+persistência+forma) para um baseline supervisionado com validação por paciente** — fechando a ponte exploratório→supervisionado.
