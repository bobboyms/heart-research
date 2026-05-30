---
title: "Detecção honesta de sopro cardíaco em fonocardiogramas: contraste sístole–diástole em banda baixa sob um protocolo de avaliação paciente-level sem reamostragem"
authors:
  - name: "[Autor 1]"
    affiliation: "[Afiliação]"
    email: "thiago.luiz.rodriguez@gmail.com"
keywords: [fonocardiograma, sopro cardíaco, CirCor DigiScope, detecção, TCN, contraste de fase, AUPRC, protocolo de avaliação]
date: 2026-05-29
---

# Resumo

A detecção automática de sopros cardíacos a partir de fonocardiogramas (PCG) é uma tarefa
clinicamente relevante para triagem cardiovascular de baixo custo. Apesar de uma literatura
crescente sobre o conjunto CirCor DigiScope, a comparação entre trabalhos é dificultada pela
**heterogeneidade dos protocolos de avaliação** — granularidade (paciente vs. segmento),
estratégia de particionamento, reamostragem da classe minoritária e métrica reportada — o que
produz números nominalmente altos, porém não comparáveis. Neste trabalho propomos um pipeline em
dois estágios — segmentação de fase cardíaca por *Temporal Convolutional Network* (TCN) seguida de
uma CNN sistólica — e introduzimos um **espectrograma de contraste de fase**, que re-referencia a
sístole pela diástole do mesmo paciente, combinado a uma restrição de **banda baixa
($\leq 300$ Hz)**. Sob um protocolo deliberadamente conservador — validação cruzada aninhada
paciente-level, sem reamostragem, com métrica primária AUPRC sob a prevalência clínica real
($\approx 20\%$ positivos) — o melhor modelo atinge **AUPRC $= 0{,}8844$** e **AUROC $= 0{,}9375$**.
Mostramos ainda, por ablação, que o teto de desempenho observado é majoritariamente
**informacional** e não de modelagem: ao excluir os sopros de grau I/VI (suaves), que concentram
$91\%$ dos falsos negativos, o AUROC sobe de $0{,}9375$ para $0{,}9932$ e a AUPRC de $0{,}8844$ para
$0{,}9617$. Discutimos como nosso protocolo se posiciona frente à literatura e
argumentamos que a falta de padronização de avaliação, e não a arquitetura, é hoje o principal
obstáculo à comparação justa nesta tarefa.

---

# 1. Introdução

Doenças cardiovasculares são a principal causa de morte global. A ausculta cardíaca permanece a
ferramenta de triagem mais acessível, mas depende de operador especializado e tem concordância
inter-observador limitada, sobretudo para sopros suaves. A detecção automática de sopros a partir
de PCG promete escalar a triagem em contextos de poucos recursos [@reyna2023challenge].

O conjunto **CirCor DigiScope** [@oliveira2022circor], usado no *George B. Moody PhysioNet
Challenge 2022* [@reyna2023challenge], tornou-se o principal *benchmark* público para a tarefa.
Contudo, a comparação entre trabalhos publicados é frequentemente enganosa: relatos de acurácia
acima de $98\%$ coexistem com a métrica oficial do desafio (*weighted accuracy*) abaixo de $0{,}80$
para os melhores times. Argumentamos que essa discrepância **não reflete diferenças de modelo**,
mas de protocolo de avaliação — em especial granularidade de predição, vazamento entre treino e
teste, e reamostragem da classe minoritária.

As contribuições deste trabalho são:

1. **Espectrograma de contraste de fase** (Seção 4.3): uma representação que cancela a coloração de
   sensor/paciente ao re-referenciar a sístole pela diástole do mesmo paciente, expondo o excesso
   de energia sistólica que caracteriza o sopro.
2. **Restrição de banda baixa** ($\leq 300$ Hz): confirmamos, de forma supervisionada e por
   varredura do corte de frequência, que o sinal discriminante reside na banda baixa.
3. **Protocolo de avaliação honesto** (Seção 4.5): validação aninhada paciente-level, sem
   reamostragem, reportando AUPRC sob prevalência real; e uma análise crítica e descritiva da
   heterogeneidade de protocolos na literatura (Seção 2).
4. **Caracterização do teto informacional** (Seção 5.3): por ablação, mostramos que os sopros I/VI
   (suaves) — e não a capacidade do modelo — limitam o desempenho.

---

# 2. Trabalhos relacionados e heterogeneidade de protocolos

A Tabela 1 resume trabalhos recentes (2020–2026) sobre a tarefa de detecção de sopro no CirCor /
PhysioNet 2022. Em vez de ordenar por desempenho nominal, ordenamos por **comparabilidade de
protocolo**, tornando explícitas as dimensões que afetam a métrica.

**Tabela 1.** Comparação de protocolos de avaliação na detecção de sopro (CirCor / PhysioNet 2022).
A única métrica diretamente comparável entre trabalhos é a AUROC; ainda assim, ela só é informativa
quando lida junto da granularidade, do particionamento e da reamostragem.

| Trabalho | Ano | Tarefa | Granularidade | Particionamento | Reamostragem | Métrica(s) reportada(s) |
|---|---|---|---|---|---|---|
| **Este trabalho** | 2026 | Present vs. Absent | **Paciente** | **Por paciente, CV aninhada** | **Nenhuma** (focal loss) | **AUPRC $0{,}8844$**, AUROC $0{,}9375$ |
| WST + 1D-CNN [@wst1dcnn2023] | 2023 | 3 classes | Segmento | Por segmento (80/20) | — | AUROC $0{,}9345$; W.Acc $0{,}863$; Acc $0{,}829$; F1 $0{,}819$ |
| HearHeart (venc. Challenge) [@reyna2023challenge] | 2022 | 3 classes | Paciente | *Test set* oculto | — | W.Acc $0{,}780$; AUROC $0{,}884$ |
| PathToMyHeart [@reyna2023challenge] | 2022 | 3 classes | Paciente | *Test set* oculto | — | W.Acc $0{,}771$; AUROC $0{,}880$ |
| Stockwell + AlexNet [@manshadi2024stockwell] | 2024 | 3 classes | Paciente | Por paciente | **SMOTE + up/down** | W.Acc $0{,}93$; "AUC" $0{,}98$; Sens. $0{,}91$; Espec. $0{,}91$ |
| EHST (Transformer) [@ehst2025] | 2025 | Caracterização¹ | n/d | n/d | — | Acc $0{,}941$; "AUC" $0{,}95$ |
| Uncertainty-aware [@uncertainty2025] | 2025 | Detecção | n/d | n/d | — | Acc $\approx 0{,}91$ |
| Training-free Transformer [@trainingfree2025] | 2025 | Detecção | n/d | n/d | — | W.Acc $0{,}786$; UAR $0{,}697$ |
| SMOTE + RNN [@ameen2026smote] | 2026 | Detecção² | Segmento | — | **SMOTE** | Acc $0{,}986$ |

¹ Caracterização de atributos do sopro (*timing*/*shape*/*pitch*), não detecção present-vs-absent.
² Os autores reportam acurácia por segmento balanceada; o número alinha-se melhor com o rótulo
*Outcome* (Normal/Anormal, $\approx$ balanceado) do que com *Murmur* ($\approx 20\%$ positivos).

Três observações decorrem da Tabela 1:

- **Granularidade infla quando é por segmento.** Particionar por segmento sem garantir separação por
  paciente permite que segmentos correlacionados do mesmo indivíduo apareçam em treino e teste. O
  trabalho de maior AUROC sem reamostragem [@wst1dcnn2023] adota exatamente esse particionamento
  (80/20 por segmento), o que torna seu $0{,}9345$ não comparável a uma AUROC paciente-level.
- **Reamostragem da classe minoritária altera a base de comparação.** Técnicas como SMOTE
  [@chawla2002smote] e *up/down-sampling* são legítimas no treino, mas quando a métrica é medida em
  um conjunto reamostrado/balanceado a prevalência deixa de ser a clínica. Um indicador objetivo:
  a *weighted accuracy* de $0{,}93$ relatada em [@manshadi2024stockwell] supera amplamente o melhor
  time do desafio ($0{,}780$ no *test set* oculto) **na mesma métrica** — diferença incompatível com
  ganho de modelagem e consistente com avaliação sob reamostragem.
- **A métrica reportada importa.** Sob forte desbalanceamento, a AUPRC é mais informativa e mais
  exigente que a AUROC ou a acurácia [@saito2015precision]. A maioria dos trabalhos não reporta
  AUPRC, o que dificulta avaliar o desempenho sobre a classe rara — justamente a de interesse
  clínico.

Concluímos que **não há, na literatura levantada, um protocolo paciente-level, sem vazamento e sem
reamostragem que reporte AUPRC/AUROC binário** além do nosso. Não fazemos juízo sobre a validade
interna dos trabalhos citados; apenas explicitamos que suas métricas não são diretamente
comparáveis às nossas.

---

# 3. Dados

Utilizamos o CirCor DigiScope v1.0.3 [@oliveira2022circor], correspondente ao conjunto de treino
público do PhysioNet Challenge 2022 [@reyna2023challenge]. Após o pré-processamento, o conjunto
contém **942 pacientes** com $\approx 3163$ gravações em até quatro focos de ausculta (aórtico —
AV, pulmonar — PV, tricúspide — TV, mitral — MV). A distribuição do rótulo `Murmur` é:

| Classe | Pacientes |
|---|---:|
| Present | 179 |
| Absent | 695 |
| Unknown | 68 (excluídos da tarefa) |

A prevalência de positivos é de $\approx 20\%$, configurando um problema desbalanceado. Cada gravação
acompanha anotações de segmentação de fase cardíaca e, para os positivos, atributos do sopro
(*grading* I–VI, *timing*, *shape*, *pitch* e foco mais audível), que usamos para **diagnóstico**
(Seção 5.3), não como alvo.

---

# 4. Métodos

## 4.1 Visão geral do pipeline

Adotamos um pipeline em dois estágios. Primeiro, uma **TCN** segmenta cada gravação em fases
cardíacas (S1, sístole, S2, diástole). Segundo, uma **CNN sistólica** classifica a presença de
sopro a partir do espectrograma da janela sistólica. As probabilidades por gravação são agregadas
ao nível do paciente por **máximo** (`max`), refletindo que basta um foco com sopro audível para o
paciente ser positivo.

## 4.2 Segmentação por TCN

A TCN [@bai2018tcn] é treinada por *fold* apenas com os pacientes de treino daquele *fold* (sem
vazamento), no modo `cardiac-phase`. A janela sistólica predita é a entrada do estágio seguinte.
Avaliamos também o uso de segmentação *ground-truth* (GT) como teto de segmentação; observamos que
a GT não melhora o desempenho agregado, indicando que o segmentador não é o gargalo dominante
(exceto em casos pontuais — Seção 5.3).

## 4.3 Espectrograma de contraste de fase

A representação central deste trabalho re-referencia a sístole pela diástole **do mesmo paciente**,
por frequência:

$$
C[f,t] = \log|S_{\text{sístole}}[f,t]| - \operatorname{median}_t \log|S_{\text{diástole}}[f,t]|,
$$

onde $S$ é a STFT. A intuição é que a coloração espectral de sensor e paciente afeta sístole e
diástole de modo semelhante; subtrair a diástole cancela esse confundidor e expõe o **excesso de
energia sistólica** característico do sopro. Para pacientes `Absent`, sístole $\approx$ diástole, de
modo que $C \approx 0$: o contraste **não infla a classe negativa** (o ganho é de *ranking*, não de
deslocamento do *operating point*).

Dois pré-requisitos não-óbvios são necessários (cada um, isoladamente, anula o efeito):

1. **Normalização global de frequência** (`--freq-norm global`): a normalização por *bin*
   re-branqueia o espectro e cancela o contraste.
2. **Retenção da diástole na segmentação**: o pipeline deve preservar os quadros diastólicos para
   compor o termo de referência, sem alterar a janela sistólica.

## 4.4 Restrição de banda baixa

Restringimos a entrada a $\leq 300$ Hz (`--high-hz 300`). Uma varredura do corte
($200/250/300/400/\text{full}$) revela um **platô em 200–300 Hz** (AUPRC $0{,}879$–$0{,}884$),
com regressão em 400 Hz ($0{,}874$) e *full-band* ($0{,}877$). Surpreendentemente, restringir a banda
**não** prejudica os sopros de *pitch* alto ($0{,}968 \rightarrow 0{,}971$), indicando que mesmo
sopros agudos têm energia relevante na banda baixa.

## 4.5 Protocolo de avaliação

- **Validação cruzada aninhada paciente-level**, 5 *folds*, semente $42$, **sem vazamento**: a TCN é
  treinada apenas nos pacientes de treino de cada *fold*; a CNN tem *split* interno para *tuning* e
  calibração.
- **Sem reamostragem do conjunto de avaliação.** O desbalanceamento é tratado **dentro do modelo**,
  via *focal loss* [@lin2017focal] ($\gamma = 2$), sem alterar a prevalência das predições OOF.
- **Métrica primária: AUPRC** OOF calibrada (calibração de Platt), por ser a mais informativa sob
  prevalência baixa [@saito2015precision]; **AUROC** como métrica secundária comparável.
- **Agregação** gravação $\rightarrow$ paciente por `max`.

## 4.6 Configuração do melhor modelo

A receita do melhor modelo (`phase_contrast_lowband300`) parte de um *baseline* forte
(`bc`) e adiciona contraste de fase e banda baixa:

- Rótulos *location-aware*; STFT por segmento ($n_{\text{fft}} = 128$, *hop* $= 32$, $f_s = 4000$ Hz).
- *Encoder* `multiscale` com dilatações $1,2,4,8,16,32$ e *pooling* por atenção.
- *Focal loss* $\gamma = 2$ (sem `pos_weight`); normalização **global** de frequência.
- Sístole via TCN (`--systole-threshold 0.45 --systole-margin-ms 50`).
- **Contraste de fase** + **`--high-hz 300`**.

---

# 5. Experimentos e resultados

## 5.1 Resultado principal

**Tabela 2.** Progressão do desempenho paciente-level (AUPRC/AUROC OOF calibrado, 5 *folds*).

| Modelo | Segmentação | AUPRC | AUROC |
|---|---|---:|---:|
| `bc` (baseline) | TCN | $0{,}8459$ | $0{,}9115$ |
| Contraste de fase (*full-band*) | TCN | $0{,}8771$ | $0{,}9338$ |
| **Contraste de fase + banda $\leq 300$ Hz** | TCN | **$0{,}8844$** | **$0{,}9375$** |

O contraste de fase isolado eleva a AUPRC em $+0{,}031$ sobre o *baseline*, com todos os *folds*
$\geq$ *baseline*. Adicionar a banda baixa melhora tanto o agregado quanto casos difíceis: *pitch*
baixo $0{,}469 \rightarrow 0{,}498$; sopros II/VI $0{,}738 \rightarrow 0{,}800$.

## 5.2 Ablações

- **Dual-canal** [sístole crua, contraste]: AUPRC $0{,}8554$ — **pior** que o contraste puro; a
  sístole crua reintroduz o confundidor que o contraste removeu.
- **Contraste robusto** ($\div$MAD): $0{,}8533$ — pior; a CNN já normaliza, e o $\div$MAD amplifica
  *bins* de diástole silenciosa (ruído).
- **Resolução de frequência** ($n_{\text{fft}} = 256$): $0{,}8750$ — pior; janela maior degrada a
  resolução temporal, e o sinal está na textura temporal.
- **Demografia** (ramo somado, idade/sexo/etc.): $0{,}8759$ — não ajuda a tarefa `Murmur`.
- **Multi-task de *pitch*** e **norma global pura** (Tier 1/2): empatam ou pioram; ver Seção 5.4.

## 5.3 O teto é informacional, não de modelagem

A análise por *grading* no melhor modelo (`lb300`) mostra que o desempenho cresce monotonicamente
com a intensidade do sopro e que o erro se concentra quase inteiramente no grau I/VI (Tabela 3).

**Tabela 3.** Desempenho por *grading* no modelo `lb300` (probabilidade calibrada média e falsos
negativos ao limiar $0{,}5$).

| Grading | $n$ | Prob. média | FN@$0{,}5$ |
|---|---:|---:|---:|
| I/VI | 104 | $0{,}59$ | 40 ($38\%$) |
| II/VI | 28 | $0{,}80$ | 3 ($11\%$) |
| III/VI | 46 | $0{,}98$ | 0 ($0\%$) |

Os sopros mais audíveis são detectados de forma praticamente perfeita (III/VI: $0/46$ perdidos),
enquanto **$91\%$ dos falsos negativos são I/VI** ($40$ de $44$). Quando se removem os I/VI do
conjunto de positivos, o desempenho do `lb300` salta para quase perfeito: **AUROC
$0{,}9375 \rightarrow 0{,}9932$** e **AUPRC $0{,}8844 \rightarrow 0{,}9617$**. O teto de desempenho,
portanto, reside **inteiramente nos sopros I/VI (suaves)**, não nos audíveis.

Nuance importante: o sopro I/VI **é audível** para um cardiologista ao vivo (por isso rotulado
Present); o que ocorre é que o modelo não o separa do normal **nestas gravações fixas**. As causas —
sinal fraco/mascarado, limitação de *features*, ou ruído de rótulo (grau I tem a menor concordância
inter-observador) — **não são distinguíveis apenas com os dados**. Portanto, o teto não é um piso
irredutível absoluto, e sim condicionado a estas gravações e a este *pipeline*, mantendo abertas
alavancas de **dados** (denoising, filtro de qualidade, limpeza de rótulo) em vez de arquitetura.

Análise por gravação corrobora que o **pitch baixo é o separador mais forte de dificuldade** (mais
que o *grading*): probabilidade calibrada média Low $0{,}349$ < Medium $0{,}679$ < High $0{,}817$.

## 5.4 O que não funcionou

Em regime de poucos positivos ($\approx 150$ I/VI), aumentar capacidade ou sofisticação não ajudou.
Empataram ou pioraram frente ao *baseline*: *data augmentation* (SpecAugment/mixup), ramos de
*features* temporais, segmentos *ground-truth*, RNN (GRU bidirecional), janelas de 1 s, arquiteturas
2D de frequência (Conv2d residual e *multiscale*), ramos de frequência (MLP e Transformer), fusão
sístole+diástole, *quality-gate* de segmentação na agregação `max`, e *multi-task* de *pitch*. O
padrão é consistente: **o que injeta domínio (DSP fixo: contraste + banda baixa) supera o que apenas
adiciona capacidade aprendível**.

---

# 6. Discussão

Dois resultados sustentam a tese central. Primeiro, o ganho vem de **DSP que injeta conhecimento de
domínio** (re-referência pela diástole) e não de capacidade de modelo; parte do que se supunha "teto
informacional" era, na verdade, sinal **mascarado** e recuperável. Segundo, o teto remanescente é
dominado pelos sopros I/VI, cuja separabilidade depende de **dados**, não de arquitetura.

Sobre comparabilidade: nosso AUROC ($0{,}9375$) é nominalmente próximo do maior valor sem
reamostragem da literatura ($0{,}9345$ [@wst1dcnn2023]), mas obtido sob protocolo mais conservador
(paciente-level, sem vazamento), o que torna a equivalência aparente favorável ao nosso método.
Frente ao melhor time do desafio (AUROC $0{,}884$ [@reyna2023challenge]), ficamos acima, com a
ressalva honesta de que nossa tarefa binária (excluindo `Unknown`) é mais simples que a de três
classes do desafio. Recomenda-se, como trabalho futuro, reportar também o desempenho sob o protocolo
oficial do desafio para blindar a comparação.

---

# 7. Limitações

- A tarefa é binária (`Present` vs. `Absent`, excluindo `Unknown`), mais simples que a de três
  classes do desafio; comparações de AUROC com trabalhos 3-classe são apenas indicativas.
- A validação usa o conjunto público; falta avaliação em *test set* oculto / coorte externa.
- Os números atribuídos a trabalhos de terceiros (Tabela 1) foram extraídos de resumos/relatos e
  **devem ser confirmados na fonte primária** antes da submissão (ver nota em Referências).
- O contraste de fase pressupõe segmentação que preserve a diástole; gravações mal segmentadas
  degradam o termo de referência.

---

# 8. Conclusão

Apresentamos um pipeline TCN + CNN sistólica com um espectrograma de **contraste de fase** em
**banda baixa** que atinge AUPRC $0{,}8844$ / AUROC $0{,}9375$ na detecção de sopro no CirCor, sob um
**protocolo de avaliação paciente-level, sem vazamento e sem reamostragem**. Mostramos que o teto de
desempenho é majoritariamente informacional (sopros I/VI) e que o principal obstáculo à comparação
justa nesta área não é a arquitetura, mas a **ausência de um protocolo de avaliação padronizado**.
Defendemos a AUPRC sob prevalência real como métrica primária e a explicitação de granularidade,
particionamento e reamostragem em todo relato futuro.

---

# Reprodutibilidade

Código e configuração do pipeline: `nested_tcn_systole_cnn/` (entrypoint
`uv run python -m nested_tcn_systole_cnn.train`). Receita do melhor modelo: *baseline* `bc` +
`--phase-contrast --freq-norm global --high-hz 300`. Dados: CirCor DigiScope v1.0.3
[@oliveira2022circor]. Detalhamento de experimentos e ablações: `MEMORY.md` e
`papers/comparacao_sota_murmur_detection.md` neste repositório.

---

# Referências

> **Nota de verificação (remover antes da submissão):** as entradas marcadas com ⚠ tiveram metadados
> (autores/veículo) extraídos de buscas e **devem ser confirmadas na fonte primária**. Os valores
> numéricos de terceiros na Tabela 1 também precisam de conferência no texto original.

1. <a id="ref-oliveira2022circor"></a> Oliveira J. et al. *The CirCor DigiScope Dataset: From
   Murmur Detection to Murmur Classification.* IEEE Journal of Biomedical and Health Informatics,
   2022. arXiv:2108.00813. https://arxiv.org/abs/2108.00813
2. <a id="ref-reyna2023challenge"></a> Reyna M.A. et al. *Heart Murmur Detection from Phonocardiogram
   Recordings: The George B. Moody PhysioNet Challenge 2022.* PLOS Digital Health, 2023.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10495026/
3. <a id="ref-wst1dcnn2023"></a> ⚠ *Heart Murmur and Abnormal PCG Detection via Wavelet Scattering
   Transform & a 1D-CNN.* arXiv:2303.11423, 2023. https://arxiv.org/abs/2303.11423
4. <a id="ref-manshadi2024stockwell"></a> ⚠ Manshadi O.D., Mihandoost S. *Murmur identification and
   outcome prediction in phonocardiograms using deep features based on Stockwell transform.*
   Scientific Reports, 2024. doi:10.1038/s41598-024-58274-6.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10981708/
5. <a id="ref-ehst2025"></a> ⚠ *Explainable attention-based deep learning for classification and
   interpretation of heart murmurs using phonocardiograms.* Scientific Reports, 2025.
   doi:10.1038/s41598-025-21971-x. https://pmc.ncbi.nlm.nih.gov/articles/PMC12575608/
6. <a id="ref-ameen2026smote"></a> ⚠ Ameen A., Fattoh I.E., Abd El-Hafeez T., Ahmed K. *Advancing
   cardiovascular screening: deep learning-based heart-sound classification using SMOTE and temporal
   modeling.* Scientific Reports, 2026. doi:10.1038/s41598-026-45276-9. (Cópia local em
   `papers/2026_10.1038_s41598-026-45276-9.md`.)
7. <a id="ref-uncertainty2025"></a> ⚠ *Optimizing Uncertainty-Aware Deep Learning for On-the-edge
   Heart-Sound Classification.* arXiv:2511.00966, 2025. https://arxiv.org/abs/2511.00966
8. <a id="ref-trainingfree2025"></a> ⚠ *A Training-Free Transformer Architecture for Heart Murmur
   Detection.* arXiv:2509.18424, 2025. https://arxiv.org/abs/2509.18424
9. <a id="ref-saito2015precision"></a> Saito T., Rehmsmeier M. *The Precision-Recall Plot Is More
   Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS
   ONE, 2015. doi:10.1371/journal.pone.0118432.
10. <a id="ref-chawla2002smote"></a> Chawla N.V. et al. *SMOTE: Synthetic Minority Over-sampling
    Technique.* Journal of Artificial Intelligence Research, 2002. doi:10.1613/jair.953.
11. <a id="ref-lin2017focal"></a> Lin T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV,
    2017. arXiv:1708.02002.
12. <a id="ref-bai2018tcn"></a> Bai S., Kolter J.Z., Koltun V. *An Empirical Evaluation of Generic
    Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271, 2018.

---

# Apêndice A — BibTeX (para o LaTeX)

```bibtex
@article{oliveira2022circor,
  title   = {The {CirCor} {DigiScope} Dataset: From Murmur Detection to Murmur Classification},
  author  = {Oliveira, Jorge and others},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2022},
  note    = {arXiv:2108.00813},
  url     = {https://arxiv.org/abs/2108.00813}
}

@article{reyna2023challenge,
  title   = {Heart Murmur Detection from Phonocardiogram Recordings: The {George B. Moody} {PhysioNet} Challenge 2022},
  author  = {Reyna, Matthew A. and others},
  journal = {PLOS Digital Health},
  year    = {2023},
  url     = {https://pmc.ncbi.nlm.nih.gov/articles/PMC10495026/}
}

% VERIFICAR autores/veiculo
@misc{wst1dcnn2023,
  title        = {Heart Murmur and Abnormal {PCG} Detection via Wavelet Scattering Transform and a 1D-CNN},
  howpublished = {arXiv:2303.11423},
  year         = {2023},
  url          = {https://arxiv.org/abs/2303.11423}
}

% VERIFICAR
@article{manshadi2024stockwell,
  title   = {Murmur identification and outcome prediction in phonocardiograms using deep features based on {Stockwell} transform},
  author  = {Manshadi, Omid Dezhkam and Mihandoost, Sara},
  journal = {Scientific Reports},
  year    = {2024},
  doi     = {10.1038/s41598-024-58274-6}
}

% VERIFICAR autores
@article{ehst2025,
  title   = {Explainable attention-based deep learning for classification and interpretation of heart murmurs using phonocardiograms},
  author  = {Anonymous},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.1038/s41598-025-21971-x}
}

% VERIFICAR
@article{ameen2026smote,
  title   = {Advancing cardiovascular screening: deep learning-based heart-sound classification using {SMOTE} and temporal modeling},
  author  = {Ameen, Asmaa and Fattoh, Ibrahim Eldesouky and Abd El-Hafeez, Tarek and Ahmed, Kareem},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {10.1038/s41598-026-45276-9}
}

% VERIFICAR autores
@misc{uncertainty2025,
  title        = {Optimizing Uncertainty-Aware Deep Learning for On-the-edge Heart-Sound Classification},
  howpublished = {arXiv:2511.00966},
  year         = {2025},
  url          = {https://arxiv.org/abs/2511.00966}
}

% VERIFICAR autores
@misc{trainingfree2025,
  title        = {A Training-Free Transformer Architecture for Heart Murmur Detection},
  howpublished = {arXiv:2509.18424},
  year         = {2025},
  url          = {https://arxiv.org/abs/2509.18424}
}

@article{saito2015precision,
  title   = {The Precision-Recall Plot Is More Informative than the {ROC} Plot When Evaluating Binary Classifiers on Imbalanced Datasets},
  author  = {Saito, Takaya and Rehmsmeier, Marc},
  journal = {PLOS ONE},
  year    = {2015},
  doi     = {10.1371/journal.pone.0118432}
}

@article{chawla2002smote,
  title   = {{SMOTE}: Synthetic Minority Over-sampling Technique},
  author  = {Chawla, Nitesh V. and Bowyer, Kevin W. and Hall, Lawrence O. and Kegelmeyer, W. Philip},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {16},
  pages   = {321--357},
  year    = {2002},
  doi     = {10.1613/jair.953}
}

@inproceedings{lin2017focal,
  title     = {Focal Loss for Dense Object Detection},
  author    = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017},
  note      = {arXiv:1708.02002}
}

@misc{bai2018tcn,
  title        = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
  author       = {Bai, Shaojie and Kolter, J. Zico and Koltun, Vladlen},
  howpublished = {arXiv:1803.01271},
  year         = {2018}
}
```

---

# Apêndice B — Notas para conversão em LaTeX

- O *front-matter* YAML mapeia para `\title`, `\author`, `\date`.
- Equações já estão em LaTeX (`$...$` e `$$...$$`).
- Tabelas em Markdown convertem para `tabular`/`booktabs` (sugerido: Pandoc
  `pandoc paper.md -o paper.tex --bibliography=refs.bib --citeproc` ou `--natbib`).
- Citações usam o estilo `[@chave]`; com Pandoc + `.bib` (Apêndice A) viram `\cite{chave}`.
- Substituir os marcadores `[Autor 1]`/`[Afiliação]` e remover a **Nota de verificação** e os ⚠
  após conferir as fontes primárias.
