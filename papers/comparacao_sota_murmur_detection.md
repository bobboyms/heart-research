# Comparação com a literatura — detecção de sopro (CirCor / PhysioNet 2022)

> Comparação do nosso melhor modelo (`phase_contrast_lowband300_reuse_tcn`, "lb300") com trabalhos
> de 2020–2026 na **mesma task**: `Murmur` Present vs Absent no CirCor DigiScope. Gerado em 2026-05-29.
> Fonte canônica do nosso resultado: [`MEMORY.md`](../MEMORY.md).

## O problema da comparabilidade (ler antes da tabela)

Quase ninguém reporta exatamente a nossa métrica. Existem três "dialetos" na literatura, e
misturá-los engana. **A única métrica diretamente comparável entre trabalhos é o AUROC.**

| Eixo | Nosso modelo | Maioria dos papers |
|---|---|---|
| Métrica primária | AUPRC / AUROC | *Weighted Accuracy* (oficial do Challenge) ou accuracy |
| Task | Binária (Present vs Absent, **exclui Unknown**) | 3 classes (Present/Unknown/Absent) |
| Granularidade | **Paciente-level**, nested CV 5-fold | Mistura: paciente-level (Challenge) vs **segmento-level** |
| Anti-vazamento | Sim (split por paciente) | Frequentemente **não** (split por segmento infla a métrica) |

Nosso AUROC = **0.9375** · AUPRC = **0.8844** (paciente-level, nested CV, sem vazamento).

## Tabela comparativa (ordenada por comparabilidade)

| Trabalho | Ano | Método | Nível | AUROC | W.Acc | Outras | Comparável? |
|---|---|---|---|---|---|---|---|
| **NOSSO `lb300`** | 2026 | TCN + CNN sístole, phase-contrast + banda ≤300 Hz | **Paciente, nested CV** | **0.9375** | — | **AUPRC 0.8844** | — (referência) |
| WST + 1D-CNN (arXiv 2303.11423) | 2023 | Wavelet scattering + 1D-CNN, PCG 2022 "clean" | **Segmento, split 80-20 por segmento** | 0.9345 | 0.863 | Acc 82.9%, F1 81.9% | ⚠️ número alto MAS **com vazamento provável** (sem split por paciente) |
| HearHeart (vencedor Challenge) | 2022 | — (test set oculto) | Paciente | 0.884 | **0.780** | — | ⚠️ 3-classe, test oculto |
| PathToMyHeart (Challenge #4) | 2022 | — | Paciente | 0.880 | 0.771 | — | ⚠️ 3-classe |
| HearTech+ (Challenge #2) | 2022 | — | Paciente | 0.771 | 0.776 | — | ⚠️ 3-classe |
| CUED_Acoustics (Challenge #2) | 2022 | — | Paciente | 0.757 | 0.776 | — | ⚠️ 3-classe |
| Uncertainty-aware (arXiv 2511.00966) | 2025 | MC-Dropout, modelo leve | n/d | — | — | Acc 91% | ⚠️ só accuracy |
| Training-free transformer (arXiv 2509.18424) | 2025 | WST + transformer sem treino | n/d | — | 0.786 | UAR 0.697 | ⚠️ só W.Acc |
| Heart Murmur Quality + Attention (MDPI Appl.Sci. 2024) | 2024 | DNN + atenção | recording | — | — | (não extraído) | ⚠️ foco em *quality* |
| Stockwell + AlexNet (Sci Rep 2024, PMC10981708) | 2024 | Detecção 3-classe, **SMOTE + up/down-sampling** | Paciente, split por paciente | "AUC" 0.98 | 0.93 | sens 0.91, spec 0.91, F1 0.91 | ❌ **SMOTE infla** (W.Acc 0.93 >> Challenge 0.78 na MESMA métrica = artefato de reamostragem) |
| EHST attention transformer (Sci Rep 2025, PMC12575608) | 2025 | **Caracterização** (timing/shape/pitch), não detecção | n/d | "AUC" 0.95 | — | Acc 0.941, MacroF1 0.926 | ❌ **task diferente** (não é present-vs-absent) |
| Murmur grading (PMC10482086) | 2023 | **Grading I–VI**, não detecção | recording | — | — | — | ❌ task diferente |
| SMOTE + RNN (Sci Rep 2026, no repo) | 2026 | MFCC + SMOTE + RNN | **Segmento (SMOTE-balanceado)** | — | — | Acc **98.6%** | ❌ **NÃO comparável** (ver MEMORY.md) |

### Achado da 2ª busca: somos o único protocolo rigoroso para a task
Todo número da literatura ≥ ao nosso cai em uma de três armadilhas, nenhuma comparável:
1. **SMOTE/balanceamento que infla** — Stockwell 2024 (AUC 0.98, mas W.Acc 0.93 >> vencedor do Challenge 0.78); SMOTE+RNN 2026 (Acc 98.6%, segmento).
2. **Task diferente** — caracterização (EHST 2025), grading (2023).
3. **Só weighted-accuracy sem AUROC** — times do Challenge.

O **WST+1D-CNN é o ÚNICO paper com a task exata (binário present vs absent) que reporta AUROC sem
balanceamento** — e mesmo ele é segment-level com vazamento (split por segmento). **Não existe na
literatura encontrada um benchmark paciente-level, sem vazamento, sem SMOTE, reportando AUROC/AUPRC
binário além do nosso.**

## Veredito: o modelo está **bom — no nível do SOTA honesto, provavelmente acima**

1. **Vs a comparação mais justa (WST+1D-CNN, binário, AUROC):** o número é quase idêntico
   (0.9375 vs 0.9345), **mas eles avaliam segment-level com split 80-20 por segmento — sem
   separação por paciente** (vazamento provável que infla a métrica). Nós fazemos nested CV
   paciente-level sem vazamento. Logo, a igualdade aparente **favorece o nosso** sob protocolo
   rigoroso.
2. **Vs o vencedor oficial do Challenge 2022 (HearHeart):** AUROC 0.9375 > 0.884. Ressalva: eles
   competiram em test set oculto e 3 classes (task mais difícil), otimizando *weighted accuracy*.
   Não é vitória limpa, mas **nenhum time do top-5 passa de AUROC 0.884**.
3. **O 98.6% do SMOTE+RNN não derruba o nosso:** é accuracy **por segmento** com classes
   balanceadas via SMOTE (bate melhor com `Outcome`, não `Murmur`). Métrica e granularidade
   diferentes — não comparável. Ver nota em `MEMORY.md`.
4. **Ponto a favor que a tabela não mostra:** somos dos poucos a reportar **AUPRC** (0.8844). Com
   prevalência ~20% Present, AUPRC é a métrica honesta e mais difícil — a maioria a omite.

**Resumo:** `lb300` (AUROC ~0.94, AUPRC 0.88) está **igual ou acima do SOTA reproduzível e honesto
do CirCor**; os únicos números maiores vêm de protocolos não comparáveis (segmento-level + SMOTE).

## Fontes
- PhysioNet/CinC Challenge 2022 overview — https://pmc.ncbi.nlm.nih.gov/articles/PMC10495026/
- WST + 1D-CNN — https://arxiv.org/abs/2303.11423
- Uncertainty-aware DL — https://arxiv.org/abs/2511.00966
- Training-free transformer — https://arxiv.org/abs/2509.18424
- Heart Murmur Quality + Attention — https://www.mdpi.com/2076-3417/14/15/6825
- CirCor DigiScope dataset paper — https://pmc.ncbi.nlm.nih.gov/articles/PMC9253493/
- SMOTE+RNN — `papers/2026_10.1038_s41598-026-45276-9.md` (no repo)
