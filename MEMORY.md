# MEMORY — memória de experimentos (detecção de sopro / CirCor)

> **Protocolo (obrigatório):**
> 1. **ANTES** de propor/rodar qualquer experimento de modelagem, **leia este arquivo inteiro** — evita repetir caminhos já descartados.
> 2. **DEPOIS** de concluir qualquer experimento, **anexe a conclusão** na seção "Log de experimentos" (run name, config, AUPRC/AUROC paciente-level, veredito em 1 linha). Atualize "Melhor modelo" e "Achado central" se mudarem.
> Mantenha sintético. Detalhe extra vai em `experiments/nested_tcn_systole_cnn/resumo_achados_modelagem.md`.

## Tarefa e métrica
- **Tarefa:** `Murmur` Present vs Absent (CirCor), **paciente-level**.
- **Métrica primária:** AUPRC OOF calibrado; **AUROC** como métrica comparável quando a prevalência muda.
- **Validação:** nested por paciente, 5 folds, sem vazamento (TCN treina só nos pacientes de treino do fold).
- **Dados:** 942 pacientes; Murmur Present 179 / Absent 695 / Unknown 68. ~3163 gravações (AV/PV/TV/MV).

## Melhor modelo
**`phase_contrast_lowband300_reuse_tcn` — AUPRC 0.8844 · AUROC 0.9375** (contraste + **banda baixa `--high-hz 300`**). Receita do `bc` + **`--phase-contrast --freq-norm global --high-hz 300`**.
- vs phase-contrast full-band (0.8771/0.9338): **melhora agregado E casos difíceis**: low-pitch 0.469→0.498, II/VI 0.738→**0.800**, `29045` 0.53→**0.909**, `50277` 0.878→0.935; Absent 0.0896→0.0810.
- **Sweep do corte (200/250/300/400/full) → PLATÔ em 200–300 Hz** (AUPRC 0.879–0.884, dentro do ruído de fold; 300 marginalmente melhor), **400 regride** (0.874), full 0.877. Ou seja: o sinal está em **≤~300 Hz**; o corte exato não é crítico, mas descartar a banda média/alta limpa ruído que diluía o low-pitch.
- **Surpresa:** restringir não derruba o high-pitch (0.968→0.971) — sopros agudos têm energia na banda baixa também. Confirma a v3.1 (banda baixa = onde está o sinal) no supervisionado.

### Phase-contrast full-band (variante anterior)
`phase_contrast_reuse_tcn` — AUPRC 0.8771 · AUROC 0.9338. Receita do `bc` + **`--phase-contrast --freq-norm global`**.
- **Phase-contrast** (`cnn/spectrogram.py::phase_contrast_spectrogram`): re-referencia a sístole pela diástole do **mesmo paciente** por frequência: `C[f,t] = log|sístole| − mediana_t log|diástole|`. Cancela coloração sensor/paciente, expõe o excesso sistólico (= o sopro). Para Absent, sístole≈diástole → C≈0 → **não infla Absent** (ganho de RANKING, não shift: Absent 0.1015→0.0896 enquanto Present sobe).
- **DOIS pré-requisitos não-óbvios** (cada um sozinho zera o efeito): (1) **`--freq-norm global`** — o per-bin re-branqueia e cancela o contraste; (2) **segmentação retém diástole** — o `bc` com `cnn-phase-mode=systole`+threshold zerava tudo que não era sístole; corrigido em `predict_tcn_segments` (retém diástole por argmax sem mexer na sístole; flag `phase_contrast`).
- Recupera os outliers diagnosticados: `29045` (low-pitch/representação) 0.022→**0.53** (vira TP); `50277` (segmentação) 0.369→**0.878**. Low-pitch +0.066, II/VI +0.041.

### Baseline anterior (referência)
`bc_locaware_perseg_focalfix_perfreqnorm` — AUPRC 0.8459 · AUROC 0.9115 · std-fold 0.064. Receita: labels **location-aware** · **STFT per-segment** · encoder `multiscale` `--dilations 1,2,4,8,16,32` pooling `attention` · loss `focal` γ=2 sem pos_weight · norma por bin · sístole via TCN `--systole-threshold 0.45 --systole-margin-ms 50`.

## ★ Achado central — o teto ~0.85 são os sopros suaves (I/VI)
Experimento `floor_audible_only_gt_systole`: excluindo os 104 sopros **I/VI** (de 179 Present), mantendo só Absent + audíveis (II/VI=28, III/VI=46), o **AUROC sobe de 0.90 → 0.955** (4/5 folds em 0.98-0.999). Os audíveis (II/VI+) já são detectados quase perfeitamente; **os I/VI seguram o teto**.

**Nuance importante (não simplificar):** I/VI **É audível** para um cardiologista ao vivo (por isso está rotulado Present). O que ocorre é que **o modelo não consegue separá-lo do normal NESTAS gravações**. Causas possíveis, **não distinguíveis só com os dados**:
1. sinal fraco/mascarado por ruído, ou não capturado no clipe (exame ao vivo usa manobras/posições/tempo que a gravação fixa não tem);
2. nossas features (STFT/log-mel + envelope) não surfam o sinal;
3. **ruído de rótulo** — grau I tem a menor concordância inter-observador.
→ Portanto NÃO é "piso irredutível absoluto", e sim "piso dado estas gravações + este pipeline". Isso mantém **abertas** alavancas de dados (denoising, filtro de qualidade, limpeza/incerteza de rótulo, gravações melhores).

**Implicação prática:** parar de caçar I/VI com truques de modelo/arquitetura (esgotado). Ganho real só via **dados/informação**. Clinicamente o modelo já é útil (pega bem II/VI+, os relevantes).

### Exceções diagnosticadas — dois falsos-negativos de alto grading (não são I/VI)
Investigação dos 2 outliers altos da lista de FN do `bc` (cruzando predição TCN vs ground-truth + energia sistólica/diastólica do sinal). **As duas causas que não são "piso informacional" existem e são distintas:**
- **`50277` (III/VI Harsh, mais audível PV) = ERRO DE SEGMENTAÇÃO DO TCN.** Predição vira com a segmentação: TCN cal **0.369** → GT cal **0.984** (mesmo fold). Sinal tem o sopro (E_sís/E_diá=7.9 no PV); TCN sub-segmentou a sístole (conf. 0.54-0.69; `50277_AV` só 31% anotado, 6 seg sístole vs 23 diástole) e contaminou a agregação paciente-level. **Recuperável** → reforça lever #6 (melhorar TCN) ou quality-gate por gravação antes de agregar.
- **`29045` (II/VI Blowing, Low pitch, mais audível TV) = FALHA DE REPRESENTAÇÃO (não seg., não rótulo).** Errado com TCN (0.022) **e com GT** (0.013; floor 0.001) — seg. perfeita não salva. Rótulo corroborado pelo sinal: gradiente de energia sistólica TV **20.3**≫PV 3.1>MV 2.4>AV 1.1 casa com "most audible TV". Raw baixo (0.30-0.33) em todos os folds. Hipótese: CNN aprendeu sopro *harsh* de média-freq; este é *blowing/low-pitch* (banda baixa) e a normalização por bin pode apagá-lo. **Exceção ao achado central:** existem II/VI audíveis que o modelo erra por representação, não por piso informacional. (Pendente: confirmação auditiva do `29045`; o ratio 20.3 merece descartar spike pontual.)

## Descartado — NÃO repetir (tudo empatou ou piorou vs bc 0.8459)
| Tentativa | AUPRC | Veredito |
|---|---:|---|
| Augmentation (SpecAugment+mixup CNN + SpecAug TCN) | 0.8436 | redistribui, colapsa fold fácil |
| Ramo features temporais (fill/flux/shape) — TCN | 0.8216 | tóxico c/ segmentação imperfeita |
| Ramo features temporais — ground-truth | 0.8371 | só empata com GT |
| Segmentos ground-truth (sem TCN) | 0.8387 | segmentador não limita o ranking |
| RNN (GRU bidir) | 0.7751 | pior em tudo; arq. errada p/ textura |
| peak1s (janelas 1s no onset) | 0.8274 | S1/S2/diástole diluem a sístole |
| freq2d Conv2d residual STFT(33) | 0.6298 | eixo freq raso + poucos dados |
| freq2d ms+aug+emph+attn log-mel(64) | 0.4779 | pior; Conv2d faminto por dados |
| Ramo MLP de frequência — GT | 0.8375 | empata, std cai p/ 0.033 |
| Ramo MLP de frequência — TCN | 0.8216 | piora c/ segmentação imperfeita |
| Ramo transformer de frequência — TCN | 0.8148 | capacidade demais p/ poucos dados |
| Fusão sístole+diástole | — | inviável (só 5 pacientes diastólicos) |
| **Quality-gate de segmentação na agregação** (descartar gravações mal-segmentadas antes de agregar) | 0.7775 vs 0.7742 baseline (Grupo I audio-level) | **inútil sob agregação `max`** |
| Norma **global** vs per-bin de frequência (`--freq-norm global`, Tier 1) | 0.8456 vs 0.8459 | **empata agregado**, mas confirma o mecanismo (ver nota abaixo) |
| **TCN melhorado** (épocas 10→20 + systole-weight 2→3) + phase-contrast | fold-1 0.889 (abortado no fold 1) | **subtreino REFUTADO**: TCN converge na ép.~3, overfita depois (val_macro_f1 travado em **0.766**, val_loss sobe); fold-1 a jusante 0.889 < phase-contrast 0.922 → não ajuda |

**Padrão:** mais capacidade/sofisticação não ajuda neste regime (~150 I/VI). Várias reduzem variância de fold mas às custas do AUPRC.

**Lição transversal (clustering → supervisionado):** ideias da linha Grupo B v3.x que ajudam o **clustering** (÷MAD robusto, center-crop, rectify/smooth/threshold, n_fft maior) **NÃO transferem** para o CNN supervisionado — ele já aprende essas transformações e já tem `--freq-norm global`. Só o **conceito** transferiu (contraste sístole-diástole + banda baixa ≤300 Hz = `lb300` 0.8844). Não re-testar refinamentos de feature hand-crafted no supervisionado.

**NB sobre agregação:** a agregação gravação→paciente é **`max`** (`cnn/cli.py:371`, `cnn/aggregate.py:14`; `bc` tem `patient_mil_attention=False`). Implicações: (1) gravações mal-segmentadas **já são ignoradas** — não puxam um Present para baixo; FN como `50277` só se conserta com **re-segmentação**, não com gate. (2) Testado quality-gate por plausibilidade de segmentação (sd_ratio, systole_segments) na agregação max do Grupo I: AUPRC 0.7742→0.7775 (ruído); gates agressivos pioram. **Descartado.** (3) `bc` não salva `recording_oof_predictions.csv` → não dá p/ re-agregar bc post-hoc sem retreinar.

## Eixos de dificuldade em nível de ÁUDIO (modelo audio-level, Grupo I)
Análise das predições por gravação (`Grupo I .../recording_oof_predictions.csv`, 497 áudios com sopro, 179 com prob_cal<0.5). NB: Grupo I é mais fraco que o `bc` paciente-level (canônico) — serve p/ achar *quais arquivos*, não como métrica.
- **Pitch baixo é o separador MAIS forte** (mais que grading): prob_cal média **Low 0.349 · Medium 0.679 · High 0.817**. Os piores áudios são quase todos *Low pitch* → reforça hipótese de **lacuna de representação na banda baixa** (norm. por bin pode apagá-la; ver `29045`).
- Grading em nível de áudio: I/VI 0.445 · II/VI 0.481 · III/VI 0.807 (confirma achado central por gravação).
- **Localização NÃO explica dificuldade:** gravação no ponto mais audível 0.578 vs longe dele 0.591 (refuta a intuição de que recordings distantes do foco seriam mais difíceis).
- Áudios mais difíceis (top): `85080_MV` II/VI (0.003), `29045_AV` II/VI (0.007), `50258_TV` I/VI (0.008), `50129_MV` II/VI (0.008), **`49754_TV` III/VI (0.015)** — forte e perdido, candidato a inspeção TCN→GT como 50277/29045.

**Tier 1 testado — norma `global` vs per-bin (`freqnorm_global_reuse_tcn`, reuso de TCN do bc):** mecanismo **confirmado mas não vira ganho agregado**. AUPRC OOF 0.8456 vs 0.8459 (empate). Porém **todas as fatias Present sobem**: pitch Low +0.047 (0.403→0.449), I/VI +0.048; e o `29045` (caso limpo de representação) tem o **raw quase dobrado** (0.351→0.628, cruza 0.5) — prova de que a norma por bin suprimia o sopro grave. Não chega ao agregado porque os **Absent também sobem** (desloca o operating point, não melhora o ranking — "redistribui, não soma"). `50277` (segmentação) piora (0.369→0.266), consistente. **Implicação:** o swap puro não vale; o sinal é recuperável → motiva (a) `dual` (per-bin p/ ranking + canal global p/ forma espectral) e sobretudo (b) **head auxiliar de pitch (Tier 2 multi-task)**. Infra: bug corrigido — `make_cnn_args` (`models/systole_cnn.py`) não propagava novas flags ao namespace da CNN; agora propaga `freq_norm` (e `aux_pitch_loss_weight`).

**Tier 2 testado — head auxiliar de pitch (`--aux-pitch-loss-weight`, CE mascarada nos Present sobre o pooled do encoder):** λ=0.3 **piora forte** (AUPRC 0.705 vs 0.846, −0.141) e — contraintuitivo — derruba mais o **low-pitch** (−0.166), a classe-alvo; FN@0.5 51→85. Causa provável: encoder pequeno (`base_channels=16`) + sinal de pitch esparso (~150 Present) → a CE de pitch (tarefa difícil *entre* sopros) domina e distorce o encoder compartilhado para discriminar pitch em vez de murmúrio-vs-normal; low-pitch (pitch mais difícil) gera o gradiente mais ruidoso. Sweep de λ feito: λ=0.1 → AUPRC 0.7615 (−0.084), low-pitch 0.295 (ainda pior que bc 0.403). **Monotônico em λ: quanto menor, mais converge de volta ao bc, nunca supera.** → **Tier 2 multi-task pitch DESCARTADO** (encoder pequeno `base_channels=16` + rótulos de pitch esparsos ~150 Present não dão para compartilhar capacidade sem degradar a tarefa principal). Implementação fica atrás de flag (default 0, backward-compat); validações: requer arch cnn, incompat SMOTE/mixup/MIL. **Padrão (revisado):** global/quality-gate/multi-task não bateram o bc, MAS o **phase-contrast (contraste sístole/diástole + global norm) bate** — AUPRC 0.8771 (ver "Melhor modelo"). Lição: o que funciona é **DSP fixo que injeta domínio** (re-referência por diástole), não capacidade/loss aprendíveis. Parte do "teto informacional" era na verdade sinal **mascarado** (recuperável: 29045/50277 recuperados), não ausente. Dual-canal `[sístole, contraste]` JÁ TESTADO → **pior** (0.8554 vs 0.8771): a sístole crua devolve o confound que o contraste removeu (29045 0.53→0.266). Pure contrast vence. Frentes ainda abertas **sobre o phase-contrast simples como base**: **reframe do alvo** (II/VI+ ou `Outcome` — hoje alvo é só `Murmur`); transfer PhysioNet 2016; demografia. **Melhorar segmentação TCN: a via fácil (mais épocas) está REFUTADA** — o TCN converge na ép.~3 e overfita (macro-f1 ~0.766); ganho só viria de levers diferentes (transfer/SSL, label-smoothing p/ as bordas que são o maior erro — recall 0.50 a <10ms, mais dados), não de treino mais longo. Regra de ouro descoberta: **qualquer normalização/realce por gravação exige `--freq-norm global`** (per-bin cancela).

## Exploração (overlap por grading) — `analise_exploratoria_sistole_overlap/`
- ~50% dos murmúrios (250/497 gravações) têm ≥80% de vizinhos Absent no espaço de envelope espectral da sístole.
- Correlação monotônica com grading: difíceis = 130 I/VI, 82 II/VI, 36 III/VI. Fáceis dominados por III/VI.

## Comparação com paper "SMOTE+RNN" (98.6%) — NÃO comparável
`1617/1546` deles bate com o rótulo **Outcome (Normal/Abnormal, ~balanceado)**, não Murmur (~20%). Reportam **accuracy por SEGMENTO** (balanceado via SMOTE), não AUPRC paciente-level. AUROC paciente-level do `bc` (0.91) está no nível do SOTA honesto do CirCor.

## Caminhos com fundamento (ainda não esgotados) — alavancas de DADOS, não arquitetura
1. **Supervisão auxiliar (multi-task):** prever `grading`/`timing`/`shape` junto — sinal mais rico p/ a classe difícil (usa rótulos que já existem).
2. **Demografia como entrada:** `Age/Sex/Height/Weight/Pregnancy` (não depende de segmentação).
3. **MIL multi-local de verdade** (`--patient-mil-attention`, nunca usado a sério).
4. **Transfer / pré-treino** (PhysioNet 2016, SSL em PCG) p/ escassez de dados.
5. **Denoising / filtro de qualidade** das gravações; **limpeza/incerteza de rótulo** nos I/VI.
6. **Melhorar a segmentação do TCN** (destrava ramos de banda/dinâmica que hoje são tóxicos com TCN-pred).
7. **Ensemble por seed** do `bc` (antes inviável por variância alta).

## Infra / código
- Arquitetura mora em `nested_tcn_systole_cnn/cnn/` e `.../tcn/` (subpacotes por responsabilidade); `modeling/Grupo G|E/` são shims de compatibilidade.
- Arquiteturas: `--model-arch {cnn,rnn,freq2d}`. Ramo de freq: `--freq-linear-branch --freq-linear-arch {mlp,transformer}`. freq2d: `--freq-emphasis --freq-attention`. Floor: `--exclude-present-grades "I/VI"`.
- **Persistência incremental por fold** (`fold_metrics_partial.csv`, `patient_oof_partial.csv`) + prints com flush → acompanhar runs ao vivo.
- **Limpeza incremental de disco** (`try/finally` no TCN) evita cache órfão.
- **Reuso de TCN:** copiar `fold_*/tcn/best_model.pt` do `bc` (mesmos args de TCN) → pula retreino (~25min só CNN). Disco é apertado (~11 GB livres); GT evita o cache pesado do TCN.

## Log de experimentos (anexar novos aqui)
| run | segmentação | AUPRC | AUROC | nota |
|---|---|---:|---:|---|
| bc_locaware_perseg_focalfix_perfreqnorm | TCN | 0.8459 | 0.9115 | **melhor** |
| groundtruth_segments_bc | GT | 0.8387 | 0.9036 | tarefa completa GT |
| augmentation_full_bc | TCN | 0.8436 | — | aug pesada |
| temporal_branch_bc | TCN | 0.8216 | — | ramo temporal |
| temporal_branch_groundtruth | GT | 0.8371 | 0.9056 | ramo temporal |
| rnn_groundtruth | GT | 0.7751 | 0.8638 | RNN |
| peak1s_groundtruth_cnn | GT | 0.8274 | 0.8986 | janelas 1s |
| freq2d_groundtruth_systole | GT | 0.6298 | 0.7737 | Conv2d residual STFT |
| freq2d_ms_aug_emph_attn_logmel_gt_systole | GT | 0.4779 | 0.6941 | freq2d kitchen-sink |
| cnn_freqlinear_gt_systole | GT | 0.8375 | 0.8919 | ramo MLP freq (std 0.033) |
| cnn_transformer_freqbranch_tcn | TCN | 0.8148 | 0.8905 | ramo transformer freq |
| cnn_mlp_freqbranch_tcn | TCN | 0.8216 | 0.9051 | ramo MLP freq |
| floor_audible_only_gt_systole | GT | 0.8909 | 0.9550 | **exclui I/VI — mede o teto** |
| freqnorm_global_reuse_tcn | TCN | 0.8456 | 0.9097 | norma global vs per-bin: empata agregado (−0.0003) |
| aux_pitch_w03_reuse_tcn | TCN | 0.7049 | 0.8055 | Tier 2 multi-task pitch λ=0.3: **piora forte** (−0.141); λ alto demais |
| aux_pitch_w01_reuse_tcn | TCN | 0.7615 | 0.8462 | Tier 2 pitch λ=0.1: piora (−0.084); monotônico em λ → só converge ao bc, **descartado** |
| phase_contrast_reuse_tcn | TCN | **0.8771** | **0.9338** | **MELHOR**: contraste sístole/diástole + `--freq-norm global`; +0.031 vs bc, todos folds ≥ bc |
| phase_contrast_dual_reuse_tcn | TCN | 0.8554 | 0.9238 | dual `[sístole,contraste]`: **pior que o simples** (−0.022); a sístole crua dilui o contraste |
| phase_contrast_robust_reuse_tcn | TCN | 0.8533 | 0.9247 | contraste robusto ÷MAD (da v3.1): **pior que o simples** (−0.024); CNN já normaliza, ÷MAD amplifica bins de diástole quieta (ruído) |
| phase_contrast_lowband{200,250,300,400} | TCN | 0.882/0.879/**0.884**/0.874 | 0.940/0.932/0.938/0.935 | **sweep do corte**: platô 200–300 (300 melhor), 400 regride; full=0.877. **lb300 = MELHOR** |
| phase_contrast_lb300 + margin 0 (vs 50) | TCN | ~0.839 (4 folds, abortado) | — | **center-crop da v3.2 NÃO transfere**: margin 0 ~−0.04 vs lb300; a expansão de 50ms ajuda (mais cobertura da sístole; o contraste já cancela S1/S2 das bordas) |
| phase_contrast_lb300 + n_fft 256 (vs 128) | TCN | 0.8750 | 0.9315 | **resolução de freq maior NÃO transfere** (−0.009; low-pitch 0.498→0.449): janela maior piora resolução temporal; sinal está na textura temporal |
| phase_contrast_lb300 + demografia (ramo somado) | TCN | 0.8759 | 0.9322 | **demografia NÃO ajuda o Murmur** (−0.009): sinal fraco (Age fraco, Sex ~0). Ramo implementado/reutilizável |
| **reframe `Outcome`** (Normal/Abnormal, GT seg, phase-contrast+lb300) | 0.7106 | 0.6764 | **DESCARTADO pelo usuário**: teto acústico baixo (AUROC 0.68 vs Murmur 0.94) — pipeline é especializado em **evento acústico** (sopro); 263 Abnormal sem sopro não têm assinatura. Demografia ajudava aqui (AUROC ~0.71, 2 folds) mas Outcome **não é o caminho** |
| feature_b3_systole_diastole_contrast_cluster | GT `.tsv` | — | — | exploratorio nao supervisionado: contraste puro sístole−diástole gerou cluster paciente 7/7 Present (100%, captura 3.9%); visualmente limpo, mas estreito; nao altera melhor modelo |
| feature_b31_robust_cycle_contrast_cluster | GT `.tsv` | — | — | exploratorio nao supervisionado: z-contraste por ciclo + bandas; low k=10 clusters enriquecidos 109 pacientes/95 Present (87.2%, captura 53.1%); low k=2 69 pacientes/97.1%; melhor cluster exploratorio, sem AUPRC/AUROC |
| feature_b32_enhanced_murmur_map_cluster | GT `.tsv` | — | — | exploratorio nao supervisionado: corte central + mapa positivo suavizado/thresholdado; low k=10 clusters 101 pacientes/98 Present (97.0%, captura 54.7%); low k=2 85/84 Present (98.8%); melhora v3.1, sem AUPRC/AUROC |
| feature_b32_cluster_optimization_focused | GT `.tsv` | — | — | exploratorio nao supervisionado sobre features ja extraidas: melhor amplo `low_persistence`+standard+PCA50+k10 = 105 pacientes/102 Present (97.1%, captura 57.0%); PCA2+k9 = 101/99 (98.0%, captura 55.3%); cluster unico PCA2+k2 = 85/85 (100%, captura 47.5%); sem AUPRC/AUROC |
| feature_b33_lowband_texture_separability | GT `.tsv` | — | — | exploratorio: textura na banda baixa (<=260 Hz) + camada de separabilidade Present×Absent. **Feature individual mais forte = NOVA: `tex_gini_map` (AUC 0.864, d=-1.53)** → sopro = Gini BAIXO = energia espalhada/sustentada (vs normal concentrado/transiente); bate a melhor do v3.2 (energia low-mid 0.834). Textura ADICIONA pouco no multivariado (Mahalanobis combinado 3.02 vs v3.2 2.84; AUC dist-ao-Absent 0.851 vs 0.838, nivel gravacao). **Score `dist_to_absent` (Mahalanobis ao centroide Absent) ordena por grading**: Absent 20.8 < II/VI 37.9 ~ I/VI 49.8 << III/VI 181.2 — reproduz quantitativamente o achado central (suaves colados no normal). RESSALVA: metricas in-sample, sem CV; `dist_to_absent_auc=1.0` paciente e artefato p>>n (ignorar). AUC por feature = leitura robusta. Sem AUPRC/AUROC supervisionado. |
