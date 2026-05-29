# Resumo de achados — modelagem nested TCN + CNN (detecção de sopro CirCor)

Consolidação das explorações para **não repetir caminhos já descartados**. Tarefa: `Murmur`
Present vs Absent, **paciente-level**, métrica **AUPRC OOF calibrado**, validação por paciente (5 folds).

## ★ ACHADO CENTRAL — o teto ~0.85 são os sopros I/VI inaudíveis (piso irredutível medido)

Experimento `floor_audible_only_gt_systole`: treinar/avaliar excluindo os 104 sopros I/VI (suaves),
mantendo só Absent + audíveis (II/VI=28, III/VI=46). Mesma receita, ground-truth. AUROC é a métrica
de comparação (independe da prevalência, que muda de 20.5% → 9.6%).

| Tarefa | AUPRC | **AUROC** |
|---|---:|---:|
| Completa (todos os grades) | 0.8387 | 0.9036 |
| **Detectável (II/VI+III/VI, sem I/VI)** | 0.8909 | **0.9550** |

AUROC por fold detectável: [0.906, 0.999, 0.983, 0.994, 0.977] — **4 de 5 folds em 0.98-0.999**.

**Conclusão (vale para qualquer experimento futuro):**
- Os sopros **audíveis (II/VI+) já são detectados quase perfeitamente** (AUROC ~0.97). O modelo NÃO
  é o gargalo nesses casos.
- Os **I/VI seguram o teto sozinhos** (AUROC −0.05). NUANCE: I/VI **é audível** para um cardiologista
  ao vivo (por isso é Present) — o modelo é que não o separa do normal NESTAS gravações. Causas não
  distinguíveis só com os dados: sinal fraco/mascarado por ruído ou não capturado no clipe; features
  insuficientes; ou ruído de rótulo (grau I tem baixa concordância inter-observador). É piso "dado
  estas gravações + este pipeline", **não** um limite absoluto → mantém abertas alavancas de dados.
- **NÃO tentar subir o número global com truques de modelo/arquitetura** — já esgotado (ver tabela
  abaixo). O ganho real só vem de **dados/informação** (supervisão auxiliar de grading/timing,
  demografia, transfer PhysioNet 2016) ou de **aceitar** que os I/VI são piso.
- Clinicamente o modelo já é útil: triador que pega bem II/VI+ (os relevantes) e erra alguns I/VI.

## Diagnóstico dos falsos-negativos de alto grading (não-I/VI) — TCN vs GT + sinal

Dos 51 falsos-negativos do `bc` (Present com prob_cal<0.5; 47 são I/VI), os 2 outliers de alto
grading foram investigados cruzando a predição com **segmentação TCN vs ground-truth** e a **energia
da banda de sopro (100-700 Hz) em sístole vs diástole** (ground-truth `.tsv`, fs 4000 Hz). Método:
se o erro persiste com segmentos ground-truth, não é segmentação; se o gradiente de energia sistólica
por local bate com `Most audible location`, o rótulo está corroborado pelo sinal (não é ruído de rótulo).

| paciente | grading | predição TCN→GT | sinal (E_sís/E_diá por local) | veredito |
|---|---|---|---|---|
| **50277** | III/VI Harsh, audível PV | cal **0.369 → 0.984** (vira) | PV **7.9**, MV 5.3, AV 1.7, TV 1.0 | **erro de segmentação do TCN** |
| **29045** | II/VI Blowing/Low, audível TV | cal **0.022 → 0.013** (não vira; floor 0.001) | TV **20.3**, PV 3.1, MV 2.4, AV 1.1 | **falha de representação** (não seg., não rótulo) |

- **50277:** com ground-truth o III/VI é detectado quase perfeito (0.98). Com TCN cai para 0.37 porque
  o segmentador sub-anotou a sístole (`predicted_segmentation_quality`: conf. 0.54-0.69; `50277_AV`
  só 31% do sinal anotado, 6 segmentos de sístole vs 23 de diástole). As gravações mal-segmentadas
  contaminaram a agregação paciente-level. → **recuperável**: lever #6 (melhorar TCN) ou *quality-gate*
  que descarte gravações de baixa confiança de segmentação antes de agregar.
- **29045:** ground-truth perfeito **não** salva (0.013, raw 0.30-0.33 em todos os folds). O rótulo é
  corroborado pelo sinal (gradiente sistólico TV≫PV>MV>AV casa com "most audible TV"), então **não é
  ruído de rótulo**. É um **miss real do modelo** sobre sopro audível — provável lacuna de representação:
  sopro *Low pitch / Blowing* (turbulência de banda baixa) vs CNN aparentemente sintonizado em textura
  *harsh* de média-frequência; a normalização por bin de frequência pode apagar o excesso de banda baixa.
  **Abre uma exceção ao achado central:** nem todo caso difícil é I/VI informacional.

Ressalvas: (1) `predicted_segmentation_quality.csv` vem do experimento Grupo B v2 TCN-predito (config de
TCN possivelmente diferente do `bc`) — usado só como corroboração; a evidência primária é o flip TCN→GT
nas predições OOF reais. (2) O ratio 20.3 do `29045_TV` merece inspeção auditiva para descartar spike pontual.

## Eixos de dificuldade em nível de ÁUDIO (gravação) — modelo audio-level (Grupo I)

Predições por gravação em `modeling/Grupo I Nested TCN CNN audio murmur/outputs_audio_level_multiscale_dilatado_smote/recording_oof_predictions.csv`
(497 áudios com `audio_target=1`; 179 com `prob_murmur_calibrated<0.5`). O Grupo I é **mais fraco**
que o `bc` paciente-level (canônico) — vale para identificar *quais arquivos* são difíceis, não como métrica.

**Eixos (prob calibrada média por gravação):**
- **Pitch é o separador mais forte, acima do grading:** Low **0.349** · Medium 0.679 · High 0.817.
  Os sopros graves (baixa frequência) são os mais perdidos → reforça a hipótese de lacuna de
  representação na banda baixa (a normalização por bin pode apagá-los; ver `29045` acima).
- Grading por áudio: I/VI 0.445 · II/VI 0.481 · III/VI 0.807 (consistente com o achado central).
- **Localização NÃO explica dificuldade:** gravação no `Most audible location` 0.578 vs longe dele 0.591.
  Refuta a intuição de que recordings distantes do foco do sopro seriam os difíceis.

**Top áudios difíceis (sopro com menor prob):** `85080_MV` II/VI Low (0.003), `29045_AV` II/VI Low/Blowing
(0.007), `50258_TV` I/VI Low (0.008), `50129_MV` II/VI Low (0.008), **`49754_TV` III/VI Medium (0.015)**.
O `49754_TV` é um III/VI forte e perdido — candidato ao mesmo discriminador TCN→GT usado em 50277/29045.

## Quality-gate de segmentação na agregação — TESTADO E DESCARTADO

Hipótese (Tier 1b): descartar/ponderar gravações mal-segmentadas antes de agregar gravação→paciente
recuperaria FN como o `50277`. **Investigação do código refutou a premissa:** a agregação é **`max`**
(`nested_tcn_systole_cnn/cnn/cli.py:371` no caminho não-MIL, e `cnn/aggregate.py:14`; `bc` roda com
`patient_mil_attention=False`). Sob `max`, gravações fracas/mal-segmentadas **já são ignoradas** — o
paciente fica com a sua melhor gravação. Logo o gate **não recupera** FN de murmúrio (o problema do
`50277` é que a *melhor* gravação piorou com TCN → só re-segmentação conserta); só poderia aparar
falsos-positivos (Absent com `max` espúrio-alto).

Teste empírico (proxy no Grupo I audio-level — `bc` não salva `recording_oof_predictions.csv`, então
não há como re-agregar o `bc` post-hoc sem retreinar). Agregação `max` gravação→paciente, com vs sem gate:

| Configuração | AUPRC | AUROC | F1max | drop |
|---|---:|---:|---:|---|
| baseline (sem gate) | 0.7742 | 0.8900 | 0.7179 | — |
| gate sd_ratio<0.80 | 0.7743 | 0.8901 | 0.7179 | 18/3003 |
| gate sd_ratio<0.90 | 0.7775 | 0.8923 | 0.7254 | 156/3003 |
| gate systole_segments<18 | 0.7649 | 0.8799 | 0.7117 | 144/3003 |
| gate systole_segments<22 | 0.7607 | 0.8754 | 0.7105 | 292/3003 |

Melhor variante: +0.003 AUPRC (ruído), à custa de descartar 156 gravações; variantes agressivas pioram.
**Veredito: inútil sob agregação `max`.** O caminho para casos tipo `50277` é re-segmentação (melhorar o
TCN, lever #6), não gating. Caveat: proxy de qualidade por contagem de segmentos (sd_ratio/systole_segments),
não a confiança real do TCN; mas o argumento da agregação `max` limita o teto independentemente do proxy.

## Tier 1 testado — normalização `global` vs per-bin de frequência (`freqnorm_global_reuse_tcn`)

Hipótese: a normalização por bin (`training.py` `specs.mean(axis=(0,2))`) z-scoreia cada bin de
frequência, achatando a razão de energia entre bandas que codifica o *pitch* — o que cegaria o modelo
para sopros graves. Teste: flag nova `--freq-norm {perbin,global}`; `global` usa um escalar único,
preservando a forma espectral. Run reusando os 5 checkpoints TCN do `bc` (~25 min, só CNN).

**Bug encontrado e corrigido antes de concluir:** o primeiro run deu resultado **bit-idêntico** ao `bc`.
Causa: `models/systole_cnn.py:make_cnn_args` reconstrói um `Namespace` novo para a CNN e não propagava
`freq_norm` → `train_one_fold` caía no default `perbin`. Corrigido (propaga `freq_norm`). Relançado.

**Resultado (agregado OOF calibrado, paciente-level):** AUPRC **0.8456** vs `bc` 0.8459 (Δ −0.0003);
AUROC 0.9097 vs 0.9115. **Empate no agregado.** Mas o fatiamento confirma o mecanismo:

| Fatia (prob_cal média dos Present) | bc | global | Δ |
|---|---:|---:|---:|
| pitch Low | 0.403 | 0.449 | **+0.047** |
| pitch Medium | 0.845 | 0.872 | +0.026 |
| pitch High | 0.960 | 0.972 | +0.012 |
| grading I/VI | 0.516 | 0.564 | **+0.048** |
| grading II/VI | 0.697 | 0.706 | +0.008 |
| grading III/VI | 0.948 | 0.960 | +0.013 |
| **29045** (raw) | 0.351 | **0.628** | **+0.277** |

Folds global: 0.894, 0.801, 0.880, 0.845, 0.882 (média 0.860, var menor que bc 0.851). FN@0.5: 51→48.

**Interpretação:** o `29045` (caso limpo de lacuna de representação) tem o raw quase dobrado e cruza 0.5
→ **a norma por bin de fato suprimia o sinal de sopro grave**. Mas o ganho não chega ao AUPRC agregado
porque os **Absent também sobem** — a norma global desloca o operating point para cima uniformemente em
vez de melhorar o ranking ("redistribui, não soma"). O `50277` (erro de segmentação) piora (0.369→0.266),
coerente com o diagnóstico de que seu problema não é representação. **Verdito:** o swap puro perbin→global
não é ganho isolado, mas valida que o sinal de baixa frequência é recuperável do espectro → motiva (a) a
variante `dual` (per-bin para discriminação + canal global para a forma espectral) e, sobretudo, (b) o
**head auxiliar de pitch (Tier 2 multi-task)**, que imporia essa representação via loss sem deslocar o Absent.

## Tier 2 testado — head auxiliar de pitch (multi-task) — λ=0.3 NEGATIVO, sweep em andamento

Implementação: flag `--aux-pitch-loss-weight`; head linear sobre o pooled do encoder (dim `c`, antes dos
ramos) prevendo pitch Low/Med/High; `loss = focal + λ·CE(ignore_index=-1)`, supervisão **só nos Present**
(recordings Absent e pitch faltante → -1). Objetivo: forçar o encoder a representar a banda baixa que a
norma por bin apaga, via loss (sem inflar o Absent como a norma global fazia). Default 0 = backward-compat;
validações impedem combinar com SMOTE/mixup/MIL e exigem `--model-arch cnn`. Smoke test (60 pac, GT) OK.

**Resultado λ=0.3 (reuso de TCN do bc):** AUPRC OOF **0.7049** vs `bc` 0.8459 (**Δ −0.141**), AUROC 0.8055.
Folds: 0.743, 0.616, 0.787, 0.643, 0.757 (todos abaixo do bc). Contraintuitivo: o **low-pitch piora mais**
(prob_cal média 0.403→0.236, **−0.166**), justamente a classe-alvo; Medium −0.112, High ≈. FN@0.5 51→85.
`29045` quase parado (0.022→0.026).

**Diagnóstico:** encoder pequeno (`base_channels=16`) + sinal de pitch esparso (~150 Present, e pitch só
existe neles) → com λ alto a CE de pitch (tarefa difícil *entre* os sopros) domina e distorce o encoder
compartilhado para discriminar pitch em vez de murmúrio-vs-normal. O low-pitch, sendo o pitch mais difícil
de prever, gera o gradiente mais ruidoso → degrada a própria classe que se queria ajudar. **Sweep de λ (veredito):** λ=0.1 → AUPRC 0.7615 (−0.084 vs bc); low-pitch 0.295 (ainda pior que bc 0.403);
29045 0.022→0.064; FN@0.5 51→71. Comportamento **monotônico em λ**: 0.3→0.705, 0.1→0.762 — quanto menor o
λ, mais converge de volta ao bc, **nunca o supera**. → **Tier 2 multi-task pitch DESCARTADO.** Causa: encoder
pequeno (`base_channels=16`) + rótulos de pitch esparsos (~150 Present) não permitem compartilhar capacidade
sem degradar a discriminação murmúrio-vs-normal; o low-pitch (pitch mais difícil) é o mais prejudicado em
todo λ. Alavancas teóricas não exploradas (warmup de λ, base_channels maior, aux só no fim) ficam abaixo das
frentes não-model-side abaixo na ordem de prioridade.

## Estado estratégico — model-side esgotado, próximas frentes

Três levers model-side seguidos não bateram o bc 0.8459: `global` (empate), quality-gate (inútil sob max),
multi-task pitch (piora). **O teto ~0.846 é informacional**, confirmado. Frentes com fundamento, em ordem:
1. **`dual` norm** — única model-side ainda não testada; adiciona o canal de forma espectral sem competir por loss.
2. **Reframe do alvo** — hoje o classificador usa só `Murmur` Present/Absent (`pipeline.py:41`); `Outcome`
   (Normal/Abnormal) é carregado mas **não é alvo**. `Outcome` é ~balanceado, vem do exame clínico completo
   (pode furar parte do piso I/VI) e é o que o paper SMOTE bateu. Alternativa barata: reportar II/VI+ (AUROC 0.955).
3. **Transfer PhysioNet 2016** — ataca a escassez (~150 sopros audíveis); maior alavanca p/ subir o teto.
4. **Demografia** como entrada (cheap, ortogonal) e **melhor segmentação TCN** (recupera o bucket 50277).

## ★ Phase-contrast (contraste sístole/diástole) — NOVO MELHOR MODELO

`phase_contrast_reuse_tcn` (receita do bc + `--phase-contrast --freq-norm global`, reuso de TCN):
**AUPRC OOF 0.8771 · AUROC 0.9338**, vs bc 0.8459/0.9115 → **+0.031 / +0.022**. Folds: 0.922, 0.794,
0.897, 0.865, 0.929 — **todos ≥ bc**. Primeiro lever a superar o baseline.

Ideia (DSP fixo, zero params): re-referenciar a sístole pela diástole do mesmo paciente, por frequência:
`C[f,t] = log|sístole|[f,t] − mediana_t log|diástole|[f]`. Cancela a coloração sensor/paciente comum às
duas fases e expõe o excesso sistólico (= o sopro). Importado do conceito de "features relativas" do
Grupo B v2 (melhor clustering) para o input supervisionado.

**Discriminador (confirma ganho de ranking, não shift):** prob_cal média dos Absent **caiu** 0.1015→0.0896
(o Tier 1 global inflava tudo); Present subiu em todas as fatias: Low-pitch +0.066 (0.403→0.469), Medium
+0.037, II/VI +0.041, III/VI +0.021. Para Absent o contraste≈0 → prob baixa por construção.

**Outliers diagnosticados, recuperados:** `29045` (low-pitch/representação, que nem o GT consertava)
0.022→**0.53** (vira TP); `50277` (erro de segmentação) 0.369→**0.878** (a re-referência por diástole
compensou parte da segmentação ruim — e o III/VI sistólico dominou apesar do sopro diastólico III/IV).

**DOIS pré-requisitos não-óbvios (cada um sozinho ZERA o efeito), descobertos depurando 2 runs que saíram
idênticos a bc/global:**
1. **`--freq-norm global`** — a norma por bin subtrai a média por bin no treino e **re-branqueia** o
   contraste (se a referência de diástole é ~constante entre gravações, o per-bin a absorve → vira bc).
2. **Segmentação retém diástole** — `predict_tcn_segments` com `cnn-phase-mode=systole`+threshold zerava
   tudo que não era sístole → sem diástole, o contraste caía no fallback (sístole pura) → virava o run
   `global`. Corrigido: retém frames de diástole por argmax sem alterar a seleção de sístole (flag
   `phase_contrast`); verificado em `29045_TV` (47 sístole + 47 diástole vs 47/0 antes).

**Regra de ouro:** qualquer realce/normalização **por gravação** exige `--freq-norm global`.

**Dual-canal `[sístole, contraste]` — TESTADO, PIOR:** `phase_contrast_dual_reuse_tcn` AUPRC 0.8554 /
AUROC 0.9238 (−0.022 vs o simples 0.8771; ainda > bc). Empilhar a sístole crua junto do contraste
**dilui** o ganho — a sístole crua reintroduz a coloração por gravação que o contraste removeu
(`29045` 0.53→0.266, volta a FN; low-pitch 0.469→0.455). "Redistribui, não soma." **O phase-contrast
simples permanece o melhor.**

### Melhorar a segmentação do TCN (épocas + systole-weight) — subtreino REFUTADO, abortado
`phase_contrast_bettertcn_e20_sw3` (TCN 10→20 épocas + systole-weight 2→3, retreino do zero). A **curva de
treino do TCN no fold 1 refuta o subtreino**: val_macro_f1 = 0.763 (ép.1) → 0.766 (ép.10/12, pico) → 0.764
(ép.20); val_loss fundo na ép.2-3 e depois **sobe** (overfit) enquanto train_loss despenca. O TCN converge
na ép.~3; as 10 do bc já bastavam. Fold-1 a jusante **0.889 < phase-contrast 0.922** → abortei (não vale
~1.5h pelo resto). **Lição:** o teto de segmentação (~0.76 macro-f1) é limite de **modelo/dados**, não de
treino. As bordas — maior fonte de erro (recall 0.50 a <10ms, AV 0.61) — não melhoram com mais épocas;
exigiriam label-smoothing nas fronteiras, mais dados, ou transfer/SSL. E o `--systole-margin-ms 50` já
compensa boa parte da sub-detecção de borda a jusante, então o headroom de "TCN melhor" é pequeno.

### Contraste robusto ÷MAD (da v3.1) — TESTADO, PIOR no supervisionado
`phase_contrast_robust_reuse_tcn`: `C=(log|sís|−mediana_t(diá))/(MAD_t(diá)+0.03)`, clip ±12 (z robusto da
v3.1). AUPRC **0.8533** / AUROC 0.9247 — **−0.024 vs o simples 0.8771** (ainda > bc). Low-pitch 0.448
(<0.469 do simples); `29045` 0.53→0.479 (volta a FN). **Por que a v3.1 ganhou com ÷MAD e aqui não:** no
clustering o ÷MAD torna features comparáveis por distância (ajuda); no CNN supervisionado, o encoder já
aprende a escala por feature + há `--freq-norm global`, então o ÷MAD é redundante e **amplifica bins de
diástole quieta** (MAD pequeno → divisão estoura no clip → ruído). **Lição: normalização que ajuda
clustering não ajuda o supervisionado.** Contraste simples (subtração) permanece o melhor.

### Ênfase na banda baixa (`--high-hz 250`) — MELHORA os casos difíceis (3º elemento da v3.1)
`phase_contrast_lowband250_reuse_tcn`: contraste simples restrito à banda ≤250 Hz (a "low band" da v3.1).
AUPRC **0.8792** / AUROC 0.9315 — agregado ~empata o full-band (0.8771/0.9338), **mas melhora claramente
o que é difícil**: low-pitch 0.469→**0.501** (cruza 0.5), II/VI 0.738→0.776, I/VI 0.572→0.591, `29045`
0.53→**0.826**, `50277` 0.878→0.935; Absent 0.0896→**0.0767** (melhor separação); FN@0.5 48→46. **Surpresa
(ressalva refutada):** descartar 250-1000 Hz **não derruba o high-pitch** (0.968→0.971) — sopros agudos têm
energia na banda baixa também, e remover a banda alta **limpou ruído** que diluía os low-pitch. **A tese da
v3.1 (banda baixa = onde está o sinal) transfere para o supervisionado.** Novo melhor modelo.

Dos 3 elementos da v3.1 no supervisionado: subtração ✅ (phase-contrast), ÷MAD ❌ (CNN já normaliza),
**banda baixa ✅**. Falta agregação por ciclo (provável pequeno efeito — o attention pooling já agrega ciclos).

**Sweep do corte de banda (200/250/300/400/full):**

| corte | AUPRC | AUROC | low-pitch | II/VI | 29045 | 50277 |
|---|---:|---:|---:|---:|---:|---:|
| full(1000) | 0.8771 | 0.9338 | 0.469 | 0.738 | 0.530 | 0.878 |
| 200 | 0.8819 | 0.9397 | 0.513 | 0.787 | 0.688 | 0.962 |
| 250 | 0.8792 | 0.9315 | 0.501 | 0.776 | 0.826 | 0.935 |
| **300** | **0.8844** | 0.9375 | 0.498 | 0.800 | 0.909 | 0.935 |
| 400 | 0.8744 | 0.9353 | 0.507 | 0.785 | 0.619 | 0.993 |

**Platô em 200–300 Hz** (AUPRC 0.879–0.884, dentro do ruído de fold — sem pico afiado), **400 regride**
(volta a readicionar a banda média que dilui), full=0.877. `300` marginalmente melhor (AUPRC + II/VI 0.800 +
`29045` 0.909) → **corte de operação 300 Hz, novo melhor modelo (0.8844)**. O aprendizado robusto não é "250"
nem "300" — é que **o sinal discriminativo está em ≤~300 Hz e descartar o resto limpa ruído**.

Próximos passos: **reframe** (`Outcome` / II/VI+); **transfer PhysioNet 2016**; demografia. (Testar <200 Hz
é baixo valor — platô + risco de sobrepor S1/S2; agregação por ciclo da v3.1 ainda em aberto.)

### Triagem das ideias da v3.2 (murmur map realçado) p/ o supervisionado
A v3.2 melhorou o clustering (cluster único 98.8% Present) com 7 incrementos. Regra aprendida (vinda do ÷MAD):
**transformações que o CNN já aprende não transferem; só mudanças de informação transferem.**
- **Provável-redundante (CNN já faz):** rectificar `max(z,0)`, suavização gaussiana, threshold `z≥1`,
  features de persistência. ÷MAD **já testado, pior** (0.8533).
- **Mudança de informação (testável):** (A) **corte central da sístole** (remover S1/S2 das bordas);
  (C) **n_fft maior na banda baixa** (mais resolução).
- **(A) margin 0 — TESTADO, PIOR** (~0.839 em 4 folds vs lb300 0.881; abortado): a expansão de 50ms ajuda
  (o contraste já cancela S1/S2 das bordas → a expansão só dá mais cobertura de sístole). Center-crop não transfere.
- **(C) n_fft 256 — TESTADO, PIOR** (0.8750 vs 0.8844; low-pitch 0.498→0.449, `29045` 0.909→0.712):
  janela maior piora a resolução temporal; o sinal do sopro está na **textura temporal** (envelope
  sustentado), não em detalhe fino de frequência.

**Conclusão da triagem v3.2:** nenhum refinamento transfere (÷MAD, center-crop, n_fft↑ todos piores;
rectify/smooth/threshold/persistência previstos redundantes). Só o **conceito** da linha v3 transferiu
(contraste + banda baixa). **`lb300` (0.8844) permanece o melhor.**

### Ramo demográfico (Age/Sex/Height/Weight/Pregnancy → embedding → somado ao pooled) — TESTADO no Murmur, PIOR
`phase_contrast_lb300_demo_reuse_tcn`: AUPRC **0.8759** / AUROC 0.9322 (−0.009 vs lb300). Confirma a previsão
a partir dos dados: no alvo `Murmur` a demografia tem **sinal fraco** (P(Present): Age 0.17–0.22, Sex 0.189/0.191
≈0). FN@0.5 caiu 44→40 mas o ranking piorou (capacidade extra sem sinal). **Ramo implementado e reutilizável
(`--demographic`, fusão por soma); guardar para o alvo `Outcome`** (idade ↔ anormalidade; ajuda os 263 Abnormal-sem-sopro).
Flag `--demographic` (one-hot+numéricos→MLP→soma no pooled); validações: requer cnn, incompat SMOTE/mixup/MIL/temporal.

### Reframe `Outcome` (Normal/Abnormal) — explorado e DESCARTADO pelo usuário
Implementado `--target {murmur,outcome}` (outcome inclui os 68 Unknown-murmur; label = patient Outcome, sem
location-aware). Probe com phase-contrast+lb300, GT seg, 942 pacientes (48% Abnormal): **AUPRC 0.7106 /
AUROC 0.6764** — muito abaixo do Murmur (0.94). Demografia ajudava (AUROC ~0.71, parcial). **Veredito do
usuário: não é o caminho** — todo o pipeline foi otimizado para o **evento acústico** (sopro), e o Outcome
mistura anormalidades sem assinatura no PCG (263 Abnormal sem sopro = piso de outro tipo). Código `--target`/
`--demographic` fica disponível (default murmur, backward-compat), mas o foco permanece no Murmur acústico.
**Melhor modelo segue `lb300` (0.8844).** A avenida phase-contrast+low-band está
bem otimizada; ganho adicional só fora dela (reframe `Outcome` / transfer / demografia).

## Melhor modelo (baseline a bater)

**`bc_locaware_perseg_focalfix_perfreqnorm` — AUPRC 0.8459, AUROC 0.9115, F1 0.7758, std-fold 0.064.**

Receita (Conv1d 1D, pipeline com TCN):
- Labels **location-aware** (gravação só é Present se a localização está em `Murmur locations` — corrige ~19% de ruído de rótulo).
- **STFT per-segment** (STFT por segmento de sístole, depois concatena frames — evita vazamento espectral da concatenação no tempo).
- Encoder `multiscale`, `--dilations 1,2,4,8,16,32`, pooling `attention`.
- Loss `focal` γ=2 **sem `pos_weight`** (alpha auto ≈ 0.83); **normalização por bin de frequência**.
- Segmentação de sístole pelo TCN (Grupo E), `--systole-threshold 0.45 --systole-margin-ms 50`.

## O que foi testado e DESCARTADO (não repetir)

| Tentativa | Resultado | Veredito |
|---|---|---|
| Augmentation completa (SpecAugment+mixup CNN + SpecAugment TCN) | AUPRC 0.8436 (≈), colapsou fold 5 | Redistribui, não soma |
| Ramo de **features temporais** (fill/flux/shape) no CNN | TCN 0.8216 / GT 0.8371 (pior/empate) | Tóxico com segmentação imperfeita |
| **Segmentos ground-truth** (sem TCN) | 0.8387 (abaixo do TCN 0.8459) | Segmentador não limita o ranking |
| **RNN** (GRU bidir) | 0.7751 (pior em todos os folds) | Arquitetura errada para textura de murmúrio |
| **peak1s** (janelas 1s no onset, ideia do paper) | 0.8274 | Contexto S1/S2/diástole dilui a sístole |
| **freq2d** Conv2d residual (STFT 33 bins) | 0.6298 | Eixo de freq raso + poucos dados |
| **freq2d** multiscale + log-mel64 + emph + attn + aug | 0.4779 | Pior ainda; Conv2d faminto por dados |
| Ramo **MLP de frequência** (banda 100-600 Hz) | GT 0.8375 (≈, std 0.033) / **TCN 0.8216** | Só não perde com GT |
| Ramo **transformer de frequência** | TCN 0.8148 | Capacidade demais p/ poucos dados |
| Fusão sístole+diástole | inviável | Só 5 pacientes com sopro diastólico no CirCor |

**Padrão geral:** toda adição de capacidade/sofisticação empatou ou piorou. Várias reduziram a
variância entre folds (ganho de estabilidade) mas às custas do AUPRC agregado — "redistribui, não soma".

## Diagnóstico do teto (~0.85) — exploração `analise_exploratoria_sistole_overlap/`

- No espaço de envelope espectral da sístole, **~50% dos murmúrios (250/497) têm ≥80% de vizinhos
  Absent** — indistinguíveis de normal.
- Correlação **monotônica com o grading**: I/VI (suave) afunda no meio dos normais; III/VI (forte)
  se separa bem. O modelo já pega os fortes; o teto são os I/VI.
- Dinâmica temporal separa um pouco melhor que o envelope (190 vs 250 difíceis) **com ground-truth**,
  mas o CNN 1D já usa o espectrograma temporal — features explícitas são redundantes/ruidosas.

**Conclusão:** o teto é majoritariamente **informacional** (sopros suaves não têm assinatura
espectral distinguível na sístole), não falha de modelagem. Parte pode ser piso irredutível
(label do CirCor vem de exame clínico, não só do `.wav`).

## Comparação com o paper "SMOTE + RNN" (98.6%)

Não é comparável: o `1617/1546` deles bate com o rótulo **Outcome (Normal/Abnormal, ~balanceado)**,
não `Murmur Present/Absent` (~20%). Eles reportam **accuracy por SEGMENTO** (balanceado via SMOTE),
não AUPRC paciente-level. O AUROC paciente-level do `bc` (0.91) está no nível do SOTA honesto do CirCor.

## Caminhos ainda não esgotados (com fundamento)

1. **Melhorar a segmentação do TCN** — é o gargalo secundário mais acionável. Ground-truth + ramo
   de banda foi a única combinação que não perdeu AUPRC e ganhou estabilidade; com TCN melhor, os
   ramos de banda e a dinâmica temporal passariam a funcionar.
2. **Ensemble por seed** do `bc` — antes inviável (variância de fold alta = loteria). Custo: TCN
   retreina por seed (~4h, disco). Reusar TCN exigiria separar seed de fold (refactor).

## Infra / código (estado atual)

- Código reorganizado: arquitetura vive em `nested_tcn_systole_cnn/cnn/` e `nested_tcn_systole_cnn/tcn/`
  (subpacotes por responsabilidade); caminhos antigos em `modeling/Grupo G|E/` são shims.
- Arquiteturas selecionáveis: `--model-arch {cnn,rnn,freq2d}`. Ramo de frequência:
  `--freq-linear-branch --freq-linear-arch {mlp,transformer}`. freq2d: `--freq-emphasis --freq-attention`.
- **Persistência incremental por fold** ativa (`fold_metrics_partial.csv`, `patient_oof_partial.csv`)
  + prints com `flush` → dá para acompanhar runs longos ao vivo.
- **Limpeza incremental de disco** (`try/finally` no TCN) — evita cache órfão entupindo o disco.
- Reuso de TCN entre runs: copiar `fold_*/tcn/best_model.pt` do `bc` (mesmos args de TCN) → pula
  o retreino (~25min só CNN).
