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

## Melhor modelo (baseline a bater)
**`bc_locaware_perseg_focalfix_perfreqnorm` — AUPRC 0.8459 · AUROC 0.9115 · F1 0.7758 · std-fold 0.064** (Conv1d 1D, pipeline com TCN).
Receita: labels **location-aware** (gravação = Present só se a localização ∈ `Murmur locations`; corrige ~19% de ruído de rótulo) · **STFT per-segment** · encoder `multiscale`, `--dilations 1,2,4,8,16,32`, pooling `attention` · loss `focal` γ=2 **sem pos_weight** (alpha auto ≈0.83) · **normalização por bin de frequência** · sístole via TCN, `--systole-threshold 0.45 --systole-margin-ms 50`.

## ★ Achado central — o teto ~0.85 são os sopros suaves (I/VI)
Experimento `floor_audible_only_gt_systole`: excluindo os 104 sopros **I/VI** (de 179 Present), mantendo só Absent + audíveis (II/VI=28, III/VI=46), o **AUROC sobe de 0.90 → 0.955** (4/5 folds em 0.98-0.999). Os audíveis (II/VI+) já são detectados quase perfeitamente; **os I/VI seguram o teto**.

**Nuance importante (não simplificar):** I/VI **É audível** para um cardiologista ao vivo (por isso está rotulado Present). O que ocorre é que **o modelo não consegue separá-lo do normal NESTAS gravações**. Causas possíveis, **não distinguíveis só com os dados**:
1. sinal fraco/mascarado por ruído, ou não capturado no clipe (exame ao vivo usa manobras/posições/tempo que a gravação fixa não tem);
2. nossas features (STFT/log-mel + envelope) não surfam o sinal;
3. **ruído de rótulo** — grau I tem a menor concordância inter-observador.
→ Portanto NÃO é "piso irredutível absoluto", e sim "piso dado estas gravações + este pipeline". Isso mantém **abertas** alavancas de dados (denoising, filtro de qualidade, limpeza/incerteza de rótulo, gravações melhores).

**Implicação prática:** parar de caçar I/VI com truques de modelo/arquitetura (esgotado). Ganho real só via **dados/informação**. Clinicamente o modelo já é útil (pega bem II/VI+, os relevantes).

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

**Padrão:** mais capacidade/sofisticação não ajuda neste regime (~150 I/VI). Várias reduzem variância de fold mas às custas do AUPRC.

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
