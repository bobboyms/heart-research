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
