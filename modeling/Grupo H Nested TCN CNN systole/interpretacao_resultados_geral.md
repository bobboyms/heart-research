# Interpretacao geral dos resultados - Grupo H Nested TCN + CNN systole

Este documento consolida os experimentos feitos ate agora em
`modeling/Grupo H Nested TCN CNN systole/`.

O objetivo do Grupo H e avaliar um pipeline profundo nested para predizer
`Murmur = Present` usando trechos sistolicos:

```text
fold externo de sopro
=> TCN treinado apenas nos pacientes de treino externo
=> TCN prediz trechos sistolicos em treino e validacao externa
=> treino externo dividido em cnn_fit e cnn_tune
=> CNN treinada apenas em cnn_fit
=> cnn_tune usado para early stopping, calibracao Platt e threshold
=> validacao externa usada uma unica vez para metricas OOF
```

Assim, os `.tsv` e rotulos dos pacientes de validacao externa nao entram no
treino do TCN, na selecao da CNN, na calibracao ou na escolha de threshold.

## Experimentos completos

| Run | Variacao principal | Pacientes OOF | Present | Absent |
|---|---|---:|---:|---:|
| `outputs_nested` | TCN por fase cardiaca, extracao por argmax | 836 | 170 | 666 |
| `outputs_nested_systole_binary` | TCN binario `non_systole` vs `systole`, extracao por argmax | 854 | 174 | 680 |
| `outputs_nested_threshold` | TCN por fase cardiaca, extracao com `systole_threshold=0.55` | 829 | 168 | 661 |
| `outputs_nested_threshold_other_mode` | TCN por fase cardiaca, `other_mode=ignore`, `systole_threshold=0.55` | 874 | 179 | 695 |
| `outputs_nested_weight-multiplier` | Igual ao anterior, mas com `systole_weight_multiplier=2.0`, `boundary_ignore_ms=10` e `systole_margin_ms=50` | 874 | 179 | 695 |
| `outputs_nested_weight-multiplier_v2` | Teste com mais contexto: `systole_threshold=0.45`, `systole_margin_ms=100` | 874 | 179 | 695 |
| `outputs_nested_weak_weights_only` | Baseline com pesos na CNN para sopro `I/VI=3.0` e `II/VI=1.5`, sem calibracao por local | 874 | 179 | 695 |
| `outputs_nested_weighted_location_calibration` | Pesos fraco/moderado + calibracao por local, mas com configuracao TCN/CNN diferente do baseline | 874 | 179 | 695 |
| `outputs_nested_mil_attention` | Agregacao MIL attention paciente-level sobre gravacoes/localizacoes | 874 | 179 | 695 |
| `outputs_nested_mil_multscale` | MIL attention paciente-level com encoder CNN `multiscale` e `mil_instance_loss_weight=0.0` | 874 | 179 | 695 |
| `outputs_nested_mil_multiscale_v2` | CNN sem MIL paciente-level, mas com encoder `multiscale`, `systole_threshold=0.45` e `systole_margin_ms=50` | 874 | 179 | 695 |
| `outputs_nested_multiscale` | CNN sem MIL paciente-level, com encoder `multiscale`, `systole_threshold=0.45` e `systole_margin_ms=50` | 874 | 179 | 695 |

A pasta `outputs_nested_threshold_other-mode`, com hifen em `other-mode`, parece
ser uma tentativa parcial/interrompida e nao entra na comparacao principal. A
pasta `outputs_nested_weight-multiplier` foi completada e continua sendo o
melhor resultado atual do Grupo H. Os experimentos posteriores testaram mais
contexto, pesos por intensidade, calibracao por local, MIL attention e encoder
multiescala. O `outputs_nested_multiscale` e agora o melhor candidato geral em
AUPRC, Brier e F1 calibrado em `0.5`; o `outputs_nested_mil_multiscale_v2`
continua sendo o ponto mais conservador entre os runs multiscale no threshold
Youden por fold.

## Tabela comparativa principal

### Score calibrado com Platt em threshold `0.5`

| Run | AUROC | AUPRC | Brier | BA | Sens | Spec | Precision | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs_nested` | 0.861 | 0.777 | 0.0856 | 0.773 | 0.576 | 0.970 | 0.831 | 0.681 | 98 | 20 | 72 | 646 |
| `outputs_nested_systole_binary` | 0.853 | 0.776 | 0.0860 | 0.787 | 0.603 | 0.971 | 0.840 | 0.702 | 105 | 20 | 69 | 660 |
| `outputs_nested_threshold` | 0.884 | 0.803 | 0.0763 | 0.809 | 0.643 | 0.974 | 0.864 | 0.737 | 108 | 17 | 60 | 644 |
| `outputs_nested_threshold_other_mode` | 0.900 | 0.815 | 0.0788 | 0.809 | 0.665 | 0.954 | 0.788 | 0.721 | 119 | 32 | 60 | 663 |
| `outputs_nested_weight-multiplier` | 0.904 | 0.837 | 0.0703 | 0.826 | 0.676 | 0.976 | 0.877 | 0.763 | 121 | 17 | 58 | 678 |
| `outputs_nested_weight-multiplier_v2` | 0.910 | 0.808 | 0.0760 | 0.832 | 0.698 | 0.965 | 0.839 | 0.762 | 125 | 24 | 54 | 671 |
| `outputs_nested_weak_weights_only` | 0.888 | 0.799 | 0.0825 | 0.804 | 0.648 | 0.960 | 0.806 | 0.718 | 116 | 28 | 63 | 667 |
| `outputs_nested_weighted_location_calibration` | 0.881 | 0.781 | 0.0845 | 0.805 | 0.654 | 0.957 | 0.796 | 0.718 | 117 | 30 | 62 | 665 |
| `outputs_nested_mil_attention` | 0.882 | 0.799 | 0.0806 | 0.798 | 0.626 | 0.971 | 0.848 | 0.720 | 112 | 20 | 67 | 675 |
| `outputs_nested_mil_multscale` | 0.879 | 0.785 | 0.0798 | 0.800 | 0.637 | 0.963 | 0.814 | 0.715 | 114 | 26 | 65 | 669 |
| `outputs_nested_mil_multiscale_v2` | 0.897 | 0.836 | 0.0719 | 0.834 | 0.715 | 0.953 | 0.795 | 0.753 | 128 | 33 | 51 | 662 |
| `outputs_nested_multiscale` | 0.901 | 0.842 | 0.0688 | 0.843 | 0.721 | 0.965 | 0.843 | 0.777 | 129 | 24 | 50 | 671 |

Neste ponto fixo calibrado, `outputs_nested_multiscale` passa a ser o melhor
compromisso geral: tem o maior AUPRC (`0.842`), menor Brier (`0.0688`) e maior
F1 (`0.777`). Ele tambem reduz FN em relacao ao baseline
`outputs_nested_weight-multiplier` (`50` vs `58`), ao custo de aumentar FP de
`17` para `24`. O baseline historico ainda tem AUROC levemente maior (`0.904`
vs `0.901`) e maior especificidade/precisao, mas perde em calibracao e F1.

### Score bruto do classificador em threshold `0.5`

| Run | AUROC | AUPRC | Brier | BA | Sens | Spec | Precision | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs_nested` | 0.859 | 0.760 | 0.1795 | 0.757 | 0.800 | 0.715 | 0.417 | 0.548 | 136 | 190 | 34 | 476 |
| `outputs_nested_systole_binary` | 0.850 | 0.775 | 0.1798 | 0.767 | 0.793 | 0.741 | 0.439 | 0.566 | 138 | 176 | 36 | 504 |
| `outputs_nested_threshold` | 0.878 | 0.794 | 0.2119 | 0.765 | 0.857 | 0.673 | 0.400 | 0.545 | 144 | 216 | 24 | 445 |
| `outputs_nested_threshold_other_mode` | 0.900 | 0.812 | 0.1763 | 0.817 | 0.877 | 0.757 | 0.482 | 0.622 | 157 | 169 | 22 | 526 |
| `outputs_nested_weight-multiplier` | 0.898 | 0.830 | 0.1484 | 0.820 | 0.849 | 0.791 | 0.512 | 0.639 | 152 | 145 | 27 | 550 |
| `outputs_nested_mil_multscale` | 0.873 | 0.796 | 0.1108 | 0.813 | 0.737 | 0.888 | 0.629 | 0.679 | 132 | 78 | 47 | 617 |
| `outputs_nested_mil_multiscale_v2` | 0.892 | 0.814 | 0.1576 | 0.826 | 0.866 | 0.786 | 0.510 | 0.642 | 155 | 149 | 24 | 546 |
| `outputs_nested_multiscale` | 0.903 | 0.838 | 0.1198 | 0.839 | 0.827 | 0.850 | 0.587 | 0.687 | 148 | 104 | 31 | 591 |

No score bruto, `outputs_nested_weight-multiplier` melhora claramente a
especificidade, a precisao, o F1, o AUPRC e o Brier em relacao a
`outputs_nested_threshold_other_mode`. A sensibilidade bruta cai de `0.877` para
`0.849`, mas com reducao importante de falsos positivos, de 169 para 145. O
MIL multiescala reduziu FP brutos para `78`, mas tambem reduziu TP para `132` e
ficou abaixo do baseline em AUROC, AUPRC e F1. O score bruto ainda nao deve ser
usado como ponto operacional final sem calibracao ou escolha explicita de
threshold. O `outputs_nested_mil_multiscale_v2` teve o maior TP bruto (`155`) e
apenas `24` FN, mas com `149` FP; depois da calibracao, esse comportamento fica
mais controlado. O `outputs_nested_multiscale` tambem melhorou o ranking bruto
e reduziu FP brutos para `104`, com F1 bruto `0.687`, mas o score bruto ainda
nao deve ser usado como criterio operacional final.

### Threshold Youden por fold escolhido em `cnn_tune`

Este e o ponto operacional mais defensavel, porque o threshold nao foi escolhido
olhando o OOF agregado.

| Run | BA | Sens | Spec | Precision | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs_nested` | 0.793 | 0.706 | 0.880 | 0.600 | 0.649 | 120 | 80 | 50 | 586 |
| `outputs_nested_systole_binary` | 0.776 | 0.638 | 0.915 | 0.657 | 0.647 | 111 | 58 | 63 | 622 |
| `outputs_nested_threshold` | 0.807 | 0.679 | 0.935 | 0.726 | 0.702 | 114 | 43 | 54 | 618 |
| `outputs_nested_threshold_other_mode` | 0.832 | 0.804 | 0.859 | 0.595 | 0.684 | 144 | 98 | 35 | 597 |
| `outputs_nested_weight-multiplier` | 0.840 | 0.788 | 0.892 | 0.653 | 0.714 | 141 | 75 | 38 | 620 |
| `outputs_nested_mil_multscale` | 0.812 | 0.721 | 0.904 | 0.658 | 0.688 | 129 | 67 | 50 | 628 |
| `outputs_nested_mil_multiscale_v2` | 0.858 | 0.782 | 0.934 | 0.753 | 0.767 | 140 | 46 | 39 | 649 |
| `outputs_nested_multiscale` | 0.858 | 0.799 | 0.918 | 0.715 | 0.755 | 143 | 57 | 36 | 638 |

Neste criterio, os dois runs multiscale sem MIL sao os melhores. O
`outputs_nested_multiscale` tem maior sensibilidade e mais TP (`143`), com BA
`0.858`. O `outputs_nested_mil_multiscale_v2` e mais conservador: tem menos FP
(`46` vs `57`) e maior F1 (`0.767` vs `0.755`), mas tambem mais FN (`39` vs
`36`). Ambos superam claramente o baseline historico
`outputs_nested_weight-multiplier`, que tinha `75` FP, `38` FN e F1 `0.714`.

### Threshold operacional calibrado `0.20`

Este threshold foi adotado como ponto pratico de triagem porque reduz falsos
negativos mantendo especificidade ainda alta. Ele nao substitui o threshold
Youden por fold como criterio de validacao, mas e util para comparar o
comportamento operacional dos runs recentes.

| Run | AUROC | AUPRC | BA | Sens | Spec | Precision | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs_nested_weight-multiplier` | 0.904 | 0.837 | 0.847 | 0.793 | 0.901 | 0.673 | 0.728 | 142 | 69 | 37 | 626 |
| `outputs_nested_weight-multiplier_v2` | 0.910 | 0.808 | 0.838 | 0.804 | 0.872 | 0.618 | 0.699 | 144 | 89 | 35 | 606 |
| `outputs_nested_weak_weights_only` | 0.888 | 0.799 | 0.811 | 0.788 | 0.835 | 0.551 | 0.648 | 141 | 115 | 38 | 580 |
| `outputs_nested_weighted_location_calibration` | 0.881 | 0.781 | 0.812 | 0.793 | 0.832 | 0.548 | 0.648 | 142 | 117 | 37 | 578 |
| `outputs_nested_mil_attention` | 0.882 | 0.799 | 0.834 | 0.765 | 0.902 | 0.668 | 0.714 | 137 | 68 | 42 | 627 |
| `outputs_nested_mil_multscale` | 0.879 | 0.785 | 0.825 | 0.749 | 0.901 | 0.660 | 0.702 | 134 | 69 | 45 | 626 |
| `outputs_nested_mil_multiscale_v2` | 0.897 | 0.836 | 0.849 | 0.816 | 0.882 | 0.640 | 0.717 | 146 | 82 | 33 | 613 |
| `outputs_nested_multiscale` | 0.901 | 0.842 | 0.847 | 0.810 | 0.885 | 0.644 | 0.718 | 145 | 80 | 34 | 615 |

Neste ponto de triagem, o baseline `outputs_nested_weight-multiplier` ainda e o
melhor compromisso se F1/precisao importam mais. O
`outputs_nested_weight-multiplier_v2` reduziu FN de `37` para `35`, mas criou
`20` FP adicionais e perdeu muito em precisao. Os pesos fortes para `I/VI` e
`II/VI` nao resolveram o problema dos sopros fracos: mantiveram FN parecido e
aumentaram bastante FP. O MIL attention teve especificidade praticamente igual ao
baseline, mas perdeu `5` TP e ficou com `42` FN. O MIL multiescala manteve
exatamente os `69` FP do baseline, mas perdeu `8` TP e ficou com `45` FN,
portanto tambem nao e melhor para triagem. O
`outputs_nested_mil_multiscale_v2` teve a maior sensibilidade nesse ponto
(`0.816`) e reduziu FN para `33`, mas aumentou FP para `82`; e uma troca
defensavel para triagem mais sensivel, mas menos precisa que o baseline. O
`outputs_nested_multiscale` teve comportamento muito parecido: `34` FN e `80`
FP. Portanto, em `0.20`, os runs multiscale sao mais sensiveis, mas nao melhores
em F1/precisao que o baseline historico.

## Leitura por experimento

### `outputs_nested`

Foi o primeiro run nested corrigido, com TCN por fase cardiaca e extracao por
argmax. O resultado mostrou sinal real (`AUPRC calibrado 0.777` contra taxa base
de cerca de 20%), mas o desempenho operacional ainda era limitado:

- score calibrado `0.5`: BA `0.773`, sensibilidade `0.576`, especificidade
  `0.970`, F1 `0.681`;
- threshold por fold: BA `0.793`, sensibilidade `0.706`, especificidade
  `0.880`, F1 `0.649`.

Interpretacao: baseline profundo honesto, mas ainda com muitos falsos negativos
quando usado de forma conservadora.

### `outputs_nested_systole_binary`

Testou um TCN binario, aprendendo apenas `non_systole` vs `systole`. A ideia era
simplificar a segmentacao. O resultado melhorou levemente o ponto calibrado
`0.5` em relacao a `outputs_nested`, mas nao melhorou ranking global nem
threshold por fold:

- score calibrado `0.5`: BA `0.787`, sensibilidade `0.603`, especificidade
  `0.971`, F1 `0.702`;
- threshold por fold: BA `0.776`, sensibilidade `0.638`, especificidade
  `0.915`, F1 `0.647`.

Interpretacao: a segmentacao binaria nao dominou a segmentacao por fase. Ela e
operacionalmente conservadora, mas perde ranking e sensibilidade no threshold
interno.

### `outputs_nested_threshold`

Testou TCN por fase cardiaca com extracao de sistole por threshold
`p(systole) >= 0.55` em vez de argmax. Foi uma melhora importante sobre os dois
runs anteriores, especialmente no score calibrado:

- score calibrado `0.5`: AUROC `0.884`, AUPRC `0.803`, BA `0.809`,
  sensibilidade `0.643`, especificidade `0.974`, precision `0.864`, F1 `0.737`;
- threshold por fold: BA `0.807`, sensibilidade `0.679`, especificidade
  `0.935`, precision `0.726`, F1 `0.702`.

Interpretacao: melhor run para uma decisao conservadora com poucos falsos
positivos. Ele e mais especifico e mais preciso, mas deixa mais `Present` abaixo
do threshold.

### `outputs_nested_threshold_other_mode`

Testou TCN por fase cardiaca com `other_mode=ignore` e extracao por
`systole_threshold=0.55`. Antes do `outputs_nested_weight-multiplier`, este era
o melhor run para ranking e para uma politica mais sensivel:

- score calibrado `0.5`: AUROC `0.900`, AUPRC `0.815`, BA `0.809`,
  sensibilidade `0.665`, especificidade `0.954`, precision `0.788`, F1 `0.721`;
- score bruto `0.5`: BA `0.817`, sensibilidade `0.877`, especificidade `0.757`,
  F1 `0.622`;
- threshold por fold: BA `0.832`, sensibilidade `0.804`, especificidade
  `0.859`, precision `0.595`, F1 `0.684`.

Interpretacao: ignorar `other` no treino do TCN parece produzir trechos
sistolicos mais uteis para a CNN. A melhora e clara em ranking e sensibilidade,
mas o modelo fica menos conservador do que `outputs_nested_threshold`.

### `outputs_nested_weight-multiplier`

Testou o mesmo caminho promissor de `outputs_nested_threshold_other_mode`, mas
reforcando a classe sistole no TCN:

- `tcn_target_mode=cardiac-phase`;
- `tcn_other_mode=ignore`;
- `tcn_boundary_ignore_ms=10`;
- `tcn_systole_weight_multiplier=2.0`;
- `systole_threshold=0.55`;
- `systole_margin_ms=50`.

Este e o melhor run atual:

- score calibrado `0.5`: AUROC `0.904`, AUPRC `0.837`, Brier `0.0703`,
  BA `0.826`, sensibilidade `0.676`, especificidade `0.976`, precision
  `0.877`, F1 `0.763`;
- score bruto `0.5`: AUROC `0.898`, AUPRC `0.830`, BA `0.820`,
  sensibilidade `0.849`, especificidade `0.791`, precision `0.512`, F1
  `0.639`;
- threshold Youden por fold: BA `0.840`, sensibilidade `0.788`,
  especificidade `0.892`, precision `0.653`, F1 `0.714`.

O ganho parece vir principalmente de uma segmentacao sistolica mais forte. O F1
medio de sistole do TCN no teste subiu de cerca de `0.903` em
`outputs_nested_threshold_other_mode` para cerca de `0.946` em
`outputs_nested_weight-multiplier`. Isso provavelmente gera trechos sistolicos
mais limpos para a CNN e reduz falsos positivos depois da calibracao.

Interpretacao: `systole_weight_multiplier=2.0` e um ajuste positivo. Ele melhora
ranking, calibracao, precisao e F1 sem sacrificar demais a sensibilidade. Este
deve ser o novo baseline profundo principal do Grupo H.

### `outputs_nested_weight-multiplier_v2`

Testou mais contexto sistolico para a CNN:

- `systole_threshold=0.45`;
- `systole_margin_ms=100`.

O resultado nao substituiu o baseline:

- score calibrado `0.5`: AUROC `0.910`, AUPRC `0.808`, BA `0.832`,
  sensibilidade `0.698`, especificidade `0.965`, precision `0.839`, F1
  `0.762`;
- threshold calibrado `0.20`: BA `0.838`, sensibilidade `0.804`,
  especificidade `0.872`, precision `0.618`, F1 `0.699`, com `35` FN e `89`
  FP.

Interpretacao: o contexto maior recuperou poucos `Present`, mas adicionou ruido
e aumentou falsos positivos. Apesar do AUROC maior em `0.5`, o AUPRC caiu de
`0.837` para `0.808`, e o ponto de triagem ficou pior que o baseline.

### `outputs_nested_weak_weights_only`

Testou somente pesos maiores na loss da CNN para casos `Present` com sopro
fraco/moderado:

- `weak_murmur_weight=3.0` para `I/VI`;
- `moderate_murmur_weight=1.5` para `II/VI`;
- sem `location_aware_calibration`.

Resultado:

- score calibrado `0.5`: AUROC `0.888`, AUPRC `0.799`, BA `0.804`,
  sensibilidade `0.648`, especificidade `0.960`, precision `0.806`, F1
  `0.718`;
- threshold calibrado `0.20`: BA `0.811`, sensibilidade `0.788`,
  especificidade `0.835`, precision `0.551`, F1 `0.648`, com `38` FN e `115`
  FP.

Interpretacao: os pesos fortes nao resolveram o gargalo dos sopros `I/VI`.
No threshold `0.20`, os FN ficaram praticamente iguais ao baseline (`38` vs
`37`), mas os FP subiram de `69` para `115`. O modelo bruto ficou liberal demais
e a calibracao nao recuperou o ranking perdido.

### `outputs_nested_weighted_location_calibration`

Combinou pesos fraco/moderado com calibracao por local, mas a configuracao nao
foi uma ablation limpa do baseline: usou `systole_threshold=0.45`,
`systole_margin_ms=50` e `tcn_pooling=none`. Portanto, deve ser interpretado
como teste exploratorio, nao como medida isolada da calibracao por local.

Resultado:

- score calibrado `0.5`: AUROC `0.881`, AUPRC `0.781`, BA `0.805`,
  sensibilidade `0.654`, especificidade `0.957`, precision `0.796`, F1
  `0.718`;
- threshold calibrado `0.20`: BA `0.812`, sensibilidade `0.793`,
  especificidade `0.832`, precision `0.548`, F1 `0.648`, com `37` FN e `117`
  FP.

Interpretacao: a combinacao nao ajudou. Ela manteve o numero de FN do baseline
no threshold `0.20`, mas aumentou FP de `69` para `117`. Nao deve ser adotada.

### `outputs_nested_mil_attention`

Implementou uma agregacao paciente-level por Multiple Instance Learning:

```text
gravacoes sistolicas do paciente
=> CNN encoder por gravacao
=> embedding de localizacao AV/PV/TV/MV
=> attention entre gravacoes
=> probabilidade final do paciente
```

Configuracao principal:

- `patient_mil_attention=True`;
- `mil_max_instances=8`;
- `mil_location_embedding_dim=4`;
- `mil_instance_loss_weight=0.25`;
- `weak_murmur_weight=1.0`;
- `moderate_murmur_weight=1.0`;
- sem `location_aware_calibration`.

Resultado:

- score calibrado `0.5`: AUROC `0.882`, AUPRC `0.799`, Brier `0.0806`,
  BA `0.798`, sensibilidade `0.626`, especificidade `0.971`, precision
  `0.848`, F1 `0.720`;
- threshold calibrado `0.20`: BA `0.834`, sensibilidade `0.765`,
  especificidade `0.902`, precision `0.668`, F1 `0.714`, com `42` FN e `68`
  FP.

Comparacao direta com o baseline em `0.20`:

- o MIL recuperou `6` falsos negativos do baseline;
- mas criou `11` novos falsos negativos;
- resolveu `33` falsos positivos do baseline;
- mas criou `32` novos falsos positivos;
- saldo final: `+5` FN e `-1` FP em relacao ao baseline.

Interpretacao: o MIL attention e tecnicamente promissor e tem algum sinal
complementar, mas nessa primeira versao nao superou o baseline. Ele parece
conservador demais para casos `Present` fracos, especialmente `I/VI`. A loss
auxiliar por gravacao (`mil_instance_loss_weight=0.25`) pode estar puxando o
modelo de volta para uma classificacao recording-level e dificultando o
aprendizado de uma agregacao realmente paciente-level.

### `outputs_nested_mil_multscale`

Testou uma versao mais paciente-level do MIL attention, combinando:

- `encoder_block=multiscale`;
- `patient_mil_attention=True`;
- `mil_instance_loss_weight=0.0`;
- `mil_max_instances=8`;
- `mil_location_embedding_dim=4`;
- `weak_murmur_weight=1.0`;
- `moderate_murmur_weight=1.0`;
- sem `location_aware_calibration`;
- mesma configuracao TCN forte do baseline: `other_mode=ignore`,
  `systole_weight_multiplier=2.0`, `systole_threshold=0.55` e
  `systole_margin_ms=50`.

A ideia era remover a loss auxiliar por gravacao, que poderia atrapalhar o
aprendizado de uma agregacao realmente paciente-level, e permitir que um encoder
multiescala capturasse padroes sistolicos em diferentes duracoes.

Resultado:

- score calibrado `0.5`: AUROC `0.879`, AUPRC `0.785`, Brier `0.0798`,
  BA `0.800`, sensibilidade `0.637`, especificidade `0.963`, precision
  `0.814`, F1 `0.715`;
- threshold calibrado `0.20`: BA `0.825`, sensibilidade `0.749`,
  especificidade `0.901`, precision `0.660`, F1 `0.702`, com `45` FN e `69`
  FP;
- threshold Youden por fold: BA `0.812`, sensibilidade `0.721`,
  especificidade `0.904`, precision `0.658`, F1 `0.688`.

Comparacao direta com `outputs_nested_weight-multiplier` no threshold calibrado
`0.20`:

- o MIL multiescala recuperou apenas `1` falso negativo do baseline;
- mas criou `9` novos falsos negativos;
- resolveu `30` falsos positivos do baseline;
- mas criou `30` novos falsos positivos;
- saldo final: `+8` FN e `0` FP em relacao ao baseline.

Comparacao direta com `outputs_nested_mil_attention` no threshold calibrado
`0.20`:

- recuperou `4` falsos negativos do MIL anterior;
- mas criou `7` novos falsos negativos;
- resolveu `24` falsos positivos;
- mas criou `25` novos falsos positivos;
- saldo final: `+3` FN e `+1` FP em relacao ao MIL anterior.

Foi criado um relatorio especifico em
`outputs_nested_mil_multscale/error_analysis/error_analysis.md`. O padrao de
erro continua concentrado em sopros fracos: no threshold `0.20`, `38/45` falsos
negativos sao `I/VI`, `6/28` sao `II/VI` e `III/VI` nao teve falsos negativos.
Por local mais audivel, os FNs ficam principalmente em `TV`, `MV` e `PV`; `AV`
teve apenas `1` FN.

A exportacao `mil_instance_attention_oof.csv` mostrou que a atencao e
interpretavel, mas ainda nao resolve a decisao: os maiores pesos por paciente
aparecem com frequencia em `PV` e `TV` tanto para `Present` quanto para `Absent`.
Ou seja, o mecanismo esta agregando informacao por local, mas nao aprendeu uma
regra suficientemente melhor para separar sopros fracos de padroes auscultatorios
confundidores.

Interpretacao: zerar a loss auxiliar e usar encoder multiescala nao resolveu o
gargalo do MIL. O run ficou abaixo do baseline em ranking, calibracao, F1 e
triagem. Ele tambem nao melhorou a primeira versao do MIL attention. Portanto,
`outputs_nested_mil_multscale` deve ser tratado como ablation negativa: util para
entender que MIL paciente-level puro, treinado do zero, nao basta nesta
configuracao.

### `outputs_nested_mil_multiscale_v2`

Apesar do nome, este run nao e MIL paciente-level:

- `patient_mil_attention=False`;
- nao gerou `mil_instance_attention_oof.csv`;
- `mil_instance_loss_weight=0.25` fica sem efeito pratico como loss auxiliar de
  MIL paciente-level;
- a decisao continua seguindo o classificador por gravacao/localizacao agregado
  pelo pipeline padrao.

O que ele testa de fato e uma CNN com encoder multiescala e extracao sistolica
mais permissiva:

- `encoder_block=multiscale`;
- `systole_threshold=0.45`;
- `systole_margin_ms=50`;
- `tcn_target_mode=cardiac-phase`;
- `tcn_other_mode=ignore`;
- `tcn_boundary_ignore_ms=10`;
- `tcn_systole_weight_multiplier=2.0`;
- `weak_murmur_weight=1.0`;
- `moderate_murmur_weight=1.0`;
- sem `location_aware_calibration`.

Resultado:

- score calibrado `0.5`: AUROC `0.897`, AUPRC `0.836`, Brier `0.0719`,
  BA `0.834`, sensibilidade `0.715`, especificidade `0.953`, precision
  `0.795`, F1 `0.753`;
- threshold calibrado `0.20`: BA `0.849`, sensibilidade `0.816`,
  especificidade `0.882`, precision `0.640`, F1 `0.717`, com `33` FN e `82`
  FP;
- threshold Youden por fold: BA `0.858`, sensibilidade `0.782`,
  especificidade `0.934`, precision `0.753`, F1 `0.767`, com `39` FN e `46`
  FP.

Comparacao direta com `outputs_nested_weight-multiplier`:

- no score calibrado `0.5`, recuperou `11` pacientes `Present` que o baseline
  perdia, mas perdeu `4` que o baseline acertava; corrigiu `3` FP, mas criou
  `19` novos FP;
- no threshold calibrado `0.20`, recuperou `7` FN do baseline e criou `3` novos
  FN; corrigiu `19` FP, mas criou `32` novos FP;
- no threshold Youden por fold, recuperou `8` FN e criou `9` novos FN, mas
  corrigiu `41` FP e criou apenas `12` novos FP.

O ponto forte do run e justamente o threshold Youden por fold. Ele manteve quase
o mesmo numero de verdadeiros positivos do baseline (`140` vs `141`), mas reduziu
falsos positivos de `75` para `46`. Isso elevou o F1 de `0.714` para `0.767` e a
balanced accuracy de `0.840` para `0.858`. Como esse threshold e escolhido em
`cnn_tune`, sem olhar o OOF agregado, este e um ganho operacional relevante.

No threshold calibrado fixo `0.20`, o comportamento muda: o run fica mais
sensivel que o baseline (`0.816` vs `0.793`) e reduz FN de `37` para `33`, mas
aumenta FP de `69` para `82` e reduz precision/F1. Portanto, e util se a
prioridade for triagem mais sensivel, mas nao substitui o baseline quando a
precisao e a calibracao importam.

Foi criado um relatorio especifico em
`outputs_nested_mil_multiscale_v2/error_analysis/error_analysis.md`. Em `0.20`,
os falsos negativos continuam concentrados em sopros fracos: `31/33` sao `I/VI`,
`2/28` sao `II/VI` e `III/VI` nao teve falsos negativos. Por local mais audivel,
os FNs ficam em `MV` (`12`), `TV` (`11`), `PV` (`9`) e `AV` (`1`). O padrao
continua indicando que o gargalo residual e reconhecer `I/VI`, especialmente em
locais nao-AV.

Interpretacao: este run e uma ablation positiva do encoder `multiscale` com
extracao sistolica mais permissiva, mas nao uma melhora definitiva do baseline.
Ele nao vence em AUROC, Brier ou F1 no threshold fixo `0.5`, porem e o melhor
resultado ate agora no threshold Youden por fold e tambem reduz FN no threshold
operacional `0.20` com custo moderado de FP. Deve ser mantido como candidato
forte para uma politica operacional calibrada por fold e como direcao promissora
para novos testes com seeds.

### `outputs_nested_multiscale`

Este run repete a ideia multiescala sem usar MIL paciente-level:

- `patient_mil_attention=False`;
- `encoder_block=multiscale`;
- `systole_threshold=0.45`;
- `systole_margin_ms=50`;
- `tcn_target_mode=cardiac-phase`;
- `tcn_other_mode=ignore`;
- `tcn_boundary_ignore_ms=10`;
- `tcn_systole_weight_multiplier=2.0`;
- `weak_murmur_weight=1.0`;
- `moderate_murmur_weight=1.0`;
- sem `location_aware_calibration`.

A configuracao e essencialmente a versao limpa do teste multiscale sem o nome
confuso `mil`. Ela nao gerou `mil_instance_attention_oof.csv`, porque nao ha
agregacao MIL paciente-level ativa.

Resultado:

- score calibrado `0.5`: AUROC `0.901`, AUPRC `0.842`, Brier `0.0688`,
  BA `0.843`, sensibilidade `0.721`, especificidade `0.965`, precision
  `0.843`, F1 `0.777`, com `129` TP, `24` FP, `50` FN e `671` TN;
- threshold calibrado `0.20`: BA `0.847`, sensibilidade `0.810`,
  especificidade `0.885`, precision `0.644`, F1 `0.718`, com `34` FN e `80`
  FP;
- threshold Youden por fold: BA `0.858`, sensibilidade `0.799`,
  especificidade `0.918`, precision `0.715`, F1 `0.755`, com `143` TP, `57`
  FP, `36` FN e `638` TN.

Comparacao direta com `outputs_nested_weight-multiplier`:

- no score calibrado `0.5`, recuperou `15` pacientes `Present` que o baseline
  perdia, mas perdeu `7` que o baseline acertava; corrigiu `4` FP, mas criou
  `11` novos FP;
- no threshold calibrado `0.20`, recuperou `6` FN do baseline e criou `3` novos
  FN; corrigiu `23` FP, mas criou `34` novos FP;
- no threshold Youden por fold, recuperou `8` FN e criou `6` novos FN; corrigiu
  `43` FP e criou `25` novos FP.

Comparado ao `outputs_nested_mil_multiscale_v2`, o novo `outputs_nested_multiscale`
melhorou o ponto fixo calibrado `0.5`: AUPRC subiu de `0.836` para `0.842`,
Brier caiu de `0.0719` para `0.0688`, F1 subiu de `0.753` para `0.777`, e os FP
caíram de `33` para `24`. No threshold Youden por fold, porem, a v2 continua
mais conservadora e com maior F1 (`0.767` vs `0.755`), enquanto o multiscale novo
tem maior sensibilidade (`0.799` vs `0.782`) e mais TP (`143` vs `140`).

Foi criado um relatorio especifico em
`outputs_nested_multiscale/error_analysis/error_analysis.md`. Em `0.20`, os FN
continuam concentrados em sopros fracos: `30/34` sao `I/VI`, `4/28` sao `II/VI`
e `III/VI` nao teve FN. Por local mais audivel, os FNs ficam em `MV` (`13`),
`TV` (`11`), `PV` (`9`) e `AV` (`1`). Isso significa que o modelo ja reconhece
bem sopros mais evidentes, mas ainda perde sopros sutis, principalmente fora de
`AV`.

Interpretacao: `outputs_nested_multiscale` e a melhor ablation positiva ate
agora e provavelmente o novo melhor candidato geral do Grupo H. Ele supera o
baseline historico em AUPRC, Brier, BA e F1 no threshold fixo calibrado `0.5`.
No threshold `0.20`, ele troca alguns FN por FP e nao melhora F1/precisao; no
Youden por fold, fica muito competitivo, mas a v2 ainda e mais conservadora. A
proxima decisao deve ser repetir este setup com varias seeds antes de declarar
substituicao definitiva do baseline.

## Analise de erros do `outputs_nested_weight-multiplier`

Foi criado um relatorio especifico em
`outputs_nested_weight-multiplier/error_analysis/error_analysis.md`, cruzando
`patient_oof_predictions.csv`, metadados clinicos do CirCor,
`fold_*/recording_metadata.csv` e proxies simples de qualidade dos `.wav`.

Foram analisados dois thresholds da probabilidade calibrada:

| Threshold calibrado | BA | Sens | Spec | Precision | TP | FP | FN | TN | Leitura |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.50 | 0.826 | 0.676 | 0.976 | 0.877 | 121 | 17 | 58 | 678 | decisao conservadora, poucos FP |
| 0.20 | 0.847 | 0.793 | 0.901 | 0.673 | 142 | 69 | 37 | 626 | melhor triagem, menos FN |

Para uso como triagem, o threshold calibrado `0.20` e mais adequado que `0.50`,
porque reduz falsos negativos de 58 para 37 mantendo especificidade de `0.901`.
Para uso conservador, `0.50` e forte pela precisao alta (`0.877`) e apenas 17
falsos positivos.

O padrao dos falsos negativos e claro:

- no threshold `0.20`, `33/37` falsos negativos sao sopros sistolicos `I/VI`;
- `II/VI` tem `4/28` falsos negativos;
- `III/VI` nao teve falsos negativos;
- por local mais audivel, os FNs restantes ficam principalmente em `TV` e `MV`,
  depois `PV`; `AV` tem apenas 1 FN.

Isso sugere que o gargalo atual nao e primariamente a segmentacao do TCN, mas a
capacidade da CNN de reconhecer sopros fracos, especialmente `I/VI`, em alguns
locais. As metricas simples de qualidade de sinal nao indicaram um problema
obvio de clipping ou silencio que explique sozinho os erros.

## Significancia da melhora de `other_mode=ignore`

Foi feita uma comparacao pareada exploratoria por bootstrap estratificado nos
pacientes em comum entre os runs. Esta analise avaliou especificamente a mudanca
para `other_mode=ignore`; ela ainda nao inclui uma comparacao bootstrap do
`outputs_nested_weight-multiplier`.

No score calibrado:

| Comparacao | Pacientes em comum | AUROC diff | AUPRC diff | Leitura |
|---|---:|---:|---:|---|
| `other_mode=ignore` vs `outputs_nested_threshold` | 829 | +0.015 | +0.016 | melhora pequena; intervalo inclui zero |
| `other_mode=ignore` vs `outputs_nested_systole_binary` | 854 | +0.044 | +0.036 | melhora com intervalo acima de zero |
| `other_mode=ignore` vs `outputs_nested` | 836 | +0.039 | +0.042 | melhora com intervalo acima de zero |

Portanto, a melhora de `other_mode=ignore` e forte contra os runs mais antigos,
mas ainda nao e estatisticamente conclusiva contra `outputs_nested_threshold`, que
e o comparador mais proximo. A diferenca contra `outputs_nested_threshold` e
praticamente relevante, mas precisa de repeticao com outras seeds ou validacao
mais formal.

## Conclusao atual

O melhor experimento depende do objetivo operacional:

| Objetivo | Melhor run atual | Motivo |
|---|---|---|
| Melhor AUROC | `outputs_nested_weight-multiplier` | AUROC calibrado `0.904`, levemente acima do multiscale `0.901` |
| Melhor AUPRC | `outputs_nested_multiscale` | AUPRC calibrado `0.842` |
| Melhor calibracao | `outputs_nested_multiscale` | menor Brier calibrado: `0.0688` |
| Melhor F1 em `0.5` calibrado | `outputs_nested_multiscale` | F1 `0.777` |
| Menos falsos positivos em `0.5` calibrado | `outputs_nested_weight-multiplier` e `outputs_nested_threshold` | ambos com 17 FP; o weight-multiplier tem mais TP |
| Melhor triagem calibrada simples | `outputs_nested_weight-multiplier` com threshold `0.20` | sensibilidade `0.793`, especificidade `0.901`, BA `0.847` |
| Melhor threshold Youden por fold conservador | `outputs_nested_mil_multiscale_v2` | F1 `0.767`, apenas `46` FP com `140` TP |
| Melhor threshold Youden por fold sensivel | `outputs_nested_multiscale` | BA `0.858`, sensibilidade `0.799`, `143` TP |
| Baseline historico de comparacao | `outputs_nested_weight-multiplier` | comparador principal para novas ablations |
| Melhor candidato geral atual | `outputs_nested_multiscale` | melhor AUPRC, Brier, BA e F1 em `0.5` calibrado |

A decisao recomendada e manter `outputs_nested_weight-multiplier` como baseline
historico de comparacao, mas tratar `outputs_nested_multiscale` como melhor
candidato geral atual. Para relatar resultado calibrado conservador, o
`outputs_nested_multiscale` agora e mais forte em F1 e Brier. Para relatar uma
configuracao de triagem simples em `0.20`, o baseline historico ainda tem melhor
F1/precisao, enquanto os runs multiscale reduzem FN ao custo de mais FP. Para uma
politica operacional com threshold escolhido internamente por fold,
`outputs_nested_mil_multiscale_v2` e a opcao mais conservadora e
`outputs_nested_multiscale` e a opcao mais sensivel.

`outputs_nested_threshold` ainda pode ser mantido como referencia historica
conservadora, mas foi superado pelo weight-multiplier em quase todos os criterios
principais.

Os experimentos posteriores mudam a leitura dos proximos passos:

- pesos fortes para sopro fraco/moderado (`I/VI=3.0`, `II/VI=1.5`) nao devem ser
  adotados, porque aumentaram muito FP sem reduzir FN;
- mais contexto sistolico com `systole_threshold=0.45` e margem de `100 ms`
  tambem nao deve substituir o baseline, porque melhorou pouco os FN e piorou
  precisao/AUPRC;
- o MIL attention ainda nao venceu, mas mostrou sinal complementar: recuperou
  alguns FN que o baseline perdia, embora tenha criado outros. Ele merece nova
  rodada com arquitetura ou inicializacao diferente;
- MIL multiescala com `mil_instance_loss_weight=0.0` tambem nao deve substituir o
  baseline: manteve os mesmos `69` FP do baseline em `0.20`, mas aumentou FN de
  `37` para `45`;
- encoder `multiscale` sem MIL paciente-level, com `systole_threshold=0.45` e
  margem de `50 ms`, e a direcao mais promissora ate agora: o run
  `outputs_nested_multiscale` melhorou AUPRC, Brier e F1 em `0.5`, enquanto o
  `outputs_nested_mil_multiscale_v2` continua sendo mais conservador no Youden
  por fold.

## Proximos passos recomendados

1. Manter `outputs_nested_weight-multiplier` como baseline historico de
   comparacao, mas priorizar `outputs_nested_multiscale` como novo candidato
   geral.
2. Repetir `outputs_nested_multiscale` com pelo menos 3 seeds para separar ganho
   real de variancia de treino.
3. Repetir `outputs_nested_mil_multiscale_v2` com pelo menos 3 seeds, porque ele
   e a variante multiscale mais conservadora no threshold Youden por fold.
4. Manter `--decision-threshold 0.20` como ponto operacional de triagem nos
   relatorios, alem do `0.5` calibrado.
5. Nao usar `--weak-murmur-weight 3.0` e `--moderate-murmur-weight 1.5` como
   configuracao padrao; se pesos forem revisitados, testar valores bem menores
   ou outra estrategia de amostragem.
6. Para o MIL attention, nao insistir apenas em `mil_instance_loss_weight=0.0`
   com encoder treinado do zero; esse caminho ja foi testado no
   `outputs_nested_mil_multscale` e piorou FN.
7. Considerar uma segunda versao do MIL usando a CNN baseline como encoder
   inicial, em vez de treinar encoder + attention do zero.
8. Usar `mil_instance_attention_oof.csv` para auditar casos em que a atencao
   seleciona localizacoes plausiveis mas a probabilidade final ainda fica abaixo
   do threshold, especialmente `I/VI` em `TV`, `MV` e `PV`.
9. Comparar `outputs_nested_weight-multiplier`, `outputs_nested_multiscale` e
   `outputs_nested_mil_multiscale_v2` contra Grupo B v2 em um mesmo protocolo
   supervisionado por paciente.
10. Testar ensemble simples entre features Grupo B v2, o score calibrado da CNN
   sistolica e, se continuar complementar, o score MIL.
11. Manter validacao externa como requisito antes de interpretar robustez fora do
   CirCor.
