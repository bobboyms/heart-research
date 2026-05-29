# Interpretacao dos resultados - Grupo B v3

Este experimento testou explicitamente a intuicao:

> A diastole e uma fotografia do ruido de fundo da mesma gravacao. Se subtrairmos essa referencia da sistole, o excesso sistolico deve destacar o sopro.

A feature central foi:

```text
contraste[f,t] = log(1 + |STFT(sistole)[f,t]|) - mediana_t(log(1 + |STFT(diastole)[f,t]|))
```

Foram geradas features de resumo desse contraste por bandas de frequencia e por perfil espectral, seguidas de PCA, UMAP e k-means.

## Resultado geral

| Item | Valor |
|---|---:|
| Gravacoes analisadas | 3002 |
| Pacientes analisados | 874 |
| Features por gravacao | 144 |
| Features por paciente | 290 |
| Proporcao base de `Present` por gravacao | 20.5% |
| Proporcao base de `Present` por paciente | 20.5% |

## Resultado dos clusters

| Nivel | Cluster enriquecido | % Present | Captura dos Present |
|---|---:|---:|---:|
| Gravacao global | 11 gravacoes | 100.0% | 1.8% |
| AV | 6 gravacoes | 100.0% | 3.9% |
| PV | 2 gravacoes | 100.0% | 1.4% |
| TV | 3 gravacoes | 100.0% | 2.1% |
| MV | 7 gravacoes | 85.7% | 3.5% |
| Paciente agregado | 7 pacientes | 100.0% | 3.9% |

## Leitura

O contraste sistole menos diastole gerou um cluster extremamente puro, mas muito pequeno.

Isso significa que:

1. A subtracao da diastole realmente revela alguns sopros de forma muito clara.
2. Esses casos ficam bem separados no PCA/UMAP.
3. Como visualizacao exploratoria, o metodo mostra que existe um "nucleo" acustico de sopro forte.
4. Como cluster geral de deteccao, ele e incompleto: recupera so cerca de 3.9% dos pacientes `Present`.

O cluster paciente-level enriquecido tem 7/7 pacientes `Present`. No nivel de gravacao, o cluster global tem 11/11 gravacoes `Present`. A maioria desses casos e de sopro `High pitch` e/ou `III/VI`, ou seja, o contraste puro esta encontrando principalmente os casos acusticamente mais evidentes.

## Comparacao com Grupo B v2

O Grupo B v2 continua melhor como visualizacao de separacao ampla:

| Experimento | Cluster paciente-level | % Present | Captura aproximada |
|---|---:|---:|---:|
| Grupo B v2 features relativas | 86 pacientes | 89.5% | muito maior |
| Grupo B v3 contraste puro | 7 pacientes | 100.0% | 3.9% dos `Present` |

Interpretacao:

- O contraste puro maximiza pureza, mas vira um detector de extremos.
- O Grupo B v2, por combinar razoes/deltas entre varias fases (`sistole/diastole`, `sistole/S1`, `sistole/S2`, bandas e MFCCs), preserva um cluster muito mais abrangente.
- Isso tambem combina com a memoria dos modelos supervisionados: o phase-contrast foi excelente quando usado dentro da CNN, mas a clusterizacao manual com resumos agregados perde bastante informacao temporal/espectral.

## Conclusao

A intuicao esta correta, mas com uma nuance:

- **Sim:** subtrair a diastole separa de forma muito limpa alguns sopros.
- **Nao:** essa representacao resumida, sozinha, nao separa a maioria dos murmurios em cluster nao supervisionado.

O resultado deve ser usado como visualizacao demonstrativa do mecanismo e como confirmacao qualitativa do achado `phase_contrast`, nao como substituto do Grupo B v2 para feature extraction ampla.
