# Interpretacao dos resultados - Grupo B v3.1

Este experimento incrementou o v3 para tentar destacar mais pacientes com `Murmur = Present`.

Mudancas principais:

1. Contraste robusto:

```text
z[f,t] = (log(1 + |STFT(sistole)[f,t]|) - mediana_t(diastole[f])) / MAD_t(diastole[f])
```

2. Features por ciclo sistolico, nao apenas pela gravacao inteira.
3. Agregacao `mean`, `max`, `p90` e `top3_mean`.
4. Visoes separadas por banda: `low`, `mid`, `high`, `profile` e `all`.
5. Sweep de k-means com `k = 2, 4, 6, 8, 10`.

## Resultado principal

O v3.1 melhorou muito em relacao ao v3.

Comparacao paciente-level:

| Experimento | Leitura | Pacientes no grupo | Present | % Present | Captura dos Present |
|---|---|---:|---:|---:|---:|
| Grupo B v3 | contraste puro, k=2 | 7 | 7 | 100.0% | 3.9% |
| Grupo B v3.1 | low band, k=2, melhor cluster | 69 | 67 | 97.1% | 37.4% |
| Grupo B v3.1 | mid band, k=6, clusters enriquecidos | 78 | 76 | 97.4% | 42.5% |
| Grupo B v3.1 | low band, k=10, clusters enriquecidos | 109 | 95 | 87.2% | 53.1% |

Leitura:

- O contraste robusto deixou de capturar apenas extremos.
- A banda baixa (`25-200 Hz`) foi a mais abrangente.
- A banda media (`200-600 Hz`) foi a mais limpa entre as configuracoes amplas.
- A divisao por bandas foi essencial: usar todas as features juntas nao foi sempre melhor.

## Comparacao com Grupo B v2

O Grupo B v2 tinha o melhor cluster paciente-level anterior:

| Experimento | Pacientes no grupo | Present | % Present | Captura aproximada |
|---|---:|---:|---:|---:|
| Grupo B v2 | 86 | 77 | 89.5% | 43.0% |
| Grupo B v3.1 low k=10, clusters enriquecidos | 109 | 95 | 87.2% | 53.1% |
| Grupo B v3.1 low k=2, melhor cluster | 69 | 67 | 97.1% | 37.4% |

Interpretacao:

- Se o objetivo for pureza, `low k=2` e muito forte: 69 pacientes com 97.1% `Present`.
- Se o objetivo for destacar mais pacientes mantendo pureza alta, `low k=10` e o melhor resumo: 109 pacientes, 87.2% `Present`.
- Isso sugere que o contraste robusto por ciclo realmente surfou um sinal que o v3 puro estava comprimindo demais.

## Resultado por gravacao

No nivel de gravacao, a mesma tendencia apareceu:

| Visao | k | Grupo enriquecido | % Present / captura |
|---|---:|---|---:|
| low | 2 | 224 gravacoes, 208 Present | 92.9% / 33.9% |
| low | 10 | clusters enriquecidos: 345 gravacoes, 305 Present | 88.4% / 49.7% |
| mid | 2 | 687 gravacoes, 293 Present | 42.6% / 47.7% |

A banda baixa foi de novo a mais util para criar um grupo amplo e enriquecido.

## Conclusao

O incremento funcionou.

O v3.1 troca a leitura "microcluster perfeito" do v3 por uma separacao mais util:

- v3: 7 pacientes, 100% Present, mas captura de apenas 3.9%;
- v3.1: ate 109 pacientes, 87.2% Present, captura de 53.1%.

O melhor caminho agora e usar essas features como score/entrada supervisionada simples, ou combinar Grupo B v2 + v3.1 para preservar tanto as razoes clinicas entre fases quanto o contraste robusto por ciclo.
