# Interpretacao dos resultados - Grupo B v3.2

Este experimento testou se realcar explicitamente a regiao do murmurio melhora a separacao vista no v3.1.

Incrementos sobre v3.1:

- corte central da sistole, removendo 15% de cada borda para reduzir vazamento de S1/S2;
- mapa positivo `max(z, 0)`;
- suavizacao leve em tempo/frequencia;
- threshold do mapa (`z >= 1.0`) para remover ruido fraco;
- features de persistencia temporal;
- STFT de maior resolucao para a banda baixa;
- mapa compacto da banda baixa (`16 x 32`) para preservar a localizacao tempo-frequencia.

## Resultado principal

O v3.2 melhorou o melhor resultado amplo do v3.1.

| Experimento | Leitura paciente-level | Pacientes no grupo enriquecido | Present | % Present | Captura dos Present |
|---|---|---:|---:|---:|---:|
| v3.1 | low, k=10, clusters enriquecidos | 109 | 95 | 87.2% | 53.1% |
| v3.2 | low, k=10, clusters enriquecidos | 101 | 98 | 97.0% | 54.7% |

O v3.2 capturou 3 pacientes `Present` a mais, com 8 pacientes a menos no grupo e pureza muito maior.

## Melhor cluster unico

| Experimento | Leitura paciente-level | Pacientes no cluster | Present | % Present | Captura dos Present |
|---|---|---:|---:|---:|---:|
| v3.1 | low, k=2 | 69 | 67 | 97.1% | 37.4% |
| v3.2 | low, k=2 | 85 | 84 | 98.8% | 46.9% |

Este e o ganho mais claro: o cluster unico ficou maior, mais puro e capturou mais `Present`.

## Leituras adicionais

Outras configuracoes paciente-level fortes:

| Leitura | Pacientes no grupo | Present | % Present | Captura |
|---|---:|---:|---:|---:|
| mid, k=10, clusters enriquecidos | 102 | 93 | 91.2% | 52.0% |
| low, k=6, clusters enriquecidos | 96 | 94 | 97.9% | 52.5% |
| mid, k=2, melhor cluster | 79 | 74 | 93.7% | 41.3% |
| all, k=2, melhor cluster | 97 | 78 | 80.4% | 43.6% |

## Heatmap

O arquivo `murmur_map_heatmap_comparison.png` mostra:

- media do mapa nos clusters enriquecidos;
- media dos `Present` nesses clusters;
- media dos `Absent` fora dos clusters;
- diferenca `Present - Absent`.

A diferenca fica concentrada na banda baixa/mid-baixa e se mantem ao longo de boa parte da sistole, o que apoia a ideia de que o realce esta capturando sinal sustentado, nao apenas spikes isolados.

## Conclusao

O v3.2 melhorou o v3.1.

Resumo pratico:

- v3.1 melhor amplo: 109 pacientes, 95 `Present`, 87.2% de pureza, 53.1% de captura.
- v3.2 melhor amplo: 101 pacientes, 98 `Present`, 97.0% de pureza, 54.7% de captura.
- v3.2 melhor cluster unico: 85 pacientes, 84 `Present`, 98.8% de pureza, 46.9% de captura.

O realce da regiao do murmurio foi util. O proximo passo mais defensavel e transformar essas features em score supervisionado simples e comparar por validacao paciente-level.

