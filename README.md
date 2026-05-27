# heart-research

Vault de pesquisa sobre sons cardíacos (PCG) e classificação de sopro (*murmur*) usando o dataset [CirCor DigiScope v1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/).

## Stack

Python ≥3.11, [`uv`](https://docs.astral.sh/uv/), PyTorch, scikit-learn, pandas, matplotlib, pymupdf4llm.

## Setup

```bash
uv sync
```

O dataset CirCor não está versionado neste repo. Baixe de [physionet.org/content/circor-heart-sound/1.0.3/](https://physionet.org/content/circor-heart-sound/1.0.3/) e coloque em `circor-heart-sound-1.0.3/training_data/`.

Para os scripts que usam OpenAI, copie `scripts/.env.example` para `scripts/.env` e preencha sua chave.

## Documentação

- [AGENTS.md](AGENTS.md) — guia mestre com experimentos, scripts e resultados
- [CLAUDE.md](CLAUDE.md) — contexto para o Claude Code
- [plano_sopro_poucos_dados.md](plano_sopro_poucos_dados.md) — plano de pesquisa
- [papers/resumos_papers.md](papers/resumos_papers.md) — resumos dos papers
- [feature extraction/resumo_experimentos_feature_extraction.md](feature%20extraction/resumo_experimentos_feature_extraction.md) — resumo dos experimentos de extração

## Estado dos experimentos

Melhor resultado até o momento: **Grupo B v2 (features relativas por local)** — cluster patient-level com ~89.5% `Murmur = Present`.

| Experimento | Melhor cluster `Present` |
|---|---:|
| Grupo A (clássicas globais) | ~31.7% |
| **Grupo B v2 (relativas por local)** | **~89.5%** |
| Grupo C1 (PANNs globais) | ~28.5% |
| Grupo C2 (PANNs por fase) | ~23.5% |
