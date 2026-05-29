# heart-research

Obsidian vault de pesquisa sobre sons cardíacos (PCG) e classificação de sopro (murmur) usando o dataset CirCor DigiScope v1.0.3. Stack: Python ≥3.11, `uv`, PyTorch, scikit-learn, pandas, matplotlib, pymupdf4llm.

> **Referência canônica**: [AGENTS.md](AGENTS.md) contém a documentação completa de experimentos, scripts e resultados. Consulte-o antes de criar ou rodar qualquer experimento.

## Ambiente

```bash
uv sync
```

Sempre rode comandos a partir da raiz do repositório com `uv run`. Não use `pip` ou `python` diretamente.

## Estrutura

- [AGENTS.md](AGENTS.md) — guia mestre do repositório (experimentos, comandos, resultados)
- [circor-heart-sound-1.0.3/](circor-heart-sound-1.0.3/) — dataset local; dicionário em [DICIONARIO_DE_VARIAVEIS.md](circor-heart-sound-1.0.3/DICIONARIO_DE_VARIAVEIS.md)
- [feature extraction/](feature%20extraction/) — experimentos de extração (Grupos A, B, B v2, C1, C2); resumo consolidado em [resumo_experimentos_feature_extraction.md](feature%20extraction/resumo_experimentos_feature_extraction.md)
- [nested_tcn_systole_cnn/](nested_tcn_systole_cnn/) — pacote de treino TCN + CNN de sístole; consultar [README.md](nested_tcn_systole_cnn/README.md) antes de rodar
- [papers/](papers/) — papers em `.md`; resumos em [resumos_papers.md](papers/resumos_papers.md)
- [scripts/](scripts/) — utilitários (ex.: `pdf_to_markdown.py`)
- [docs/](docs/), [experiments/](experiments/), [modeling/](modeling/), [tests/](tests/)

## Estado atual dos experimentos

Melhor resultado: **Grupo B v2 (features relativas por local)** — cluster patient-level com ~89.5% `Murmur = Present`.

| Experimento | Melhor cluster `Present` |
|---|---|
| Grupo A (clássicas globais) | ~31.7% |
| Grupo B v1 (por fase) | sem separação direta |
| **Grupo B v2 (relativas por local)** | **~89.5%** |
| Grupo C1 (PANNs globais) | ~28.5% |
| Grupo C2 (PANNs por fase) | ~23.5% |

Próximo passo recomendado: baseline supervisionado sobre Grupo B v2 com validação por paciente. Ver detalhes e comandos em [AGENTS.md](AGENTS.md).

## Comandos frequentes

```bash
# Conversão de PDF (papers) → Markdown
uv run python scripts/pdf_to_markdown.py path/to/file.pdf
uv run python scripts/pdf_to_markdown.py --all papers --mode static

# Experimento campeão (Grupo B v2)
uv run "feature extraction/Grupo B v2 features relativas por local/extract_relative_phase_features_by_location.py"

# Validação do script de PDF sem chamar API
uv run python -m py_compile scripts/pdf_to_markdown.py
```

Os comandos completos de cada experimento (Grupos A, B, B v2, C1, C2) estão em [AGENTS.md](AGENTS.md). Flags úteis: `--skip-umap`, `--skip-tsne`, `--reuse-recording-embeddings`, `--reuse-recording-features`, `--batch-size`.

## Convenções

- Idioma: documentação, resumos e interpretações em **português**; código e identificadores em inglês.
- Resultados de cada experimento ficam em `outputs/` dentro da pasta do experimento, com `summary.md` + `interpretacao_resultados.md`.
- Não rodar PANNs com batch alto no MacBook M3 Pro — usar `--batch-size 1` ou `2` em CPU para estabilidade.
- Credenciais OpenAI ficam em [scripts/.env](scripts/.env); variáveis exportadas no shell têm prioridade.
- Cache de arXiv: `.cache/arxiv-sources/`. Cache de PANNs: `~/panns_data/`.

## Não fazer

- Não sobrescrever Markdown existente sem `--overwrite`.
- Não commitar `scripts/.env` nem checkpoints PANNs.
- Não tratar PANNs global (C1) ou por fase (C2) como caminho prioritário — priorizar supervisionado sobre Grupo B v2.
- Não criar novos experimentos de modelagem TCN/CNN sem ler [nested_tcn_systole_cnn/README.md](nested_tcn_systole_cnn/README.md) primeiro.
