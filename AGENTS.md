# Repository Notes

## ⚠️ Protocolo de memória (LER PRIMEIRO)

A raiz do projeto tem um **`MEMORY.md`** com a memória sintetizada de todos os experimentos de modelagem (o que é o melhor modelo, o que já foi testado e descartado, o achado central, e os caminhos com fundamento).

- **ANTES de propor ou rodar qualquer experimento de modelagem**, leia `MEMORY.md` inteiro. Não repita caminhos já descartados lá.
- **DEPOIS de concluir qualquer experimento**, anexe a conclusão no `MEMORY.md` (run name, config, AUPRC/AUROC paciente-level, veredito em 1 linha) e atualize "Melhor modelo"/"Achado central" se mudarem.
- Mantenha o `MEMORY.md` sintético; detalhe extra vai em `experiments/nested_tcn_systole_cnn/resumo_achados_modelagem.md`.

## Paper Summaries

The file `papers/resumos_papers.md` contains Portuguese summaries for all paper Markdown files under `papers/`.

Each summary references the source `.md` file and is structured around the paper's problem/context, approach, and conclusions.

## CirCor Dataset

The local dataset is stored under `circor-heart-sound-1.0.3/`.

The raw dataset (`training_data/` and `training_data.csv`) is **not versioned in this repository** because of its size (~588 MB). Download it from PhysioNet before running any experiment:

- Source: https://physionet.org/content/circor-heart-sound/1.0.3/
- Target layout:
  - `circor-heart-sound-1.0.3/training_data/` — `.wav`, `.tsv`, `.txt`, `.hea` files
  - `circor-heart-sound-1.0.3/training_data.csv` — subject-level metadata

The text-only files that already ship in the repo (`DICIONARIO_DE_VARIAVEIS.md`, `LICENSE.txt`, `RECORDS`, `SHA256SUMS.txt`) describe and verify the download. After unpacking, you can validate file integrity against `SHA256SUMS.txt`.

Use `circor-heart-sound-1.0.3/DICIONARIO_DE_VARIAVEIS.md` as the local data dictionary for the CirCor DigiScope Phonocardiogram Dataset v1.0.3. It describes the variables in `training_data.csv`, the subject description tags in `training_data/*.txt`, the segmentation columns in `training_data/*.tsv`, and the observed WFDB header fields in `training_data/*.hea`.

## CirCor Feature Extraction Experiments

Feature extraction experiments are stored under `feature extraction/`.

Use `feature extraction/resumo_experimentos_feature_extraction.md` as the consolidated local summary of all feature extraction experiments. It explains the data used, each experiment's method, the main results, and the current conclusion. As of the latest runs, it identifies `Grupo B v2: features relativas por local` as the best experiment so far, with a patient-level cluster containing about 89.5% `Murmur = Present`.

The Grupo A classical-feature experiment is `feature extraction/Grupo A features classicas por gravacao/extract_group_a_classical_features.py`. It extracts global features from each full `.wav` recording without using `.tsv` cardiac-phase segmentations: duration, RMS, peak, crest factor, zero-crossing rate, clipping fraction, spectral descriptors, band energies, MFCCs, delta MFCCs, and log-mel statistics. It also generates PCA/UMAP projections, k-means diagnostics, per-location plots, and patient-level mean/max aggregations.

Run it from the repository root with `uv`:

```bash
uv run "feature extraction/Grupo A features classicas por gravacao/extract_group_a_classical_features.py"
```

Outputs are written to `feature extraction/Grupo A features classicas por gravacao/outputs/`.

Important local result documents:

- `feature extraction/Grupo A features classicas por gravacao/outputs/summary.md`
- `feature extraction/Grupo A features classicas por gravacao/outputs/interpretacao_resultados.md`

The current Grupo A interpretation is that global classical features provide a useful baseline but weak murmur separation: the best patient-level cluster has about 31.7% `Present`, far below the Grupo B v2 patient-level cluster with about 89.5% `Present`.

The Grupo C1 pretrained-embedding experiment is `feature extraction/Grupo C1 PANNs embeddings globais por gravacao/extract_panns_global_embeddings.py`. It uses PANNs/Cnn14 as a frozen pretrained AudioSet embedding extractor on full `.wav` recordings. It does not use `.tsv` cardiac-phase segmentations and does not fine-tune the model. Each recording is resampled to 32 kHz, split into 10-second windows with 5-second hop, embedded with PANNs, pooled with mean/std/max, projected with PCA/UMAP, and aggregated to patient-level mean/max features.

Run it from the repository root with `uv`:

```bash
uv run "feature extraction/Grupo C1 PANNs embeddings globais por gravacao/extract_panns_global_embeddings.py"
```

The PANNs checkpoint and labels are cached under `~/panns_data/`. On the MacBook M3 Pro, the completed run used CPU for stability (`--batch-size 2`). Use `--reuse-recording-embeddings` to regenerate projections without recalculating PANNs embeddings.

Outputs are written to `feature extraction/Grupo C1 PANNs embeddings globais por gravacao/outputs/`.

Important local result documents:

- `feature extraction/Grupo C1 PANNs embeddings globais por gravacao/outputs/summary.md`
- `feature extraction/Grupo C1 PANNs embeddings globais por gravacao/outputs/interpretacao_resultados.md`

The current Grupo C1 interpretation is that global PANNs embeddings are a valid baseline but weak for murmur separation: the best patient-level cluster has about 28.5% `Present`, below Grupo A and far below Grupo B v2. PANNs should next be tested by cardiac phase or used as a hybrid complement to Grupo B v2, not relied on as a global embedding alone.

The Grupo C2 pretrained-embedding-by-phase experiment is `feature extraction/Grupo C2 PANNs embeddings por fase cardiaca/extract_panns_phase_embeddings.py`. It is the PANNs analogue of Grupo B v2: it uses `.tsv` segmentations to extract frozen PANNs/Cnn14 embeddings separately for `S1`, systole, `S2`, and diastole, then builds systole-vs-other-phase deltas, absolute deltas, cosine/L2 distances, and norm ratios. It also generates PCA/UMAP globally, by auscultation location, and after patient-level mean/max aggregation.

Run a quick test from the repository root with `uv`:

```bash
uv run "feature extraction/Grupo C2 PANNs embeddings por fase cardiaca/extract_panns_phase_embeddings.py" --max-recordings 20 --skip-umap --batch-size 1
```

Recommended first full run on the MacBook M3 Pro:

```bash
uv run "feature extraction/Grupo C2 PANNs embeddings por fase cardiaca/extract_panns_phase_embeddings.py" --skip-umap --batch-size 1
```

Use `--reuse-recording-features` to regenerate projections without recalculating PANNs phase embeddings.

The current Grupo C2 interpretation is that PANNs by cardiac phase did not help: the best patient-level cluster has about 23.5% `Present`, worse than Grupo C1 global PANNs at about 28.5%, worse than Grupo A at about 31.7%, and far below Grupo B v2 at about 89.5%. The likely issue is that frozen AudioSet embeddings are not well aligned with short PCG phase segments. Prioritize supervised baselines on Grupo B v2 before further PANNs work.

The first implemented experiment is `feature extraction/Grupo B features segmentadas por fase cardiaca/extract_group_b_features.py`. It extracts Grupo B features from CirCor recordings by cardiac phase (`S1`, systole, `S2`, diastole) using the local `.tsv` segmentations, then generates PCA projections, k-means clusters, plots, and CSV outputs.

Run it from the repository root with `uv`:

```bash
uv run "feature extraction/Grupo B features segmentadas por fase cardiaca/extract_group_b_features.py"
```

Use `--skip-tsne` for the faster PCA/k-means path:

```bash
uv run "feature extraction/Grupo B features segmentadas por fase cardiaca/extract_group_b_features.py" --skip-tsne
```

Outputs are written to `feature extraction/Grupo B features segmentadas por fase cardiaca/outputs/`.

Important local result documents:

- `feature extraction/Grupo B features segmentadas por fase cardiaca/outputs/summary.md`
- `feature extraction/Grupo B features segmentadas por fase cardiaca/outputs/interpretacao_resultados.md`

The current interpretation is that direct k-means over PCA does not separate `Murmur = Present` from `Murmur = Absent`; however, `pca_2` shows a useful gradient where high values contain a larger proportion of `Present`. Future experiments should control for auscultation location and reduce dependence on absolute volume/energy features.

The second implemented experiment is `feature extraction/Grupo B v2 features relativas por local/extract_relative_phase_features_by_location.py`. It is separate from the first script and tests a cleaner feature set: ratios between phases, high-frequency systolic ratios, and systole-minus-diastole deltas. It excludes RMS, peak, absolute energy, and `MFCC_1`, generates PCA/UMAP by auscultation location (`AV`, `PV`, `TV`, `MV`), and creates patient-level mean/max aggregations.

Run it from the repository root with `uv`:

```bash
uv run "feature extraction/Grupo B v2 features relativas por local/extract_relative_phase_features_by_location.py"
```

Outputs are written to `feature extraction/Grupo B v2 features relativas por local/outputs/`.

Important local result documents:

- `feature extraction/Grupo B v2 features relativas por local/outputs/summary.md`
- `feature extraction/Grupo B v2 features relativas por local/outputs/interpretacao_resultados.md`

The current v2 interpretation is more promising than v1: after removing absolute volume/energy features, small clusters enriched for `Murmur = Present` appear globally, within each auscultation location, and after patient-level aggregation. The patient-level cluster has 86 patients with about 89.5% `Present`, so the next step should be a supervised baseline with validation by patient.

## Nested TCN + Systole CNN Experiments

The reorganized Nested TCN + Systole CNN training package is documented in `nested_tcn_systole_cnn/README.md`.

Always consult `nested_tcn_systole_cnn/README.md` before creating or running these modeling experiments. It describes the current entrypoint, accepted parameters, named experiment workflow, fold artifact cleanup, experiment registry, scoring, and static dashboard generation.

**IMPORTANTE — leia antes de propor experimentos de modelagem:** `experiments/nested_tcn_systole_cnn/resumo_achados_modelagem.md` consolida o que já foi testado e descartado, e o achado central. Resumo:

- **Melhor modelo:** `bc_locaware_perseg_focalfix_perfreqnorm` (Conv1d 1D, AUPRC paciente-level **0.8459**). É o baseline a bater.
- **Teto ~0.85 é INFORMACIONAL, não de modelagem.** Medido: excluindo os sopros I/VI (suaves, 104 de 179 Present), o AUROC sobe de 0.90 → **0.955** (4/5 folds em 0.98-0.999). Os audíveis (II/VI+) já são detectados quase perfeitamente; os I/VI seguram o teto. Nuance: I/VI **é audível** para um cardiologista ao vivo — o modelo é que não o separa do normal NESTAS gravações (causas não distinguíveis: sinal fraco/mascarado, features insuficientes, ou ruído de rótulo). Isso mantém abertas alavancas de **dados** (denoising, qualidade, limpeza de rótulo), não de arquitetura.
- **Já testado e DESCARTADO** (não repetir): augmentation (SpecAugment/mixup), ramo de features temporais, segmentos ground-truth, RNN, peak1s (janelas 1s), freq2d/Conv2d (residual e multiscale, STFT e log-mel), ramos de frequência MLP e transformer, fusão sístole+diástole. Todos empataram ou pioraram — mais capacidade/sofisticação não ajuda neste regime de poucos dados.
- **Caminhos com fundamento (ainda não esgotados):** supervisão auxiliar (grading/timing/shape), demografia como entrada, transfer do PhysioNet 2016, melhorar a segmentação do TCN, ensemble por seed. Ou seja: **alavancas de dados/informação, não de arquitetura.**

## PDF to Markdown Script

The script `scripts/pdf_to_markdown.py` converts PDFs into Markdown files.

It is useful for turning papers and technical PDFs into `.md` files that can be read, indexed, edited, or linked inside this Obsidian research vault.

For papers, the preferred path is arXiv LaTeX source when the filename contains an arXiv id such as `2410.05258`. This preserves mathematical formulas as LaTeX instead of reconstructing them from PDF text. If source is unavailable, the script falls back to local static extraction with `pymupdf4llm`. The script can optionally use the OpenAI Responses API to clean noisy extracted Markdown chunks, or it can fall back to sending the whole PDF to the model.

When arXiv source is used, the script also converts figure environments to Markdown image links, copies figure files into a sibling `_assets` directory, attempts to render `.bib` entries into a References section, and prints a short quality report.

## Configuration

Use `uv` for the Python environment:

```bash
uv sync
```

The script reads OpenAI credentials from `scripts/.env` when an OpenAI-backed mode needs them.

Used by `--mode hybrid --clean auto`, required by `--mode hybrid --clean all`, and required by `--mode llm`:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5.4-mini
```

Environment variables already exported in the shell take priority over values in `scripts/.env`.

## Usage

Run the script from the repository root with `uv`:

```bash
uv run python scripts/pdf_to_markdown.py path/to/file.pdf
```

By default, the script uses `--source auto --mode hybrid --clean auto`: it tries arXiv LaTeX source first, then falls back to PDF extraction, and only calls the LLM for chunks that look noisy.

By default, the Markdown file is written next to the PDF with the same base name:

```bash
uv run python scripts/pdf_to_markdown.py papers/encoding/2025_2511.09146_DoPE.pdf
```

This creates:

```text
papers/encoding/2025_2511.09146_DoPE.md
```

If arXiv source contains figures, assets are copied beside the Markdown file:

```text
papers/encoding/2025_2511.09146_DoPE_assets/
```

To choose a custom output path:

```bash
uv run python scripts/pdf_to_markdown.py path/to/file.pdf -o path/to/output.md
```

To convert all PDFs under `papers` and skip files that already have Markdown:

```bash
uv run python scripts/pdf_to_markdown.py --all papers
```

For the fastest no-API conversion:

```bash
uv run python scripts/pdf_to_markdown.py --all papers --mode static
```

This still tries arXiv source first by default. To force PDF-only static extraction:

```bash
uv run python scripts/pdf_to_markdown.py --all papers --mode static --source never
```

To require arXiv source and fail instead of falling back to PDF extraction:

```bash
uv run python scripts/pdf_to_markdown.py path/to/file.pdf --source require
```

For maximum cleanup quality, at higher API cost and runtime:

```bash
uv run python scripts/pdf_to_markdown.py --all papers --mode hybrid --clean all
```

To use the original whole-PDF OpenAI path as a fallback:

```bash
uv run python scripts/pdf_to_markdown.py path/to/file.pdf --mode llm
```

To overwrite an existing Markdown file:

```bash
uv run python scripts/pdf_to_markdown.py path/to/file.pdf --overwrite
```

To preserve the local static extraction beside the final file:

```bash
uv run python scripts/pdf_to_markdown.py path/to/file.pdf --keep-raw
```

When arXiv source is used, `--keep-raw` writes the expanded `.source.tex` file beside the final Markdown.

To use a different model for one run:

```bash
uv run python scripts/pdf_to_markdown.py path/to/file.pdf --model gpt-5.4
```

## Behavior

- `--source auto` detects arXiv ids in PDF filenames, downloads source from arXiv, expands simple TeX inputs/macros, and writes Markdown with formulas preserved as LaTeX.
- arXiv figures are converted to Markdown image links and copied into `<paper>_assets/`.
- arXiv `.bib` files are parsed into a simple `## References` section when possible.
- LaTeX layout wrappers such as figure/table/quote/CJK environments are cleaned where safe.
- `--mode static` extracts Markdown locally with `pymupdf4llm` and does not call OpenAI.
- `--mode hybrid` extracts Markdown locally, splits it into chunks, and optionally cleans selected chunks with the OpenAI Responses API.
- `--mode llm` uploads the whole PDF to OpenAI with `purpose=user_data` and asks the model to convert it directly.
- Each conversion prints a report with source, math source, figure count, table count, copied asset count, reference count, unresolved macro count, and LLM chunk count.
- Writes the final Markdown to the target `.md` file.
- Refuses to overwrite an existing output file unless `--overwrite` is provided.
- Rejects direct whole-PDF OpenAI inputs larger than 50 MB in `--mode llm`.
- Deletes uploaded files after `--mode llm` conversion completes or fails.
- Caches arXiv source downloads under `.cache/arxiv-sources`.

## Validation

Check the script syntax without calling the API:

```bash
uv run python -m py_compile scripts/pdf_to_markdown.py
```

Show CLI options:

```bash
uv run python scripts/pdf_to_markdown.py --help
```
