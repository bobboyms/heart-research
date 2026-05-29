# Grupo B v2 - features relativas por local com TCN predito

Este experimento repete o **Grupo B v2**, mas nao usa os `.tsv` reais do CirCor para extrair as fases cardiacas.

O fluxo e:

```text
.wav
=> TCN frame segmenter
=> .predicted.tsv com S1 / systole / S2 / diastole
=> features relativas do Grupo B v2
=> PCA / UMAP / k-means
=> agregacao por paciente
```

O objetivo e medir se o sinal forte do Grupo B v2 continua aparecendo quando a segmentacao vem de um modelo predito, que e o cenario real para audio novo.

## Execucao

Teste rapido:

```bash
uv run "feature extraction/Grupo B v2 features relativas por local com TCN predito/extract_relative_phase_features_with_tcn_segments.py" \
  --max-recordings 40 \
  --skip-umap
```

Execucao completa recomendada:

```bash
caffeinate -dimsu uv run "feature extraction/Grupo B v2 features relativas por local com TCN predito/extract_relative_phase_features_with_tcn_segments.py" \
  --checkpoint "modeling/Grupo E TCN segmentacao frame a frame/outputs_noncausal_overlap/best_model.pt" \
  --device cpu
```

Experimento filtrado recomendado, usando apenas os locais `PV` e `TV`:

```bash
caffeinate -dimsu uv run "feature extraction/Grupo B v2 features relativas por local com TCN predito/extract_relative_phase_features_with_tcn_segments.py" \
  --checkpoint "modeling/Grupo E TCN segmentacao frame a frame/outputs_noncausal_overlap/best_model.pt" \
  --predicted-tsv-dir "feature extraction/Grupo B v2 features relativas por local com TCN predito/outputs/predicted_tsvs" \
  --locations PV TV \
  --skip-umap \
  --output-dir "feature extraction/Grupo B v2 features relativas por local com TCN predito/outputs_pv_tv" \
  --device cpu
```

Por padrao, a inferencia do TCN usa CPU porque o caminho MPS com `GroupNorm` pode ser lento ou instavel em predicao por arquivo. Para forcar MPS:

```bash
uv run "feature extraction/Grupo B v2 features relativas por local com TCN predito/extract_relative_phase_features_with_tcn_segments.py" \
  --allow-mps-predict \
  --device mps
```

## Saidas

As saidas ficam em:

```text
feature extraction/Grupo B v2 features relativas por local com TCN predito/outputs/
```

Arquivos principais:

- `predicted_tsvs/*.predicted.tsv`: segmentacoes preditas usadas no experimento.
- `predicted_segmentation_quality.csv`: contagens de segmentos e confianca media da segmentacao.
- `recording_relative_phase_features.csv`: features relativas por gravacao.
- `recording_relative_phase_features_with_projection.csv`: PCA/k-means por gravacao.
- `patient_relative_phase_features.csv`: features agregadas por paciente.
- `patient_relative_phase_features_with_projection.csv`: PCA/k-means por paciente.
- `projection_metrics.csv`: metricas dos clusters e projecoes.
- `summary.md`: resumo final.
