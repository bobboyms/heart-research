# Papers sobre classificação de murmúrio cardíaco

Fonte: `Busca de papers sobre murmúrio.pdf`.

## Paper vencedor

Pela regra usada no script `scripts/analisar_papers_murmurio_metricas.py`, o melhor resultado geral foi o paper **“Advancing cardiovascular screening: deep learning-based heart-sound classification using SMOTE and temporal modeling”**, publicado em **2026**.

O ranking geral foi calculado pela média das métricas de desempenho disponíveis em cada paper, considerando apenas métricas em que valores maiores indicam melhor desempenho. Para esse paper, as métricas avaliadas foram:

| Métrica | Valor |
|---|---:|
| Sensibilidade | ≈0,9895 |
| Especificidade | ≈0,9830 |
| Precisão / PPV | ≈0,9826 |
| F1-score | ≈0,9861 |
| Score médio calculado | 0,9853 |

Observação: essa comparação depende das métricas reportadas no PDF. Como os papers não reportam exatamente o mesmo conjunto de métricas, o vencedor geral deve ser interpretado como o melhor dentro dos dados disponíveis nesta tabela. Se a comparação for feita somente por `BA / weighted accuracy`, o melhor paper é **“Development and validation of an integrated residual-recurrent neural network model for automated heart-murmur detection in pediatric populations”** com BA aproximada de **≈0,953**.

## Tabela completa

| Ano | Paper | AUROC | AUPRC | Brier | BA / weighted accuracy | UAR | Sensibilidade | Especificidade | Precisão / PPV | F1-score | Acurácia | TP | FP | FN | TN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | Heart Murmur Quality Detection Using Deep Neural Networks with Attention Mechanism | N/A | N/A | N/A | N/A | N/A | ≈0,804 | N/A | N/A | ≈0,758 | N/A | N/A | N/A | N/A | N/A |
| 2023 | Beyond Heart-Murmur Detection: Automatic Murmur Grading from Phonocardiogram | N/A | N/A | N/A | N/A | N/A | ≈0,863 | N/A | N/A | ≈0,816 | N/A | N/A | N/A | N/A | N/A |
| 2024 | Enhanced Heart Murmur Detection via Branchformer with Uncertainty Estimation | N/A | N/A | N/A | 0,798 | N/A | N/A | N/A | N/A | 0,651 | N/A | N/A | N/A | N/A | N/A |
| 2024 | Exploring Pre-trained General-purpose Audio Representations for Heart-Murmur Detection | N/A | N/A | N/A | 0,832 | 0,713 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2023 | Exploring Wav2vec 2.0 Model for Heart Murmur Detection | N/A | N/A | N/A | ≈0,80 | ≈0,70 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2026 | Advancing cardiovascular screening: deep learning-based heart-sound classification using SMOTE and temporal modeling | N/A | N/A | N/A | N/A | N/A | ≈0,9895 | ≈0,9830 | ≈0,9826 | ≈0,9861 | N/A | N/A | N/A | N/A | N/A |
| 2025 | Development and validation of an integrated residual-recurrent neural network model for automated heart-murmur detection in pediatric populations | N/A | N/A | N/A | ≈0,953 | N/A | ≈0,916 | ≈0,991 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2025 | Explainable Attention-based Deep Learning for Classification of Heart Murmurs | >0,90 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | ≈0,955 | N/A | N/A | N/A | N/A | N/A |
| 2022 | SmartBeatIT - BiLSTM classificador | N/A | N/A | N/A | 0,757 | N/A | Present: 0,827; Unknown: 0,312 | 0,801 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2022 | amc-sh - Learning time-frequency representations | N/A | N/A | N/A | 0,688 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2022 | UKJ FSU - Ensemble deep-learning model | 0,576 | N/A | N/A | 0,458 | N/A | N/A | N/A | N/A | 0,379 | N/A | N/A | N/A | N/A | N/A |
| 2022 | CUED - Parallel hidden semi-Markov model | N/A | N/A | N/A | 0,776 | N/A | ≈0,84 | ≈0,31 | ≈0,53 | N/A | N/A | N/A | N/A | N/A | N/A |
| 2022 | Heart2Beat - multiple instance learning network | 0,831 | 0,657 | N/A | 0,751 | N/A | N/A | N/A | N/A | ≈0,572 | N/A | N/A | N/A | N/A | N/A |
| 2022 | Leicester Fox - Transfer learning com mel-espectrograma | N/A | N/A | N/A | 0,536 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2022 | JUST_IT_Academy - Residual CNN + MLP | 0,797 | 0,610 | N/A | 0,671 | N/A | N/A | N/A | N/A | N/A | 0,757 | N/A | N/A | N/A | N/A |
| 2022 | Care4MyHeart - Ensemble transformer-based neural network | N/A | N/A | N/A | ≈0,757 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2022 | ISIBrno-AIMT - Expert feature classifier | N/A | N/A | N/A | 0,755 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2022 | LSMU - CNN + wavelet transform features | N/A | N/A | N/A | 0,671 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2022 | lubdub - Combined frequency-domain and physician-inspired features | N/A | N/A | N/A | 0,525 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2022 | CeZIS - Supervised contrastive learning | N/A | N/A | N/A | 0,756 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
