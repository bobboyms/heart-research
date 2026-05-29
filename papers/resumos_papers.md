# Resumos dos Papers

Este arquivo reúne resumos em português dos arquivos Markdown em `papers/`. Cada item referencia o arquivo fonte e sintetiza o problema/contexto, a abordagem e as conclusões principais.

## Audio Feature Extraction SOTA 2020-2026

### Contrastive Learning of General-Purpose Audio Representations (COLA, 2020)

**Fonte:** [2020_COLA_contrastive_general_purpose_audio_representations.md](audio_feature_extraction_sota_2020_2026/2020_COLA_contrastive_general_purpose_audio_representations.md)

- **Problema/contexto:** Métodos auto-supervisionados em áudio ainda eram fortemente concentrados em fala ou dependiam de tarefas pretexto e mineração de negativos difíceis, com transferência limitada para domínios como música, cenas acústicas e sons animais.
- **Abordagem:** O COLA aprende representações gerais por aprendizado contrastivo, aproximando segmentos extraídos da mesma gravação e afastando segmentos de gravações diferentes no batch. O encoder usa log-mel filterbanks e é pré-treinado no AudioSet, depois avaliado com classificador linear congelado e fine-tuning em nove tarefas.
- **Conclusões:** Mesmo simples, o método supera sistemas auto-supervisionados anteriores em várias tarefas e melhora modelos supervisionados quando usado como inicialização. O artigo estabelece o COLA como uma baseline prática para aprendizado auto-supervisionado geral de áudio.

### PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (2020)

**Fonte:** [2020_PANNs_large_scale_pretrained_audio_neural_networks.md](audio_feature_extraction_sota_2020_2026/2020_PANNs_large_scale_pretrained_audio_neural_networks.md)

- **Problema/contexto:** Reconhecimento de padrões de áudio envolve múltiplas tarefas, mas muitos sistemas anteriores eram treinados em datasets pequenos ou específicos. Faltavam modelos pré-treinados em larga escala para áudio comparáveis ao papel de ImageNet em visão.
- **Abordagem:** Os autores treinam PANNs no AudioSet com várias arquiteturas CNN, avaliam custo computacional e desempenho, e propõem Wavegram-Logmel-CNN combinando waveform e log-mel spectrogram. Os modelos são transferidos para tarefas como cenas acústicas, eventos sonoros, gênero musical e emoção na fala.
- **Conclusões:** O melhor sistema atinge mAP de 0,439 no AudioSet, acima de baselines anteriores, e transfere bem para seis tarefas. PANNs mostram que pré-treinamento supervisionado em larga escala é útil especialmente quando a tarefa-alvo tem poucos dados.

### wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (2020)

**Fonte:** [2020_wav2vec2_self_supervised_speech_representations.md](audio_feature_extraction_sota_2020_2026/2020_wav2vec2_self_supervised_speech_representations.md)

- **Problema/contexto:** Sistemas de reconhecimento de fala dependem de grandes volumes de áudio transcrito, algo inviável para muitas línguas e cenários de poucos recursos.
- **Abordagem:** O wav2vec 2.0 aprende diretamente de waveform não rotulada: uma CNN extrai representações latentes, partes são mascaradas, um Transformer contextualiza a sequência e o modelo resolve uma tarefa contrastiva sobre alvos quantizados aprendidos. Depois, o modelo é fine-tuned com transcrições.
- **Conclusões:** O método alcança desempenho forte com poucos rótulos, inclusive com apenas 10 minutos ou 1 hora de fala transcrita, e melhora resultados no LibriSpeech e TIMIT. A conclusão central é que pré-treinamento auto-supervisionado reduz drasticamente a dependência de dados anotados em ASR.

### AST: Audio Spectrogram Transformer (2021)

**Fonte:** [2021_AST_audio_spectrogram_transformer.md](audio_feature_extraction_sota_2020_2026/2021_AST_audio_spectrogram_transformer.md)

- **Problema/contexto:** Modelos de classificação de áudio eram dominados por CNNs ou híbridos CNN-atenção, deixando em aberto se Transformers puros seriam suficientes para espectrogramas.
- **Abordagem:** O AST trata espectrogramas como sequências de patches, inspirado em Vision Transformers, sem convoluções. O modelo usa pré-treinamento ImageNet e é avaliado em AudioSet, ESC-50 e Speech Commands.
- **Conclusões:** O AST alcança resultados de estado da arte, incluindo 0,485 mAP no AudioSet, 95,6% no ESC-50 e 98,1% no Speech Commands V2. O trabalho mostra que atenção pura pode funcionar muito bem em classificação de áudio quando aplicada a espectrogramas.

### BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation (2021)

**Fonte:** [2021_BYOL_A_self_supervised_general_purpose_audio_representation.md](audio_feature_extraction_sota_2020_2026/2021_BYOL_A_self_supervised_general_purpose_audio_representation.md)

- **Problema/contexto:** Muitos métodos auto-supervisionados de áudio dependem de relações temporais entre segmentos ou pares positivos/negativos, o que pode limitar a generalidade e exigir desenho cuidadoso de amostras.
- **Abordagem:** BYOL-A adapta Bootstrap Your Own Latent para áudio, criando pares aumentados a partir de um único segmento e treinando sem negativos explícitos. O método combina normalização, mixup, random resized crop em espectrogramas e ajustes para reduzir drift estatístico.
- **Conclusões:** BYOL-A obtém resultados competitivos ou de estado da arte em diferentes tarefas downstream e as ablações indicam que normalização e augmentations são componentes centrais. O artigo reforça que representações gerais de áudio podem ser aprendidas com objetivos não contrastivos baseados em pares aumentados.

### Efficient Training of Audio Transformers with Patchout (PaSST, 2021)

**Fonte:** [2021_PaSST_efficient_training_audio_transformers_patchout.md](audio_feature_extraction_sota_2020_2026/2021_PaSST_efficient_training_audio_transformers_patchout.md)

- **Problema/contexto:** Transformers para áudio têm bom desempenho, mas o custo quadrático da atenção torna o treinamento caro em espectrogramas com muitos patches.
- **Abordagem:** O PaSST introduz Patchout, removendo aleatoriamente patches de espectrograma durante o treinamento para reduzir custo e atuar como regularização. O trabalho também explora menor overlap de patches e redução de profundidade.
- **Conclusões:** Patchout acelera o treinamento e mantém ou melhora a performance em AudioSet e tarefas de transferência como ESC-50, OpenMIC, DCASE e FSD50K. O estudo mostra que Transformers de áudio podem ser treinados de forma mais eficiente sem abandonar a arquitetura.

### Masked Autoencoders that Listen / Audio-MAE (2022)

**Fonte:** [2022_AudioMAE_masked_autoencoders_that_listen.md](audio_feature_extraction_sota_2020_2026/2022_AudioMAE_masked_autoencoders_that_listen.md)

- **Problema/contexto:** Masked autoencoders haviam se mostrado fortes em visão, mas sua adaptação para áudio exigia entender como mascarar e reconstruir espectrogramas de forma útil para representações transferíveis.
- **Abordagem:** O Audio-MAE mascara grande parte dos patches de espectrograma e treina um encoder-decoder para reconstruí-los. O encoder pré-treinado é usado em fine-tuning e linear probing em tarefas de classificação de áudio.
- **Conclusões:** O método aprende representações robustas e competitivas, com forte desempenho em benchmarks como AudioSet e ESC-50. O artigo também mostra que a reconstrução pode capturar estrutura relevante do áudio, embora com custo de decoder no pré-treinamento.

### CLAP: Learning Audio Concepts from Natural Language Supervision (2022)

**Fonte:** [2022_CLAP_learning_audio_concepts_natural_language_supervision.md](audio_feature_extraction_sota_2020_2026/2022_CLAP_learning_audio_concepts_natural_language_supervision.md)

- **Problema/contexto:** Modelos supervisionados tradicionais dependem de rótulos fechados, enquanto muitas descrições de áudio aparecem naturalmente como texto livre.
- **Abordagem:** O CLAP aprende um espaço conjunto áudio-texto por contraste entre embeddings de áudio e de linguagem natural. O modelo é treinado com pares áudio-caption e avaliado em zero-shot, tarefas supervisionadas e variações de prompts.
- **Conclusões:** O CLAP permite classificação zero-shot e transferência por prompts textuais, além de funcionar como encoder em tarefas supervisionadas. O trabalho antecipa o papel de modelos áudio-linguagem como base para compreensão semântica flexível de sons.

### HTS-AT: Hierarchical Token-Semantic Audio Transformer (2022)

**Fonte:** [2022_HTS_AT_hierarchical_token_semantic_audio_transformer.md](audio_feature_extraction_sota_2020_2026/2022_HTS_AT_hierarchical_token_semantic_audio_transformer.md)

- **Problema/contexto:** Classificação e detecção de sons precisam capturar padrões locais e globais em espectrogramas, mas Transformers padrão podem ser caros e pouco estruturados para áudio.
- **Abordagem:** O HTS-AT usa uma arquitetura hierárquica com window attention e patch merging, inspirada em Transformers hierárquicos de visão, além de um módulo token-semantic para relacionar tokens a classes de áudio.
- **Conclusões:** O modelo alcança bons resultados em AudioSet, ESC-50, Speech Commands V2 e localização/detecção em DESED. A contribuição principal é mostrar que hierarquia e semântica de tokens ajudam tanto em classificação quanto em detecção/localização sonora.

### BEATs: Audio Pre-Training with Acoustic Tokenizers (2023)

**Fonte:** [2023_BEATs_audio_pretraining_with_acoustic_tokenizers.md](audio_feature_extraction_sota_2020_2026/2023_BEATs_audio_pretraining_with_acoustic_tokenizers.md)

- **Problema/contexto:** Masked modeling em áudio depende da qualidade dos alvos de predição; alvos contínuos ou pouco semânticos podem limitar o aprendizado de representações úteis.
- **Abordagem:** BEATs propõe pré-treinamento iterativo com tokenizadores acústicos. O processo começa com um tokenizador por projeção aleatória, depois usa tokenizadores auto-destilados para gerar alvos discretos mais informativos para o modelo SSL.
- **Conclusões:** Tokenizadores acústicos melhores produzem representações mais semânticas e transferíveis, com desempenho de ponta em múltiplos benchmarks. O artigo posiciona a escolha dos alvos discretos como peça crítica em pré-treinamento mascarado para áudio.

### GRAM: Spatial General-Purpose Audio Representations for Real-World Environments (2025)

**Fonte:** [2025_GRAM_general_purpose_audio_representation_real_world_sound_scenes.md](audio_feature_extraction_sota_2020_2026/2025_GRAM_general_purpose_audio_representation_real_world_sound_scenes.md)

- **Problema/contexto:** Muitos modelos gerais de áudio focam conteúdo semântico, mas aplicações em ambientes reais também exigem informação espacial, reverberação e robustez a cenas naturais.
- **Abordagem:** O GRAM aprende representações gerais espaciais usando entradas que combinam mel-spectrogramas e pistas espaciais, com pré-treinamento e avaliação em benchmarks como HEAR, NatHEAR e RealSELD.
- **Conclusões:** As representações GRAM transferem para tarefas semânticas e espaciais, incluindo localização, estimativa de reverberação e classificação. O artigo destaca que modelos foundation de áudio devem preservar informações espaciais, não apenas categorias sonoras.

### WavJEPA: Semantic Learning Unlocks Robust Audio Foundation Models for Raw Waveforms (2025)

**Fonte:** [2025_WavJEPA_semantic_learning_robust_audio_foundation_models_raw_waveforms.md](audio_feature_extraction_sota_2020_2026/2025_WavJEPA_semantic_learning_robust_audio_foundation_models_raw_waveforms.md)

- **Problema/contexto:** Modelos foundation para áudio em waveform cru precisam aprender representações robustas sem depender de espectrogramas ou de reconstrução detalhada de sinais locais.
- **Abordagem:** O WavJEPA adapta a ideia de joint embedding predictive architectures para áudio cru, treinando o modelo a prever representações-alvo em espaço latente. O trabalho inclui uma variante voltada a cenas naturalísticas e avalia em tarefas HEAR, NatHEAR e ARCH.
- **Conclusões:** O método enfatiza aprendizado semântico em vez de reconstrução ponto a ponto, com boa robustez em cenas naturais. A conclusão é que objetivos preditivos em espaço de embedding podem gerar modelos de áudio mais transferíveis para waveform cru.

### Transformation of Audio Embeddings into Interpretable, Concept-Based Representations (2025)

**Fonte:** [2025_audio_embeddings_interpretable_concept_based_representations.md](audio_feature_extraction_sota_2020_2026/2025_audio_embeddings_interpretable_concept_based_representations.md)

- **Problema/contexto:** Embeddings neurais de áudio têm bom desempenho, mas são caixas-pretas difíceis de interpretar, o que limita análise, auditoria e uso em contextos sensíveis.
- **Abordagem:** O trabalho usa o espaço áudio-texto do CLAP para decompor embeddings em representações esparsas baseadas em vocabulários de conceitos. Avalia qualitativamente, quantitativamente, com fine-tuning e diferentes tamanhos/construções de conjuntos conceituais.
- **Conclusões:** Representações baseadas em conceitos igualam ou superam embeddings originais em várias tarefas, ao mesmo tempo em que oferecem interpretabilidade semântica. O artigo também disponibiliza vocabulários específicos de áudio para esse tipo de análise.

### AudioMosaic: Contrastive Masked Audio Representation Learning (2026)

**Fonte:** [2026_AudioMosaic_contrastive_masked_audio_representation_learning.md](audio_feature_extraction_sota_2020_2026/2026_AudioMosaic_contrastive_masked_audio_representation_learning.md)

- **Problema/contexto:** Em áudio, métodos contrastivos são menos explorados que reconstrução mascarada, em parte porque exigem augmentations adequadas e batches grandes. Em espectrogramas, positivos muito parecidos podem tornar a tarefa fácil demais e reduzir a qualidade das representações.
- **Abordagem:** O AudioMosaic cria pares positivos por masking estruturado em tempo e frequência sobre patches de espectrograma, processando apenas patches visíveis e contrastando visões complementares da mesma amostra. O objetivo é reduzir redundância local e forçar representações globais discriminativas.
- **Conclusões:** O modelo supera métodos de masked spectrogram modeling em linear probing e é competitivo ou superior em fine-tuning em AudioSet, ESC-50, Speech Commands e detecção de deepfake ambiental. Também melhora tarefas áudio-linguagem quando usado como encoder pré-treinado.

## Audio Feature Extraction, Embeddings and Similarity 2024-2026

### EAT: Self-Supervised Pre-Training with Efficient Audio Transformer (2024)

**Fonte:** [2024_EAT_self_supervised_pretraining_efficient_audio_transformer.md](audio_feature_extraction_similarity_2024_2026/2024_EAT_self_supervised_pretraining_efficient_audio_transformer.md)

- **Problema/contexto:** Pré-treinamento auto-supervisionado em áudio aprende boas representações com dados não rotulados, mas costuma exigir muito custo computacional, dificultando iteração, aplicação prática e otimização de modelos SSL.
- **Abordagem:** O EAT adapta uma estrutura bootstrap ao domínio de áudio, usando espectrogramas em patches, teacher atualizado por EMA, máscara inversa em blocos com alta taxa de mascaramento e o objetivo Utterance-Frame Objective (UFO), que combina predição global do utterance com predição local de frames.
- **Conclusões:** O modelo melhora a eficiência de pré-treinamento, com aceleração de até cerca de 15 vezes em relação a modelos SSL anteriores, e alcança resultados fortes em AudioSet, ESC-50 e Speech Commands. O trabalho reforça que objetivo de representação e estratégia de mascaramento são decisões centrais para extração eficiente de features de áudio.

### NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics (2024)

**Fonte:** [2024_NatureLM_audio_language_foundation_model_bioacoustics.md](audio_feature_extraction_similarity_2024_2026/2024_NatureLM_audio_language_foundation_model_bioacoustics.md)

- **Problema/contexto:** Bioacústica precisa detectar, classificar e descrever vocalizações animais em grandes coleções, mas os dados anotados são escassos, específicos por espécie e difíceis de generalizar para taxa, comportamentos e tarefas novas.
- **Abordagem:** O NatureLM-audio é um foundation model áudio-linguagem treinado com pares texto-áudio de bioacústica, fala, música e áudio geral. O modelo usa instruções em linguagem natural e é avaliado em BEANS-Zero, benchmark com tarefas como classificação, detecção, captioning, tipo de chamada, estágio de vida e contagem de indivíduos.
- **Conclusões:** O modelo demonstra transferência positiva de fala e música para bioacústica, generaliza para espécies/taxa não vistos e estabelece novo estado da arte em várias tarefas de bioacústica, incluindo classificação zero-shot. A contribuição principal é mostrar que embeddings e modelos áudio-texto podem ampliar bioacústica para além de classificadores fechados por espécie.

### Musical Source Separation Bake-Off: Comparing Objective Metrics with Human Perception (2025)

**Fonte:** [2025_musical_source_separation_bakeoff_objective_metrics_human_perception.md](audio_feature_extraction_similarity_2024_2026/2025_musical_source_separation_bakeoff_objective_metrics_human_perception.md)

- **Problema/contexto:** Sistemas de separação musical são avaliados por métricas objetivas como SDR, SAR e SIR, mas esses números nem sempre acompanham a percepção humana de qualidade para stems como voz, bateria, baixo e outros instrumentos.
- **Abordagem:** O estudo coleta avaliações perceptuais em larga escala no MUSDB18 e compara métricas baseadas em energia com alternativas baseadas em embeddings, incluindo Frechet Audio Distance calculado com CLAP-LAION-music, EnCodec, VGGish, wav2vec2 e HuBERT.
- **Conclusões:** SDR continua melhor para vocais, enquanto SI-SAR prediz melhor julgamentos em bateria e baixo. FAD com CLAP-LAION-music é competitivo para bateria e baixo, mas embeddings não correlacionam bem com percepção em vocais. A conclusão prática é que comparação de áudio em música exige métricas específicas por tipo de fonte, não uma única medida universal.

### VocSim: A Training-Free Benchmark for Zero-Shot Content Identity in Single-Source Audio (2025)

**Fonte:** [2025_VocSim_zero_shot_content_identity_audio_embeddings.md](audio_feature_extraction_similarity_2024_2026/2025_VocSim_zero_shot_content_identity_audio_embeddings.md)

- **Problema/contexto:** Muitos benchmarks avaliam embeddings de áudio após linear probing ou fine-tuning, confundindo qualidade intrínseca da representação com capacidade de adaptação supervisionada. Faltava um teste direto de similaridade zero-shot entre sons de mesma identidade acústica.
- **Abordagem:** O VocSim agrega 125 mil clipes single-source de fala humana, vocalizações animais e sons ambientais, evitando o confound de separação de fontes. Avalia embeddings congelados com Precision@k e Global Separation Rate, calibrado por baseline de permutação, além de pooling tempo-frequência e PCA sem rótulos.
- **Conclusões:** Features congeladas do Whisper com pooling e PCA têm desempenho zero-shot forte, mas há queda substancial em fala cega de baixo recurso, sugerindo falha de generalização para fonotáticas não vistas. Os melhores embeddings também predizem similaridade perceptual aviária, melhoram classificação bioacústica e alcançam resultados fortes no HEAR, tornando o VocSim uma ferramenta útil para diagnosticar geometria de embeddings.

### MAEB: Massive Audio Embedding Benchmark (2026)

**Fonte:** [2026_MAEB_massive_audio_embedding_benchmark.md](audio_feature_extraction_similarity_2024_2026/2026_MAEB_massive_audio_embedding_benchmark.md)

- **Problema/contexto:** A avaliação de embeddings de áudio é fragmentada entre fala, música, eventos sonoros, recuperação, classificação zero-shot e tarefas multimodais. Isso torna difícil comparar modelos gerais e entender trade-offs entre capacidades acústicas, linguísticas e semânticas.
- **Abordagem:** O MAEB propõe um benchmark unificado com 30 tarefas em fala, música, sons ambientais, bioacústica e raciocínio áudio-texto em mais de 100 línguas, derivado de uma coleção maior com 98 tarefas. Avalia mais de 50 modelos com tarefas de classificação, clustering, pair classification, retrieval, reranking e zero-shot classification.
- **Conclusões:** Nenhum modelo domina todas as tarefas: modelos contrastivos áudio-texto se destacam em sons ambientais, enquanto modelos pré-treinados em fala são melhores em tarefas linguísticas. Clustering continua difícil para todos. O benchmark mostra que qualidade de encoder em MAEB se correlaciona com desempenho em Audio LLMs, oferecendo uma referência útil para escolher embeddings conforme o domínio.

### WavLink: Compact Audio-Text Embeddings with a Global Whisper Token (2026)

**Fonte:** [2026_WavLink_compact_audio_text_embeddings.md](audio_feature_extraction_similarity_2024_2026/2026_WavLink_compact_audio_text_embeddings.md)

- **Problema/contexto:** Whisper é amplamente usado como encoder em Audio LLMs, mas normalmente produz muitas features frame-level por clipe; já modelos compactos de embedding áudio-texto costumam usar encoders como HTS-AT ou PaSST, explorando pouco o potencial do Whisper para busca por similaridade.
- **Abordagem:** O WavLink adiciona ao encoder Whisper um token global aprendível, treinado em conjunto com um encoder textual em espaço contrastivo. O estudo compara text encoders, perdas CLIP/SigLIP, LoRA, fine-tuning completo, atualização de uma ou duas torres, e usa supervisão Matryoshka para embeddings truncáveis em várias dimensões.
- **Conclusões:** O modelo alcança desempenho de ponta em recuperação áudio-texto no AudioCaps e Clotho, é competitivo em classificação zero-shot e AIR-Bench, e preserva boa performance com embeddings até 8 vezes menores. O trabalho mostra que features ASR do Whisper podem ser adaptadas para embeddings compactos e eficientes em busca semântica de áudio.

### Temporal Pooling Strategies for Training-Free Anomalous Sound Detection with Self-Supervised Audio Embeddings (2026)

**Fonte:** [2026_temporal_pooling_training_free_anomalous_sound_detection_embeddings.md](audio_feature_extraction_similarity_2024_2026/2026_temporal_pooling_training_free_anomalous_sound_detection_embeddings.md)

- **Problema/contexto:** Detecção de som anômalo sem treinamento usa embeddings pré-treinados e amostras normais de referência, mas quase sempre agrega sequências temporais por média simples, o que pode apagar desvios curtos e relevantes para anomalias.
- **Abordagem:** O artigo avalia sistematicamente pooling temporal em embeddings self-supervised para ASD e propõe Relative Deviation Pooling (RDP), que enfatiza frames que desviam do padrão temporal típico. Também introduz uma estratégia híbrida RDP+GeM para combinar pesos adaptativos e agregação não linear.
- **Conclusões:** Apenas trocar a estratégia de pooling já melhora consistentemente vários modelos de embedding em cinco datasets DCASE, com resultados de estado da arte em ASD training-free e desempenho superior a sistemas treinados relatados no DCASE2025. A principal lição é que pooling temporal é uma parte crítica da comparação entre áudios, especialmente quando anomalias são locais.

## Clinical Heart Murmur Studies

### Does This Patient Have an Abnormal Systolic Murmur? (JAMA, 1997)

**Fonte:** [1997_jama_does_this_patient_have_abnormal_systolic_murmur.md](clinical_heart_murmur_studies/1997_jama_does_this_patient_have_abnormal_systolic_murmur.md)

- **Problema/contexto:** Sopros sistólicos são comuns e podem indicar doença estrutural, mas o exame clínico varia entre observadores e pode levar a ecocardiogramas desnecessários ou atraso em diagnósticos importantes.
- **Abordagem:** Revisão da literatura sobre precisão e acurácia do exame físico para identificar sopros sistólicos anormais, incluindo achados associados a estenose aórtica, regurgitações, cardiomiopatia hipertrófica e prolapso mitral.
- **Conclusões:** A confiabilidade geral para detectar sopros sistólicos é apenas moderada, mas alguns achados são clinicamente úteis: pulso carotídeo de subida lenta, pico tardio do sopro e B2 reduzida ajudam a sugerir estenose aórtica; ausência de irradiação para carótida direita ajuda a afastá-la. O exame por cardiologistas pode ser acurado, mas faltavam estudos robustos em não cardiologistas.

### Auscultation While Standing: A Basic and Reliable Method to Rule Out a Pathologic Heart Murmur in Children (2017)

**Fonte:** [2017_auscultation_while_standing_pathologic_murmur_children.md](clinical_heart_murmur_studies/2017_auscultation_while_standing_pathologic_murmur_children.md)

- **Problema/contexto:** Distinguir sopros inocentes de patológicos em crianças é difícil na atenção primária, o que gera encaminhamentos e exames cardiológicos frequentes.
- **Abordagem:** Estudo prospectivo com 194 crianças de 2 a 18 anos encaminhadas por sopro. Cardiologistas avaliaram o sopro em decúbito e em pé, e todos os pacientes realizaram ecocardiograma.
- **Conclusões:** O desaparecimento completo do sopro ao ficar em pé teve valor preditivo positivo de 98% para excluir sopro patológico, embora sensibilidade menor. O teste é simples e útil para reduzir encaminhamentos desnecessários em crianças a partir de 2 anos, desde que aplicado ao contexto clínico adequado.

### Heart Murmurs and Echocardiography Findings in the Normal Newborn Nursery (2018)

**Fonte:** [2018_heart_murmurs_echocardiography_findings_normal_newborn_nursery.md](clinical_heart_murmur_studies/2018_heart_murmurs_echocardiography_findings_normal_newborn_nursery.md)

- **Problema/contexto:** Em recém-nascidos assintomáticos, o significado de um sopro cardíaco e a necessidade de ecocardiograma variam entre serviços, especialmente após melhora da detecção pré-natal e triagem por oximetria.
- **Abordagem:** Coorte retrospectiva de recém-nascidos em alojamento conjunto entre 2008 e 2015, analisando ecocardiogramas realizados por sopro, desfechos clínicos e uma pesquisa com médicos do berçário.
- **Conclusões:** O uso de ecocardiografia aumentou ao longo do tempo. Entre exames por sopro, muitos eram normais ou sem necessidade de cuidado adicional, mas uma minoria clinicamente importante exigiu seguimento, UTI neonatal, cirurgia ou cateterismo. O estudo conclui que exame físico e ecocardiograma ainda têm papel relevante no berçário normal.

### Diagnosis of Cardiac Murmurs in Children (Vessel Plus, 2022)

**Fonte:** [2022_diagnosis_cardiac_murmurs_children_vessel_plus.md](clinical_heart_murmur_studies/2022_diagnosis_cardiac_murmurs_children_vessel_plus.md)

- **Problema/contexto:** Sopros são uma das formas mais frequentes de descoberta de cardiopatias em crianças, mas sua interpretação exige domínio de timing, localização, irradiação, intensidade, posição corporal e achados associados.
- **Abordagem:** Revisão didática da ausculta pediátrica, classificando sopros em sistólicos, diastólicos e contínuos, e discutindo diagnósticos diferenciais como estenoses, defeitos septais, regurgitações, canal arterial patente, zumbido venoso e shunts cirúrgicos.
- **Conclusões:** A ausculta cuidadosa, combinada com história, exame físico, ECG e radiografia quando apropriados, frequentemente orienta o diagnóstico. Ecocardiografia Doppler é confirmatória, quantifica lesões e guia o momento e tipo de intervenção, mas não substitui a formação clínica em ausculta.

### Evaluation and Management of Heart Murmurs in Children (AAFP, 2022)

**Fonte:** [2022_heart_murmurs_children_evaluation_management_aafp.md](clinical_heart_murmur_studies/2022_heart_murmurs_children_evaluation_management_aafp.md)

- **Problema/contexto:** Sopros são comuns em crianças saudáveis, mas podem ser a única manifestação de doença cardíaca estrutural. O desafio é decidir quando observar, encaminhar ou solicitar ecocardiograma.
- **Abordagem:** Revisão prática para atenção primária, enfatizando história, sinais de doença cardíaca, exame cardiovascular detalhado, características de sopros inocentes e sinais de alerta para patologia.
- **Conclusões:** Achados como sopro holossistólico ou diastólico, grau 3 ou maior, qualidade áspera, B2 anormal, clique sistólico, intensidade máxima na borda esternal superior esquerda ou aumento em pé aumentam suspeita de patologia. ECG e radiografia raramente ajudam; encaminhamento a cardiologista pediátrico e ecocardiografia são indicados quando há achados anormais, sintomas, fatores de risco ou incerteza, especialmente em neonatos.
