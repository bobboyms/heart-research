<!-- page: 1 -->

**MAEB: Massive Audio Embedding Benchmark** 

**Adnan El Assadi**[1] **Isaac Chung**[2] **Chenghao Xiao**[3] **Roman Solomatin**[4 5] **Animesh Jha**[6] **Rahul Chand**[6] **Silky Singh**[6] **Kaitlyn Wang**[6] **Ali Sartaz Khan**[6] **Marc Moussa Nasser**[6] **Sufen Fong**[6] **Pengfei He**[6] **Alan Xiao**[6] **Ayush Sunil Munot**[7] **Aditya Shrivastava**[8] **Artem Gazizov**[9] **Niklas Muennighoff**[6] **Kenneth Enevoldsen**[10] 

## **Abstract** 

We introduce the **M** assive **A** udio **E** mbedding **B** enchmark (MAEB), a large-scale benchmark covering 30 tasks across speech, music, environmental sounds, and cross-modal audio-text reasoning in 100+ languages. We evaluate 50+ models and find that no single model dominates across all tasks: contrastive audio-text models excel at environmental sound classification (e.g., ESC50) but score near random on multilingual speech tasks (e.g., SIB-FLEURS), while speech-pretrained models show the opposite pattern. Clustering remains challenging for all models, with even the best-performing model achieving only modest results. We observe that models excelling on acoustic understanding often perform poorly on linguistic tasks, and vice versa. We also show that the performance of audio encoders on MAEB correlates highly with their performance when used in audio large language models. MAEB is derived from MAEB+, a collection of 98 tasks. MAEB is designed to maintain task diversity while reducing evaluation cost, and it integrates into the MTEB ecosystem for unified evaluation across text, image, and audio modalities. We release MAEB and all 98 tasks along with code and a leaderboard at https://github.com/ embeddings-benchmark/mteb. 

## **1. Introduction** 

Audio and speech representations support diverse applications such as voice assistants and music recommendation systems. However, evaluation protocols for audio embedding models vary significantly, spanning speech recognition, 

1Carleton University 2Zendesk 3Durham University 4MIRAI 5SaluteDevices 6Stanford University 7Indian Institute of Technology, Kharagpur[8] Capital One[9] Harvard University[10] Aarhus University. Correspondence to: Adnan El Assadi <adnanelassadi@cmail.carleton.ca>. 

zero-shot classification, and audio-text retrieval. Existing audio benchmarks often focus on specific tasks (e.g., vocal sound classification (Gong et al., 2022)) or narrow domains (e.g., environmental sounds (Piczak, 2015)) while often ignoring others, limiting insight into how well embeddings transfer across different applications. Without a unified evaluation framework, the field remains fragmented, making it difficult to compare models or track meaningful progress across the full landscape of audio tasks. Additionally, the absence of integrated development and maintenance infrastructure has led to stagnation in existing benchmarks, with many becoming outdated as the field rapidly evolves. 

We introduce the **M** assive **A** udio **E** mbedding **B** enchmark (MAEB) to provide a unified, comprehensive evaluation protocol to spur the field’s advancement toward universal audio embedding models. Building on the success of MTEB (Muennighoff et al., 2023), MMTEB (Enevoldsen et al., 2025), and MIEB (Xiao et al., 2025b), which have unified and expanded evaluation of embedding models for text and image through continual development and community maintenance, we extend this proven framework to the audio domain. 

MAEB spans 30 audio tasks grouped into 7 categories. Aligning with MTEB’s approach, we include Classification, Zero-shot Classification, Clustering, Pair Classification, Retrieval, and Reranking tasks adapted for audio data. Notably, we consider audio-specific aspects such as multilingual audio understanding, long-form audio processing, and cross-modal audio-text tasks that have been largely absent from prior audio benchmarks. Beyond traditional speech recognition tasks, we emphasize comprehensive audio understanding capabilities through: 1) Diverse acoustic domains, including speech, music, environmental sounds, and bioacoustics; 2) Cross-modal abilities, particularly in zeroshot settings leveraging text descriptions; 3) Complex recognition tasks requiring fine-grained audio understanding; 4) Multilingual audio processing across various languages and dialects. 

To ensure efficient evaluation and broader adoption, MAEB allows for evaluation of a small audio-only model in 2 GPU hours while not compromising on coverage. We also 

_Preprint. February 19, 2026._ 

1

<!-- page: 2 -->

## **MAEB: Massive Audio Embedding Benchmark** 

**==> picture [462 x 201] intentionally omitted <==**

**----- Start of picture text -----**<br>
Classification Pair Classification<br>MAEB+ Audio A Audio B<br>Massive Audio Embedding Benchmark<br>Angry 94%<br>Cc Other 6% = Same Accent ✓<br>13 (13) • 5 (2) • 2 (1)<br>EXAMPLE DATASETS 165 (159) 98 (30) 11 (11) 50+ L n 2 (1) • P d 5 (3) • ® 2 (1)<br>Emotion Classification Languages Tasks Domains Models EXAMPLE DATASETS<br>Intent Classification Accent Pair Answer Relevance<br>Clustering Zero-shot Classification Multi-label Classification Retrieval Reranking<br>"Find dog barking sounds"<br>0.92<br>Vehicles Classes: "Car", "Bird", "Ocean" 1. Jazz 0.95<br>Music Car soe Bird Wind Forest aa 0.65 Cl 2. Blues 0.78<br>% 6 (5) • F es 10 (5) • ® 4 (2) % 2 (1) • = 5 (2) • ® 2 (1) % 102 (101) • = 6 (2) • ® 7 (2) % 156 (150) • 2 38 (9) • ® 7 (7) % 2 (1) • b a 5 (1) • ® 3 (1)<br>EXAMPLE DATASETS EXAMPLE DATASETS EXAMPLE DATASETS EXAMPLE DATASETS EXAMPLE DATASETS<br>Genre Clustering Vehicle Sound Environmental Sounds Urban Sounds Avian Bioacoustics Audio Event Detection Audio Description Question Answering Music Genre Reranking Urban Sounds<br>**----- End of picture text -----**<br>


_Figure 1._ overview of task types and example subtypes in MAEB+. Values in parentheses denote numbers for MAEB. 

provide MAEB(audio), a 19-task audio-only subset for evaluating audio-only models, and MAEB+, our full unfiltered collection of 98 tasks. Additionally, we provide a modular architecture that simplifies the addition of new audio models and datasets, ensuring that MAEB can evolve with the rapidly advancing field of audio representation learning. 

Our evaluation of 53 models reveals that no single model dominates across all audio domains; each excels in specific areas while underperforming in others. Preliminary evidence from four Audio LLMs suggests that MAEB encoder quality may correlate with downstream Audio LLM performance ( _R_[2] = 0.86, _n_ = 4; see Figure 3), validating the benchmark’s relevance for multimodal audio understanding. 

To summarize, MAEB makes the following key contributions: 

1. We provide the first comprehensive benchmark for audio embeddings that spans multiple domains, languages, and task types, 

2. We establish baseline evaluations using a representative set of 53 models, revealing strengths and weaknesses across different audio understanding capabilities, 

3. We identify critical areas where current models struggle, particularly in multilingual contexts and crossmodal understanding, providing clear directions for future research, 

4. We create a flexible, extensible framework that enables the audio research community to standardize evaluation practices and track progress more effectively. 

## **2. MAEB** 

MAEB is fully integrated into the MTEB ecosystem (Muennighoff et al., 2023), extending its unified evaluation framework to the audio modality alongside text (Enevoldsen et al., 2025) and image (Xiao et al., 2025b) embeddings. This integration provides several advantages: (1) _tried-and-tested implementations_ with standardized metrics and evaluation protocols validated across thousands of submissions; (2) _extensibility_ through a minimal interface that allows adding new models or tasks with minimal code changes; (3) _reproducibility_ via versioned code and artifacts, with results stored in a public repository; and (4) _long-term maintenance_ and community-driven development (Chung et al., 2025). MAEB seeks to broadly evaluate _embedding quality_ for downstream tasks–it does not assess transcription, generation, or other capabilities outside the scope of representation learning. 

## **2.1. Benchmark Construction** 

**Dataset Selection** We curate datasets according to four guiding principles: (1) _domain diversity_ across speech, music, environmental sounds, and bioacoustics; (2) _task diversity_ spanning classification, clustering, pair classification, retrieval, and reranking; (3) _linguistic diversity_ across languages and dialects; and (4) _quality and accessibility_ , prioritizing datasets with established usage, clear licensing, and public availability. 

**Task Selection** Evaluating models across our full dataset collection, MAEB+, would be prohibitively expensive for 

2

<!-- page: 3 -->

**MAEB: Massive Audio Embedding Benchmark** 

most groups. Following MMTEB and MIEB, which demonstrated that principled filtering maintains high rank correlation with exhaustive evaluation, we construct MAEB using five selection criteria: (1) _Validity_ : For directional tasks (e.g., retrieval), we prioritize the more semantically valid direction (e.g., text-to-audio over audio-to-text when text queries better reflect realistic use cases); (2) _Unique coverage_ : Tasks providing exclusive coverage of a domain or capability are retained regardless of other factors (e.g., the only bioacoustics clustering task); (3) _Linguistic breadth_ : Among comparable tasks, we retain those covering more languages; (4) _Redundancy removal_ : We compute pairwise correlation matrices across model rankings and remove tasks with Spearman _ρ >_ 0 _._ 8 to a retained task, keeping the task with broader coverage or lower runtime; (5) _Runtime efficiency_ : Among otherwise equivalent tasks, we select those with lower computational cost. 

As an intermediate step in task selection, we create MAEB(extended) with 89 tasks by applying initial validity and unique coverage filters to MAEB+. From this intermediate collection, we apply redundancy removal and runtime efficiency criteria to produce the final MAEB (30 tasks). Table 1 compares GPU runtime between MAEB and MAEB(extended) across representative models, showing a 2.2–3.3 _×_ speedup depending on model type. MAEB maintains strong correlation with MAEB(extended) in terms of model scores (Pearson _r_ =0.981) and model ranking (Spearman _ρ_ =0.912), indicating that it preserves relative model performance while substantially reducing evaluation time. 

_Table 1._ Benchmark runtime comparison (GPU hours) between MAEB and MAEB(extended). Runtime measured on a single NVIDIA A100 GPU. 

|**Model**|**Params**|**MAEB**|**Extended**|**Speedup**|
|---|---|---|---|---|
|YAMNet|3.7M|2.01|6.02|3.0_×_|
|wav2vec2-xls-r-2b|2B|26.93|45.62|1.7_×_|
|larger_clap_general|630M|11.52|32.23|2.8_×_|
|CLAP-htsat-fused|194M|13.03|35.35|2.7_×_|



For comprehensive evaluation, we release the full unfiltered collection as MAEB+. See the full dataset list in Appendix A. 

**Benchmark Ranking** Following the same protocol in MMTEB (Enevoldsen et al., 2025), we compute model ranks using a Borda count (Colombo et al., 2022) by treating each task as a preference voter over models. While the Borda count has several advantages over the mean (including scale invariance and robustness to outliers), it is not a continuous measure; thus, we provide both the Borda rank and the mean in the leaderboard. 

## **2.2. Tasks and Evaluation** 

We follow a similar approach to MMTEB and MIEB to extend tasks to the audio domain. 

**Classification** A logistic regression is trained on audio embeddings to predict labels (Alain & Bengio, 2018; Radford et al., 2021). We use few-shot linear probing (Muennighoff et al., 2023; Cherti et al., 2023) with 8 examples per class, balancing evaluation quality with computational efficiency. 

**Zero-shot Classification** Audio embeddings are directly matched to class labels converted to text prompts (e.g., “This is a sound of dog bark”) without training a classifier. We measure accuracy following Radford et al. (2021). 

**Clustering** We use MiniBatchKMeans (with k set to the number of true labels) and V-measure (Rosenberg & Hirschberg, 2007) as the main metric to evaluate whether embeddings group meaningfully according to semantic categories. 

**Retrieval** Retrieval evaluates finding relevant documents from a corpus given a query, including uni-modal (audioto-audio) and cross-modal (text-to-audio, audio-to-text) scenarios. Documents are ranked by cosine similarity, with CV Recall@5 (cross-validation recall at 5) as the main metric. 

**Pair Classification** Given two audio inputs, the task is to predict whether they are similar according to a criterion (e.g., same speaker, same sound class). Similarity is computed between embeddings, and average precision based on cosine similarity serves as the main metric. 

**Reranking** Unlike retrieval over full corpora, reranking evaluates ranking quality on pre-selected candidate sets containing relevant documents and hard negatives. This tests fine-grained discrimination, with MAP@1000 (mean average precision at 1000) as the main metric. 

## **3. Experimental Settings** 

## **3.1. Models** 

We seek to evaluate the broad category of audio embedding models, and select 50+ audio encoders representing four broad development categories. 

**Audio Encoders** includes models trained specifically on audio through various methods. Self-supervised speech models learn contextualized representations through masked prediction and clustering objectives, including Wav2Vec2/XLSR (Baevski et al., 2020; Babu et al., 2021), WavLM (Chen et al., 2022a), HuBERT (Hsu et al., 2021), Data2Vec (Baevski et al., 2022), UniSpeech (Wang et al., 2021b), 

3

<!-- page: 4 -->

**MAEB: Massive Audio Embedding Benchmark** 

SEW-D (Wu et al., 2021), and MCTCT (Lugosch et al., 2022). Transformer-based models apply vision transformer architectures to audio spectrograms, including AST (Gong et al., 2021). CNN-based models employ convolutional architectures trained on large-scale audio datasets, including CNN14 (Kong et al., 2020), YAMNet (Gemmeke et al., 2017), and VGGish (Hershey et al., 2017). Neural codec models provide audio compression through learned representations, including Encodec (Défossez et al., 2022). 

**Sequence-to-Sequence Models** includes models trained for a sequence-to-sequence objective, e.g., for speech recognition and translation. This category includes Whisper (Radford et al., 2022), MMS (Pratap et al., 2023), SeamlessM4T (Communication et al., 2023), and SpeechT5 ASR (Ao et al., 2022). 

**Contrastive Alignment Models** includes models that learn joint audio-text embedding spaces through a contrastive alignment objective, including CLAP (Wu et al., 2024), MSCLAP (Elizalde et al., 2023), Wav2CLIP (Wu et al., 2022), MuQ-MuLan (Zhu et al., 2025), and SpeechT5 Multimodal (Ao et al., 2022). 

**Large Audio-Language Models** are models derived from generative multimodal LLMs, which are then adapted for embeddings, e.g., by utilizing their hidden states or through contrastive refinement. These include Qwen2-Audio (Chu et al., 2024) and LCO-Embedding (Xiao et al., 2025a). 

Note that the categories are not perfect; for instance, LCOEmbedding (Xiao et al., 2025a) and Wav2Vec2/XLS-R (Baevski et al., 2020; Babu et al., 2021) both utilize a contrastive loss during training. Please refer to Appendix B for all model details. 

## **3.2. Implementation Details** 

All models implement consistent preprocessing with audio truncated to a maximum of 30 seconds, or shorter where required by model architecture or memory constraints. Audio is resampled to model-specific sampling rates (16kHz for speech models, 48kHz for CLAP and MS-CLAP variants, 24kHz for MuQ-MuLan and Encodec) and converted to mono when required. 

For embedding extraction, we use model-native approaches: transformer models employ mean pooling over temporal dimensions, CNN models use global average pooling, and specialized architectures follow their intended pooling strategies. Contrastive models (CLAP, MS-CLAP, Wav2CLIP, MuQ-MuLan) use their audio encoder branches with L2 normalization for retrieval compatibility. Large audio-language models extract embeddings from the final hidden layer using last-token pooling. 

## **4. Results** 

Table 2 presents the top 30 models on the MAEB benchmark. The table includes both MAEB rank (over all 30 tasks) and Audio-only rank (over the 19 audio-only subset tasks) to highlight how models perform differently across task types. LCO-Embedding-Omni-7B ranks first overall by Borda count, achieving the highest average scores (52.2% overall, 50.3% cross-modal retrieval, 64.5% zero-shot) across all categories. Qwen2-Audio-7B ranks second overall by Borda count (overall average 33.7%) but ranks first on audio-only tasks by Borda count (50.8% average) and excels in reranking (80.8%) and clustering (12.7%). Whisper-medium achieves third place overall by Borda count (overall average 46.7%) with strong audio-only performance (48.2%) but cannot perform cross-modal tasks. CLAP variants (larger_clap_general at 4th, larger_clap_music_and_speech at 6th) demonstrate balanced cross-modal capabilities. We provide detailed per-task results for each category in Appendix E. 

Figure 2 visualizes the performance of leading models on 94 tasks in MAEB+ across 5 acoustic domains (see Appendix D for task details). For each domain, we select the model achieving the highest average score across all task types. We observe distinct specialization patterns: LCO-EmbeddingOmni-7B leads in the Speech domain with an aggregate score of 68.2, driven by strong speech-text alignment, while the Audio Spectrogram Transformer (AST) dominates the Music (71.6), Environmental (63.8), and Bioacoustics (45.2) domains, likely benefiting from its AudioSet pre-training on diverse non-speech events. Qwen2-Audio establishes itself as the leader in Emotion recognition (44.7), demonstrating the advantages of multimodal instruction-tuning for paralinguistic understanding. The disjointed, non-overlapping shapes confirm that no single encoder achieves universal performance across all acoustic domains, the dashed target of 80 remains unmet in every category. This validates our finding that specialized models excel in their respective domains but fail to generalize broadly across the full acoustic spectrum. 

## **4.1. Key Findings on Model Performance** 

Our comprehensive evaluation over MAEB reveals four critical weaknesses in current audio representations, each suggesting specific directions for future model development. 

**(a) No universal audio model exists.** Speech-trained models (Wav2Vec2, Whisper) underperform on music tasks, while music-focused models (CLAP variants) struggle with speech understanding, confirming that no single architecture achieves universal audio representation. As shown in Table 2, Whisper-medium achieves strong classification performance (51.7%) but struggles with clustering (5.0%), 

4

<!-- page: 5 -->

## **MAEB: Massive Audio Embedding Benchmark** 

_Table 2._ Top 30 models on the MAEB benchmark (30 tasks spanning audio-only and audio-text evaluation). Results are ranked using Borda count. The “Audio” column shows the model’s rank on MAEB(audio-only) for reference. We provide averages across all tasks, and per task category. “Eng.” shows the average for English-only tasks, “Multi.” shows the average excluding tasks with no linguistic content (zxx), and “Aud.” shows the average for audio-only tasks. Task categories are abbreviated as: Classification (Clf), Multilabel Classification (M.Clf), Pair Classification (PC), Reranking (Rrnk), Clustering (Clust), Audio Retrieval (A. Rtrvl), Cross-modal Retrieval (X. Rtrvl), Zero-shot Classification (Zero Clf.). We highlight the best score in **bold** and the best score with each model category using a grey cell. 

||**Rank**(_↓_)|**Average**|**Average per Category**|
|---|---|---|---|
|**Model**|MAEB<br>Audio|All<br>Cat.<br>Eng.<br>Multi.<br>Aud.|Clf<br>M.Clf<br>PC<br>Rrnk<br>Clust<br>A. Rtrvl<br>X. Rtrvl<br>Zero Clf.|
|||MAEB||
|Number of datasets||(30)<br>(30)<br>(15)<br>(23)<br>(19)|(10)<br>(2)<br>(3)<br>(1)<br>(3)<br>(1)<br>(8)<br>(2)|
|**Large audio-language models**||||
|LCO-Embedding-Omni-7B<br>Qwen2-Audio-7B<br>LCO-Embedding-Omni-3B|1<br>5|**52.2**<br>**55.6**<br>**50.9**<br>**53.6**<br>**52.2**|58.0<br>**45.7**<br>**67.3**<br>78.7<br>1.7<br>78.2<br>**50.3**<br>**64.5**|
||2<br>1<br>5<br>11|33.7<br>34.0<br>30.1<br>27.6<br>50.8|**62.7**<br>10.7<br>56.9<br>80.8<br>12.7<br>33.9<br>1.6<br>12.4<br>56.4<br>41.6<br>66.7<br>75.4<br>1.3<br>67.7<br>50.3<br>62.2|
|||50.7<br>52.7<br>49.0<br>52.0<br>50.0||
|**Contrastive Alignment Models**||||
|larger_clap_general<br>larger_clap_music_and_speech<br>clap-htsat-unfused<br>clap-htsat-fused<br>msclap-2023<br>wav2clip<br>MuQ-MuLan-large<br>msclap-2022|4<br>3|32.2<br>37.1<br>29.8<br>28.3<br>45.1|51.7<br>2.3<br>51.9<br>66.8<br>6.6<br>93.2<br>9.8<br>14.9|
||6<br>4<br>7<br>9<br>10<br>14<br>12<br>12<br>14<br>13<br>16<br>16<br>19<br>28|31.9<br>37.0<br>29.7<br>28.1<br>45.1<br><br>|51.3<br>2.7<br>52.1<br>65.6<br>7.7<br>94.3<br>9.3<br>13.2<br>45.2<br>1.8<br>52.6<br>66.5<br>12.5<br>88.8<br>8.8<br>11.3<br>44.5<br>4.0<br>52.0<br>61.3<br>**22.7**<br>82.8<br>9.2<br>13.2<br>45.0<br>5.8<br>53.6<br>75.4<br>15.2<br>87.3<br>9.4<br>12.6<br>39.4<br>13.0<br>53.6<br>68.9<br>6.0<br>68.9<br>1.0<br>10.8<br>40.7<br>10.3<br>51.9<br>**85.4**<br>4.3<br>**95.2**<br>1.1<br>12.6<br>38.3<br>7.6<br>51.7<br>62.9<br>19.9<br>82.4<br>13.7<br>12.1|
|||30.0<br>35.9<br>29.1<br>25.9<br>42.4||
|||30.7<br>36.2<br>29.0<br>27.3<br>43.2||
|||31.1<br>38.0<br>28.7<br>26.7<br>43.7||
|||25.5<br>32.7<br>23.2<br>21.5<br>38.8||
|||27.0<br>37.7<br>22.2<br>22.3<br>40.9||
|||29.8<br>36.1<br>29.7<br>27.3<br>39.9||
|**Sequence-to-sequence Models**||||
|whisper-medium<br>whisper-base<br>whisper-small<br>whisper-large-v3<br>whisper-tiny<br>speecht5_multimodal<br>mms-1b-l1107<br>mms-1b-all|3<br>2|46.7<br>46.0<br>41.7<br>44.2<br>48.2|57.5<br>22.3<br>53.9<br>67.6<br>5.0<br>69.5<br>-<br>-<br>53.0<br>11.7<br>52.1<br>65.0<br>5.0<br>64.5<br>-<br>-<br>53.4<br>15.5<br>52.6<br>64.2<br>3.9<br>66.2<br>-<br>-<br>50.7<br>17.1<br>52.5<br>63.9<br>3.4<br>69.1<br>-<br>-<br>51.0<br>14.9<br>51.5<br>63.4<br>7.4<br>62.7<br>-<br>-|
||8<br>6<br>9<br>7<br>11<br>8<br>13<br>10<br>22<br>37<br>25<br>27<br>29<br>29|42.7<br>41.9<br>38.7<br>39.6<br>44.4<br>43.2<br>42.6<br>38.8<br>40.5<br>44.8<br>42.1<br>42.8<br>37.3<br>40.0<br>43.8||
|||42.1<br>41.8<br>37.0<br>39.0<br>44.0||
|||25.8<br>29.6<br>23.2<br>23.5<br>38.4<br><br><br>|42.9<br>5.9<br>57.9<br>56.5<br>1.1<br>55.6<br>1.3<br>15.9<br><br><br>|
|||38.6<br>37.0<br>32.5<br>37.4<br>40.5<br>38.8<br>37.5<br>33.3<br>38.0<br>40.6|48.1<br>12.4<br>51.5<br>58.8<br>1.0<br>50.3<br>-<br>-<br>47.4<br>14.9<br>52.8<br>59.5<br>1.6<br>48.8<br>-<br>-|
|**Audio Encoders**||||
|ast-fnetuned-audioset-10-10-0.4593<br>vggish<br>wavlm-large<br>hubert-base-ls960<br>yamnet<br>wav2vec2-lv-60-espeak-cv-ft<br>wav2vec2-xls-r-2b<br>cnn14-esc50|15<br>15|44.2<br>50.1<br>40.4<br>36.8<br>44.5|48.9<br>26.1<br>51.2<br>77.6<br>6.9<br>90.2<br>-<br>-<br>41.8<br>9.7<br>52.8<br>78.7<br>7.8<br>83.8<br>-<br>-<br>43.9<br>7.1<br>52.3<br>68.8<br>2.4<br>71.8<br>-<br>-<br>43.2<br>8.3<br>51.9<br>66.3<br>2.7<br>70.7<br>-<br>-<br>40.1<br>16.6<br>54.6<br>81.7<br>1.6<br>74.5<br>-<br>-<br>48.6<br>8.2<br>53.7<br>55.6<br>1.6<br>46.9<br>-<br>-<br>48.4<br>7.8<br>50.8<br>62.9<br>1.4<br>53.7<br>-<br>-<br>33.5<br>9.4<br>54.2<br>53.8<br>7.4<br>72.3<br>-<br>-|
||17<br>17<br>18<br>18<br>20<br>19<br>21<br>20<br>23<br>22<br>24<br>23<br>30<br>21|39.1<br>45.8<br>38.0<br>34.9<br>40.9<br><br>||
|||37.9<br>41.1<br>35.4<br>36.6<br>39.7<br>37.5<br>40.5<br>36.7<br>35.6<br>39.3||
|||38.0<br>44.9<br>37.1<br>32.6<br>39.0<br><br><br>||
|||38.5<br>35.8<br>34.9<br>36.6<br>40.4<br>38.7<br>37.5<br>35.8<br>34.1<br>40.5<br>33.2<br>38.4<br>34.0<br>31.8<br>35.0||



whereas CLAP variants show more balanced performance across categories but lower peak scores on speech-specific tasks. 

Models pretrained on massively multilingual automatic speech recognition data (SeamlessM4T, MMS) substantially outperform other approaches on multilingual classification— SeamlessM4T-v2-large achieves the best performance on 10 of 12 languages in MInDS-14 (Table 9). Yet this strength does not transfer to music or environmental sound tasks. Conversely, audio-text models like CLAP variants, despite their strength on environmental audio, score below 15% across all languages on MInDS-14, near random chance for intent classification. 

While LCO-Embedding-Omni-7B and Qwen2-Audio-7B both rank at the top and leverage similar training approaches, they obtain drastically different scores on cross-modal re- 

trieval tasks (50.3% and 1.6%, respectively). This highlights that scale and multimodal pretraining do not guarantee balanced performance. This indicates that training paradigm, data curation, and architectural choices matter more than parameter count for general audio embedding quality, echoing findings from text embedding research. 

_Direction:_ The specialization gap calls for domain-agnostic architectures that generalize across speech, music, and environmental sound without sacrificing domain-specific capabilities. Future work should explore unified training objectives and architectural innovations that maintain strong performance across the full acoustic spectrum. 

**(b) Multilingual audio understanding remains unsolved.** Despite evaluation across 200+ languages via SIBFLEURS (Adelani et al., 2024) (94 languages), CommonVoice (Ardila et al., 2020) (43 languages), MInDS-14 (Gerz 

5

<!-- page: 6 -->

## **MAEB: Massive Audio Embedding Benchmark** 

**==> picture [211 x 214] intentionally omitted <==**

**----- Start of picture text -----**<br>
Speech<br>100<br>80<br>60<br>40<br>Emotion Music<br>20<br>Bioacoustics Environmental<br>Universal Target (80%) LCO-7B AST Qwen2-Audio<br>**----- End of picture text -----**<br>


_Figure 2._ Domain-level performance on 94 tasks in MAEB+. Radial plot shows the top-performing model for each of the five acoustic domains: Speech (44 tasks), Music (13), Environmental (29), Bioacoustics (2), and Emotion (6). The dashed line represents an 80 target for universal performance, which remains unmet. Scores are averaged across all available task types (classification, clustering, retrieval, reranking). See Appendix D for methodology. 

et al., 2021) (14 languages), VoxPopuli (Wang et al., 2021a) (5 languages), and FLEURS (Schmidt et al., 2025) (102 languages), models demonstrate a strong bias toward highresource languages with severely degraded performance on African, Indigenous, and minority languages. On SIBFLEURS classification (Table 10), high-resource European languages achieve 40–60% accuracy while low-resource languages like Umbundu, Yoruba, and Xhosa remain below 20% even for the best models. 

This disparity becomes catastrophic for cross-modal tasks. While audio-to-audio retrieval maintains reasonable performance across languages (50–99% on JamAlt, Table 37), cross-modal audio-text retrieval collapses in multilingual settings. On FLEURS retrieval across 102 languages (Tables 27–33), even the best CLAP models achieve below 3% for most language pairs, with audio-to-text and text-to-audio retrieval scores often below 1%. Current audio-text alignment approaches, trained predominantly on English data, fail completely to generalize to multilingual scenarios—a critical gap for global audio retrieval applications. 

_Direction:_ We recommend extending contrastive audio-text pretraining to multilingual corpora and implementing crosslingual transfer learning to leverage high-resource language knowledge for the 100+ languages where current models achieve near-random performance. 

**(c) Acoustic versus linguistic representations trade off.** Multilingual evaluation reveals fundamental trade-offs between acoustic and linguistic representations that current architectures cannot reconcile. On VoxPopuli tasks (Table 17), CLAP-htsat-unfused achieves 94.4% on gender identification but only 30.0% on language identification, while Whisper-medium shows the inverse pattern (59.2% vs 99.4%). This suggests that models optimized for acoustic properties (timbre, speaker characteristics) develop fundamentally different representations than those optimized for linguistic content. 

This trade-off extends to audio-text alignment more broadly. The performance gap between audio-only and audio-text tasks is substantial: as shown in Table 2, AST achieves 44.2% overall but cannot perform cross-modal tasks (showing “-” for Retrieval and Zero-shot Classification), while CLAP variants achieve around 30-32% overall despite enabling cross-modal tasks. Within audio-text tasks, most models show weak retrieval performance (CLAP variants around 8-14%), though LCO-Embedding-Omni-7B achieves 50.3% cross-modal retrieval and 64.5% zeroshot classification, demonstrating that stronger cross-modal alignment is possible with appropriate training. Models struggle especially with complex audio scenes and abstract musical concepts, suggesting current training objectives fail to capture deeper semantic relationships beyond surfacelevel correspondences. 

_Direction:_ Future architectures should explore disentangled representations or multi-task learning approaches that capture both acoustic properties (speaker, timbre) and linguistic content simultaneously, enabling models to perform well on both gender identification and language identification without sacrificing one for the other. 

**(d) Clustering exposes fundamental representation gaps.** Clustering tasks prove universally challenging across all evaluated models, revealing a consistent weakness in semantic structure. Even the best-performing model on clustering (clap-htsat-fused) achieves only 22.7%, while top-ranked models show inconsistent clustering performance: Qwen2Audio-7B (2nd overall) scores 12.7%, LCO-EmbeddingOmni-7B (1st overall, highest average scores) achieves only 1.7%, and whisper-medium (3rd overall) reaches just 5.0%. This disconnect between supervised and unsupervised task performance suggests that current audio embeddings lack the semantic organization necessary for grouping related audio without explicit labels—a fundamental limitation for applications requiring audio organization, discovery, or similarity-based retrieval at scale. 

_Direction:_ Incorporating clustering-aware losses or contrastive objectives that explicitly encourage semantically coherent embedding neighborhoods could address this gap, 

6

<!-- page: 7 -->

## **MAEB: Massive Audio Embedding Benchmark** 

**==> picture [211 x 172] intentionally omitted <==**

**----- Start of picture text -----**<br>
50 R [2] = 0.86 Qwen2-Audio (Qwen2-Audio)<br>p  = 0.072<br>40<br>SALMONN (Whisper)<br>30<br>20<br>LTU (AST)<br>10<br>Pengi (CLAP)<br>0<br>45 50 55 60 65 70<br>MAEB+ Embedding Score (%)<br>Audio LLM Score (MMAU %)<br>**----- End of picture text -----**<br>


_Figure 3._ **MAEB+ embedding quality correlates with Audio LLM performance.** MMAU evaluates Audio LLMs across Speech, Music, and Sound, the same domains covered by MAEB+. Each point plots an Audio LLM’s overall MMAU score (y-axis, averaged across domains) against its encoder’s MAEB+ score (x-axis, computed from 26 classification tasks aligned with MMAU domains). Preliminary correlation (R²=0.86, p=0.072, n=4) suggests a positive relationship between embedding quality and downstream reasoning, though the small sample size and statistical marginality warrant caution in interpreting this relationship. 

enabling applications that require audio organization without explicit labels. 

## **4.2. Correlation with Audio LLM Performance** 

To assess whether MAEB scores translate to real-world multimodal capabilities, we examine the relationship between encoder quality and Audio LLM performance on the MMAU benchmark (Sakshi et al., 2024). MMAU evaluates multimodal audio understanding through expert-annotated questions organized into three domains: Speech, Music, and Sound. To ensure a direct comparison, we compute the encoder’s embedding quality using a subset of 26 classification tasks from MAEB+ selected to align with these three domains (see Appendix C for the full task list). 

We compare four Audio LLMs that use different encoder architectures: Qwen2-Audio (Qwen2-Audio encoder), SALMONN (Whisper), LTU (AST), and Pengi (CLAP). Figure 3 shows a preliminary positive correlation across four models. Given the strong correlation between MAEB and MAEB(extended) established in subsection 2.1, this result suggests that the efficient MAEB benchmark serves as a reliable predictive signal for downstream Audio LLM performance. 

## **5. Limitations** 

**Technical Constraints** While our evaluation includes 50+ models spanning multiple architectures, this represents only 

a subset of available models. Audio length management poses challenges: models with native limits below 30 seconds retain those settings, while others are limited to 30 seconds for memory management, restricting applicability to long-form content like podcasts or lectures. While future standardization around pre-processing pipelines could streamline evaluation, our approach currently reflects the diverse sampling rate requirements inherent to different audio domains rather than a benchmark limitation. Large-scale models (Whisper-large-v3: 1.55B parameters, Wav2Vec2XLS-R-2B: 2B parameters) require substantial computational resources, limiting accessibility. 

**Dataset Coverage Limitations** The benchmark exhibits several coverage gaps. Domain representation skews toward Western musical traditions and standard speech patterns. Language coverage, while spanning 100+ languages, remains limited for many underrepresented language families, with some languages appearing in only a single datasets, preventing comprehensive cross-task evaluation. The language distribution of MAEB is shown in Figure 4. 

Task coverage across 30 tasks in MAEB (98 in MAEB+) and 7 categories still lacks certain capabilities including audio generation quality assessment and real-time processing evaluation. Ecological validity is limited as many tasks use clean, studio-recorded audio that does not reflect realworld conditions with noise, reverberation, and compression artifacts. 

## **6. Related Work** 

**Text Embedding Benchmarks** Large, standardized benchmarks have been critical for driving progress in representation learning. For text, MTEB provides a comprehensive evaluation suite spanning 8 task families across 58 datasets and 112 languages, enabling systematic assessment of generalization beyond task-specific setups (Muennighoff et al., 2023). Recent expansions toward massive multilingual and multimodal evaluations such as MMTEB for multilingual text embeddings and MIEB for image embeddings reinforce the value of broad, regularly maintained leaderboards with consistent protocols (Enevoldsen et al., 2025; Xiao et al., 2025b). These efforts motivate analogous, up-to-date benchmarking for audio embeddings. 

**Audio Representation Benchmarks** HEAR (Turian et al., 2022) represents one of the first attempts to evaluate generalpurpose audio embeddings across diverse domains such as speech recognition, music tagging, and environmental sound classification. Evaluating 29 models on 19 downstream tasks, HEAR primarily tests pretrained features with simple classifiers like multilayer perceptrons (MLPs), leaving room for exploration with more complex architectures. 

7

<!-- page: 8 -->

## **MAEB: Massive Audio Embedding Benchmark** 

**==> picture [487 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
70 Total Languages: 165<br>60 Average Tasks per Language: 4.8Total Task-Language Pairs: 796<br>50 Most Common: eng (70 tasks)<br>40<br>30<br>20<br>10<br>0<br>Language Code<br>Number of Tasks<br>engzxxfradeu spapolkor rustel tamjpnfinurdmal marbenhin cesitanld porslvindkatvielitoci swe ronmkdukr turtha srphauslkhun astdan est cymbulbelellglgara monswh panasm catmltckbamhluoafrgujhebluguzn yorglekirkazkanhyeaze swaori pusbrefas yuelavzhoxhotgkwolzulumbsnd som snalinceb peshrviboisljavkamboskhmlaokealtzmri myanobnpinsolvs nyasat tatsrdsqiskr nanroh nepsah nno osstoktigfil tsntukuigyidzghzza cmnmrjful msa orm acm apcarqmyvinamhr eusfroabkbakbaschvcnhmdfdivdyuepodavfryklnltggrnkmrlijkinkabhsb arzazjarbfuv gazzsmkhktglorypbt<br>**----- End of picture text -----**<br>


_Figure 4._ Language distribution in the MAEB+ collection. English dominates with 70 tasks. We use zxx (No Linguistic Content) to tag datasets with no languages present. 

Despite this progress, comprehensive evaluation of audio embeddings remains limited. Task coverage is narrow, focusing primarily on classification while neglecting systematic evaluation across fundamental applications such as retrieval, and clustering. Similarly, zero-shot performance testing remains fragmented with prior work exploring approaches such as using textual label embeddings, sentence descriptions, or even image embeddings of sound classes (Xie et al., 2021; Mercea et al., 2022), but these efforts are isolated and not integrated into comprehensive evaluation frameworks. Large-scale multilingual support also remains an outstanding issue despite the importance of supporting diverse languages and accents (Xu et al., 2024). Maintenance and reproducibility pose ongoing challenges, with outdated datasets and inconsistent evaluation protocols hindering fair model comparison of current models. MAEB addresses these limitations by building into an existing and maintained framework for evaluating embeddings, drawing on lessons from MTEB while adapting to the unique challenges of audio representation learning. Separately, AudioBench (Wang et al., 2024) and MMAU (Sakshi et al., 2024) focus on evaluating AudioLLMs rather than embedding models. AudioBench evaluates instructionfollowing capabilities across eight tasks using 26 datasets, while MMAU introduces multimodal benchmarks requiring reasoning across speech, sound, and music domains. 

tering (12.7%). Speech-pretrained models (e.g., Whisper) perform strongly on audio-only tasks but cannot support cross-modal evaluation, while contrastive audio-text models (e.g., CLAP variants) provide cross-modal capabilities but remain weak on multilingual speech tasks. 

Clustering proves universally challenging (best model: 22.7%), exposing fundamental limitations in semantic structure. We observe stark trade-offs between acoustic and linguistic features, with models excelling at gender identification struggling on language identification and vice versa. Cross-modal multilingual retrieval reveals a stark capability gap: LCO models achieve 50%+ accuracy across 100+ languages, while most other models (CLAP, Whisper, ASR encoders) remain below 2%, highlighting the critical role of speech-text alignment for this task. Preliminary analysis across four Audio LLMs suggests a positive relationship between MAEB encoder quality and downstream performance, validating the benchmark’s relevance for multimodal audio understanding. 

MAEB integrates into the MTEB ecosystem, enabling unified evaluation across text, image, and audio modalities. We release code, tasks, and leaderboards to support communitydriven progress toward robust, multilingual audio representations. 

## **Impact Statement** 

## **7. Conclusion** 

We introduce the Massive Audio Embedding Benchmark (MAEB), comprising 30 tasks across 100+ languages with baselines from 50+ models. 

Our evaluation reveals critical gaps in current audio representations. No single model achieves universal performance: LCO-Embedding-Omni-7B ranks first overall, achieving the strongest cross-modal retrieval (50.3%) and zero-shot classification (64.5%) averages in our MAEB evaluation. Qwen2Audio-7B ranks second overall and ranks first on audio-only tasks, excelling particularly in reranking (80.8%) and clus- 

Large benchmarks create barriers for low-resource communities and incur high environmental costs. We have reduced large datasets to reasonable sizes and include kilogram _CO_ 2 measures per task, allowing users to assess environmental benchmarking costs. 

## **References** 

- Adelani, D. I., Liu, H., Shen, X., Vassilyev, N., Alabi, J. O., Mao, Y., Gao, H., and Lee, A. E.-S. Sib-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects, 2024. URL 

8

<!-- page: 9 -->

## **MAEB: Massive Audio Embedding Benchmark** 

https://arxiv.org/abs/2309.07445. 

- Adigwe, A., Tits, N., Haddad, K. E., Ostadabbas, S., and Dutoit, T. The emotional voices database: Towards controlling the emotion dimension in voice generation systems, 2018. URL https://arxiv.org/abs/ 1806.09514. 

- Agostinelli, A., Denk, T. I., Borsos, Z., Engel, J., Verzetti, M., Caillon, A., Huang, Q., Jansen, A., Roberts, A., Tagliasacchi, M., Sharifi, M., Zeghidour, N., and Frank, C. Musiclm: Generating music from text, 2023. URL https://arxiv.org/abs/2301.11325. 

   - Baevski, A., Zhou, Y., Mohamed, A., and Auli, M. wav2vec 2.0: A framework for self-supervised learning of speech representations. _Advances in neural information processing systems_ , 33:12449–12460, 2020. 

   - Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., and Auli, M. data2vec: A general framework for self-supervised learning in speech, vision and language, 2022. URL https://arxiv.org/abs/2202.03555. 

   - Bakhturina, E., Lavrukhin, V., Ginsburg, B., and Zhang, Y. Hi-Fi Multi-Speaker English TTS Dataset. _arXiv preprint arXiv:2104.01497_ , 2021. 

- Alain, G. and Bengio, Y. Understanding intermediate layers using linear classifier probes, 2018. URL https:// arxiv.org/abs/1610.01644. 

- Allauzen, C., Heigold, G., Ma, J., Variani, E., Riley, M., and Bagby, T. Massive sound embedding benchmark (MSEB). In _The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , 2025. URL https://neurips.cc/ virtual/2025/poster/121597. 

- Anantapadmanabhan, A., Bellur, A., and Murthy, H. A. Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization. In _2013 IEEE International Conference on Acoustics, Speech and Signal Processing_ , pp. 181–185, 2013. doi: 10.1109/ ICASSP.2013.6637633. 

   - Bazilinskyy, P., van der Aa, A., Schoustra, M., Spruit, J., Staats, L., van der Vlist, K. J., and de Winter, J. An auditory dataset of passing vehicles recorded with a smartphone. In _12th International Symposium on Tools and Methods of Competitive Engineering (TMCE 2018)_ , pp. 417–422, 2018. 

   - Busso, C., Bulut, M., Lee, C.-C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J. N., Lee, S., and Narayanan, S. S. Iemocap: Interactive emotional dyadic motion capture database. _Language resources and evaluation_ , 42(4): 335–359, 2008. 

   - Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., and Verma, R. CREMA-D: Crowd-sourced emotional multimodal actors dataset. _IEEE Trans. Affect. Comput._ , 5(4):377–390, oct 2014. 

- Ao, J., Wang, R., Zhou, L., Wang, C., Ren, S., Wu, Y., Liu, S., Ko, T., Li, Q., Zhang, Y., Wei, Z., Qian, Y., Li, J., and Wei, F. SpeechT5: Unified-modal encoder-decoder pre-training for spoken language processing. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 5723–5738, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.393. URL https: //aclanthology.org/2022.acl-long.393/. 

- Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J., Morais, R., Saunders, L., Tyers, F. M., and Weber, G. Common voice: A massively-multilingual speech corpus, 2020. URL https://arxiv.org/ abs/1912.06670. 

- Babu, A., Wang, C., Tjandra, A., Lakhotia, K., Xu, Q., Goyal, N., Singh, K., von Platen, P., Saraf, Y., Pino, J., Baevski, A., Conneau, A., and Auli, M. Xls-r: Selfsupervised cross-lingual speech representation learning at scale, 2021. URL https://arxiv.org/abs/ 2111.09296. 

- Chen, G., Chai, S., Wang, G., Du, J., Zhang, W.-Q., Weng, C., Su, D., Povey, D., Trmal, J., Zhang, J., Jin, M., Khudanpur, S., Watanabe, S., Zhao, S., Zou, W., Li, X., Yao, X., Wang, Y., Wang, Y., You, Z., and Yan, Z. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. In _Proc. Interspeech 2021_ , 2021. 

- Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Qian, Y., Wu, J., Zeng, M., Yu, X., and Wei, F. Wavlm: Large-scale selfsupervised pre-training for full stack speech processing. _IEEE Journal of Selected Topics in Signal Processing_ , 16 (6):1505–1518, October 2022a. ISSN 1941-0484. doi: 10.1109/jstsp.2022.3188113. URL http://dx.doi. org/10.1109/JSTSP.2022.3188113. 

- Chen, S., Wu, Y., Wang, C., Chen, Z., Chen, Z., Liu, S., Wu, J., Qian, Y., Wei, F., Li, J., and Yu, X. Unispeech-sat: Universal speech representation learning with speaker aware pre-training. In _ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 6152–6156, 2022b. doi: 10.1109/ICASSP43922.2022.9747077. 

9

<!-- page: 10 -->

## **MAEB: Massive Audio Embedding Benchmark** 

- Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 2818–2829, 2023. 

   - Lee, A., Ma, X., Mourachko, A., Peloquin, B., Pino, J., Popuri, S., Ropers, C., Saleem, S., Schwenk, H., Sun, A., Tomasello, P., Wang, C., Wang, J., Wang, S., and Williamson, M. Seamless: Multilingual expressive and streaming speech translation, 2023. URL https://arxiv.org/abs/2312.05187. 

- Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., Zhou, C., and Zhou, J. Qwen2audio technical report, 2024. URL https://arxiv. org/abs/2407.10759. 

   - Conneau, A., Baevski, A., Collobert, R., Mohamed, A.-r., and Auli, M. Unsupervised cross-lingual representation learning for speech recognition. In _Proc. Interspeech 2020_ , pp. 2426–2430, 2020. 

- Chung, I., Kerboua, I., Kardos, M., Solomatin, R., and Enevoldsen, K. Maintaining mteb: Towards long term usability and reproducibility of embedding benchmarks, 2025. URL https://arxiv.org/abs/ 2506.21182. 

   - Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V., Dalmia, S., Riesa, J., Rivera, C., and Bapna, A. Fleurs: Few-shot learning evaluation of universal representations of speech. In _2022 IEEE Spoken Language Technology Workshop (SLT)_ , pp. 798–805. IEEE, 2023. 

- Chung, J. S., Nagrani, A., and Zisserman, A. Voxceleb2: Deep speaker recognition. In _Proceedings of Interspeech_ , 2018. 

   - Drossos, K., Lipping, S., and Virtanen, T. Clotho: An audio captioning dataset, 2019. URL https://arxiv. org/abs/1910.09387. 

- Cífka, O., Schreiber, H., Miner, L., and Stöter, F. Lyrics transcription for humans: A readability-aware benchmark. In _Proceedings of the 25th International Society for Music Information Retrieval Conference_ , pp. 737–744. ISMIR, 2024. doi: 10.5281/ZENODO.14877443. URL https: //doi.org/10.5281/zenodo.14877443. 

   - Défossez, A., Copet, J., Synnaeve, G., and Adi, Y. High fidelity neural audio compression, 2022. URL https: //arxiv.org/abs/2210.13438. 

   - Elizalde, B., Deshmukh, S., Al Ismail, M., and Wang, H. Clap: Learning audio concepts from natural language supervision. In _ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 1–5. IEEE, 2023. 

- Clark, R. and Richmond, K. A detailed report on the cmu arctic speech database. Technical Report CMU-LTI-03177, Carnegie Mellon University, Language Technologies Institute, 2003. 

   - Enevoldsen, K., Chung, I., Kerboua, I., Kardos, M., Mathur, A., Stap, D., Gala, J., Siblini, W., Krzemi´nski, D., Winata, G. I., Sturua, S., Utpala, S., Ciancone, M., Schaeffer, M., Sequeira, G., Misra, D., Dhakal, S., Rystrøm, J., Solomatin, R., Ömer Ça˘gatan, Kundu, A., Bernstorff, M., Xiao, S., Sukhlecha, A., Pahwa, B., Po´swiata, R., GV, K. K., Ashraf, S., Auras, D., Plüster, B., Harries, J. P., Magne, L., Mohr, I., Hendriksen, M., Zhu, D., Gisserot-Boukhlef, H., Aarsen, T., Kostkan, J., Wojtasik, K., Lee, T., Šuppa, M., Zhang, C., Rocca, R., Hamdy, M., Michail, A., Yang, J., Faysse, M., Vatolin, A., Thakur, N., Dey, M., Vasani, D., Chitale, P., Tedeschi, S., Tai, N., Snegirev, A., Günther, M., Xia, M., Shi, W., Lù, X. H., Clive, J., Krishnakumar, G., Maksimova, A., Wehrli, S., Tikhonova, M., Panchal, H., Abramov, A., Ostendorff, M., Liu, Z., Clematide, S., Miranda, L. J., Fenogenova, A., Song, G., Safi, R. B., Li, W.D., Borghini, A., Cassano, F., Su, H., Lin, J., Yen, H., Hansen, L., Hooker, S., Xiao, C., Adlakha, V., Weller, O., Reddy, S., and Muennighoff, N. Mmteb: Massive multilingual text embedding benchmark, 2025. URL https://arxiv.org/abs/2502.13595. 

- Colombo, P., Noiry, N., Irurozki, E., and Clémençon, G. I., S. What are the best systems? new perspectives M., on nlp benchmarking. In Koyejo, S., Mohamed, S., Solomatin, Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), M., Xiao, _Advances in Neural Information Processing Systems_ , GV, K. K., volume 35, pp. 26915–26932. Curran Associates, Inc., J. P., 2022. URL https://proceedings.neurips. cc/paper_files/paper/2022/file/ tasik, K., ac4920f4085b5662133dd751493946a6-Paper-Conference. pdf. 

- Communication, S., Barrault, L., Chung, Y.-A., Meglioli, M. C., Dale, D., Dong, N., Duppenthaler, M., Duquenne, P.-A., Ellis, B., Elsahar, H., Haaheim, J., Hoffman, J., Hwang, M.-J., Inaguma, H., Klaiber, C., Kulikov, I., Li, P., Licht, D., Maillard, J., Mavlyutov, R., Rakotoarison, A., Sadagopan, K. R., Ramakrishnan, A., Tran, T., Wenzek, G., Yang, Y., Ye, E., Evtimov, I., Fernandez, P., Gao, C., Hansanti, P., Kalbassi, E., Kallet, A., Kozhevnikov, A., Gonzalez, G. M., Roman, R. S., Touret, C., Wong, C., Wood, C., Yu, B., Andrews, P., Balioglu, C., Chen, P.-J., Costa-jussà, M. R., Elbayad, M., Gong, H., Guzmán, F., Heffernan, K., Jain, S., Kao, J., 

- Engel, J., Resnick, C., Roberts, A., Dieleman, S., Eck, D., Simonyan, K., and Norouzi, M. Neural audio synthesis 

10

<!-- page: 11 -->

**MAEB: Massive Audio Embedding Benchmark** 

- of musical notes with wavenet autoencoders, 2017. URL https://arxiv.org/abs/1704.01279. 

- Fonseca, E., Plakal, M., Ellis, D. P. W., Font, F., Favory, X., and Serra, X. Learning sound event classifiers from web audio with noisy labels. In _ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 21–25. IEEE, 2019. 

- Fonseca, E., Favory, X., Pons, J., Font, F., and Serra, X. Fsd50k: an open dataset of human-labeled sound events. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 30:829–852, 2021. 

- Gemmeke, J. F., Ellis, D. P. W., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., Plakal, M., and Ritter, M. Audio set: An ontology and human-labeled dataset for audio events. In _Proc. IEEE ICASSP 2017_ , New Orleans, LA, 2017. 

- Gerz, D., Su, P., Kusztos, R., Mondal, A., Lis, M., Singhal, E., Mrkši´c, N., Wen, T., and Vulic, I. Multilingual and cross-lingual intent detection from spoken data. _CoRR_ , abs/2104.08524, 2021. URL https://arxiv.org/ abs/2104.08524. 

- Gong, Y., Chung, Y.-A., and Glass, J. Ast: Audio spectrogram transformer, 2021. URL https://arxiv.org/ abs/2104.01778. 

- Gong, Y., Yu, J., and Glass, J. Vocalsound: A dataset for improving human vocal sounds recognition. In _ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ . IEEE, May 2022. doi: 10.1109/icassp43922.2022. 9746828. URL http://dx.doi.org/10.1109/ ICASSP43922.2022.9746828. 

- Groh, R., Goes, N., and Kist, A. M. Spoken-100: A cross-lingual benchmarking dataset for the classification of spoken numbers in different languages, 2024. URL https://arxiv.org/abs/2403.09753. 

- Hershey, S., Chaudhuri, S., Ellis, D. P. W., Gemmeke, J. F., Jansen, A., Moore, R. C., Plakal, M., Platt, D., Saurous, R. A., Seybold, B., Slaney, M., Weiss, R. J., and Wilson, K. Cnn architectures for large-scale audio classification. In _2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 131–135. IEEE Press, 2017. doi: 10.1109/ ICASSP.2017.7952132. URL https://doi.org/ 10.1109/ICASSP.2017.7952132. 

- Hershey, S., Ellis, D. P. W., Fonseca, E., Jansen, A., Liu, C., Moore, R. C., and Plakal, M. The benefit of temporallystrong labels in audio event classification, 2021. URL https://arxiv.org/abs/2105.07031. 

- Homburg, H., Mierswa, I., Möller, B., Morik, K., and Wurst, M. A benchmark dataset for audio classification and clustering. In _ISMIR_ , volume 2005, pp. 528–31, 2005. 

- Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., and Mohamed, A.-r. Hubert: Selfsupervised speech representation learning by masked prediction of hidden units. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 29:3451–3460, 2021. 

- James, J., Li, T., and Watson, C. An open source emotional speech corpus for human robot interaction applications. In _Proc. Interspeech 2018_ , 2018. 

- Kim, C. D., Kim, B., Lee, H., and Kim, G. Audiocaps: Generating captions for audios in the wild. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pp. 119–132, 2019. 

- Klinck, H., Cañas, J. S., Demkin, M., Dane, S., Kahl, S., and Denton, T. Birdclef+ 2025. https://kaggle.com/ competitions/birdclef-2025, 2025. Kaggle. 

- Koepke, A., Oncescu, A.-M., Henriques, J., Akata, Z., and Albanie, S. Audio retrieval with natural language queries: A benchmark study. In _IEEE Transactions on Multimedia_ , 2022. 

- Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., and Plumbley, M. D. Panns: Large-scale pretrained audio neural networks for audio pattern recognition, 2020. URL https://arxiv.org/abs/1912.10211. 

- Li, C.-H., Ma, S.-L., Zhang, H.-W., Lee, H.-y., and Lee, L.-s. Spoken squad: A study of mitigating the impact of speech recognition errors on listening comprehension. In _Interspeech_ , pp. 3459–3463, 2018. 

- Lin, G.-T., Chuang, Y.-S., Chung, H.-L., wen Yang, S., Chen, H.-J., Dong, S., Li, S.-W., Mohamed, A., yi Lee, H., and shan Lee, L. Dual: Discrete spoken unit adaptive learning for textless spoken question answering, 2022. URL https://arxiv.org/abs/2203.04911. 

- Livingstone, S. R. and Russo, F. A. The ryerson audio-visual database ofal speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english. _PLOS ONE_ , 13(5):1–35, 05 2018. doi: 10.1371/journal.pone.0196391. URL https://doi. org/10.1371/journal.pone.0196391. 

- Lugosch, L., Likhomanenko, T., Synnaeve, G., and Collobert, R. Pseudo-labeling for massively multilingual speech recognition, 2022. URL https://arxiv. org/abs/2111.00161. 

11

<!-- page: 12 -->

**MAEB: Massive Audio Embedding Benchmark** 

- Martin-Morato, I. and Mesaros, A. What is the ground truth? reliability of multi-annotator data for audio tagging, 2021. URL https://arxiv.org/abs/2104.04214. 

- Mercea, O.-B., Riesch, L., Koepke, A. S., and Akata, Z. Audio-visual generalised zero-shot learning with crossmodal attention and language. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 10553–10563, June 2022. 

- Mesaros, A., Heittola, T., and Virtanen, T. A multi-device dataset for urban acoustic scene classification. In _Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018)_ , Tampere, Finland, 2018. Tampere University of Technology. URL https://arxiv.org/abs/1807.09840. 

- Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. Mteb: Massive text embedding benchmark, 2023. URL https://arxiv.org/abs/2210.07316. 

- Park, C., Min, C., Bhattacharya, S., and Kawsar, F. Augmenting conversational agents with ambient acoustic contexts. In _22nd International Conference on HumanComputer Interaction with Mobile Devices and Services_ , MobileHCI ’20, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450375160. doi: 10.1145/3379503.3403535. URL https://doi. org/10.1145/3379503.3403535. 

- Piczak, K. J. Esc: Dataset for environmental sound classification. In _Proceedings of the 23rd ACM International Conference on Multimedia_ , MM ’15, pp. 1015–1018, New York, NY, USA, 2015. Association for Computing Machinery. ISBN 9781450334594. doi: 10. 1145/2733373.2806390. URL https://doi.org/ 10.1145/2733373.2806390. 

- Pratap, V., Tjandra, A., Shi, B., Tomasello, P., Babu, A., Kundu, S., Elkahky, A., Ni, Z., Vyas, A., Fazel-Zarandi, M., Baevski, A., Adi, Y., Zhang, X., Hsu, W.-N., Conneau, A., and Auli, M. Scaling speech technology to 1,000+ languages, 2023. URL https://arxiv. org/abs/2305.13516. 

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020. 

- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., and Sutskever, I. Robust speech recognition via largescale weak supervision, 2022. URL https://arxiv. org/abs/2212.04356. 

- Raponi, S., Ali, I., and Oligeri, G. Sound of guns: Digital forensics of gun audio samples meets artificial intelligence, 2021. URL https://arxiv.org/abs/ 2004.07948. 

- Rauch, L., Schwinger, R., Wirth, M., Heinrich, R., Huseljic, D., Herde, M., Lange, J., Kahl, S., Sick, B., Tomforde, S., and Scholz, C. Birdset: A large-scale dataset for audio classification in avian bioacoustics, 2024. URL https://arxiv.org/abs/2403.10380. 

- Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., Subakan, C., Dawid, N., Heba, A., Zhong, J., et al. Speechbrain: A general-purpose speech toolkit. _arXiv preprint arXiv:2106.04624_ , 2021. 

- Rosenberg, A. and Hirschberg, J. V-measure: A conditional entropy-based external cluster evaluation measure. In Eisner, J. (ed.), _Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLPCoNLL)_ , pp. 410–420, Prague, Czech Republic, June 2007. Association for Computational Linguistics. URL https://aclanthology.org/D07-1043/. 

- Sakshi, S., Tyagi, U., Kumar, S., Seth, A., Selvakumar, R., Nieto, O., Duraiswami, R., Ghosh, S., and Manocha, D. Mmau: A massive multi-task audio understanding and reasoning benchmark, 2024. URL https://arxiv. org/abs/2410.19168. 

- Salamon, J., Jacoby, C., and Bello, J. P. A dataset and taxonomy for urban sound research. In _Proceedings of the 22nd ACM international conference on Multimedia_ , pp. 1041–1044. ACM, 2014. 

- Schmidt, F. D., Vuli´c, I., Glavaš, G., and Adelani, D. I. Fleurs-slu: A massively multilingual benchmark for spoken language understanding, 2025. URL https: //arxiv.org/abs/2501.06117. 

- Shon, S., Pasad, A., Wu, F., Brusco, P., Artzi, Y., Livescu, K., and Han, K. J. Slue: New benchmark tasks for spoken language understanding evaluation on natural speech, 2022. URL https://arxiv.org/abs/2111.10367. 

- Shon, S., Arora, S., Lin, C.-J., Pasad, A., Wu, F., Sharma, R., Wu, W.-L., Lee, H.-Y., Livescu, K., and Watanabe, S. Slue phase-2: A benchmark suite of diverse spoken language understanding tasks, 2023. URL https:// arxiv.org/abs/2212.10525. 

- Sinisetty, G., Ruban, P., Dymov, O., and Ravanelli, M. Commonlanguage, June 2021. URL https://doi.org/ 10.5281/zenodo.5036977. 

12

<!-- page: 13 -->

**MAEB: Massive Audio Embedding Benchmark** 

- Stoter, F.-R., Chakrabarty, S., Edler, B., and Habets, E. A. P. Classification vs. regression in supervised learning for single channel speaker count estimation. In _2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 436–440. IEEE, April 2018. doi: 10.1109/icassp.2018.8462159. URL http://dx. doi.org/10.1109/ICASSP.2018.8462159. 

- Tian, M., Srinivasamurthy, A., Sandler, M., and Serra, X. A study of instrument-wise onset detection in beijing opera percussion ensembles. In _2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 2159–2163, 2014. doi: 10.1109/ICASSP. 2014.6853981. 

- Turian, J., Shier, J., Khan, H. R., Raj, B., Schuller, B. W., Steinmetz, C. J., Malloy, C., Tzanetakis, G., Velarde, G., McNally, K., Henry, M., Pinto, N., Noufi, C., Clough, C., Herremans, D., Fonseca, E., Engel, J., Salamon, J., Esling, P., Manocha, P., Watanabe, S., Jin, Z., and Bisk, Y. Hear: Holistic evaluation of audio representations, 2022. URL https://arxiv.org/abs/2203.03022. 

- Tzanetakis, G. and Cook, P. Musical genre classification of audio signals. _IEEE Transactions on Speech and Audio Processing_ , 10(5):293–302, 2002. doi: 10.1109/TSA. 2002.800560. 

- Valk, J. and Alumäe, T. Voxlingua107: a dataset for spoken language recognition, 2020. URL https://arxiv. org/abs/2011.12998. 

- Wang, B., Zou, X., Lin, G., Sun, S., Liu, Z., Zhang, W., Liu, Z., Aw, A., and Chen, N. F. Audiobench: A universal benchmark for audio large language models. _arXiv preprint arXiv:2406.16020_ , 2024. 

- Wang, C., Riviere, M., Lee, A., Wu, A., Talnikar, C., Haziza, D., Williamson, M., Pino, J., and Dupoux, E. VoxPopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 993–1003, Online, August 2021a. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.80. URL https: //aclanthology.org/2021.acl-long.80. 

- Wang, C., Wu, Y., Qian, Y., Kumatani, K., Liu, S., Wei, F., Zeng, M., and Huang, X. Unispeech: Unified speech representation learning with labeled and unlabeled data, 2021b. URL https://arxiv.org/ abs/2101.07597. 

- Wang, Z., Subakan, C., Jiang, X., Wu, J., Tzinis, E., Ravanelli, M., and Smaragdis, P. Learning repre- 

sentations for new sound classes with continual selfsupervised learning. _IEEE Signal Processing Letters_ , 29:2607–2611, 2022. ISSN 1558-2361. doi: 10.1109/ lsp.2022.3229643. URL http://dx.doi.org/10. 1109/LSP.2022.3229643. 

- Warden, P. Speech commands: A dataset for limitedvocabulary speech recognition. _CoRR_ , abs/1804.03209, 2018. URL http://arxiv.org/abs/1804. 03209. 

- Wu, F., Kim, K., Pan, J., Han, K., Weinberger, K. Q., and Artzi, Y. Performance-efficiency trade-offs in unsupervised pre-training for speech recognition, 2021. URL https://arxiv.org/abs/2109.06870. 

- Wu, H.-H., Seetharaman, P., Kumar, K., and Bello, J. P. Wav2clip: Learning robust audio representations from clip. In _ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2022. 

- Wu, Y., Chen, K., Zhang, T., Hui, Y., Nezhurina, M., Berg-Kirkpatrick, T., and Dubnov, S. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation, 2024. URL https://arxiv.org/abs/2211.06687. 

- Xiao, C., Chan, H. P., Zhang, H., Xu, W., Aljunied, M., and Rong, Y. Scaling language-centric omnimodal representation learning. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_ , 2025a. 

- Xiao, C., Chung, I., Kerboua, I., Stirling, J., Zhang, X., Kardos, M., Solomatin, R., Al Moubayed, N., Enevoldsen, K., and Muennighoff, N. Mieb: Massive image embedding benchmark. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 22187–22198, 2025b. 

- Xie, H., Räsänen, O., and Virtanen, T. Zero-shot audio classification with factored linear and nonlinear acousticsemantic projections. In _ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 326–330. IEEE, 2021. 

- Xu, S., Dong, W., Guo, Z., Wu, X., and Xiong, D. Exploring multilingual concepts of human values in large language models: Is value alignment consistent, transferable and controllable across languages? In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2024_ , pp. 1771– 1793, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-emnlp.96. URL https://aclanthology. org/2024.findings-emnlp.96/. 

13

<!-- page: 14 -->

**MAEB: Massive Audio Embedding Benchmark** 

- Zen, H., Dang, V., Clark, R., Zhang, Y., Weiss, R. J., Jia, Y., Chen, Z., and Wu, Y. Libritts: A corpus derived from librispeech for text-to-speech, 2019. URL https: //arxiv.org/abs/1904.02882. 

- Zhu, H., Zhou, Y., Chen, H., Yu, J., Ma, Z., Gu, R., Luo, Y., Tan, W., and Chen, X. Muq: Self-supervised music representation learning with mel residual vector quantization, 2025. URL https://arxiv.org/abs/2501. 01108. 

- Zohar, J., Cãar, S., Jason, F., Yuxin, P., Hereman, N., and Adhish, T. Jakobovski/free-spoken-digit-dataset: V1.0.8, aug 2018. URL https://doi.org/10. 5281/zenodo.1342401. 

14

<!-- page: 15 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 3._ MAEB+ Audio-Only Tasks Overview. Tasks are grouped by type and show MAEB benchmark membership, dataset size, total audio duration, language coverage, domains, and main evaluation metric. * denotes values from huge datasets. 

|**Dataset**|**Citation**|**MAEB**|**N. Samples**|**Total Duration(s)**|**N. Langs**|**Domains**|**Main Metric**|
|---|---|---|---|---|---|---|---|
|_Any2AnyRetrieval_||||||||
|JamAltArtistA2ARetrieval|(Cífka et al.,2024)|✓|6.7k|22992|4|Music|ndcg_at_10|
|_Classifcation_||||||||
|AmbientAcousticContext|(Park et al.,2020)||1k|1046|1|Spoken, Speech|accuracy|
|BeijingOpera|(Tian et al.,2014)|✓|236|393|1|Music|accuracy|
|BirdCLEF|(Klinck et al.,2025)|✓|1k|33602|1|Spoken, Speech, Bioacoustics|accuracy|
|CREMA_D|(Cao et al.,2014)|✓|7.4k|18924|1|Emotion|accuracy|
|CommonLanguageAgeDetection|(Sinisetty et al.,2021)|✓|2k|8685|1|Spoken, Scene, Speech|accuracy|
|CommonLanguageGenderDetection|(Sinisetty et al.,2021)||2k|8777|1|Spoken, Scene, Speech|accuracy|
|CommonLanguageLanguageDetection|(Sinisetty et al.,2021)||2k|8637|1|Spoken, Scene, Speech|accuracy|
|ESC50|(Piczak,2015)||2k|10000|1|Spoken|accuracy|
|FSDD|(Zohar et al.,2018)||300|129|1|Music|accuracy|
|GTZANGenre|(Tzanetakis & Cook,2002)|✓|1k|30024|1|Music|accuracy|
|GunshotTriangulation|(Raponi et al.,2021)||88|132|1||accuracy|
|IEMOCAPEmotion|(Busso et al.,2008)||10k|44775|1|Spoken, Emotion|accuracy|
|IEMOCAPGender|(Busso et al.,2008)|✓|10k|44775|1|Spoken, Speech|accuracy|
|LibriCount|(Stoter et al.,2018)||5.7k|28600|1|Speech|accuracy|
|MInDS14|(Gerz et al.,2021)|✓|7k|78225|12|Speech, Spoken|accuracy|
|MridinghamStroke|(Anantapadmanabhan et al.,2013)||7k|2462|1|Music|accuracy|
|MridinghamTonic|(Anantapadmanabhan et al.,2013)|✓|7k|2462|1|Music|accuracy|
|NSynth|(Engel et al.,2017)||3k|12008|1|Music|accuracy|
|SpeechCommands|(Warden,2018)||4.9k|4890|1|Speech|accuracy|
|SpokeNEnglish|(Groh et al.,2024)||3.2k|2829|1|Spoken|accuracy|
|SpokenQAForIC|(Shon et al.,2023)||6.1k|12967|1|Spoken|accuracy|
|TUTAcousticScenes|(Mesaros et al.,2018)||2k|20000|1|AudioScene|accuracy|
|UrbanSound8k|(Salamon et al.,2014)||8.7k|31501|1|AudioScene|accuracy|
|VocalSound|(Gong et al.,2022)||3.6k|14934|1|Spoken|accuracy|
|VoxCelebSA|(Shon et al.,2022)|✓|3.4k|27337|1|Spoken|accuracy|
|VoxLingua107_Top10|(Valk & Alumäe,2020)||972|9634|1|Speech|accuracy|
|VoxPopuliAccentID|(Wang et al.,2021a)||2k|22381|1|Spoken, Speech|accuracy|
|VoxPopuliGenderID|(Wang et al.,2021a)||500|5122|5|Spoken, Speech|accuracy|
|VoxPopuliLanguageID|(Wanget al.,2021a)|✓|500|5122|5|Spoken,Speech|accuracy|
|_Clustering_||||||||
|AmbientAcousticContextClustering|(Park et al.,2020)||1k|1046|1|Spoken, Speech|v_measure|
|CREMA_DClustering|(Cao et al.,2014)|✓|2k|5246|1|Speech|v_measure|
|ESC50Clustering|(Piczak,2015)||2k|10000|1|Spoken, Speech|v_measure|
|GTZANGenreClustering|(Tzanetakis & Cook,2002)||1k|30024|1|Music|v_measure|
|MusicGenreClustering|(Homburg et al.,2005)||1.9k|18965|1|Music|v_measure|
|VehicleSoundClustering|(Bazilinskyy et al.,2018)|✓|1.7k|6819|1|Scene|v_measure|
|VoiceGenderClustering|(Chung et al.,2018)||2k|14559|1|Spoken|v_measure|
|VoxCelebClustering|(Shon et al.,2022)||2k|16124|1|Spoken, Speech|v_measure|
|VoxPopuliAccentClustering|(Wang et al.,2021a)||2k|23097|1|Spoken, Speech|v_measure|
|VoxPopuliGenderClustering|(Wanget al.,2021a)|✓|500|5122|5|Spoken,Speech|v_measure|
|_MultilabelClassifcation_||||||||
|AudioSet|(Gemmeke et al.,2017)||*|*|1|Web, Music, Speech...|lrap|
|AudioSetMini|(Gemmeke et al.,2017)||2.2k|21316|1|Web, Music, Speech...|lrap|
|BirdSet|(Rauch et al.,2024)||*|*|1|Spoken, Speech, Bioacoustics|accuracy|
|FSD2019Kaggle|(Fonseca et al.,2021)|✓|9k|92834|1|Web|accuracy|
|FSD50K|(Fonseca et al.,2021)||2k|21157|1|Web|accuracy|
|SIBFLEURS|(Schmidt et al.,2025)|✓|11.4k|152396|101|Encyclopaedic|accuracy|
|_PairClassifcation_||||||||
|CREMADPairClassifcation|(Cao et al.,2014)|✓|7.4k|37858|1|Spoken|max_ap|
|ESC50PairClassifcation|(Piczak,2015)||2k|20000|1|Encyclopaedic|max_ap|
|NMSQAPairClassifcation|(Lin et al.,2022)|✓|171|3245|1|Spoken|max_ap|
|VocalSoundPairClassifcation|(Gong et al.,2022)||720|6010|1|Spoken|max_ap|
|VoxPopuliAccentPairClassifcation|(Wanget al.,2021a)|✓|7.4k|169638|1|Spoken|max_ap|
|_Reranking_||||||||
|ESC50AudioReranking|(Piczak,2015)||4.4k|22000|1|AudioScene|map_at_1000|
|FSDnoisy18kAudioReranking|(Fonseca et al.,2019)||4.2k|21924|1|AudioScene|map_at_1000|
|GTZANAudioReranking|(Tzanetakis & Cook,2002)|✓|1.4k|42033|1|Music|map_at_1000|
|UrbanSound8KAudioReranking|(Salamon et al.,2014)||5.2k|17904|1|Spoken|map_at_1000|
|VocalSoundAudioReranking|(Gong et al.,2022)||4.2k|17371|1|Spoken|map_at_1000|



15

<!-- page: 16 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 4._ MAEB+ Audio-Text Cross-Modal Tasks Overview. Tasks include zero-shot classification and bidirectional retrieval between audio and text modalities, with dataset size, total audio duration, and main evaluation metric. * denotes values from huge datasets. 

|**Dataset**|**Citation**|**MAEB**|**N. Samples**|**Total Secs**|**N. Langs**|**Modality**|**Domains**|**Main Metric**|
|---|---|---|---|---|---|---|---|---|
|_Audio-to-Text Retrieval_|||||||||
|AudioCapsA2TRetrieval|(Kim et al.,2019)||5.3k|8708|2|a2t|Encyclopaedic, Written|cv_recall_at_5|
|AudioSetStrongA2TRetrieval|(Hershey et al.,2021)||1k|5065|1|a2t|AudioScene|cv_recall_at_5|
|CMUArcticA2TRetrieval|(Clark & Richmond,2003)||2.6k|4134|1|a2t|Spoken|cv_recall_at_5|
|ClothoA2TRetrieval|(Drossos et al.,2019)||6.6k|23636|1|a2t|Encyclopaedic, Written|cv_recall_at_5|
|CommonVoiceMini17A2TRetrieval|(Ardila et al.,2020)||46.8k|120220|50|a2t|Spoken|cv_recall_at_5|
|CommonVoiceMini21A2TRetrieval|(Ardila et al.,2020)||58.5k|149040|114|a2t|Spoken|cv_recall_at_5|
|EmoVDBA2TRetrieval|(Adigwe et al.,2018)||2.9k|7231|1|a2t|Spoken|cv_recall_at_5|
|FleursA2TRetrieval|(Conneau et al.,2023)||155620|1018098|102|a2t|Spoken|cv_recall_at_5|
|GigaSpeechA2TRetrieval|(Chen et al.,2021)||13.5k|44982|1|a2t|Spoken|cv_recall_at_5|
|GoogleSVQA2TRetrieval|(Allauzen et al.,2025)||342.9k|879901|20|a2t|Spoken|cv_recall_at_5|
|HiFiTTSA2TRetrieval|(Bakhturina et al.,2021)||600|1280|1|a2t|Spoken|cv_recall_at_5|
|JLCorpusA2TRetrieval|(James et al.,2018)||2.5k|5083|1|a2t|Spoken|cv_recall_at_5|
|JamAltLyricA2TRetrieval|(Cífka et al.,2024)|✓|6.7k|11496|4|a2t|Music|ndcg_at_10|
|LibriTTSA2TRetrieval|(Zen et al.,2019)||9.4k|30433|1|a2t|Spoken|cv_recall_at_5|
|MACSA2TRetrieval|(Martin-Morato & Mesaros,2021)||786|3930|1|a2t|AudioScene|cv_recall_at_5|
|MusicCapsA2TRetrieval|(Agostinelli et al.,2023)||8.6k|42844|1|a2t|Music|cv_recall_at_5|
|SoundDescsA2TRetrieval|(Koepke et al.,2022)||*|*|1|a2t|Encyclopaedic, Written|cv_recall_at_5|
|UrbanSound8KA2TRetrieval|(Salamon et al.,2014)||10.2k|18334|1|a2t|AudioScene|cv_recall_at_5|
|_Text-to-Audio Retrieval_|||||||||
|AudioCapsT2ARetrieval|(Kim et al.,2019)||5.3k|8708|2|t2a|Encyclopaedic, Written|cv_recall_at_5|
|AudioSetStrongT2ARetrieval|(Hershey et al.,2021)||1k|5065|1|t2a|AudioScene|cv_recall_at_5|
|CMUArcticT2ARetrieval|(Clark & Richmond,2003)||2.6k|4134|1|t2a|Spoken|cv_recall_at_5|
|ClothoT2ARetrieval|(Drossos et al.,2019)|✓|6.6k|23636|1|t2a|Encyclopaedic, Written|cv_recall_at_5|
|CommonVoiceMini17T2ARetrieval|(Ardila et al.,2020)||46.8k|120220|50|t2a|Spoken|cv_recall_at_5|
|CommonVoiceMini21T2ARetrieval|(Ardila et al.,2020)|✓|58.5k|149040|114|t2a|Spoken|cv_recall_at_5|
|EmoVDBT2ARetrieval|(Adigwe et al.,2018)||2.9k|7231|1|t2a|Spoken|cv_recall_at_5|
|FleursT2ARetrieval|(Conneau et al.,2023)|✓|155620|1018098|102|t2a|Spoken|cv_recall_at_5|
|GigaSpeechT2ARetrieval|(Chen et al.,2021)|✓|13.5k|44982|1|t2a|Spoken|cv_recall_at_5|
|GoogleSVQT2ARetrieval|(Allauzen et al.,2025)||342.9k|879901|20|t2a|Spoken|cv_recall_at_5|
|HiFiTTST2ARetrieval|(Bakhturina et al.,2021)||600|1280|1|t2a|Spoken|cv_recall_at_5|
|JLCorpusT2ARetrieval|(James et al.,2018)||2.5k|5083|1|t2a|Spoken|cv_recall_at_5|
|JamAltLyricT2ARetrieval|(Cífka et al.,2024)||6.7k|11496|4|t2a|Music|ndcg_at_10|
|LibriTTST2ARetrieval|(Zen et al.,2019)||9.4k|30433|1|t2a|Spoken|cv_recall_at_5|
|MACST2ARetrieval|(Martin-Morato & Mesaros,2021)|✓|786|3930|1|t2a|AudioScene|cv_recall_at_5|
|MusicCapsT2ARetrieval|(Agostinelli et al.,2023)||8.6k|42844|1|t2a|Music|cv_recall_at_5|
|SoundDescsT2ARetrieval|(Koepke et al.,2022)||*|*|1|t2a|Encyclopaedic, Written|cv_recall_at_5|
|SpokenSQuADT2ARetrieval|(Li et al.,2018)|✓|600|3557|1|t2a|Academic, Encyclopaedic, Non-fction|cv_recall_at_5|
|UrbanSound8KT2ARetrieval|(Salamon et al.,2014)|✓|10.2k|18334|1|t2a|AudioScene|cv_recall_at_5|
|_Zero-shot Classifcation_|||||||||
|ESC50_Zeroshot|(Piczak,2015)||2k|10000|1|a2t|Spoken|accuracy|
|RavdessZeroshot|(Livingstone & Russo,2018)|✓|1.4k|5329|1|a2t|Spoken|accuracy|
|SpeechCommandsZeroshotv0.01|(Warden,2018)||2.6k|2567|1|a2t|Spoken|accuracy|
|SpeechCommandsZeroshotv0.02|(Warden,2018)|✓|4.1k|4074|1|a2t|Spoken|accuracy|
|UrbanSound8kZeroshot|(Salamon et al.,2014)||2k|7378|1|a2t|AudioScene|accuracy|



16

<!-- page: 17 -->

**==> picture [183 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
MAEB: Massive Audio Embedding Benchmark<br>**----- End of picture text -----**<br>


**==> picture [487 x 154] intentionally omitted <==**

**----- Start of picture text -----**<br>
MAEB+ MAEB MAEB(audio-only)<br>(98 tasks) (30 tasks) (19 tasks)<br>Domains<br>Spoken<br>Speech<br>Music<br>Encyclopaedic<br>AudioScene<br>Scene<br>Written<br>Web<br>Bioacoustics<br>Academic<br>Non-fiction<br>None<br>**----- End of picture text -----**<br>


_Figure 5._ Domain distributions in the MAEB+ collection, MAEB, and MAEB(audio-only). 

## **A. Tasks overview** 

This appendix provides detailed information on all tasks within MAEB, including size, language, metrics, and other relevant details in Table 3 and Table 4. The domain distribution of MAEB is shown in Figure 5. 

## **B. Overview of Models** 

All models used in the evaluations are listed in Table 5. 

## **B.1. Audio Encoders** 

**Transformer-based Models:** AST (Audio Spectrogram Transformer) (Gong et al., 2021) applies vision transformer architecture to mel-spectrograms. For retrieval evaluation, we extract the pooler output embedding (768-dim), which corresponds to the [CLS] token representation that captures global audio characteristics. 

**Self-supervised Speech Models:** Wav2Vec2 (Baevski et al., 2020) learns contextualized speech representations through masked prediction on quantized latent speech units. We evaluate ten variants ranging from base (95M) to XLS-R 2B (2B parameters), extracting embeddings from the final transformer layer with mean pooling across the temporal dimension. The XLS-R variants (Babu et al., 2021) extend this to 128 languages through multilingual pre-training on 436k hours of speech. 

WavLM (Chen et al., 2022a) enhances Wav2Vec2 with masked speech prediction and denoising objectives, showing particular strength on noisy audio. We evaluate seven specialized variants: base models, speaker verification (SV), speaker diarization (SD), and combinations thereof. The denoising pre-training makes WavLM particularly robust for retrieval tasks involving real-world audio conditions. 

HuBERT (Hsu et al., 2021) learns discrete speech units through iterative k-means clustering and masked prediction. We evaluate base (95M) and large fine-tuned (317M) variants, using the final layer representations which capture both acoustic and linguistic information through the learned discrete units. 

Data2Vec (Baevski et al., 2022) provides a unified self-supervised framework using the same learning objective across modalities. For audio, we extract contextualized embeddings from the transformer encoder with mean pooling, leveraging representations that benefit from cross-modal learning insights. 

SEW-D (Wu et al., 2021) offers performance-efficiency trade-offs through squeezed and efficient transformer architectures. We evaluate three variants (tiny: 20M, mid: 139M, base: 95M parameters), extracting embeddings from the final hidden layer with mean pooling. 

UniSpeech (Wang et al., 2021b) combines self-supervised pre-training with multi-task fine-tuning for universal speech representations. 

MCTCT (Lugosch et al., 2022) supports 60 languages through multilingual connectionist temporal classification, using pseudo-labeling for low-resource language adaptation. We extract embeddings from the final hidden states with mean 

17

<!-- page: 18 -->

**MAEB: Massive Audio Embedding Benchmark** 

## pooling. 

**CNN-based Models:** CNN14 (Kong et al., 2020) employs a 14-layer CNN with global average pooling, trained on AudioSet’s 2M audio clips. We extract 2048-dimensional embeddings from the penultimate layer before classification. YAMNet (Gemmeke et al., 2017) uses MobileNet architecture optimized for mobile deployment, providing 1024-dimensional features from efficient depthwise separable convolutions. VGGish (Hershey et al., 2017) adapts VGG for audio through mel-spectrogram processing, yielding compact 128-dimensional embeddings. **Neural Codec Models:** Encodec (Défossez et al., 2022) provides neural audio compression through residual vector quantization. For retrieval evaluation, we extract continuous embeddings from the encoder before quantization (128-dim), applying mean pooling over the temporal dimension. 

## **B.2. Sequence-to-Sequence Models** 

Whisper (Radford et al., 2022) provides robust multilingual speech recognition across 99 languages. For retrieval, we extract embeddings from the encoder at the final layer, using mean pooling across the sequence dimension. We evaluate five model sizes (tiny: 39M to large-v3: 1.55B parameters). 

MMS (Pratap et al., 2023) supports over 1,000 languages through massive multilingual pre-training. We evaluate three variants (1B-all, 1B-fl102, 1B-l1107) differing in language coverage, using the Wav2Vec2-style encoder with languagespecific adapter loading when available. 

SeamlessM4T (Communication et al., 2023) provides unified speech-text translation across 100+ languages. For retrieval, we extract embeddings from the speech encoder component before translation processing, capturing multilingual audio semantics. 

SpeechT5 ASR (Ao et al., 2022) provides speech recognition through unified encoder-decoder architecture (152M parameters). We extract embeddings from the encoder representations. 

## **B.3. Contrastive Alignment Models** 

CLAP (Wu et al., 2024) learns joint audio-text representations through contrastive learning on 633k audio-text pairs. We evaluate five LAION variants: htsat-fused/unfused (153M parameters) and larger variants (193M) specialized for general audio, music, and combined music-speech. The key implementation detail is using the audio encoder branch with L2 normalization. 

MS-CLAP (Elizalde et al., 2023) (2022: 196M, 2023: 160M parameters) uses different architectures and training data, providing complementary audio-text alignment capabilities. 

Wav2CLIP (Wu et al., 2022) bridges audio and vision by learning audio representations that align with CLIP’s visual embedding space. For retrieval, we extract features from the audio encoder (11.7M parameters) while text encoding uses the standard CLIP text encoder (151M parameters). 

MuQ-MuLan (Zhu et al., 2025) specializes in joint music-text understanding through contrastive learning on music data. We extract 512-dimensional embeddings from the audio encoder branch. 

SpeechT5 Multimodal (Ao et al., 2022) provides unified speech-text modeling through shared encoder-decoder architecture (298M parameters). We extract embeddings from the shared encoder representations. 

## **B.4. Large Audio-Language Models** 

Qwen2-Audio (Chu et al., 2024) integrates audio understanding into large language models (7B parameters). We extract embeddings from the final hidden layer using last-token pooling, selecting the embedding at the last non-padding position for each sample. 

LCO-Embedding (Xiao et al., 2025a) provides language-centric omnimodal representations through contrastive learning on multimodal data. We evaluate two variants (3B: 4.7B parameters, 7B: 8.9B parameters), extracting embeddings from the final hidden layer using last-token pooling. 

18

<!-- page: 19 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 5._ List of all models evaluated in MAEB. Model sizes are in millions of parameters. 

|**Model Name**|**Model Size**|**Modalities**|
|---|---|---|
|laion/clap-htsat-fused(Wu et al.,2024)|153|audio, text|
|laion/clap-htsat-unfused(Wu et al.,2024)|153|audio, text|
|laion/larger_clap_general(Wu et al.,2024)|193|audio, text|
|laion/larger_clap_music(Wu et al.,2024)|193|audio, text|
|laion/larger_clap_music_and_speech(Wu et al.,2024)|193|audio, text|
|MIT/ast-fnetuned-audioset-10-10-0.4593(Gong et al.,2021)|86|audio|
|speechbrain/cnn14-esc50(Wang et al.,2022)|80|audio|
|facebook/data2vec-audio-base-960h(Baevski et al.,2022)|93|audio|
|facebook/data2vec-audio-large-960h(Baevski et al.,2022)|313|audio|
|facebook/encodec_24khz(Défossez et al.,2022)|23|audio|
|facebook/hubert-base-ls960(Hsu et al.,2021)|95|audio|
|facebook/hubert-large-ls960-ft(Hsu et al.,2021)|317|audio|
|speechbrain/m-ctc-t-large(Ravanelli et al.,2021)|1058|audio|
|facebook/mms-1b-all(Pratap et al.,2023)|1000|audio|
|facebook/mms-1b-f102(Pratap et al.,2023)|1000|audio|
|facebook/mms-1b-l1107(Pratap et al.,2023)|1000|audio|
|microsoft/msclap-2022(Elizalde et al.,2023)|196|audio, text|
|microsoft/msclap-2023(Elizalde et al.,2023)|160|audio, text|
|OpenMuQ/MuQ-MuLan-large(Zhu et al.,2025)|630|audio, text|
|Qwen/Qwen2-Audio-7B(Chu et al.,2024)|7000|audio, text|
|LCO-Embedding/LCO-Embedding-Omni-3B(Xiao et al.,2025a)|4703|audio, text|
|LCO-Embedding/LCO-Embedding-Omni-7B(Xiao et al.,2025a)|8932|audio, text|
|facebook/seamless-m4t-v2-large(Communication et al.,2023)|2300|audio|
|asapp/sew-d-base-plus-400k-ft-ls100h(Wu et al.,2021)|95|audio|
|asapp/sew-d-tiny-100k-ft-ls100h(Wu et al.,2021)|19|audio|
|asapp/sew-d-mid-400k-ft-ls100h(Wu et al.,2021)|139|audio|
|microsoft/speecht5_asr(Ao et al.,2022)|151|audio|
|microsoft/speecht5_tts(Ao et al.,2022)|146|text|
|microsoft/speecht5_multimodal(Ao et al.,2022)|297|audio, text|
|microsoft/unispeech-sat-base-100h-libri-ft(Chen et al.,2022b)|94|audio|
|google/vggish(Hershey et al.,2017)|72|audio|
|lyrebird/wav2clip(Wu et al.,2022)|163|audio, text|
|facebook/wav2vec2-xls-r-300m(Babu et al.,2021)|300|audio|
|vitouphy/wav2vec2-xls-r-300m-phoneme(Babu et al.,2021)|300|audio|
|facebook/wav2vec2-xls-r-1b(Babu et al.,2021)|1000|audio|
|facebook/wav2vec2-xls-r-2b(Babu et al.,2021)|2000|audio|
|facebook/wav2vec2-xls-r-2b-21-to-en(Babu et al.,2021)|2000|audio|
|facebook/wav2vec2-base(Baevski et al.,2020)|95|audio|
|facebook/wav2vec2-base-960h(Baevski et al.,2020)|95|audio|
|facebook/wav2vec2-large(Baevski et al.,2020)|317|audio|
|facebook/wav2vec2-large-xlsr-53(Conneau et al.,2020)|317|audio|
|facebook/wav2vec2-lv-60-espeak-cv-ft(Baevski et al.,2020)|317|audio|
|microsoft/wavlm-base(Chen et al.,2022a)|94|audio|
|microsoft/wavlm-base-sd(Chen et al.,2022a)|94|audio|
|microsoft/wavlm-base-plus(Chen et al.,2022a)|94|audio|
|microsoft/wavlm-base-plus-sv(Chen et al.,2022a)|94|audio|
|microsoft/wavlm-base-plus-sd(Chen et al.,2022a)|94|audio|
|microsoft/wavlm-base-sv(Chen et al.,2022a)|94|audio|
|microsoft/wavlm-large(Chen et al.,2022a)|316|audio|
|openai/whisper-tiny(Radford et al.,2022)|39|audio|
|openai/whisper-base(Radford et al.,2022)|74|audio|
|openai/whisper-small(Radford et al.,2022)|244|audio|
|openai/whisper-medium(Radford et al.,2022)|769|audio|
|openai/whisper-large-v3(Radford et al.,2022)|1550|audio|
|google/yamnet(Gemmeke et al.,2017)|3|audio|



19

<!-- page: 20 -->

**MAEB: Massive Audio Embedding Benchmark** 

## **C. Correlation Analysis Tasks** 

For the correlation analysis presented in Figure 3, we utilized the following subset of 26 classification tasks from MAEB+, grouped by domain to align with the MMAU benchmark: 

- **Speech (13 tasks):** SpeechCommands, FSDD, CommonLanguage (Age, Gender, Language), VoxPopuli (Accent, Gender, Language), VoxLingua107, LibriCount, VocalSound, VoxCelebSA, SpokeNEnglish. 

- **Music (5 tasks):** GTZAN Genre, Beijing Opera, Mridingham (Stroke, Tonic), NSynth. 

- **Sound (8 tasks):** ESC50, UrbanSound8k, TUT Acoustic Scenes, Ambient Acoustic Context, Gunshot Triangulation, AudioSet Mini, FSD50K, FSD2019 Kaggle. 

## **D. Domain Radar Chart Methodology** 

The domain radar chart (Figure 2) visualizes model performance across five core acoustic domains. 94 tasks from MAEB+ are assigned to domains based on their primary audio content and intended application. 

**Score Computation** For each model and domain, we compute the arithmetic mean of the main scores across all tasks assigned to that domain. All metrics (e.g., Accuracy, v_measure, nDCG, AP), which are natively in the [0, 1] range, are aggregated on a shared 0–100 scale for consistent visualization. This aggregation ensures that different task types contribute equally to the domain average. 

**Full Task Breakdown per Domain** Below we list all 94 tasks contributing to the domain scores, categorized by their acoustic content: 

- **Speech** (44 tasks): SpeechCommands, FSDD, CommonLanguageAgeDetection, CommonLanguageGenderDetection, CommonLanguageLanguageDetection, VoxPopuliAccentID, VoxPopuliGenderID, VoxPopuliLanguageID, VoxLingua107_Top10, LibriCount, VocalSound, VoxCelebSA, SpokeNEnglish, SpokenQAForIC, MInDS14, IEMOCAPGender, VoiceGenderClustering, VoxCelebClustering, VoxPopuliAccentClustering, VoxPopuliGenderClustering, VocalSoundPairClassification, VoxPopuliAccentPairClassification, VocalSoundAudioReranking, CMUArcticA2TRetrieval, CMUArcticT2ARetrieval, EmoVDBA2TRetrieval, EmoVDBT2ARetrieval, GigaSpeechA2TRetrieval, GigaSpeechT2ARetrieval, HiFiTTSA2TRetrieval, HiFiTTST2ARetrieval, JLCorpusA2TRetrieval, JLCorpusT2ARetrieval, LibriTTSA2TRetrieval, LibriTTST2ARetrieval, CommonVoiceMini17A2TRetrieval, CommonVoiceMini17T2ARetrieval, CommonVoiceMini21A2TRetrieval, CommonVoiceMini21T2ARetrieval, FleursA2TRetrieval, FleursT2ARetrieval, SpokenSQuADT2ARetrieval, SpeechCommandsZeroshotv0.01, and SpeechCommandsZeroshotv0.02. 

- **Music** (13 tasks): GTZANGenre, BeijingOpera, MridinghamStroke, MridinghamTonic, NSynth, GTZANGenreClustering, MusicGenreClustering, GTZANAudioReranking, JamAltArtistA2ARetrieval, JamAltLyricA2T, JamAltLyricT2A, MusicCapsA2TRetrieval, and MusicCapsT2ARetrieval. 

- **Environmental** (29 tasks): ESC50, UrbanSound8k, TUTAcousticScenes, AmbientAcousticContext, GunshotTriangulation, AudioSetMini, FSD50K, FSD2019Kaggle, ESC50Clustering, AmbientAcousticContextClustering, VehicleSoundClustering, ESC50PairClassification, ESC50AudioReranking, UrbanSound8KAudioReranking, FSDnoisy18kAudioReranking, AudioCapsA2T, AudioCapsT2A, AudioSetStrongA2T, AudioSetStrongT2A, ClothoA2T, ClothoT2A, MACSA2T, MACST2A, SoundDescsA2T, SoundDescsT2A, UrbanSound8KA2T, UrbanSound8KT2A, ESC50_Zeroshot, and UrbanSound8kZeroshot. 

- **Bioacoustics** (2 tasks): BirdCLEF and BirdSet. 

- **Emotion** (6 tasks): CREMA_D (Classification, Clustering, PairClassification), IEMOCAP Emotion, NMSQA PairClassification, and Ravdess Zeroshot. 

**Model Selection & Visualization** To maintain clarity, the chart displays only representative models that achieve the highest average score in at least one domain. This highlights both domain specialists and generalists. 

20

<!-- page: 21 -->

**MAEB: Massive Audio Embedding Benchmark** 

**Missing Results and Task Averaging** Domain-averaged scores are computed using the arithmetic mean of all tasks within a domain for which results are available. If a model cannot perform a specific task type (e.g., an audio-only encoder evaluated on text-to-audio retrieval), those tasks are omitted from the average rather than being treated as a zero-score. This approach ensures the radar chart reflects the performance quality of a model’s existing capabilities within a domain. 

## **E. Per Task Category Results** 

## **E.1. Zero-Shot Classification** 

Table 22 presents results of zero-shot classification tasks. LCO models (LCO-Embedding-Omni-7B) achieve the highest overall zero-shot performance (76.2%), significantly outperforming other models. Specifically, LCO models excel on speech commands (SpeechCmd v0.01, v0.02) with near-perfect scores ( _>_ 96%) and show strong performance on emotional speech (Ravdess). CLAP models (larger_clap_general, msclap-2023) excel on environmental sound tasks (ESC50), with larger_clap_general achieving the top score (90.5%), demonstrating the effectiveness of contrastive audio-text pretraining for open-vocabulary environmental sound classification. However, CLAP models generally underperform LCO models on speech-specific tasks. Msclap-2023 achieves the strongest performance on UrbanSound8k (83.0%). Overall, while CLAP models are robust for environmental sounds, LCO-Embedding models demonstrate superior generalization across the broader diverse set of zero-shot tasks, particularly in the speech domain. 

## **E.2. Linear Probe Classification** 

Table 6, Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13, Table 14, Table 15, Table 16, and Table 17 present results of classification tasks. 

As shown in Tables, Qwen2-Audio-7B achieves the highest classification average (61.7%), surpassing the previously reported baselines. The audio-LLM model (Qwen2-Audio-7B) achieves top performance on a wide range of tasks including emotion recognition (CREMA-D, IEMOCAPEmotion), music tasks (BeijingOpera, GTZANGenre, MridinghamStroke, MridinghamTonic, NSynth), and vocal sound classification (VocalSound). LCO-Embedding models also demonstrate exceptional performance, particularly dominating language and speaker tasks such as MInDS14 ( _>_ 98%) and VoxCelebSA, where they outperform ASR-based models. Only on specific environmental tasks does the AudioSet-finetuned model (ast-finetuned-audioset-10-10-0.4593) retain dominance (AmbientAcousticContext, BirdCLEF). Whisper models (whispermedium) perform well on accent classification (VoxPopuliAccent) but are generally outperformed by Audio-LLMs and LCO models on broader semantic classification benchmarks. 

## **E.3. Multilabel Classification** 

Table 21 presents results of multilabel classification tasks. The LCO-Embedding model (LCO-Embedding-Omni-7B) achieves top performance on FSD2019Kaggle, while Qwen2-Audio-7B leads on FSD50K, leveraging its broad semantic understanding for complex multi-tag scenarios. This contrasts with earlier findings where AudioSet-finetuned models were dominant; here, large-scale trained multimodal models show superior capability in handling diverse acoustic tagging tasks. 

## **E.4. Clustering** 

Table 18 and Table 19 present results of clustering tasks. The CLAP variant larger_clap_music_and_speech achieves the highest clustering average (35.3%), closely followed by clap-htsat-unfused (35.0%). These models excel because their contrastive objectives naturally structure the embedding space to group semantically similar audio clips, which is ideal for clustering. ASR encoders and Audio-LLMs generally trail behind contrastive models in this category, as their representations are either too phonetically granular (ASR) or generation-oriented (LLM) rather than density-optimized for unsupervised grouping. 

## **E.5. Pair Classification** 

Table 20 presents results of pair classification tasks. LCO-Embedding-Omni-7B achieves the highest pair classification score (79.2%), significantly outperforming whisper-medium (59.9%). This dominance suggests that LCO models capture highly discriminative features suitable for determining verification and similarity across diverse audio pairs. While CLAP models show competence in environmental sound pairs, the LCO model’s robust performance across speech and mixed 

21

<!-- page: 22 -->

**MAEB: Massive Audio Embedding Benchmark** 

domains drives its superior average. 

## **E.6. Retrieval** 

Table 24, Table 25, Table 27, Table 28, Table 29, Table 31, Table 32, Table 33, Table 35, Table 35, and Table 36 present results of retrieval tasks. Results indicate a strong split by domain. LCO-Embedding models achieve near-perfect performance on speech-text retrieval tasks (CMU Arctic, EmoVDB, HiFiTTS, LibriTTS), likely due to extensive speech-text alignment during training. In contrast, CLAP models (larger_clap_general) remain superior for environmental sound retrieval (AudioCaps, AudioSetStrong, Clotho), where their specific training on general audio-text pairs provides an advantage. UrbanSound8K retrieval is an exception where LCO models outperform CLAP substantially. Overall, LCO models dominate the speech retrieval landscape, while CLAP retains the edge in general acoustic event retrieval. 

## **E.7. Reranking** 

Table 23 presents results of reranking tasks. LCO-Embedding-Omni-7B achieves the highest average performance (86.0%), demonstrating exceptional capability in distinguishing relevant from non-relevant audio candidates. It tops not only vocal tasks (VocalSound, UrbanSound8K) but also proves highly effective generally. Microsoft’s msclap-2023 is the top performer on specific environmental reranking tasks like ESC50AudioReranking and FSDnoisy18kAudioReranking. The results highlight that while specialized models like MSCLAP are powerful for specific acoustic domains, recent multimodal embeddings like LCO provide a more versatile and high-performance solution across diverse reranking challenges. 

22

<!-- page: 23 -->

_Table 6._ English classification results (datasets 1–8 of 23). 

|Model|AmbientAcoustic|BeijingOpera|BirdCLEF|CommonLangAge|CommonLangGender|CREMA-D|ESC50|FSDD|
|---|---|---|---|---|---|---|---|---|
|Qwen/Qwen2-Audio-7B|45.33|**97.45**|37.10|17.59|48.83|**73.99**|96.30|90.33|
|LCO-Embedding/LCO-Embedding-Omni-7B|39.67|92.79|34.10|16.16|30.70|36.05|94.15|**99.00**|
|LCO-Embedding/LCO-Embedding-Omni-3B|38.42|93.22|31.60|16.87|33.22|31.03|94.40|98.27|
|openai/whisper-medium|39.85|91.54|29.00|17.63|36.63|53.98|84.00|87.73|
|openai/whisper-small|36.89|88.13|26.40|15.85|45.87|49.19|77.35|91.47|
|MIT/ast-fnetuned-audioset-10-10-0.4593|**48.86**|97.03|**45.20**|12.82|52.33|37.84|96.20|64.27|
|facebook/wav2vec2-xls-r-2b|32.88|83.06|31.20|17.36|39.80|45.94|73.45|95.53|
|openai/whisper-base|32.05|89.39|27.50|18.18|46.23|48.05|72.35|82.67|
|microsoft/msclap-2023|46.07|91.11|17.30|15.35|62.20|37.06|97.40|56.37|
|laion/clap-htsat-unfused|46.72|91.96|16.40|15.98|**73.26**|37.56|97.05|49.33|
|laion/clap-htsat-fused|42.78|92.77|18.00|15.27|70.90|38.75|96.50|45.67|
|openai/whisper-tiny|30.17|83.88|25.10|16.45|45.89|45.93|64.90|83.60|
|laion/larger_clap_general|47.80|93.62|17.00|**20.51**|43.71|39.83|**97.45**|40.20|
|openai/whisper-large-v3|35.31|86.01|21.60|19.28|33.13|48.94|71.65|77.00|
|microsoft/wavlm-large|29.61|57.68|19.80|16.86|40.58|38.93|64.20|91.47|
|laion/larger_clap_music_and_speech|47.18|91.52|16.40|16.59|43.87|40.18|97.20|43.53|
|facebook/mms-1b-l1107|24.71|78.88|21.00|18.48|34.05|29.21|55.25|95.13|
|facebook/wav2vec2-lv-60-espeak-cv-ft|25.25|87.32|20.50|15.43|40.34|34.95|52.00|88.40|
|facebook/hubert-base-ls960|25.87|65.76|18.30|15.00|46.67|40.49|59.40|93.73|
|facebook/mms-1b-f102|26.85|77.98|19.90|17.03|33.41|30.68|57.10|92.93|
|facebook/wav2vec2-xls-r-1b|31.29|70.74|16.80|17.98|33.12|41.95|64.45|82.80|
|facebook/seamless-m4t-v2-large|24.36|66.15|11.10|16.56|33.26|28.11|44.60|92.40|
|microsoft/wavlm-base-sv|23.17|65.71|10.20|15.37|41.45|40.11|50.40|96.40|
|microsoft/wavlm-base-sd|23.17|65.71|10.20|15.37|41.45|40.11|50.40|96.40|
|microsoft/wavlm-base|23.17|65.71|10.20|15.37|41.45|40.11|50.40|96.40|
|facebook/mms-1b-all|23.01|78.81|21.10|15.84|34.94|31.63|52.30|93.53|
|vitouphy/wav2vec2-xls-r-300m-phoneme|25.25|86.01|17.50|19.96|37.80|33.71|54.55|94.07|
|microsoft/wavlm-base-plus-sd|27.20|62.76|12.00|18.53|35.82|33.71|57.50|91.40|
|microsoft/wavlm-base-plus-sv|27.20|62.76|12.00|18.53|35.82|33.71|57.50|91.40|
|microsoft/wavlm-base-plus|27.20|62.76|12.00|18.53|35.82|33.71|57.50|91.40|
|microsoft/speecht5_multimodal|18.77|77.13|15.80|19.50|29.98|32.52|46.55|89.37|
|facebook/hubert-large-ls960-ft|22.76|58.55|14.90|19.37|32.63|31.67|46.20|98.00|
|facebook/wav2vec2-base|26.91|75.41|11.10|15.02|39.29|37.97|46.95|54.80|
|microsoft/msclap-2022|43.86|93.20|13.20|14.27|58.32|27.94|90.95|29.07|
|google/vggish|38.49|87.70|10.50|14.70|60.45|34.79|61.15|27.23|
|google/yamnet|40.46|89.39|16.80|17.18|40.12|25.87|79.70|34.50|
|microsoft/unispeech-sat-base-100h-libri-ft|21.72|58.49|9.20|15.89|38.43|33.43|41.95|91.87|
|asapp/sew-d-tiny-100k-ft-ls100h|15.14|56.79|9.00|19.25|36.75|30.03|39.55|85.87|
|lyrebird/wav2clip|34.61|88.10|9.70|13.11|39.84|44.45|72.10|21.37|
|facebook/data2vec-audio-base-960h|17.86|53.01|7.40|18.53|34.65|27.98|31.45|68.87|
|facebook/data2vec-audio-large-960h|15.12|70.81|9.50|17.18|32.86|26.19|29.60|63.33|
|facebook/wav2vec2-base-960h|16.68|49.14|7.10|16.37|35.83|29.47|27.00|82.87|
|facebook/wav2vec2-xls-r-300m|27.55|81.78|7.80|15.39|29.25|35.34|43.05|64.47|
|OpenMuQ/MuQ-MuLan-large|22.34|76.32|7.40|19.34|33.21|34.35|38.50|27.33|
|asapp/sew-d-mid-400k-ft-ls100h|14.03|43.72|5.50|16.27|39.79|31.46|25.45|69.53|
|speechbrain/m-ctc-t-large|12.15|43.24|12.30|17.35|35.91|26.77|17.05|59.50|
|facebook/wav2vec2-large|18.47|48.75|7.80|17.89|30.15|35.46|32.30|49.33|
|speechbrain/cnn14-esc50|18.13|83.86|11.20|14.52|40.27|34.53|63.50|17.20|
|asapp/sew-d-base-plus-400k-ft-ls100h|11.64|40.29|4.50|19.97|42.13|34.44|21.30|24.67|
|laion/larger_clap_music|4.83|62.23|3.70|17.62|47.31|30.87|9.75|10.00|
|facebook/encodec_24khz|13.05|42.35|1.80|19.04|31.87|29.83|11.50|24.27|
|facebook/wav2vec2-large-xlsr-53|9.29|54.65|2.30|15.69|36.55|28.33|6.70|18.93|

<!-- page: 24 -->

_Table 7._ English classification results (datasets 9–16 of 23). 

|Model|GTZANGenre|GunshotTri|IEMOCAPEmo|IEMOCAPGender|LibriCount|MridinghamStroke|MridinghamTonic|NSynth|
|---|---|---|---|---|---|---|---|---|
|Qwen/Qwen2-Audio-7B|**93.10**|**100.00**|**29.96**|92.96|49.60|**84.33**|**61.17**|**63.04**|
|LCO-Embedding/LCO-Embedding-Omni-7B|82.30|96.67|24.35|70.75|33.22|61.89|42.07|58.09|
|LCO-Embedding/LCO-Embedding-Omni-3B|81.00|96.60|22.53|59.42|30.35|61.52|39.54|59.02|
|openai/whisper-medium|76.00|94.25|25.60|76.05|**57.87**|69.21|49.51|51.33|
|openai/whisper-small|71.50|94.44|24.21|69.93|53.37|69.06|45.49|49.88|
|MIT/ast-fnetuned-audioset-10-10-0.4593|80.70|98.82|20.49|87.00|42.15|79.20|54.16|56.26|
|facebook/wav2vec2-xls-r-2b|73.20|94.31|23.86|70.27|50.24|70.59|44.62|45.02|
|openai/whisper-base|70.90|89.80|23.82|72.16|50.21|60.68|44.67|47.07|
|microsoft/msclap-2023|78.10|87.45|22.10|85.86|42.06|79.46|52.09|62.64|
|laion/clap-htsat-unfused|74.90|69.41|22.61|92.58|48.69|71.66|49.52|59.80|
|laion/clap-htsat-fused|67.90|70.59|21.17|**93.62**|47.81|74.09|47.00|60.23|
|openai/whisper-tiny|68.40|90.92|23.38|68.88|50.17|61.70|44.03|46.47|
|laion/larger_clap_general|84.50|86.27|23.08|89.28|48.50|71.05|49.45|59.31|
|openai/whisper-large-v3|71.90|81.90|22.18|60.25|57.22|54.84|37.54|46.04|
|microsoft/wavlm-large|67.70|**100.00**|20.25|63.05|52.26|60.87|31.42|47.14|
|laion/larger_clap_music_and_speech|83.50|77.25|24.07|93.12|47.99|70.99|48.99|58.64|
|facebook/mms-1b-l1107|58.30|88.56|16.85|54.69|41.21|61.02|30.70|44.29|
|facebook/wav2vec2-lv-60-espeak-cv-ft|55.20|87.45|19.02|68.31|45.30|66.60|37.29|42.62|
|facebook/hubert-base-ls960|69.00|98.89|20.59|76.19|48.92|55.41|31.58|46.36|
|facebook/mms-1b-f102|56.80|90.98|16.35|52.44|39.60|54.84|30.89|43.84|
|facebook/wav2vec2-xls-r-1b|66.10|91.90|23.30|61.82|49.79|65.17|39.85|46.65|
|facebook/seamless-m4t-v2-large|52.50|72.61|22.79|53.72|44.14|39.39|35.37|43.30|
|microsoft/wavlm-base-sv|63.00|95.49|21.20|67.32|50.73|46.68|29.68|43.10|
|microsoft/wavlm-base-sd|63.00|95.49|21.20|67.32|50.73|46.68|29.68|43.10|
|microsoft/wavlm-base|63.00|95.49|21.20|67.32|50.73|46.68|29.68|43.10|
|facebook/mms-1b-all|57.10|91.90|17.97|56.23|41.78|51.63|27.42|43.25|
|vitouphy/wav2vec2-xls-r-300m-phoneme|60.90|85.36|20.79|51.16|42.54|57.39|33.97|42.02|
|microsoft/wavlm-base-plus-sd|62.90|97.71|18.42|55.00|52.06|42.25|26.47|46.38|
|microsoft/wavlm-base-plus-sv|62.90|97.71|18.42|55.00|52.06|42.25|26.47|46.38|
|microsoft/wavlm-base-plus|62.90|97.71|18.42|55.00|52.06|42.25|26.47|46.38|
|microsoft/speecht5_multimodal|52.50|93.20|19.79|52.14|43.25|42.43|30.74|40.13|
|facebook/hubert-large-ls960-ft|50.60|93.14|17.84|54.25|44.27|47.93|24.22|40.26|
|facebook/wav2vec2-base|62.40|97.71|20.23|68.88|54.95|57.98|35.14|42.65|
|microsoft/msclap-2022|58.70|57.97|15.54|89.35|39.74|47.08|29.14|50.21|
|google/vggish|79.30|86.41|19.32|91.54|45.61|50.48|32.77|43.80|
|google/yamnet|79.30|80.46|14.84|76.91|41.33|56.01|35.16|45.81|
|microsoft/unispeech-sat-base-100h-libri-ft|51.60|91.05|18.17|60.45|45.35|39.79|25.34|42.50|
|asapp/sew-d-tiny-100k-ft-ls100h|51.10|89.80|17.01|52.57|43.43|30.49|23.65|39.91|
|lyrebird/wav2clip|59.10|76.27|16.33|65.83|34.65|46.52|40.28|46.04|
|facebook/data2vec-audio-base-960h|40.30|81.76|15.49|55.84|48.44|33.27|20.77|40.42|
|facebook/data2vec-audio-large-960h|43.60|70.52|15.58|54.59|42.48|28.48|21.21|38.22|
|facebook/wav2vec2-base-960h|42.70|79.41|15.11|51.82|46.15|36.43|24.32|38.89|
|facebook/wav2vec2-xls-r-300m|42.60|70.46|14.68|53.12|37.54|55.84|35.83|43.06|
|OpenMuQ/MuQ-MuLan-large|88.30|51.05|16.07|57.45|36.70|42.07|38.25|52.97|
|asapp/sew-d-mid-400k-ft-ls100h|43.40|76.14|16.00|56.26|43.13|29.28|19.38|36.68|
|speechbrain/m-ctc-t-large|39.20|57.78|16.17|50.96|34.86|21.43|24.27|40.72|
|facebook/wav2vec2-large|54.30|84.18|16.78|54.28|45.17|44.46|26.64|40.34|
|speechbrain/cnn14-esc50|41.40|83.20|18.09|59.37|28.37|30.59|26.06|39.83|
|asapp/sew-d-base-plus-400k-ft-ls100h|40.10|52.48|18.68|55.73|42.74|22.69|19.49|36.46|
|laion/larger_clap_music|25.80|60.13|11.58|65.75|20.49|38.86|18.39|38.27|
|facebook/encodec_24khz|29.90|46.60|10.57|53.46|21.24|18.35|24.04|37.30|
|facebook/wav2vec2-large-xlsr-53|22.10|33.07|16.06|50.10|25.14|24.65|19.13|32.78|

<!-- page: 25 -->

_Table 8._ English classification results (datasets 17–23 of 23). 

|Model|SpeechCommands|SpokenQA|TUTAcoustic|VocalSound|VoxCelebSA|VoxPopuliAccent|MInDS14|
|---|---|---|---|---|---|---|---|
|Qwen/Qwen2-Audio-7B|75.60|21.35|**34.30**|**91.82**|29.54|39.35|25.51|
|LCO-Embedding/LCO-Embedding-Omni-7B|94.21|36.58|25.35|91.77|43.40|10.33|98.14|
|LCO-Embedding/LCO-Embedding-Omni-3B|**94.40**|**37.00**|25.00|91.31|**48.97**|8.97|**98.48**|
|openai/whisper-medium|72.43|21.74|26.55|80.13|33.92|**54.04**|48.30|
|openai/whisper-small|73.80|19.96|23.55|77.59|32.45|38.85|35.64|
|MIT/ast-fnetuned-audioset-10-10-0.4593|24.27|15.86|30.45|82.11|28.33|23.61|7.94|
|facebook/wav2vec2-xls-r-2b|65.13|17.76|24.95|68.98|28.73|34.29|15.21|
|openai/whisper-base|62.18|17.92|25.10|72.11|31.26|28.47|29.56|
|microsoft/msclap-2023|29.77|14.52|28.20|78.41|26.18|17.69|7.43|
|laion/clap-htsat-unfused|23.33|14.46|30.95|81.53|33.72|18.00|9.29|
|laion/clap-htsat-fused|24.86|14.30|30.15|80.88|29.29|17.59|10.30|
|openai/whisper-tiny|60.03|18.36|25.45|68.14|29.66|27.52|31.25|
|laion/larger_clap_general|10.50|15.06|30.75|74.46|30.47|23.86|8.11|
|openai/whisper-large-v3|65.17|19.21|22.65|76.76|31.46|28.87|31.92|
|microsoft/wavlm-large|83.46|20.68|24.15|66.68|33.44|49.82|20.77|
|laion/larger_clap_music_and_speech|10.51|15.68|29.85|73.64|33.78|24.06|7.77|
|facebook/mms-1b-l1107|83.85|27.17|20.55|66.83|30.62|44.36|64.02|
|facebook/wav2vec2-lv-60-espeak-cv-ft|83.89|23.64|22.80|64.70|28.12|27.07|42.07|
|facebook/hubert-base-ls960|79.65|20.47|24.35|58.84|32.91|28.07|15.53|
|facebook/mms-1b-f102|80.88|25.73|19.15|64.46|31.95|21.00|77.03|
|facebook/wav2vec2-xls-r-1b|62.09|18.02|23.00|63.14|28.62|26.27|13.34|
|facebook/seamless-m4t-v2-large|84.55|32.22|15.60|64.62|43.55|13.28|89.18|
|microsoft/wavlm-base-sv|83.23|22.59|24.00|53.00|33.61|23.51|21.78|
|microsoft/wavlm-base-sd|83.23|22.59|24.00|53.00|33.61|23.51|21.78|
|microsoft/wavlm-base|83.23|22.59|24.00|53.00|33.61|23.51|21.78|
|facebook/mms-1b-all|75.37|24.64|19.55|59.97|28.36|15.59|58.46|
|vitouphy/wav2vec2-xls-r-300m-phoneme|83.37|20.08|20.70|59.65|27.14|15.64|27.37|
|microsoft/wavlm-base-plus-sd|83.06|18.22|27.15|55.63|30.30|17.69|12.84|
|microsoft/wavlm-base-plus-sv|83.06|18.22|27.15|55.63|30.30|17.69|12.84|
|microsoft/wavlm-base-plus|83.06|18.22|27.15|55.63|30.30|17.69|12.84|
|microsoft/speecht5_multimodal|78.59|23.52|20.65|53.22|28.39|20.90|41.89|
|facebook/hubert-large-ls960-ft|85.28|23.69|19.65|57.06|33.64|14.34|35.63|
|facebook/wav2vec2-base|40.35|15.03|21.25|54.45|28.88|26.07|10.13|
|microsoft/msclap-2022|19.04|13.61|26.30|71.94|25.54|14.74|7.77|
|google/vggish|12.64|14.13|26.15|39.46|27.08|18.45|7.26|
|google/yamnet|14.47|13.10|25.10|47.82|27.46|18.80|5.91|
|microsoft/unispeech-sat-base-100h-libri-ft|81.86|18.54|20.50|47.62|30.13|17.59|16.05|
|asapp/sew-d-tiny-100k-ft-ls100h|80.39|22.19|22.80|48.96|29.75|15.79|26.01|
|lyrebird/wav2clip|10.51|13.31|22.40|38.29|25.78|14.19|16.04|
|facebook/data2vec-audio-base-960h|74.43|20.45|19.45|55.42|32.07|13.33|37.50|
|facebook/data2vec-audio-large-960h|69.45|21.65|15.85|53.82|30.56|12.38|47.81|
|facebook/wav2vec2-base-960h|77.05|16.55|18.95|51.99|30.42|10.08|23.13|
|facebook/wav2vec2-xls-r-300m|20.18|13.59|19.25|39.75|17.69|9.07|10.47|
|OpenMuQ/MuQ-MuLan-large|11.10|14.25|18.45|34.99|25.11|11.53|18.92|
|asapp/sew-d-mid-400k-ft-ls100h|65.20|18.13|15.85|40.23|30.65|13.68|15.36|
|speechbrain/m-ctc-t-large|76.00|20.83|11.95|49.20|30.36|13.13|52.88|
|facebook/wav2vec2-large|15.93|14.05|16.20|40.77|31.17|12.03|11.65|
|speechbrain/cnn14-esc50|8.96|13.17|21.25|42.58|23.07|10.58|10.80|
|asapp/sew-d-base-plus-400k-ft-ls100h|51.67|17.58|14.05|37.18|32.13|14.59|15.85|
|laion/larger_clap_music|4.71|13.76|19.35|30.54|27.05|10.53|9.63|
|facebook/encodec_24khz|9.66|11.81|18.20|26.90|23.55|9.42|8.61|
|facebook/wav2vec2-large-xlsr-53|4.32|11.65|16.60|24.62|12.09|8.07|7.44|

<!-- page: 26 -->

**MAEB: Massive Audio Embedding Benchmark** 

. 

_Table 9._ MInDS-14 classification results across languages. Best result per language in bold. 

|Model|cs|de|en|es|fr|it|ko|nl|pl|pt|ru|zh|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|72.3|**91.5**|98.1|**97.9**|97.4|84.2|92.9|**94.2**|68.7|**82.5**|**95.6**|97.8|
|LCO-Embedding/LCO-Embedding-Omni-3B|72.5|89.8|**98.5**|96.9|**97.4**|**84.8**|**93.2**|92.1|67.6|81.0|95.2|**98.2**|
|facebook/seamless-m4t-v2-large|**92.3**|89.4|89.2|92.0|90.4|78.9|90.5|90.2|67.8|63.4|90.4|93.0|
|facebook/mms-1b-f102|77.0|78.4|77.0|76.8|75.3|71.8|70.3|78.7|**69.8**|66.2|66.6|71.3|
|facebook/mms-1b-all|64.3|54.7|58.5|46.1|62.0|51.1|69.9|64.2|47.3|53.5|54.0|76.9|
|facebook/mms-1b-l1107|55.6|57.4|64.0|58.2|63.1|51.9|44.8|56.7|43.8|46.5|52.9|37.6|
|openai/whisper-medium|50.5|53.0|48.3|57.6|44.9|48.3|53.2|54.1|43.8|47.8|47.3|47.4|
|speechbrain/m-ctc-t-large|49.8|53.7|52.9|43.0|60.3|46.8|30.1|41.7|30.6|32.9|47.5|45.6|
|openai/whisper-small|35.9|38.0|35.6|40.9|36.7|35.5|38.4|40.4|29.2|35.9|37.5|39.4|
|openai/whisper-large-v3|34.3|31.2|31.9|30.9|34.5|32.8|40.2|38.1|24.9|33.1|31.2|31.4|
|openai/whisper-base|30.5|35.2|29.6|35.4|26.0|31.2|29.4|29.5|26.3|31.0|23.9|28.3|
|facebook/wav2vec2-lv-60-espeak-cv-ft|27.5|21.9|42.1|22.6|31.2|20.0|21.3|28.3|19.9|25.0|25.2|28.5|
|openai/whisper-tiny|24.6|28.3|31.3|25.3|23.4|27.2|25.3|28.1|21.9|26.5|18.7|21.5|
|Qwen/Qwen2-Audio-7B|20.9|28.5|25.5|22.0|28.0|28.0|27.5|27.4|20.7|27.2|20.4|16.9|
|facebook/data2vec-audio-large-960h|26.1|17.2|47.8|17.1|18.4|22.4|14.4|22.9|22.6|22.2|20.2|13.5|
|vitouphy/wav2vec2-xls-r-300m-phoneme|23.9|20.5|27.4|20.4|25.1|16.7|19.4|26.8|15.8|21.4|20.4|22.9|
|microsoft/speecht5_multimodal|19.2|20.1|41.9|14.2|17.8|14.1|16.9|24.9|12.8|18.0|17.6|15.5|
|asapp/sew-d-tiny-100k-ft-ls100h|19.0|18.5|26.0|10.1|14.3|16.2|18.9|20.6|15.8|18.4|13.5|14.7|
|facebook/hubert-large-ls960-ft|15.8|14.1|35.6|11.5|15.6|13.9|14.5|15.8|15.1|17.7|13.2|14.5|
|facebook/data2vec-audio-base-960h|15.5|10.6|37.5|13.2|16.0|13.9|13.8|17.4|17.3|12.9|11.5|11.3|
|OpenMuQ/MuQ-MuLan-large|17.2|16.5|18.9|13.8|22.1|11.2|19.9|13.3|9.6|14.9|12.1|18.7|
|facebook/wav2vec2-base-960h|16.9|11.9|23.1|11.7|16.0|13.8|12.2|12.1|11.2|17.0|11.7|12.8|
|facebook/wav2vec2-xls-r-2b|13.9|14.1|15.2|8.8|13.2|15.7|12.5|16.8|11.7|19.7|9.7|12.3|
|microsoft/wavlm-large|14.1|12.1|20.8|10.1|11.5|12.1|13.5|14.7|14.9|14.6|10.8|12.4|
|microsoft/wavlm-base-sv|11.5|14.2|21.8|9.9|11.1|13.6|12.0|14.1|14.4|13.4|12.1|12.2|
|microsoft/wavlm-base-sd|11.5|14.2|21.8|9.9|11.1|13.6|12.0|14.1|14.4|13.4|12.1|12.2|
|microsoft/wavlm-base|11.5|14.2|21.8|9.9|11.1|13.6|12.0|14.1|14.4|13.4|12.1|12.2|
|facebook/hubert-base-ls960|13.9|14.4|15.5|9.9|13.2|12.5|11.2|12.8|11.0|14.4|11.7|10.4|
|facebook/wav2vec2-xls-r-1b|12.0|14.2|13.3|9.1|12.4|14.1|12.7|12.5|10.0|18.0|10.9|10.2|
|lyrebird/wav2clip|9.8|14.2|16.0|13.2|15.2|12.6|12.7|10.2|7.8|16.4|8.9|9.2|
|microsoft/wavlm-base-plus-sv|11.9|10.3|12.8|8.6|12.2|11.9|11.8|11.0|14.4|14.6|10.0|11.8|
|microsoft/wavlm-base-plus|11.9|10.3|12.8|8.6|12.2|11.9|11.8|11.0|14.4|14.6|10.0|11.8|
|microsoft/wavlm-base-plus-sd|11.9|10.3|12.8|8.6|12.2|11.9|11.8|11.0|14.4|14.6|10.0|11.8|
|asapp/sew-d-base-plus-400k-ft-ls100h|13.1|11.9|15.9|8.8|9.8|11.8|9.5|9.9|10.3|12.9|14.3|9.4|
|asapp/sew-d-mid-400k-ft-ls100h|12.7|11.9|15.4|9.3|10.6|11.1|8.6|10.6|9.6|13.9|9.7|10.8|
|microsoft/unispeech-sat-base-100h-libri-ft|8.9|7.4|16.0|8.6|12.2|9.6|9.6|8.1|9.3|13.7|7.6|8.4|
|facebook/wav2vec2-large|11.2|11.3|11.6|9.1|8.4|9.3|8.9|9.3|9.1|14.1|7.6|9.0|
|laion/clap-htsat-unfused|9.6|11.0|9.3|5.1|10.2|9.8|9.3|9.5|11.6|12.6|9.8|7.2|
|facebook/wav2vec2-base|9.8|10.5|10.1|6.8|9.5|10.5|9.0|9.2|9.4|13.7|9.1|6.2|
|facebook/wav2vec2-xls-r-300m|9.6|8.3|10.5|7.8|8.3|10.2|7.4|7.9|8.5|15.2|8.5|9.0|
|laion/clap-htsat-fused|8.9|10.5|10.3|8.8|6.7|9.3|8.9|9.2|12.3|13.1|5.4|7.2|
|speechbrain/cnn14-esc50|8.4|8.3|10.8|8.4|7.8|8.5|6.8|10.2|8.7|13.7|7.8|8.8|
|facebook/encodec_24khz|9.2|8.7|8.6|8.2|9.7|8.2|8.1|8.1|10.9|9.6|6.3|10.0|
|laion/larger_clap_music|8.2|9.2|9.6|8.2|8.0|9.2|9.6|8.9|8.9|9.9|7.8|6.8|
|facebook/wav2vec2-xls-r-2b-21-to-en|8.0|8.5|7.8|5.8|8.5|10.1|7.4|7.5|6.8|15.2|6.7|7.6|
|facebook/wav2vec2-large-xlsr-53|7.5|7.9|7.4|9.3|8.7|6.9|7.3|8.7|9.1|8.1|5.0|10.3|
|laion/larger_clap_general|6.4|12.1|8.1|6.0|7.6|10.6|6.3|8.0|8.2|11.8|5.0|5.8|
|MIT/ast-fnetuned-audioset-10-10-0.4593|6.4|9.7|7.9|5.8|6.1|10.3|9.3|7.3|7.5|10.4|4.6|5.4|
|microsoft/msclap-2023|7.1|9.3|7.4|5.4|4.5|9.6|8.6|6.6|8.7|9.3|7.6|6.6|
|laion/larger_clap_music_and_speech|5.0|9.0|7.8|4.3|5.9|11.5|5.2|5.4|9.8|11.3|5.8|3.6|
|google/vggish|6.8|8.8|7.3|5.8|6.5|8.0|4.9|6.1|7.3|10.8|5.2|5.4|
|microsoft/msclap-2022|7.0|6.5|7.8|4.9|7.8|5.7|8.3|5.0|5.7|9.6|6.3|7.8|
|google/yamnet|6.6|7.0|5.9|4.1|5.2|7.5|8.1|7.6|7.8|8.8|6.1|5.8|



26

<!-- page: 27 -->

_Table 10._ SIB-FLEURS classification results (languages 1–15 of 102). Best per language in bold. 

|Model|afr_Latn|amh_Ethi|arb_Arab|asm_Beng|ast_Latn|azj_Latn|bel_Cyrl|ben_Beng|bos_Latn|bul_Cyrl|cat_Latn|ceb_Latn|ces_Latn|ckb_Arab|cym_Latn|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|**47.3**|39.3|**71.5**|32.1|**70.6**|50.9|**70.5**|41.1|51.7|**55.4**|**74.3**|**53.4**|45.5|28.5|32.0|
|LCO-Embedding/LCO-Embedding-Omni-3B|40.2|35.7|65.2|36.6|65.4|**53.6**|63.3|32.9|**51.8**|49.2|68.8|48.1|**53.6**|21.4|21.4|
|facebook/seamless-m4t-v2-large|42.0|**41.9**|54.6|**51.7**|51.8|41.0|43.8|**46.5**|48.1|53.6|45.7|24.2|46.6|**32.9**|**34.8**|
|openai/whisper-medium|26.8|18.8|28.6|15.9|44.5|36.5|39.1|24.0|29.5|29.6|32.1|31.1|30.4|11.6|21.4|
|facebook/mms-1b-f102|17.9|23.1|30.4|34.9|27.7|20.5|34.0|27.7|22.3|16.1|26.8|19.6|27.6|23.2|25.8|
|facebook/mms-1b-all|25.0|27.7|21.5|26.0|27.7|17.8|29.4|25.9|26.9|21.5|18.7|21.4|18.7|20.4|24.9|
|OpenMuQ/MuQ-MuLan-large|20.6|15.3|28.7|26.7|41.1|18.6|24.9|23.2|35.9|21.6|17.8|32.8|28.5|15.9|14.2|
|openai/whisper-large-v3|25.0|22.4|19.6|9.7|37.5|30.2|25.9|25.0|22.3|19.7|22.1|32.9|17.7|11.5|18.7|
|lyrebird/wav2clip|26.0|21.3|23.2|26.7|31.4|30.3|28.7|25.9|35.7|18.7|15.1|16.2|22.5|15.9|22.4|
|speechbrain/m-ctc-t-large|21.5|18.8|18.7|24.9|31.3|16.9|19.6|24.0|19.7|20.6|32.9|18.7|17.0|16.2|20.4|
|facebook/mms-1b-l1107|22.4|17.9|20.6|20.6|29.5|15.2|22.4|27.7|24.2|14.3|17.0|11.5|23.2|9.9|17.9|
|openai/whisper-small|17.0|22.3|14.3|16.8|28.5|18.5|27.6|18.7|14.3|16.0|15.1|18.7|11.6|9.8|14.3|
|facebook/data2vec-audio-large-960h|12.5|14.3|18.7|26.8|9.8|17.7|18.7|27.8|17.9|19.8|20.4|20.6|11.6|18.7|16.0|
|speechbrain/cnn14-esc50|18.9|22.2|8.9|18.7|25.0|26.8|14.4|21.4|23.2|21.5|19.7|17.0|12.6|22.2|22.2|
|openai/whisper-base|13.4|17.9|10.8|17.7|28.5|16.0|21.3|20.4|14.3|16.0|15.9|20.4|10.7|9.9|8.9|
|vitouphy/wav2vec2-xls-r-300m-phoneme|15.2|18.7|10.7|20.5|14.3|11.6|11.5|14.3|16.1|10.7|16.9|12.5|14.3|9.9|19.6|
|microsoft/speecht5_multimodal|10.7|22.2|18.9|14.2|15.1|18.7|10.8|24.1|23.3|13.4|18.7|11.5|14.5|8.9|19.5|
|facebook/data2vec-audio-base-960h|16.8|17.9|16.2|20.5|21.4|10.7|15.1|15.3|18.8|16.1|8.0|11.6|11.7|9.9|18.7|
|openai/whisper-tiny|22.4|13.4|14.3|14.1|19.6|18.7|13.4|24.9|16.1|12.4|15.0|22.2|14.3|11.5|17.0|
|Qwen/Qwen2-Audio-7B|14.3|16.9|16.1|15.1|7.2|15.1|16.0|16.0|20.7|13.4|18.9|14.3|15.1|14.3|11.7|
|facebook/wav2vec2-lv-60-espeak-cv-ft|12.6|20.5|8.1|17.7|19.6|16.0|16.0|20.6|8.9|13.4|5.3|8.9|17.9|15.2|7.1|
|facebook/encodec_24khz|13.4|15.3|14.3|15.2|17.9|18.7|10.7|21.5|7.9|12.6|16.0|13.4|12.6|17.0|15.1|
|microsoft/msclap-2022|15.2|8.9|8.9|14.3|16.2|14.4|16.1|16.1|17.1|16.9|14.2|9.9|10.8|21.3|12.4|
|facebook/wav2vec2-xls-r-300m|17.0|10.8|20.4|15.2|8.0|18.0|19.6|12.5|12.5|16.0|15.2|9.8|11.8|14.2|17.9|
|facebook/wav2vec2-xls-r-2b-21-to-en|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|
|MIT/ast-fnetuned-audioset-10-10-0.4593|15.2|15.1|13.3|17.8|17.8|14.3|10.7|15.1|15.2|12.6|22.1|15.1|18.0|9.8|13.3|
|microsoft/msclap-2023|11.5|11.7|14.4|16.0|17.0|10.7|13.4|15.2|8.9|19.6|9.7|16.1|11.7|16.8|9.8|
|google/vggish|15.1|16.1|12.5|16.1|12.5|11.7|9.8|19.6|14.3|19.7|14.2|14.3|17.1|16.0|13.3|
|asapp/sew-d-tiny-100k-ft-ls100h|14.3|12.5|9.8|8.9|14.3|10.6|8.1|20.5|13.3|6.2|14.2|18.7|10.8|10.7|10.7|
|google/yamnet|14.3|14.3|12.6|15.2|17.0|11.7|12.5|11.6|15.1|12.6|18.7|17.9|16.2|9.6|9.7|
|asapp/sew-d-base-plus-400k-ft-ls100h|18.8|10.6|15.2|8.9|17.8|7.2|10.7|17.9|16.9|9.8|17.7|15.1|8.9|10.8|10.6|
|facebook/wav2vec2-xls-r-2b|11.6|16.9|12.5|16.0|13.4|16.1|15.2|20.5|9.8|7.2|13.3|14.3|9.0|8.9|13.4|
|facebook/hubert-large-ls960-ft|13.4|14.2|9.9|12.4|8.9|8.9|8.9|22.2|18.6|8.1|16.9|13.4|9.0|8.1|6.3|
|microsoft/wavlm-base-sd|9.8|13.4|14.3|15.1|17.8|14.2|9.8|16.1|17.7|5.3|15.9|15.1|10.7|12.4|10.8|
|microsoft/wavlm-base|9.8|13.4|14.3|15.1|17.8|14.2|9.8|16.1|17.7|5.3|15.9|15.1|10.7|12.4|10.8|
|microsoft/wavlm-base-sv|9.8|13.4|14.3|15.1|17.8|14.2|9.8|16.1|17.7|5.3|15.9|15.1|10.7|12.4|10.8|
|facebook/wav2vec2-base|12.5|15.1|15.1|10.6|20.4|13.4|13.5|17.0|13.3|6.2|18.6|9.7|14.4|10.7|8.0|
|facebook/hubert-base-ls960|14.3|14.3|14.3|6.1|13.3|10.8|14.3|17.8|13.4|8.9|14.1|11.7|12.6|8.9|7.2|
|facebook/wav2vec2-xls-r-1b|17.0|17.8|11.5|15.2|16.8|9.8|17.9|15.1|9.8|9.0|11.5|12.5|9.0|7.1|15.1|
|asapp/sew-d-mid-400k-ft-ls100h|16.0|10.6|14.3|7.1|9.8|11.6|8.9|16.8|13.2|9.0|18.6|15.2|10.8|12.5|7.2|
|microsoft/wavlm-large|11.7|11.5|10.8|9.9|14.3|10.7|11.7|14.2|14.9|10.7|14.2|11.6|8.9|9.8|11.7|
|microsoft/unispeech-sat-base-100h-libri-ft|14.3|9.8|11.6|8.8|10.7|7.1|11.6|15.2|15.1|6.3|14.2|10.6|8.1|8.8|11.6|
|facebook/wav2vec2-base-960h|14.2|19.6|8.9|11.6|12.5|11.5|9.8|13.3|14.2|7.2|8.0|12.5|13.4|8.9|16.0|
|microsoft/wavlm-base-plus-sv|15.2|11.5|15.1|10.8|10.7|8.9|8.9|17.8|7.9|2.7|13.4|11.6|10.8|8.9|5.5|
|microsoft/wavlm-base-plus|15.2|11.5|15.1|10.8|10.7|8.9|8.9|17.8|7.9|2.7|13.4|11.6|10.8|8.9|5.5|
|microsoft/wavlm-base-plus-sd|15.2|11.5|15.1|10.8|10.7|8.9|8.9|17.8|7.9|2.7|13.4|11.6|10.8|8.9|5.5|
|facebook/wav2vec2-large|8.9|10.6|6.2|5.3|14.4|6.2|6.2|17.8|10.6|8.0|10.6|10.6|4.4|10.8|10.7|
|laion/larger_clap_general|4.5|9.0|15.1|7.1|11.5|8.8|8.9|9.7|6.1|5.3|10.6|9.8|5.4|6.2|7.1|
|laion/clap-htsat-fused|8.0|10.8|9.8|12.5|13.4|9.8|4.5|11.6|6.1|7.9|7.1|5.4|4.5|9.8|9.9|
|laion/larger_clap_music_and_speech|5.4|10.0|12.5|8.9|8.9|5.3|8.9|13.3|7.1|8.9|8.9|9.8|2.7|8.9|4.4|
|laion/clap-htsat-unfused|3.6|8.0|8.8|7.0|11.5|7.1|7.1|7.1|12.3|8.9|10.6|8.9|4.5|7.1|4.4|
|laion/larger_clap_music|4.4|6.2|6.2|5.3|7.9|5.3|7.9|5.3|5.3|7.0|7.0|3.6|4.4|3.6|7.9|
|facebook/wav2vec2-large-xlsr-53|7.1|4.4|7.9|3.6|5.3|7.1|7.9|5.3|4.4|4.4|7.0|6.2|5.3|5.3|3.6|

<!-- page: 28 -->

_Table 11._ SIB-FLEURS classification results (languages 16–30 of 102). Best per language in bold. 

|Model|dan_Latn|deu_Latn|ell_Grek|eng_Latn|est_Latn|fn_Latn|fra_Latn|fuv_Latn|gaz_Latn|gle_Latn|glg_Latn|guj_Gujr|hau_Latn|heb_Hebr|hin_Deva|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|39.3|**71.3**|29.4|**70.6**|34.8|40.2|**81.3**|25.8|**39.4**|28.6|**72.2**|**64.3**|22.4|28.5|**73.2**|
|LCO-Embedding/LCO-Embedding-Omni-3B|43.8|68.7|**42.0**|68.9|36.6|32.1|74.1|18.7|31.2|27.7|67.6|49.1|19.6|31.2|65.2|
|facebook/seamless-m4t-v2-large|**49.1**|52.7|38.4|57.2|**49.1**|**47.2**|57.3|15.2|11.5|**33.0**|53.6|49.8|16.0|**43.6**|46.5|
|openai/whisper-medium|31.3|41.0|27.7|28.6|28.5|28.6|34.8|19.6|16.1|20.4|22.3|26.0|17.8|15.9|31.3|
|facebook/mms-1b-f102|25.8|29.6|17.7|20.6|27.7|21.3|20.6|24.2|30.3|32.2|26.8|25.8|25.9|25.8|25.1|
|facebook/mms-1b-all|17.0|33.2|18.7|21.5|19.7|26.8|27.6|16.1|19.7|23.2|22.3|26.8|21.5|19.6|28.7|
|OpenMuQ/MuQ-MuLan-large|26.9|36.6|23.2|16.1|21.4|18.7|22.3|17.9|13.5|32.9|26.7|26.7|33.0|20.6|35.8|
|openai/whisper-large-v3|23.2|26.9|18.0|25.8|19.5|33.0|33.0|19.7|16.9|23.1|24.1|17.9|20.4|20.4|31.2|
|lyrebird/wav2clip|24.9|29.4|14.3|22.3|17.8|24.1|24.0|17.8|14.3|31.4|26.6|20.3|**34.0**|24.2|26.8|
|speechbrain/m-ctc-t-large|19.8|33.0|19.5|31.2|22.4|22.3|29.4|16.9|12.5|23.4|24.1|24.2|15.1|17.9|23.1|
|facebook/mms-1b-l1107|18.7|24.9|13.4|26.0|25.1|24.0|26.8|20.6|17.0|25.0|15.9|21.3|18.7|17.7|27.0|
|openai/whisper-small|22.3|26.7|15.2|16.1|13.3|17.8|20.4|**26.9**|17.8|14.2|17.9|17.8|16.9|16.9|26.0|
|facebook/data2vec-audio-large-960h|19.6|14.2|16.9|16.9|12.5|19.6|21.5|12.6|12.5|17.8|21.3|17.0|20.5|12.5|25.0|
|speechbrain/cnn14-esc50|15.1|15.1|17.9|20.5|14.3|20.7|17.7|11.5|13.4|25.8|21.4|20.4|17.0|16.0|28.5|
|openai/whisper-base|17.0|31.2|9.8|19.6|16.9|16.9|17.0|22.3|13.4|15.1|13.3|12.4|9.7|11.6|21.5|
|vitouphy/wav2vec2-xls-r-300m-phoneme|22.3|15.2|22.3|20.5|14.2|14.2|22.2|17.1|15.2|17.1|16.9|21.5|16.0|15.1|17.0|
|microsoft/speecht5_multimodal|16.9|17.0|9.8|21.3|14.3|16.0|21.5|23.2|10.8|18.0|14.3|17.1|17.8|14.3|13.4|
|facebook/data2vec-audio-base-960h|13.5|17.8|16.0|16.0|13.4|15.2|12.5|17.0|19.6|17.0|16.9|13.4|21.3|22.4|15.1|
|openai/whisper-tiny|16.1|22.3|8.1|12.5|17.8|16.0|13.3|23.1|13.4|12.5|15.2|11.6|7.1|10.7|17.9|
|Qwen/Qwen2-Audio-7B|13.3|16.2|8.9|18.7|16.0|16.0|13.4|25.1|13.4|12.5|15.1|11.7|17.7|19.6|22.3|
|facebook/wav2vec2-lv-60-espeak-cv-ft|17.0|19.6|10.7|18.7|12.4|13.4|10.7|17.0|8.9|17.0|8.0|20.4|17.9|14.3|11.6|
|facebook/encodec_24khz|15.3|16.0|9.8|19.5|15.1|15.2|9.8|15.1|8.9|13.4|12.5|14.3|16.0|8.9|18.6|
|microsoft/msclap-2022|14.2|14.3|7.1|11.5|17.9|17.8|17.8|14.3|15.2|13.4|11.7|7.9|15.2|8.9|17.9|
|facebook/wav2vec2-xls-r-300m|15.2|13.4|14.2|20.6|15.1|12.6|11.6|16.1|18.7|15.2|12.5|20.5|17.9|12.5|23.3|
|facebook/wav2vec2-xls-r-2b-21-to-en|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|
|MIT/ast-fnetuned-audioset-10-10-0.4593|12.4|13.3|20.6|10.6|16.1|9.8|17.7|14.3|12.5|20.6|12.6|15.1|13.3|13.3|18.7|
|microsoft/msclap-2023|10.7|16.9|11.6|14.3|20.5|11.6|15.2|16.1|10.7|17.9|16.0|10.7|11.6|8.0|16.1|
|google/vggish|6.2|16.9|5.4|15.1|17.8|14.2|20.6|14.3|16.2|15.3|17.0|7.9|14.3|7.1|13.3|
|asapp/sew-d-tiny-100k-ft-ls100h|16.0|15.3|8.8|25.1|13.3|15.0|13.2|23.3|19.6|13.4|9.8|16.9|14.2|8.1|13.5|
|google/yamnet|14.2|12.5|11.7|14.2|16.2|9.9|20.6|9.8|12.5|16.1|10.6|8.9|9.0|15.2|10.7|
|asapp/sew-d-base-plus-400k-ft-ls100h|7.1|14.3|7.0|21.4|15.1|11.6|21.1|17.0|23.2|17.9|20.5|10.8|15.1|6.3|14.3|
|facebook/wav2vec2-xls-r-2b|14.2|16.0|7.0|15.1|10.7|9.8|15.9|17.8|13.2|13.4|13.3|12.5|14.2|9.7|14.3|
|facebook/hubert-large-ls960-ft|14.2|17.9|5.3|18.8|8.1|8.0|16.0|15.3|19.6|10.6|10.7|10.7|15.2|11.6|16.1|
|microsoft/wavlm-base-sd|10.8|13.4|7.2|15.1|14.2|10.6|16.0|20.5|15.3|17.0|15.8|8.0|9.7|12.5|12.5|
|microsoft/wavlm-base|10.8|13.4|7.2|15.1|14.2|10.6|16.0|20.5|15.3|17.0|15.8|8.0|9.7|12.5|12.5|
|microsoft/wavlm-base-sv|10.8|13.4|7.2|15.1|14.2|10.6|16.0|20.5|15.3|17.0|15.8|8.0|9.7|12.5|12.5|
|facebook/wav2vec2-base|10.7|12.6|9.0|16.8|16.9|10.7|17.7|12.4|9.8|14.3|12.4|8.1|12.5|11.6|10.8|
|facebook/hubert-base-ls960|7.1|18.8|9.0|19.6|14.2|8.9|13.2|20.6|13.3|15.2|8.0|15.1|13.3|8.9|15.1|
|facebook/wav2vec2-xls-r-1b|8.1|17.8|4.5|10.7|18.7|8.9|12.5|16.1|16.0|14.3|10.7|12.5|9.7|6.3|17.0|
|asapp/sew-d-mid-400k-ft-ls100h|10.6|9.8|4.4|14.3|12.5|15.1|13.3|13.4|12.4|12.5|11.5|8.1|10.6|5.3|9.8|
|microsoft/wavlm-large|8.9|15.3|12.5|13.3|6.2|10.8|14.1|20.6|15.1|13.4|6.2|9.8|7.1|7.9|12.5|
|microsoft/unispeech-sat-base-100h-libri-ft|6.2|18.9|9.8|12.4|10.7|14.2|15.1|18.8|14.2|8.9|8.9|10.7|14.2|10.6|15.2|
|facebook/wav2vec2-base-960h|14.3|17.8|7.9|18.7|8.9|5.4|15.1|16.1|16.9|14.3|17.8|15.2|14.2|14.3|15.1|
|microsoft/wavlm-base-plus-sv|11.5|16.2|7.2|13.4|8.8|17.9|12.4|17.9|15.1|16.1|9.8|8.1|12.4|4.4|11.5|
|microsoft/wavlm-base-plus|11.5|16.2|7.2|13.4|8.8|17.9|12.4|17.9|15.1|16.1|9.8|8.1|12.4|4.4|11.5|
|microsoft/wavlm-base-plus-sd|11.5|16.2|7.2|13.4|8.8|17.9|12.4|17.9|15.1|16.1|9.8|8.1|12.4|4.4|11.5|
|facebook/wav2vec2-large|5.3|11.6|9.0|8.9|15.1|8.0|18.6|12.5|9.8|13.4|9.9|10.8|8.0|10.6|10.7|
|laion/larger_clap_general|13.4|14.2|3.6|8.8|12.4|7.2|8.0|13.4|9.8|12.5|10.7|4.5|11.6|10.7|11.6|
|laion/clap-htsat-fused|10.7|16.8|8.0|12.4|13.4|6.2|9.8|7.9|10.7|10.7|15.1|8.1|8.1|5.4|5.3|
|laion/larger_clap_music_and_speech|7.1|8.9|1.7|9.8|11.6|6.2|8.9|10.6|9.8|10.8|7.1|5.4|12.5|3.5|8.9|
|laion/clap-htsat-unfused|6.2|15.1|4.5|13.4|8.9|5.4|8.9|7.1|12.4|8.1|6.2|3.5|7.1|4.4|10.7|
|laion/larger_clap_music|5.3|5.3|7.0|5.3|5.3|5.3|5.3|6.2|7.0|3.6|7.0|7.0|6.2|5.3|5.3|
|facebook/wav2vec2-large-xlsr-53|5.3|5.3|4.4|4.4|4.4|4.4|3.6|6.2|7.0|6.2|7.9|5.3|7.0|7.0|4.4|

<!-- page: 29 -->

_Table 12._ SIB-FLEURS classification results (languages 31–45 of 102). Best per language in bold. 

|Model|hrv_Latn|hun_Latn|hye_Armn|ibo_Latn|ind_Latn|isl_Latn|ita_Latn|jav_Latn|jpn_Jpan|kam_Latn|kan_Knda|kat_Geor|kaz_Cyrl|kea_Latn|khk_Cyrl|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|51.8|**44.6**|25.9|25.0|**82.9**|33.0|**72.3**|**50.2**|**74.2**|26.0|**51.0**|34.9|44.7|**72.4**|20.5|
|LCO-Embedding/LCO-Embedding-Omni-3B|55.4|29.4|29.4|22.3|78.4|24.0|67.9|43.8|72.3|**29.5**|44.7|21.4|41.0|65.2|24.2|
|facebook/seamless-m4t-v2-large|**56.2**|40.1|**48.4**|20.4|54.4|**61.6**|49.1|44.7|49.4|14.3|49.1|**57.2**|**48.1**|36.6|**34.7**|
|openai/whisper-medium|31.2|37.4|20.5|19.6|33.8|15.1|40.9|19.6|26.8|24.1|17.8|16.1|24.0|28.5|23.1|
|facebook/mms-1b-f102|28.5|27.7|22.3|**27.6**|30.3|29.5|25.9|19.6|19.8|24.2|20.5|23.1|26.8|32.1|31.3|
|facebook/mms-1b-all|27.7|26.8|23.1|22.3|27.6|25.9|31.1|22.4|19.8|18.8|23.3|24.9|24.1|25.9|18.8|
|OpenMuQ/MuQ-MuLan-large|24.2|23.2|22.4|13.5|22.4|21.3|24.9|17.0|17.9|25.0|26.8|30.5|20.5|33.9|18.8|
|openai/whisper-large-v3|30.3|24.8|17.8|13.5|32.1|21.4|31.9|23.2|19.6|14.2|17.7|13.4|24.9|26.6|21.4|
|lyrebird/wav2clip|37.5|14.2|25.9|16.0|14.2|19.7|35.7|22.3|22.3|13.3|21.4|20.5|18.7|25.1|25.8|
|speechbrain/m-ctc-t-large|23.2|24.1|18.7|22.3|23.2|15.8|33.0|17.2|14.3|18.9|11.6|21.4|17.0|20.5|22.3|
|facebook/mms-1b-l1107|28.5|25.8|21.3|17.8|22.3|17.8|29.5|22.5|18.9|18.9|15.2|17.0|23.2|18.7|25.9|
|openai/whisper-small|22.3|19.6|15.2|16.0|24.0|25.0|26.7|20.5|24.2|24.0|12.5|11.6|20.5|16.0|18.8|
|facebook/data2vec-audio-large-960h|14.2|15.3|15.2|18.8|15.2|15.2|19.7|14.3|15.3|25.2|11.7|14.3|15.2|19.5|18.8|
|speechbrain/cnn14-esc50|22.3|18.7|21.4|12.6|15.1|16.1|24.9|16.0|18.0|12.5|8.9|17.9|21.5|19.6|15.2|
|openai/whisper-base|18.7|13.2|19.6|15.1|11.5|18.7|27.6|17.9|16.2|18.7|13.4|13.5|18.7|16.1|17.0|
|vitouphy/wav2vec2-xls-r-300m-phoneme|14.3|15.1|22.3|13.4|12.5|22.3|18.7|18.8|13.5|18.7|12.5|18.7|22.3|13.3|13.4|
|microsoft/speecht5_multimodal|25.9|16.8|19.7|15.3|14.2|19.6|14.2|15.1|22.5|13.4|7.2|16.1|15.3|15.1|10.7|
|facebook/data2vec-audio-base-960h|21.4|12.5|21.3|21.3|14.2|12.5|19.6|17.1|10.7|11.6|9.8|18.8|15.0|13.3|13.3|
|openai/whisper-tiny|18.7|16.8|10.6|19.6|13.2|16.0|25.8|14.3|15.2|15.1|6.2|9.0|19.5|16.1|14.2|
|Qwen/Qwen2-Audio-7B|12.5|20.5|16.9|12.5|16.9|13.3|18.8|14.3|15.2|17.0|15.2|14.3|12.6|25.0|12.5|
|facebook/wav2vec2-lv-60-espeak-cv-ft|16.0|15.0|17.0|13.4|10.8|12.5|21.3|20.6|15.2|20.5|10.7|11.7|17.0|14.3|15.2|
|facebook/encodec_24khz|18.8|17.8|17.8|11.7|16.9|13.4|21.4|14.3|19.8|15.2|13.3|9.8|13.4|12.6|7.0|
|microsoft/msclap-2022|14.4|17.0|13.3|8.9|14.2|14.2|13.3|16.9|11.7|12.4|16.2|12.5|20.6|13.3|16.0|
|facebook/wav2vec2-xls-r-300m|13.3|12.4|20.6|8.1|13.3|20.7|19.7|16.8|13.5|14.3|9.7|18.8|16.0|12.4|17.0|
|facebook/wav2vec2-xls-r-2b-21-to-en|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|
|MIT/ast-fnetuned-audioset-10-10-0.4593|13.4|14.3|5.3|15.3|17.0|15.3|16.0|20.6|16.0|12.4|9.9|11.6|18.8|12.6|12.6|
|microsoft/msclap-2023|23.2|16.0|8.0|9.9|15.2|11.6|18.7|18.6|15.3|11.5|16.1|10.7|16.1|13.4|18.8|
|google/vggish|14.3|14.2|8.1|17.1|16.0|16.0|18.8|21.4|14.3|12.4|15.2|11.5|16.2|9.8|16.9|
|asapp/sew-d-tiny-100k-ft-ls100h|17.8|20.3|17.9|12.5|12.5|16.8|19.6|16.9|11.7|13.3|7.1|8.9|12.5|9.8|15.1|
|google/yamnet|13.4|12.4|10.6|17.0|14.3|17.9|17.0|17.9|14.2|8.8|12.5|15.1|18.9|13.4|16.0|
|asapp/sew-d-base-plus-400k-ft-ls100h|18.7|18.6|17.0|14.3|14.2|15.9|20.4|10.7|9.0|15.1|7.1|10.7|5.4|16.0|13.4|
|facebook/wav2vec2-xls-r-2b|11.6|13.3|12.5|13.4|10.6|13.3|14.2|16.0|11.7|13.4|6.3|15.2|18.7|13.5|19.7|
|facebook/hubert-large-ls960-ft|19.5|13.4|15.2|13.3|15.2|14.2|19.6|10.7|9.8|13.4|8.0|12.5|16.0|15.1|13.4|
|microsoft/wavlm-base-sd|9.0|12.3|19.7|13.5|10.6|15.1|18.6|17.0|6.4|17.0|11.5|9.8|15.9|9.0|16.1|
|microsoft/wavlm-base|9.0|12.3|19.7|13.5|10.6|15.1|18.6|17.0|6.4|17.0|11.5|9.8|15.9|9.0|16.1|
|microsoft/wavlm-base-sv|9.0|12.3|19.7|13.5|10.6|15.1|18.6|17.0|6.4|17.0|11.5|9.8|15.9|9.0|16.1|
|facebook/wav2vec2-base|16.1|10.6|15.3|14.3|13.4|17.8|15.1|19.7|8.0|15.1|8.1|15.1|20.4|16.1|11.7|
|facebook/hubert-base-ls960|15.2|7.1|9.0|12.5|10.7|16.9|17.7|16.1|10.8|11.5|7.9|8.0|15.1|11.6|18.8|
|facebook/wav2vec2-xls-r-1b|15.1|13.2|9.8|11.7|12.5|16.9|18.8|17.0|8.9|11.5|13.2|12.6|9.8|13.4|14.3|
|asapp/sew-d-mid-400k-ft-ls100h|12.5|18.6|20.6|16.8|18.7|16.9|18.6|11.6|8.1|13.4|7.9|7.2|12.5|14.3|11.6|
|microsoft/wavlm-large|12.5|13.2|10.8|12.5|15.1|17.0|21.3|18.7|9.1|11.6|6.3|5.3|14.1|14.3|11.6|
|microsoft/unispeech-sat-base-100h-libri-ft|9.8|11.4|12.5|9.1|12.5|13.4|14.2|15.1|8.1|12.4|9.8|13.4|17.7|10.7|12.5|
|facebook/wav2vec2-base-960h|18.7|10.7|13.4|16.1|12.5|15.1|11.5|13.4|4.5|11.5|9.0|10.8|14.2|10.6|7.1|
|microsoft/wavlm-base-plus-sv|8.9|10.6|11.6|9.8|13.3|16.8|17.0|14.2|7.2|11.5|6.3|8.0|14.1|17.0|12.5|
|microsoft/wavlm-base-plus|8.9|10.6|11.6|9.8|13.3|16.8|17.0|14.2|7.2|11.5|6.3|8.0|14.1|17.0|12.5|
|microsoft/wavlm-base-plus-sd|8.9|10.6|11.6|9.8|13.3|16.8|17.0|14.2|7.2|11.5|6.3|8.0|14.1|17.0|12.5|
|facebook/wav2vec2-large|9.8|10.6|13.4|8.0|17.0|12.5|13.4|15.9|16.0|11.6|10.0|8.8|13.2|16.8|12.5|
|laion/larger_clap_general|7.9|7.9|7.0|9.8|7.1|14.2|13.3|7.9|4.5|7.2|9.8|8.1|13.3|8.9|9.8|
|laion/clap-htsat-fused|12.5|6.2|9.7|8.1|4.4|10.6|8.1|10.6|7.2|9.8|12.5|9.8|11.5|11.5|9.8|
|laion/larger_clap_music_and_speech|11.5|8.9|8.0|8.0|8.9|11.5|12.4|7.0|8.1|4.4|7.1|7.2|13.3|8.9|12.4|
|laion/clap-htsat-unfused|12.5|8.9|6.2|11.5|7.9|13.2|14.2|9.7|7.1|7.1|9.8|8.1|8.8|7.2|7.0|
|laion/larger_clap_music|3.6|8.8|5.3|4.4|8.8|4.4|6.2|7.9|7.0|7.0|4.4|4.4|5.3|7.0|6.2|
|facebook/wav2vec2-large-xlsr-53|5.3|5.3|4.4|6.2|5.3|5.3|5.3|4.4|6.2|6.2|5.3|6.2|6.2|2.6|4.4|

<!-- page: 30 -->

_Table 13._ SIB-FLEURS classification results (languages 46–60 of 102). Best per language in bold. 

|Model|khm_Khmr|kir_Cyrl|kor_Hang|lao_Laoo|lin_Latn|lit_Latn|ltz_Latn|lug_Latn|luo_Latn|lvs_Latn|mal_Mlym|mar_Deva|mkd_Cyrl|mlt_Latn|mri_Latn|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|23.3|43.8|68.7|**49.1**|32.9|42.7|**65.2**|21.4|**36.6**|39.2|38.3|**56.2**|49.8|**59.7**|21.4|
|LCO-Embedding/LCO-Embedding-Omni-3B|22.3|48.1|**70.5**|42.7|**33.0**|39.1|59.9|24.2|29.4|41.9|33.0|52.8|58.9|56.2|23.3|
|facebook/seamless-m4t-v2-large|**30.4**|**55.1**|40.4|33.2|23.2|**55.3**|37.5|**35.7**|21.4|**52.8**|**44.5**|41.9|**64.3**|43.8|25.9|
|openai/whisper-medium|26.8|20.4|27.7|24.0|30.4|32.8|32.0|20.6|19.5|34.9|25.0|34.7|32.2|21.3|**28.6**|
|facebook/mms-1b-f102|27.7|24.9|19.7|30.4|26.7|32.0|26.0|27.7|25.7|35.6|21.3|27.6|25.1|24.9|24.2|
|facebook/mms-1b-all|18.7|16.1|29.6|23.2|23.1|23.3|22.3|25.0|22.3|26.6|25.0|32.9|26.1|28.7|22.4|
|OpenMuQ/MuQ-MuLan-large|15.2|14.4|18.7|25.0|25.1|27.7|23.2|11.6|25.8|27.7|16.0|15.1|28.7|17.9|19.8|
|openai/whisper-large-v3|21.4|20.4|22.3|18.7|22.4|31.0|29.3|10.8|20.5|29.5|19.6|33.9|28.6|17.9|25.0|
|lyrebird/wav2clip|26.9|13.4|17.0|23.3|17.9|41.1|28.7|23.2|30.3|25.0|18.7|19.6|19.7|17.9|9.8|
|speechbrain/m-ctc-t-large|24.3|17.7|21.3|17.9|21.3|24.0|20.6|26.8|25.0|32.1|15.3|24.9|25.1|26.8|20.6|
|facebook/mms-1b-l1107|16.9|17.7|17.9|17.0|19.6|30.2|24.9|19.7|22.3|34.7|23.2|29.4|24.1|19.7|19.8|
|openai/whisper-small|24.9|9.8|16.9|17.8|21.4|18.7|26.7|19.7|15.2|22.4|14.3|24.1|25.1|9.7|18.0|
|facebook/data2vec-audio-large-960h|10.6|15.1|16.2|16.2|27.0|17.0|18.7|24.1|20.6|25.1|20.5|19.6|28.5|11.6|17.9|
|speechbrain/cnn14-esc50|19.7|20.5|25.1|14.3|11.5|16.2|18.8|19.6|16.1|10.8|13.4|23.2|12.6|14.4|13.3|
|openai/whisper-base|17.8|9.8|19.6|11.5|26.8|16.0|22.3|16.1|15.1|18.8|13.3|17.0|20.6|10.7|16.9|
|vitouphy/wav2vec2-xls-r-300m-phoneme|13.5|13.3|15.3|16.2|19.6|10.7|19.6|15.2|17.9|14.2|19.6|16.0|13.4|9.8|17.0|
|microsoft/speecht5_multimodal|19.7|11.6|20.5|14.3|21.4|13.3|17.0|17.0|17.0|18.6|19.6|18.7|13.3|13.4|20.6|
|facebook/data2vec-audio-base-960h|15.2|17.0|15.2|14.3|14.2|20.5|17.8|17.2|12.5|13.3|20.6|20.5|11.5|9.9|16.2|
|openai/whisper-tiny|18.7|11.5|16.1|13.3|24.9|13.3|16.1|13.4|21.4|23.2|11.6|17.9|17.7|8.1|18.9|
|Qwen/Qwen2-Audio-7B|15.3|10.7|23.2|18.9|16.0|12.4|25.8|11.6|15.2|18.7|12.5|9.8|15.1|13.4|9.0|
|facebook/wav2vec2-lv-60-espeak-cv-ft|17.0|13.4|18.7|10.7|17.8|17.7|9.8|13.5|17.9|13.5|16.9|12.3|14.3|10.7|17.2|
|facebook/encodec_24khz|15.2|12.6|14.3|9.8|16.0|17.8|14.2|14.2|21.3|19.6|9.9|15.2|21.4|13.5|11.6|
|microsoft/msclap-2022|16.9|9.8|10.8|9.8|18.5|17.0|9.8|21.4|16.1|11.5|13.4|17.9|11.5|14.3|14.3|
|facebook/wav2vec2-xls-r-300m|28.7|12.5|11.6|13.4|8.9|12.5|10.7|12.4|18.8|12.5|15.1|14.3|14.3|9.8|15.1|
|facebook/wav2vec2-xls-r-2b-21-to-en|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|
|MIT/ast-fnetuned-audioset-10-10-0.4593|14.2|10.7|20.4|14.3|12.5|12.5|14.3|15.3|9.0|10.8|16.2|15.9|15.1|10.7|13.4|
|microsoft/msclap-2023|17.0|14.3|17.0|15.1|7.2|7.1|14.3|12.5|10.6|12.5|8.1|19.7|16.9|10.7|9.8|
|google/vggish|15.3|10.8|15.3|13.4|11.5|8.1|16.1|18.8|11.7|12.5|13.4|16.9|12.5|12.6|13.5|
|asapp/sew-d-tiny-100k-ft-ls100h|16.1|10.7|10.8|14.2|14.3|17.8|20.3|16.9|18.7|14.3|16.1|13.2|15.1|8.9|21.3|
|google/yamnet|9.9|7.2|15.2|14.3|13.3|12.5|15.2|18.8|9.9|18.7|17.0|13.4|16.0|12.5|14.3|
|asapp/sew-d-base-plus-400k-ft-ls100h|21.5|11.6|9.8|13.3|14.3|15.1|15.9|18.7|14.3|13.4|19.8|15.1|8.0|12.4|13.3|
|facebook/wav2vec2-xls-r-2b|17.0|11.6|16.0|12.5|17.7|14.2|16.1|18.7|15.1|12.5|16.0|9.8|14.3|12.5|16.1|
|facebook/hubert-large-ls960-ft|19.6|14.4|13.4|12.4|20.6|13.3|15.8|18.7|16.0|11.7|11.7|8.9|19.6|13.4|15.2|
|microsoft/wavlm-base-sd|18.8|14.2|19.6|17.8|21.3|8.9|21.5|13.3|12.5|17.1|16.2|17.9|8.9|8.9|12.5|
|microsoft/wavlm-base|18.8|14.2|19.6|17.8|21.3|8.9|21.5|13.3|12.5|17.1|16.2|17.9|8.9|8.9|12.5|
|microsoft/wavlm-base-sv|18.8|14.2|19.6|17.8|21.3|8.9|21.5|13.3|12.5|17.1|16.2|17.9|8.9|8.9|12.5|
|facebook/wav2vec2-base|16.1|10.7|15.3|12.4|15.1|10.6|23.2|12.6|7.1|16.2|13.4|16.0|6.2|10.7|9.9|
|facebook/hubert-base-ls960|17.9|9.8|14.3|9.8|24.0|9.8|20.5|9.0|15.9|15.2|15.2|16.9|12.5|10.7|11.6|
|facebook/wav2vec2-xls-r-1b|12.4|12.5|14.3|10.8|12.5|7.1|17.7|8.9|13.3|16.0|11.6|15.1|14.2|8.9|17.9|
|asapp/sew-d-mid-400k-ft-ls100h|21.5|13.3|13.3|13.3|16.1|16.0|19.5|10.7|16.0|13.4|14.3|13.4|10.8|8.9|12.5|
|microsoft/wavlm-large|25.0|11.6|17.0|8.9|18.6|12.4|15.1|11.6|16.0|12.5|9.8|13.3|11.6|9.8|13.4|
|microsoft/unispeech-sat-base-100h-libri-ft|17.0|10.7|16.2|8.9|20.4|13.3|16.0|15.2|14.3|13.4|12.5|12.5|8.9|9.8|11.5|
|facebook/wav2vec2-base-960h|15.1|14.2|9.8|7.9|19.6|15.1|15.1|14.2|14.3|13.4|11.6|17.7|12.5|11.5|9.8|
|microsoft/wavlm-base-plus-sv|20.5|6.2|15.2|8.0|16.9|11.5|13.2|15.3|16.8|15.2|9.8|13.3|14.3|11.5|9.8|
|microsoft/wavlm-base-plus|20.5|6.2|15.2|8.0|16.9|11.5|13.2|15.3|16.8|15.2|9.8|13.3|14.3|11.5|9.8|
|microsoft/wavlm-base-plus-sd|20.5|6.2|15.2|8.0|16.9|11.5|13.2|15.3|16.8|15.2|9.8|13.3|14.3|11.5|9.8|
|facebook/wav2vec2-large|19.6|9.0|13.4|12.5|10.7|11.6|15.2|17.0|9.8|17.9|14.3|8.9|11.6|10.8|8.8|
|laion/larger_clap_general|12.5|7.0|12.4|9.8|9.8|12.4|10.7|11.6|9.8|12.5|10.7|10.6|11.6|10.6|7.1|
|laion/clap-htsat-fused|13.3|5.3|13.4|5.3|8.0|6.2|8.0|10.7|6.2|7.2|11.7|8.8|8.0|11.6|7.2|
|laion/larger_clap_music_and_speech|13.3|3.6|8.0|7.9|8.0|8.0|7.1|14.3|8.9|13.4|9.8|12.4|9.9|9.8|8.9|
|laion/clap-htsat-unfused|12.4|6.2|12.4|9.7|10.6|7.1|11.6|7.1|5.3|10.7|8.1|10.7|8.9|14.2|7.9|
|laion/larger_clap_music|7.0|3.6|2.7|5.3|7.0|4.4|7.9|6.2|2.7|4.4|5.3|7.0|5.3|4.4|6.2|
|facebook/wav2vec2-large-xlsr-53|7.0|5.3|4.4|5.3|5.3|4.4|5.3|5.3|7.9|6.2|6.2|4.4|4.4|5.3|4.4|

<!-- page: 31 -->

_Table 14._ SIB-FLEURS classification results (languages 61–75 of 102). Best per language in bold. 

|Model|mya_Mymr|nld_Latn|nob_Latn|npi_Deva|nso_Latn|nya_Latn|oci_Latn|ory_Orya|pan_Guru|pbt_Arab|pes_Arab|pol_Latn|por_Latn|ron_Latn|rus_Cyrl|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|15.3|**71.5**|42.0|**55.3**|23.3|23.1|**56.3**|**49.0**|**53.7**|29.3|38.5|66.2|71.4|**58.0**|69.7|
|LCO-Embedding/LCO-Embedding-Omni-3B|20.5|66.0|**50.9**|46.5|23.3|26.8|55.2|37.7|41.1|**29.4**|27.5|**71.3**|**75.9**|57.2|**75.0**|
|facebook/seamless-m4t-v2-large|**34.0**|52.9|46.5|42.0|20.5|**45.6**|30.4|40.2|50.0|25.9|**55.3**|43.8|50.9|44.9|52.7|
|openai/whisper-medium|8.1|31.3|38.3|26.7|17.7|17.8|26.7|24.9|33.0|21.4|24.8|36.6|42.8|33.2|33.0|
|facebook/mms-1b-f102|27.7|32.2|30.4|24.2|30.4|33.0|26.8|26.8|25.0|25.0|21.3|22.4|33.8|18.0|29.5|
|facebook/mms-1b-all|25.0|30.4|27.6|20.6|26.8|25.0|29.4|15.2|27.6|20.6|24.8|30.3|30.3|18.8|21.5|
|OpenMuQ/MuQ-MuLan-large|22.4|28.6|20.6|18.9|14.3|33.1|20.6|29.4|23.1|17.9|31.1|9.8|18.7|20.8|23.2|
|openai/whisper-large-v3|12.5|25.9|27.6|28.5|24.0|21.4|25.8|14.2|25.8|21.3|24.0|33.1|29.4|25.2|28.6|
|lyrebird/wav2clip|9.8|26.0|14.4|14.3|**35.7**|18.7|29.6|13.4|18.7|16.2|26.7|10.7|17.0|13.5|19.6|
|speechbrain/m-ctc-t-large|15.2|19.6|27.6|12.5|21.3|25.1|26.8|12.5|22.5|22.2|21.3|24.1|33.8|26.0|22.4|
|facebook/mms-1b-l1107|18.9|24.2|21.4|19.8|22.3|15.1|20.4|20.5|21.3|20.6|25.0|25.9|32.0|25.1|20.6|
|openai/whisper-small|15.2|21.3|19.7|15.1|11.5|22.2|18.6|18.8|23.2|16.1|20.5|24.9|25.9|20.7|15.1|
|facebook/data2vec-audio-large-960h|14.3|14.2|11.7|17.0|14.2|17.0|13.4|14.3|14.3|22.3|22.3|16.0|21.3|13.5|9.7|
|speechbrain/cnn14-esc50|12.6|18.7|20.6|10.8|14.2|17.7|17.9|17.0|14.3|11.7|20.4|9.8|16.0|16.1|9.8|
|openai/whisper-base|8.9|17.8|22.4|10.8|14.2|11.5|18.7|17.0|19.5|14.3|19.7|20.5|16.1|12.5|18.8|
|vitouphy/wav2vec2-xls-r-300m-phoneme|11.6|21.6|11.7|13.4|15.1|17.0|20.5|11.7|18.8|18.9|14.2|10.8|16.0|17.0|17.0|
|microsoft/speecht5_multimodal|11.7|19.8|13.5|10.8|20.5|17.0|19.5|11.6|16.0|11.7|15.2|8.9|23.1|11.7|13.4|
|facebook/data2vec-audio-base-960h|14.4|17.0|15.2|16.1|24.1|19.6|17.8|14.3|14.2|16.1|17.8|17.8|15.2|13.6|7.9|
|openai/whisper-tiny|8.1|20.5|12.6|14.3|16.8|19.6|18.8|18.7|16.9|17.0|15.1|16.0|16.9|15.2|14.2|
|Qwen/Qwen2-Audio-7B|9.9|11.7|19.7|12.6|10.6|12.5|22.4|17.8|20.5|17.8|15.2|10.8|20.6|11.7|15.2|
|facebook/wav2vec2-lv-60-espeak-cv-ft|13.5|16.1|15.2|17.0|16.8|8.9|14.3|17.9|24.8|16.0|10.8|10.7|20.5|18.9|17.9|
|facebook/encodec_24khz|9.7|17.7|17.1|8.9|14.2|18.8|16.0|15.1|14.3|11.7|14.2|9.7|9.8|15.3|9.8|
|microsoft/msclap-2022|12.5|20.5|10.8|11.5|17.0|17.0|15.1|9.8|16.1|15.2|7.2|11.6|9.9|16.1|11.6|
|facebook/wav2vec2-xls-r-300m|10.7|12.6|16.0|10.7|10.8|10.6|14.3|13.4|11.7|18.0|14.3|9.8|12.5|6.3|18.0|
|facebook/wav2vec2-xls-r-2b-21-to-en|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|
|MIT/ast-fnetuned-audioset-10-10-0.4593|9.8|18.7|10.7|12.5|12.5|15.1|12.5|21.5|13.4|17.0|17.8|7.2|12.4|13.4|12.5|
|microsoft/msclap-2023|13.4|16.9|10.7|12.5|15.2|15.0|11.6|9.9|16.9|15.3|12.5|11.7|9.8|15.1|17.1|
|google/vggish|6.2|17.0|9.8|14.2|17.0|15.1|12.6|13.4|15.2|16.1|14.2|9.8|12.5|9.8|16.0|
|asapp/sew-d-tiny-100k-ft-ls100h|13.3|22.3|12.5|10.6|11.5|12.3|17.7|18.7|13.3|14.3|9.0|5.4|14.3|18.8|14.3|
|google/yamnet|10.8|19.6|13.5|9.9|17.8|9.9|12.5|11.5|18.7|13.4|14.3|7.2|9.0|10.8|17.0|
|asapp/sew-d-base-plus-400k-ft-ls100h|7.2|13.4|10.7|10.5|9.7|12.5|12.6|15.1|8.8|15.1|11.7|8.9|17.7|11.7|14.2|
|facebook/wav2vec2-xls-r-2b|15.2|17.0|15.2|11.5|20.5|11.5|22.1|9.8|8.1|18.8|15.2|6.2|24.1|10.8|13.3|
|facebook/hubert-large-ls960-ft|8.9|21.3|17.0|12.6|11.4|15.2|18.7|6.2|12.5|11.6|10.8|9.8|17.0|13.4|10.6|
|microsoft/wavlm-base-sd|9.8|22.3|16.0|13.3|11.6|11.6|15.0|12.3|8.8|15.1|10.7|6.2|16.0|14.3|11.6|
|microsoft/wavlm-base|9.8|22.3|16.0|13.3|11.6|11.6|15.0|12.3|8.8|15.1|10.7|6.2|16.0|14.3|11.6|
|microsoft/wavlm-base-sv|9.8|22.3|16.0|13.3|11.6|11.6|15.0|12.3|8.8|15.1|10.7|6.2|16.0|14.3|11.6|
|facebook/wav2vec2-base|4.5|18.9|13.5|9.8|15.0|13.3|17.0|12.5|7.9|16.9|16.9|4.4|10.8|13.4|12.5|
|facebook/hubert-base-ls960|9.8|17.9|19.7|13.4|10.6|10.8|18.7|9.7|8.9|14.3|15.2|5.3|18.7|12.5|13.3|
|facebook/wav2vec2-xls-r-1b|6.3|17.0|12.5|14.2|9.8|12.4|18.7|8.0|15.1|10.8|11.5|4.4|19.6|10.8|15.9|
|asapp/sew-d-mid-400k-ft-ls100h|6.2|17.0|12.5|3.6|16.8|7.0|16.8|5.4|7.2|16.8|11.6|7.9|16.0|11.7|8.9|
|microsoft/wavlm-large|6.3|20.6|15.3|10.8|9.7|15.1|21.3|12.3|10.6|13.4|10.7|4.4|15.1|13.4|13.3|
|microsoft/unispeech-sat-base-100h-libri-ft|8.9|19.7|16.0|8.9|15.1|11.6|17.8|12.5|15.9|12.5|12.4|8.9|11.7|10.7|17.8|
|facebook/wav2vec2-base-960h|10.7|17.0|5.5|6.2|9.8|12.4|10.7|8.9|8.0|16.0|9.0|2.7|14.2|13.4|12.4|
|microsoft/wavlm-base-plus-sv|6.2|10.7|12.5|9.8|13.4|15.1|14.2|9.8|10.6|13.4|10.7|4.4|12.6|11.6|10.6|
|microsoft/wavlm-base-plus|6.2|10.7|12.5|9.8|13.4|15.1|14.2|9.8|10.6|13.4|10.7|4.4|12.6|11.6|10.6|
|microsoft/wavlm-base-plus-sd|6.2|10.7|12.5|9.8|13.4|15.1|14.2|9.8|10.6|13.4|10.7|4.4|12.6|11.6|10.6|
|facebook/wav2vec2-large|9.8|8.9|8.1|5.3|11.5|16.0|15.1|10.7|12.4|9.8|8.9|7.1|11.6|11.6|8.9|
|laion/larger_clap_general|6.2|8.9|5.3|7.9|8.9|10.7|13.3|8.9|8.9|10.7|5.3|4.4|7.0|9.8|8.9|
|laion/clap-htsat-fused|5.4|5.3|6.3|12.4|9.8|8.8|9.7|9.8|12.3|5.3|7.9|3.6|16.8|8.0|7.2|
|laion/larger_clap_music_and_speech|2.7|9.8|5.3|8.8|12.5|8.8|10.7|9.7|10.6|11.6|4.5|4.4|15.0|11.6|8.9|
|laion/clap-htsat-unfused|5.3|8.9|5.3|8.0|10.6|7.9|9.8|5.4|6.2|9.7|9.7|5.3|11.5|8.1|14.3|
|laion/larger_clap_music|4.4|6.2|3.6|4.4|5.3|5.3|4.4|6.2|6.2|4.4|7.0|6.2|6.2|4.4|6.2|
|facebook/wav2vec2-large-xlsr-53|7.0|7.1|5.3|4.4|6.2|6.2|6.2|4.4|8.8|7.0|5.3|6.2|5.3|6.2|6.2|

<!-- page: 32 -->

_Table 15._ SIB-FLEURS classification results (languages 76–90 of 102). Best per language in bold. 

|Model|slk_Latn|slv_Latn|sna_Latn|snd_Arab|som_Latn|spa_Latn|srp_Cyrl|swe_Latn|swh_Latn|tam_Taml|tel_Telu|tgk_Cyrl|tgl_Latn|tha_Thai|tur_Latn|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|**51.8**|**55.3**|32.3|**38.4**|22.2|**75.1**|**59.8**|50.9|29.5|24.0|**50.8**|34.7|**50.2**|**72.2**|43.9|
|LCO-Embedding/LCO-Embedding-Omni-3B|49.0|44.5|**40.2**|37.5|25.8|69.6|51.8|**52.7**|29.5|31.3|50.0|28.5|43.0|65.8|**49.1**|
|facebook/seamless-m4t-v2-large|50.1|53.6|24.2|20.6|**30.3**|53.4|56.2|45.6|**41.8**|**41.1**|48.2|**42.6**|37.7|44.5|49.1|
|openai/whisper-medium|33.1|29.3|28.6|23.2|15.2|30.3|33.0|27.7|26.8|27.7|29.4|28.5|26.6|34.7|33.0|
|facebook/mms-1b-f102|37.5|30.3|24.2|26.6|21.4|19.6|23.2|22.2|37.5|26.7|23.2|26.0|20.6|32.8|28.7|
|facebook/mms-1b-all|22.3|24.1|27.8|25.8|28.6|21.4|25.0|18.7|33.1|24.0|24.9|24.1|15.1|24.0|24.3|
|OpenMuQ/MuQ-MuLan-large|32.3|31.2|33.1|17.1|15.2|22.3|14.3|27.8|24.1|34.0|26.8|21.6|17.8|22.5|28.7|
|openai/whisper-large-v3|30.4|21.3|21.6|19.6|13.3|22.3|30.4|27.6|22.3|25.0|25.8|15.1|16.8|30.3|22.3|
|lyrebird/wav2clip|23.3|26.7|22.3|13.5|17.0|17.8|17.0|17.9|26.8|26.8|26.0|18.0|22.3|26.0|24.1|
|speechbrain/m-ctc-t-large|19.6|24.0|28.7|13.4|23.3|28.7|23.1|18.7|20.6|22.2|23.1|19.6|18.7|20.6|26.0|
|facebook/mms-1b-l1107|23.2|25.0|31.3|20.5|19.6|23.2|24.2|20.6|31.3|18.8|14.2|25.0|14.2|16.9|25.8|
|openai/whisper-small|18.9|13.2|30.6|24.2|17.0|19.6|22.3|15.2|19.6|24.1|22.3|18.8|15.9|22.3|24.2|
|facebook/data2vec-audio-large-960h|9.8|13.4|16.1|19.7|18.9|13.4|25.1|17.7|19.7|24.1|16.0|22.3|16.2|12.5|16.0|
|speechbrain/cnn14-esc50|17.0|19.6|23.3|17.0|18.7|18.0|22.3|16.9|20.6|19.8|16.1|25.9|17.0|17.9|21.5|
|openai/whisper-base|16.9|12.4|16.1|13.4|13.3|15.2|17.9|12.5|26.8|18.6|11.5|14.3|16.8|14.3|20.6|
|vitouphy/wav2vec2-xls-r-300m-phoneme|19.7|18.6|17.8|16.8|21.4|13.4|18.0|14.4|10.8|17.0|20.5|9.7|18.9|19.8|22.4|
|microsoft/speecht5_multimodal|11.7|19.6|20.7|15.1|16.1|15.2|17.9|14.3|19.7|18.8|15.1|17.0|16.9|12.5|17.9|
|facebook/data2vec-audio-base-960h|16.9|13.4|21.4|9.9|15.2|10.6|16.2|16.2|24.2|18.7|17.0|14.3|16.0|17.9|19.8|
|openai/whisper-tiny|11.6|13.4|17.9|10.8|14.2|12.5|16.1|13.4|21.4|22.3|14.2|14.2|17.6|10.8|22.4|
|Qwen/Qwen2-Audio-7B|13.3|18.7|17.9|20.6|15.2|16.1|10.7|18.8|14.2|17.7|14.3|9.8|17.8|17.0|12.5|
|facebook/wav2vec2-lv-60-espeak-cv-ft|12.5|11.6|18.7|13.3|16.0|10.6|7.1|13.4|18.9|15.1|17.7|15.2|15.1|13.4|12.5|
|facebook/encodec_24khz|17.1|18.8|16.1|9.9|11.6|14.4|16.1|22.3|18.9|11.6|16.0|13.5|18.0|13.4|16.9|
|microsoft/msclap-2022|12.5|25.1|18.9|13.4|12.5|12.4|13.4|12.6|19.7|16.2|17.0|23.2|15.2|16.2|23.1|
|facebook/wav2vec2-xls-r-300m|9.8|14.3|15.2|9.8|16.0|10.0|18.7|8.1|8.9|11.7|13.4|14.3|14.3|15.2|15.2|
|facebook/wav2vec2-xls-r-2b-21-to-en|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|
|MIT/ast-fnetuned-audioset-10-10-0.4593|14.3|18.8|17.1|15.2|8.1|15.2|20.6|14.4|17.9|10.7|15.1|10.8|10.6|20.6|11.6|
|microsoft/msclap-2023|13.5|26.0|22.2|14.2|15.2|14.2|15.2|9.0|17.9|13.3|10.8|17.8|18.8|19.8|17.0|
|google/vggish|18.9|18.8|16.0|9.8|11.6|14.3|16.1|15.3|22.3|8.1|11.6|10.6|19.7|12.6|14.4|
|asapp/sew-d-tiny-100k-ft-ls100h|12.3|15.9|17.9|12.5|14.3|13.3|19.7|18.7|14.2|11.5|15.1|9.8|9.7|12.5|15.2|
|google/yamnet|22.5|14.2|17.0|18.8|11.7|15.2|14.2|16.2|23.3|10.8|17.9|13.3|18.7|16.2|12.5|
|asapp/sew-d-base-plus-400k-ft-ls100h|14.2|12.5|19.5|11.6|20.6|13.4|15.9|16.1|14.3|16.0|12.5|12.5|14.2|13.5|13.5|
|facebook/wav2vec2-xls-r-2b|11.6|13.4|16.8|12.5|7.1|11.5|18.9|9.9|20.7|16.8|16.0|7.2|15.0|11.6|17.0|
|facebook/hubert-large-ls960-ft|17.8|11.6|17.9|12.5|15.2|9.0|20.4|17.0|14.3|12.4|12.4|15.1|5.3|10.7|14.4|
|microsoft/wavlm-base-sd|10.7|11.6|11.6|11.6|14.2|9.8|9.7|14.3|14.3|13.4|9.9|11.7|15.9|11.6|10.8|
|microsoft/wavlm-base|10.7|11.6|11.6|11.6|14.2|9.8|9.7|14.3|14.3|13.4|9.9|11.7|15.9|11.6|10.8|
|microsoft/wavlm-base-sv|10.7|11.6|11.6|11.6|14.2|9.8|9.7|14.3|14.3|13.4|9.9|11.7|15.9|11.6|10.8|
|facebook/wav2vec2-base|11.7|13.4|19.6|7.2|9.8|12.3|19.6|10.0|22.3|17.8|17.8|11.7|13.3|8.0|11.7|
|facebook/hubert-base-ls960|14.3|13.4|14.2|9.8|14.3|10.8|12.5|14.3|15.1|10.6|17.8|9.8|15.1|16.0|12.6|
|facebook/wav2vec2-xls-r-1b|12.5|13.3|17.0|15.2|9.9|10.6|12.6|5.3|15.9|12.5|13.3|8.9|13.3|11.5|11.7|
|asapp/sew-d-mid-400k-ft-ls100h|9.7|10.6|24.2|7.2|7.1|9.8|15.2|17.0|18.7|11.5|15.0|9.8|11.5|8.9|10.8|
|microsoft/wavlm-large|11.6|16.0|17.9|11.6|8.9|7.9|16.0|11.6|12.5|11.6|11.7|12.5|9.6|8.9|15.4|
|microsoft/unispeech-sat-base-100h-libri-ft|10.7|18.8|19.6|8.9|10.8|7.1|17.0|15.2|9.7|9.7|15.1|8.1|16.8|10.7|11.6|
|facebook/wav2vec2-base-960h|11.6|14.3|19.7|14.4|17.8|17.7|11.5|11.5|8.9|10.7|17.7|8.9|9.8|11.6|10.6|
|microsoft/wavlm-base-plus-sv|11.6|16.0|14.3|8.0|7.1|8.0|14.2|14.3|8.9|11.6|14.2|11.6|11.5|5.4|12.6|
|microsoft/wavlm-base-plus|11.6|16.0|14.3|8.0|7.1|8.0|14.2|14.3|8.9|11.6|14.2|11.6|11.5|5.4|12.6|
|microsoft/wavlm-base-plus-sd|11.6|16.0|14.3|8.0|7.1|8.0|14.2|14.3|8.9|11.6|14.2|11.6|11.5|5.4|12.6|
|facebook/wav2vec2-large|11.7|22.3|13.5|9.9|6.2|11.6|16.1|17.8|8.9|11.5|14.3|11.6|13.3|7.1|5.3|
|laion/larger_clap_general|7.2|8.0|11.6|9.9|13.4|6.2|10.8|10.7|19.6|10.6|9.8|12.5|9.7|10.6|7.2|
|laion/clap-htsat-fused|5.3|11.5|9.8|10.7|9.8|5.3|9.8|9.0|9.7|7.1|8.9|10.7|12.3|11.4|11.6|
|laion/larger_clap_music_and_speech|7.1|7.9|12.5|6.2|11.6|4.4|12.6|6.3|13.3|7.9|14.2|8.0|12.3|9.6|7.1|
|laion/clap-htsat-unfused|3.5|15.9|12.5|9.8|11.5|8.0|8.1|6.2|8.9|6.2|8.9|9.8|10.6|12.5|7.2|
|laion/larger_clap_music|6.2|5.3|4.4|4.4|6.2|3.6|6.2|3.6|3.6|7.0|5.3|6.2|6.2|6.2|6.2|
|facebook/wav2vec2-large-xlsr-53|5.3|6.2|5.3|7.0|4.4|4.4|5.3|4.4|3.5|6.2|6.2|7.0|5.3|3.6|2.7|

<!-- page: 33 -->

_Table 16._ SIB-FLEURS classification results (languages 91–102 of 102). Best per language in bold. 

|Model|ukr_Cyrl|umb_Latn|urd_Arab|uzn_Latn|vie_Latn|wol_Latn|xho_Latn|yor_Latn|zho_Hans|zho_Hant|zsm_Latn|zul_Latn|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|68.6|17.8|55.5|51.1|**69.6**|**34.0**|31.1|27.8|**67.7**|**64.2**|**63.4**|26.0|
|LCO-Embedding/LCO-Embedding-Omni-3B|**71.3**|20.6|**57.2**|40.2|66.0|33.0|25.0|32.1|65.9|62.5|55.5|26.7|
|facebook/seamless-m4t-v2-large|54.4|15.3|43.8|**58.9**|50.8|25.1|19.7|**33.8**|49.0|33.1|51.8|16.1|
|openai/whisper-medium|32.1|17.8|21.5|23.0|30.4|14.2|15.2|16.9|22.3|29.5|30.5|16.0|
|facebook/mms-1b-f102|32.0|27.7|24.9|31.4|22.4|28.3|22.4|26.7|26.0|24.2|26.7|26.8|
|facebook/mms-1b-all|27.8|25.8|23.1|31.2|22.4|18.7|24.0|29.4|22.3|19.8|27.7|**28.7**|
|OpenMuQ/MuQ-MuLan-large|23.3|8.9|21.5|25.1|25.1|23.2|**33.0**|24.1|10.8|25.1|28.6|25.0|
|openai/whisper-large-v3|29.4|18.6|23.3|19.5|25.1|12.4|23.1|14.2|16.1|22.3|27.7|19.8|
|lyrebird/wav2clip|23.4|22.5|13.4|12.6|23.3|23.3|30.4|24.9|11.6|17.9|30.5|19.7|
|speechbrain/m-ctc-t-large|17.8|**29.5**|19.6|18.7|15.1|20.3|26.0|27.5|16.0|19.7|23.3|13.4|
|facebook/mms-1b-l1107|31.2|20.5|18.7|23.1|23.2|20.6|19.7|25.8|14.3|20.6|22.4|19.7|
|openai/whisper-small|24.0|20.6|17.9|16.8|17.7|14.3|17.8|13.3|16.2|23.2|19.7|15.1|
|facebook/data2vec-audio-large-960h|23.2|19.6|16.1|17.0|21.4|21.4|18.5|14.2|16.0|17.0|15.2|21.5|
|speechbrain/cnn14-esc50|15.2|12.5|10.7|18.8|16.0|18.8|14.3|21.4|10.8|16.0|21.4|8.9|
|openai/whisper-base|18.7|13.4|13.5|15.1|16.9|11.6|17.8|15.0|13.5|15.2|21.5|16.1|
|vitouphy/wav2vec2-xls-r-300m-phoneme|20.4|18.8|19.6|13.2|14.3|17.9|19.8|14.2|17.8|13.5|18.0|17.0|
|microsoft/speecht5_multimodal|16.0|10.6|19.6|11.6|24.0|15.2|14.2|16.8|15.3|16.1|17.9|17.1|
|facebook/data2vec-audio-base-960h|17.8|16.0|11.7|17.9|22.2|19.7|9.8|17.8|19.7|17.1|17.9|14.3|
|openai/whisper-tiny|18.5|15.0|14.4|15.9|17.0|14.2|18.6|15.9|10.8|16.0|18.8|17.0|
|Qwen/Qwen2-Audio-7B|19.6|16.0|9.8|16.2|20.5|12.4|16.8|11.6|12.4|17.7|14.3|13.4|
|facebook/wav2vec2-lv-60-espeak-cv-ft|16.9|13.3|17.0|11.6|15.2|16.1|19.7|14.3|10.7|13.4|18.8|16.2|
|facebook/encodec_24khz|22.4|12.5|13.4|7.1|13.3|20.6|23.4|19.6|9.8|10.7|10.6|13.3|
|microsoft/msclap-2022|11.5|14.3|8.9|13.4|21.4|10.8|19.8|19.6|9.8|14.3|15.3|16.9|
|facebook/wav2vec2-xls-r-300m|12.5|21.4|12.6|18.8|13.4|9.9|18.9|10.6|13.5|16.8|19.8|8.0|
|facebook/wav2vec2-xls-r-2b-21-to-en|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|14.2|
|MIT/ast-fnetuned-audioset-10-10-0.4593|13.4|12.5|8.9|12.5|23.2|19.7|8.0|15.2|9.0|18.8|6.3|10.7|
|microsoft/msclap-2023|9.8|10.8|12.5|16.9|15.2|13.4|16.1|11.6|17.9|11.6|11.7|13.4|
|google/vggish|12.5|10.7|12.5|12.5|16.0|11.7|19.7|16.0|18.8|11.7|15.1|9.9|
|asapp/sew-d-tiny-100k-ft-ls100h|18.7|9.8|16.0|15.1|13.3|10.7|9.8|9.7|10.8|9.8|19.6|13.3|
|google/yamnet|4.5|17.0|8.9|8.1|10.7|19.6|15.3|14.2|16.1|12.6|10.0|15.2|
|asapp/sew-d-base-plus-400k-ft-ls100h|13.3|14.1|13.4|14.2|14.3|8.0|20.4|10.7|8.1|5.4|14.3|18.7|
|facebook/wav2vec2-xls-r-2b|15.1|10.8|16.1|10.8|16.0|10.7|16.1|7.1|9.0|9.8|12.6|9.8|
|facebook/hubert-large-ls960-ft|12.5|10.6|15.1|8.0|18.6|17.9|15.9|11.5|10.8|13.4|10.7|19.6|
|microsoft/wavlm-base-sd|15.1|9.0|19.6|8.1|16.1|10.6|18.6|8.9|11.7|11.6|12.5|12.6|
|microsoft/wavlm-base|15.1|9.0|19.6|8.1|16.1|10.6|18.6|8.9|11.7|11.6|12.5|12.6|
|microsoft/wavlm-base-sv|15.1|9.0|19.6|8.1|16.1|10.6|18.6|8.9|11.7|11.6|12.5|12.6|
|facebook/wav2vec2-base|11.5|9.0|12.5|16.8|10.7|6.2|18.7|14.2|8.1|10.8|11.5|18.6|
|facebook/hubert-base-ls960|14.2|6.3|16.9|10.7|18.8|8.0|13.4|15.1|11.7|11.6|14.4|12.5|
|facebook/wav2vec2-xls-r-1b|17.8|10.7|14.3|12.5|14.3|13.4|14.3|9.7|9.0|11.6|10.8|9.8|
|asapp/sew-d-mid-400k-ft-ls100h|21.4|15.1|20.5|13.3|14.3|6.2|17.7|12.4|14.3|11.6|10.7|11.6|
|microsoft/wavlm-large|14.1|8.1|13.4|11.6|22.4|9.8|16.8|12.5|8.1|12.5|9.9|14.3|
|microsoft/unispeech-sat-base-100h-libri-ft|14.2|15.2|17.9|11.5|16.1|7.2|10.6|8.9|9.9|11.7|9.9|15.2|
|facebook/wav2vec2-base-960h|12.4|13.4|13.4|12.4|16.9|11.5|15.9|15.0|7.2|9.1|7.2|10.6|
|microsoft/wavlm-base-plus-sv|15.1|8.9|17.8|10.7|19.7|9.8|16.0|12.5|10.7|7.2|11.6|16.1|
|microsoft/wavlm-base-plus|15.1|8.9|17.8|10.7|19.7|9.8|16.0|12.5|10.7|7.2|11.6|16.1|
|microsoft/wavlm-base-plus-sd|15.1|8.9|17.8|10.7|19.7|9.8|16.0|12.5|10.7|7.2|11.6|16.1|
|facebook/wav2vec2-large|14.1|6.2|11.6|12.5|10.7|7.0|14.2|8.1|9.8|12.6|9.0|16.1|
|laion/larger_clap_general|8.0|7.1|9.8|5.3|12.4|7.1|14.2|4.4|6.3|8.9|9.8|8.1|
|laion/clap-htsat-fused|7.0|7.2|9.7|9.7|12.4|7.2|9.8|8.9|6.3|7.1|8.9|7.1|
|laion/larger_clap_music_and_speech|6.2|5.3|13.3|7.1|15.1|7.1|16.0|6.2|6.3|7.1|4.5|10.6|
|laion/clap-htsat-unfused|8.8|4.5|9.8|7.1|11.5|7.1|13.3|6.2|6.3|8.9|4.5|10.7|
|laion/larger_clap_music|7.9|5.3|6.2|2.7|4.4|7.0|3.6|4.4|3.6|3.6|6.2|7.0|
|facebook/wav2vec2-large-xlsr-53|4.5|7.0|4.4|3.6|5.3|4.4|4.4|7.0|4.4|4.4|7.9|5.3|

<!-- page: 34 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 17._ VoxPopuli classification results. GenderID = gender classification, LanguageID = language identification. Both tasks are evaluated on multilingual audio samples containing English, French, Spanish, Polish, and German. Best result per task in bold. 

|Model|GenderID|LanguageID|
|---|---|---|
|laion/larger_clap_general|84.6|84.6|
|Qwen/Qwen2-Audio-7B|68.2|99.0|
|laion/larger_clap_music_and_speech|84.4|81.4|
|openai/whisper-tiny|66.6|96.6|
|openai/whisper-small|62.8|99.2|
|openai/whisper-base|63.6|98.0|
|openai/whisper-medium|59.2|**99.4**|
|speechbrain/m-ctc-t-large|56.0|**99.4**|
|microsoft/wavlm-large|64.4|89.0|
|facebook/wav2vec2-lv-60-espeak-cv-ft|56.2|97.0|
|facebook/mms-1b-all|53.2|**99.4**|
|openai/whisper-large-v3|53.0|98.2|
|facebook/wav2vec2-xls-r-2b|76.0|74.8|
|facebook/mms-1b-l1107|54.0|95.2|
|facebook/mms-1b-f102|49.8|97.2|
|facebook/hubert-base-ls960|76.4|67.6|
|facebook/hubert-large-ls960-ft|56.2|86.8|
|facebook/seamless-m4t-v2-large|52.4|89.0|
|facebook/data2vec-audio-large-960h|52.2|87.6|
|microsoft/wavlm-base-sv|71.8|65.4|
|microsoft/wavlm-base|71.8|65.4|
|microsoft/wavlm-base-sd|71.8|65.4|
|microsoft/speecht5_multimodal|56.8|78.8|
|facebook/wav2vec2-xls-r-1b|71.2|62.8|
|facebook/data2vec-audio-base-960h|56.8|73.6|
|vitouphy/wav2vec2-xls-r-300m-phoneme|53.0|77.2|
|asapp/sew-d-base-plus-400k-ft-ls100h|57.6|71.6|
|microsoft/msclap-2023|86.2|39.6|
|laion/clap-htsat-fused|93.2|32.0|
|facebook/wav2vec2-base|76.2|48.8|
|MIT/ast-fnetuned-audioset-10-10-0.4593|86.8|37.8|
|laion/clap-htsat-unfused|**94.4**|30.0|
|asapp/sew-d-tiny-100k-ft-ls100h|57.6|66.2|
|microsoft/wavlm-base-plus-sv|57.2|64.4|
|microsoft/wavlm-base-plus-sd|57.2|64.4|
|microsoft/wavlm-base-plus|57.2|64.4|
|microsoft/unispeech-sat-base-100h-libri-ft|60.0|58.6|
|asapp/sew-d-mid-400k-ft-ls100h|53.8|63.2|
|LCO-Embedding/LCO-Embedding-Omni-7B|52.2|64.2|
|LCO-Embedding/LCO-Embedding-Omni-3B|51.8|63.4|
|google/vggish|82.0|32.0|
|microsoft/msclap-2022|89.6|24.2|
|facebook/wav2vec2-base-960h|51.0|58.8|
|google/yamnet|79.4|27.2|
|lyrebird/wav2clip|74.2|31.4|
|OpenMuQ/MuQ-MuLan-large|58.6|41.4|
|speechbrain/cnn14-esc50|70.0|29.8|
|laion/larger_clap_music|69.2|28.2|
|facebook/wav2vec2-large|53.8|30.0|
|facebook/wav2vec2-xls-r-300m|52.0|29.4|
|facebook/encodec_24khz|50.2|25.6|
|facebook/wav2vec2-large-xlsr-53|51.6|19.4|



34

<!-- page: 35 -->

_Table 18._ English clustering results. 

|Model|AmbientAcoustic|CREMA-D|ESC50|GTZANGenre|MusicGenre|VehicleSound|VoiceGender|VoxCeleb|VoxPopuliAccent|
|---|---|---|---|---|---|---|---|---|---|
|laion/larger_clap_music_and_speech|43.51|14.43|94.46|63.82|45.30|4.72|**71.33**|0.70|10.61|
|laion/clap-htsat-unfused|43.68|12.78|94.76|55.70|40.07|2.66|68.11|1.06|9.35|
|laion/larger_clap_general|43.78|13.17|94.10|65.09|43.38|3.37|26.63|0.68|**11.20**|
|Qwen/Qwen2-Audio-7B|39.92|**32.37**|88.69|73.94|41.84|5.52|13.43|0.18|4.25|
|microsoft/msclap-2023|44.12|10.73|**95.33**|61.91|36.57|2.86|31.43|0.78|9.58|
|MIT/ast-fnetuned-audioset-10-10-0.4593|**46.76**|7.20|91.36|61.60|47.20|**13.37**|14.07|1.01|7.55|
|laion/clap-htsat-fused|40.43|10.82|93.30|47.10|37.00|4.71|40.80|0.81|9.34|
|microsoft/msclap-2022|44.02|5.36|88.66|45.46|25.84|7.98|53.15|0.64|8.90|
|LCO-Embedding/LCO-Embedding-Omni-7B|42.12|0.62|92.35|62.86|45.06|3.63|10.47|6.69|3.22|
|LCO-Embedding/LCO-Embedding-Omni-3B|41.29|0.62|92.37|59.33|45.25|2.18|3.90|**7.64**|3.04|
|google/vggish|37.38|10.79|54.20|61.34|39.65|4.46|26.80|0.51|10.72|
|google/yamnet|38.09|4.02|63.94|61.75|40.04|0.35|21.93|0.14|6.47|
|OpenMuQ/MuQ-MuLan-large|23.13|11.18|43.55|**76.01**|**53.89**|0.92|0.02|0.39|6.65|
|lyrebird/wav2clip|36.35|13.24|69.29|44.01|29.43|3.97|0.69|0.38|6.63|
|openai/whisper-medium|29.90|14.02|55.02|38.81|26.85|1.08|13.58|1.07|3.28|
|openai/whisper-large-v3|32.26|8.70|54.55|32.25|24.50|1.38|11.85|1.13|3.06|
|openai/whisper-base|25.44|12.54|47.33|33.86|24.28|2.50|13.85|1.01|2.97|
|openai/whisper-small|26.93|10.33|46.48|34.24|24.39|1.29|12.19|1.07|2.99|
|openai/whisper-tiny|24.05|9.82|40.45|32.64|23.97|12.39|11.11|0.93|2.75|
|microsoft/wavlm-large|26.88|7.10|44.10|36.94|21.70|0.18|15.26|0.51|3.35|
|speechbrain/cnn14-esc50|22.29|11.92|59.82|22.02|14.40|2.47|14.45|0.59|4.47|
|facebook/hubert-base-ls960|25.94|7.93|42.08|35.69|18.47|0.13|14.45|0.45|2.85|
|vitouphy/wav2vec2-xls-r-300m-phoneme|23.29|4.87|42.29|28.92|19.31|1.24|21.93|0.09|4.43|
|microsoft/wavlm-base-plus|27.89|5.65|44.69|35.27|20.48|1.79|5.32|0.40|2.84|
|microsoft/wavlm-base-plus-sd|27.89|5.65|44.69|35.27|20.48|1.79|5.32|0.40|2.84|
|microsoft/wavlm-base-plus-sv|27.89|5.65|44.69|35.27|20.48|1.79|5.32|0.40|2.84|
|facebook/mms-1b-all|22.00|4.54|37.42|27.01|18.59|0.32|17.81|0.64|2.70|
|microsoft/wavlm-base-sv|25.12|6.50|39.71|26.40|16.01|0.41|12.94|0.43|3.13|
|microsoft/wavlm-base-sd|25.12|6.50|39.71|26.40|16.01|0.41|12.94|0.43|3.13|
|microsoft/wavlm-base|25.12|6.50|39.71|26.40|16.01|0.41|12.94|0.43|3.13|
|facebook/wav2vec2-base|25.49|12.42|36.72|26.65|12.79|3.09|6.59|0.39|4.21|
|facebook/seamless-m4t-v2-large|26.40|0.83|42.19|25.65|18.25|0.42|9.13|0.38|3.01|
|facebook/mms-1b-f102|23.77|3.21|38.85|24.93|15.96|0.15|11.02|0.58|2.99|
|facebook/wav2vec2-xls-r-2b|21.60|3.31|33.75|29.62|5.74|0.94|21.68|0.44|2.81|
|facebook/encodec_24khz|23.76|11.61|37.56|25.57|11.16|3.80|0.10|0.47|3.46|
|facebook/wav2vec2-lv-60-espeak-cv-ft|23.48|2.32|39.40|21.21|11.66|2.37|7.18|0.08|7.33|
|facebook/wav2vec2-xls-r-300m|25.02|7.10|35.17|12.34|7.36|2.30|21.93|0.71|2.57|
|microsoft/speecht5_multimodal|19.88|2.69|39.82|23.25|18.94|0.47|0.99|0.43|6.48|
|facebook/mms-1b-l1107|21.40|2.25|35.76|28.04|16.96|0.44|2.87|0.64|3.74|
|microsoft/unispeech-sat-base-100h-libri-ft|21.82|4.14|38.38|17.35|14.17|0.33|11.41|0.43|3.48|
|facebook/wav2vec2-large|18.93|7.79|31.91|16.64|8.12|1.41|21.93|0.52|3.18|
|facebook/wav2vec2-xls-r-1b|21.85|2.71|34.38|16.27|7.86|0.13|21.93|1.08|3.29|
|asapp/sew-d-tiny-100k-ft-ls100h|18.12|5.59|32.84|19.61|14.17|0.34|9.92|0.61|3.91|
|asapp/sew-d-mid-400k-ft-ls100h|16.39|5.56|29.74|15.11|11.33|0.63|22.00|0.67|2.90|
|facebook/wav2vec2-large-xlsr-53|17.01|7.81|27.42|14.49|10.44|1.72|21.93|0.33|3.06|
|speechbrain/m-ctc-t-large|17.51|2.32|27.66|19.71|12.79|0.97|19.26|0.76|3.23|
|facebook/hubert-large-ls960-ft|21.83|3.02|37.25|19.68|13.91|0.59|3.24|0.53|3.18|
|laion/larger_clap_music|18.36|10.85|33.01|18.28|10.42|3.73|1.88|0.27|6.18|
|facebook/wav2vec2-base-960h|19.54|2.85|31.10|19.58|15.39|0.61|9.09|0.39|3.57|
|facebook/data2vec-audio-base-960h|19.29|2.12|32.09|18.52|16.90|1.33|6.21|0.34|2.73|
|asapp/sew-d-base-plus-400k-ft-ls100h|15.06|4.51|28.05|16.05|13.07|0.63|17.90|0.45|2.79|
|facebook/data2vec-audio-large-960h|17.39|2.31|31.86|21.62|16.11|0.15|4.05|0.44|2.83|

<!-- page: 36 -->

## **MAEB: Massive Audio Embedding Benchmark** 

_Table 19._ VoxPopuli gender clustering results. Task clusters audio samples by speaker gender across multilingual audio samples containing German, English, French, Spanish, and Polish. Best result in bold. 

|Model|VoxPopuliGender|
|---|---|
|laion/clap-htsat-fused|**52.68**|
|microsoft/msclap-2022|46.37|
|microsoft/msclap-2023|32.09|
|laion/clap-htsat-unfused|22.20|
|google/vggish|8.20|
|speechbrain/cnn14-esc50|7.72|
|laion/larger_clap_music_and_speech|3.80|
|laion/larger_clap_general|3.33|
|laion/larger_clap_music|1.40|
|LCO-Embedding/LCO-Embedding-Omni-3B|1.02|
|facebook/wav2vec2-large-xlsr-53|0.94|
|OpenMuQ/MuQ-MuLan-large|0.76|
|LCO-Embedding/LCO-Embedding-Omni-7B|0.73|
|lyrebird/wav2clip|0.71|
|microsoft/unispeech-sat-base-100h-libri-ft|0.57|
|facebook/wav2vec2-xls-r-1b|0.33|
|google/yamnet|0.29|
|vitouphy/wav2vec2-xls-r-300m-phoneme|0.25|
|facebook/wav2vec2-xls-r-300m|0.21|
|facebook/mms-1b-l1107|0.16|
|asapp/sew-d-tiny-100k-ft-ls100h|0.12|
|facebook/seamless-m4t-v2-large|0.11|
|asapp/sew-d-base-plus-400k-ft-ls100h|0.10|
|speechbrain/m-ctc-t-large|0.08|
|Qwen/Qwen2-Audio-7B|0.07|
|microsoft/wavlm-large|0.07|
|facebook/hubert-base-ls960|0.07|
|facebook/wav2vec2-base-960h|0.06|
|facebook/mms-1b-all|0.06|
|facebook/data2vec-audio-base-960h|0.04|
|microsoft/wavlm-base-sd|0.04|
|microsoft/wavlm-base|0.04|
|microsoft/wavlm-base-sv|0.04|
|facebook/hubert-large-ls960-ft|0.04|
|openai/whisper-base|0.04|
|asapp/sew-d-mid-400k-ft-ls100h|0.04|
|openai/whisper-tiny|0.04|
|MIT/ast-fnetuned-audioset-10-10-0.4593|0.03|
|microsoft/wavlm-base-plus-sd|0.03|
|microsoft/wavlm-base-plus|0.03|
|microsoft/wavlm-base-plus-sv|0.03|
|facebook/wav2vec2-xls-r-2b|0.02|
|facebook/wav2vec2-large|0.02|
|facebook/wav2vec2-lv-60-espeak-cv-ft|0.02|
|facebook/data2vec-audio-large-960h|0.01|
|facebook/mms-1b-f102|0.01|
|openai/whisper-small|0.01|
|microsoft/speecht5_multimodal|0.01|
|facebook/wav2vec2-base|0.01|
|facebook/encodec_24khz|0.01|
|openai/whisper-large-v3|0.01|
|openai/whisper-medium|0.01|



36

<!-- page: 37 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 20._ Pair classification results. Models are evaluated on audio pair similarity tasks. Best result per task in bold. 

|Model|CREMA-D|ESC50|NMSQA|VocalSound|VoxPopuliAccent|
|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|53.56|99.44|**97.60**|94.47|50.73|
|LCO-Embedding/LCO-Embedding-Omni-3B|52.40|**99.54**|96.92|**94.65**|50.67|
|microsoft/msclap-2023|57.65|99.22|51.09|84.22|52.05|
|laion/clap-htsat-unfused|56.91|99.31|46.98|81.58|53.96|
|laion/larger_clap_music_and_speech|56.66|99.35|46.14|80.36|53.49|
|laion/larger_clap_general|54.71|99.14|47.42|81.09|53.55|
|Qwen/Qwen2-Audio-7B|**68.87**|92.54|48.85|71.99|52.93|
|laion/clap-htsat-fused|54.95|98.79|47.85|79.95|53.18|
|microsoft/msclap-2022|53.06|98.36|50.09|74.51|52.08|
|MIT/ast-fnetuned-audioset-10-10-0.4593|54.26|95.49|47.56|71.45|51.83|
|speechbrain/cnn14-esc50|57.07|90.39|53.02|65.55|52.58|
|lyrebird/wav2clip|58.76|93.09|49.47|58.84|52.56|
|google/yamnet|53.15|84.82|58.28|59.67|52.42|
|microsoft/speecht5_multimodal|54.38|72.01|66.28|60.84|53.14|
|openai/whisper-medium|59.95|82.02|46.45|55.88|**55.40**|
|google/vggish|55.79|83.52|47.78|57.06|54.83|
|openai/whisper-small|58.49|76.59|46.81|57.47|52.36|
|OpenMuQ/MuQ-MuLan-large|57.44|75.89|46.19|59.32|52.04|
|openai/whisper-large-v3|57.51|78.37|48.41|54.69|51.55|
|microsoft/wavlm-large|55.80|79.55|48.98|53.13|52.16|
|facebook/mms-1b-all|53.25|74.04|53.90|56.03|51.24|
|openai/whisper-base|57.80|76.38|46.99|55.44|51.54|
|facebook/wav2vec2-lv-60-espeak-cv-ft|53.21|75.31|52.73|51.59|55.08|
|facebook/mms-1b-f102|52.56|75.16|52.06|56.25|51.40|
|microsoft/wavlm-base-plus-sd|54.86|79.07|48.42|53.51|50.91|
|microsoft/wavlm-base-plus-sv|54.86|79.07|48.42|53.51|50.91|
|microsoft/wavlm-base-plus|54.86|79.07|48.42|53.51|50.91|
|microsoft/wavlm-base-sd|54.85|74.21|53.00|52.83|51.31|
|microsoft/wavlm-base|54.85|74.21|53.00|52.83|51.31|
|microsoft/wavlm-base-sv|54.85|74.21|53.00|52.83|51.31|
|facebook/hubert-base-ls960|56.05|76.66|48.43|53.76|51.29|
|microsoft/unispeech-sat-base-100h-libri-ft|54.35|73.88|51.83|54.40|51.27|
|facebook/data2vec-audio-large-960h|53.96|70.38|52.38|57.79|50.74|
|vitouphy/wav2vec2-xls-r-300m-phoneme|53.10|78.21|47.08|53.74|51.03|
|speechbrain/m-ctc-t-large|52.74|55.57|58.98|62.41|51.98|
|facebook/wav2vec2-xls-r-2b|52.83|71.49|48.61|57.08|50.97|
|openai/whisper-tiny|57.52|73.03|45.06|53.49|51.87|
|facebook/wav2vec2-large|55.63|64.11|54.16|55.48|51.44|
|facebook/hubert-large-ls960-ft|53.88|72.70|50.00|52.69|50.66|
|asapp/sew-d-tiny-100k-ft-ls100h|54.01|69.94|48.21|56.34|51.21|
|facebook/wav2vec2-base|56.03|72.45|47.00|51.60|52.15|
|facebook/mms-1b-l1107|53.02|65.65|50.07|58.33|51.40|
|facebook/wav2vec2-xls-r-300m|54.09|72.67|49.32|51.30|50.24|
|facebook/seamless-m4t-v2-large|53.27|68.20|49.13|53.34|51.49|
|facebook/encodec_24khz|55.41|70.44|47.54|50.32|50.93|
|facebook/data2vec-audio-base-960h|52.35|64.46|50.35|55.43|50.97|
|facebook/wav2vec2-base-960h|53.26|65.34|51.92|51.04|51.09|
|facebook/wav2vec2-large-xlsr-53|54.12|62.75|53.78|49.96|51.65|
|asapp/sew-d-mid-400k-ft-ls100h|54.20|62.06|48.42|52.18|50.70|
|facebook/wav2vec2-xls-r-1b|52.56|64.15|49.32|49.94|51.51|
|asapp/sew-d-base-plus-400k-ft-ls100h|54.40|61.22|49.41|52.19|50.11|
|laion/larger_clap_music|55.18|61.25|44.06|52.52|51.60|



37

<!-- page: 38 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 21._ Multilabel classification results. Models are evaluated on audio tagging tasks where each sample can have multiple labels. Best result per task in bold. 

|Model|AudioSetMini|BirdSet|FSD2019Kaggle|FSD50K|
|---|---|---|---|---|
|MIT/ast-fnetuned-audioset-10-10-0.4593|**55.10**|**6.55**|36.90|2.79|
|LCO-Embedding/LCO-Embedding-Omni-7B|43.60|3.03|**44.19**|4.90|
|LCO-Embedding/LCO-Embedding-Omni-3B|42.75|3.05|42.97|4.22|
|google/yamnet|46.46|2.51|18.95|1.34|
|Qwen/Qwen2-Audio-7B|46.28|0.81|7.05|**13.78**|
|openai/whisper-medium|38.15|2.77|17.84|3.25|
|openai/whisper-small|36.19|5.35|14.03|1.94|
|openai/whisper-large-v3|35.36|5.50|9.17|1.22|
|openai/whisper-base|33.59|3.83|9.95|1.55|
|openai/whisper-tiny|31.94|4.24|7.53|1.25|
|laion/larger_clap_general|43.37|0.00|0.00|0.73|
|google/vggish|38.69|0.00|4.25|0.47|
|laion/clap-htsat-unfused|41.57|0.00|0.00|1.03|
|laion/larger_clap_music_and_speech|40.23|0.00|0.00|1.06|
|laion/clap-htsat-fused|37.73|0.00|0.00|0.76|
|microsoft/msclap-2023|38.28|0.00|0.00|0.00|
|facebook/wav2vec2-xls-r-2b|32.39|0.43|4.06|0.12|
|microsoft/msclap-2022|36.02|0.00|0.01|0.02|
|facebook/mms-1b-f102|24.95|3.43|6.14|1.23|
|facebook/hubert-base-ls960|27.52|5.03|2.34|0.56|
|facebook/wav2vec2-xls-r-1b|30.33|0.62|2.77|0.09|
|microsoft/wavlm-large|29.71|0.15|2.55|0.59|
|lyrebird/wav2clip|32.35|0.00|0.00|0.00|
|facebook/mms-1b-all|24.73|1.59|4.82|1.16|
|facebook/data2vec-audio-large-960h|22.84|6.03|0.86|0.40|
|microsoft/wavlm-base|25.92|2.34|1.07|0.52|
|microsoft/wavlm-base-sd|25.92|2.34|1.07|0.52|
|microsoft/wavlm-base-sv|25.92|2.34|1.07|0.52|
|facebook/mms-1b-l1107|26.28|0.35|2.45|0.50|
|facebook/wav2vec2-lv-60-espeak-cv-ft|24.99|0.21|3.91|0.14|
|speechbrain/m-ctc-t-large|19.94|-|1.69|0.04|
|facebook/wav2vec2-base|26.03|0.59|1.89|0.07|
|OpenMuQ/MuQ-MuLan-large|28.38|0.00|0.00|0.00|
|microsoft/wavlm-base-plus-sv|27.17|0.16|0.46|0.24|
|microsoft/wavlm-base-plus-sd|27.17|0.16|0.46|0.24|
|microsoft/wavlm-base-plus|27.17|0.16|0.46|0.24|
|facebook/seamless-m4t-v2-large|25.56|0.03|1.50|0.91|
|microsoft/speecht5_multimodal|26.54|0.31|1.10|0.01|
|vitouphy/wav2vec2-xls-r-300m-phoneme|26.49|0.27|0.67|0.02|
|facebook/hubert-large-ls960-ft|24.70|0.22|0.84|0.37|
|asapp/sew-d-tiny-100k-ft-ls100h|24.09|0.95|0.69|0.28|
|speechbrain/cnn14-esc50|24.63|0.00|0.04|0.00|
|microsoft/unispeech-sat-base-100h-libri-ft|24.37|0.04|0.10|0.02|
|facebook/data2vec-audio-base-960h|22.97|0.25|0.55|0.44|
|facebook/wav2vec2-xls-r-300m|24.01|0.00|0.02|0.00|
|facebook/wav2vec2-large|23.34|0.01|0.25|0.00|
|asapp/sew-d-mid-400k-ft-ls100h|22.35|0.20|0.17|0.06|
|facebook/wav2vec2-base-960h|22.57|0.06|0.07|0.00|
|asapp/sew-d-base-plus-400k-ft-ls100h|21.68|0.16|0.08|0.07|
|laion/larger_clap_music|20.49|0.00|0.00|0.00|
|facebook/wav2vec2-large-xlsr-53|19.67|0.00|0.00|0.00|
|facebook/encodec_24khz|19.56|0.00|0.00|0.00|
|facebook/wav2vec2-xls-r-2b-21-to-en|17.81|0.00|0.00|0.00|



38

<!-- page: 39 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 22._ Zero-shot classification results. Models classify audio using text descriptions without task-specific training. Best result per task in bold. 

|Model|ESC50|Ravdess|SpeechCmd v0.01|SpeechCmd v0.02|UrbanSound8k|
|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|87.80|**31.67**|**96.96**|**97.42**|67.38|
|LCO-Embedding/LCO-Embedding-Omni-3B|87.65|26.94|96.61|97.40|68.80|
|laion/larger_clap_general|**90.50**|17.29|12.39|12.44|79.64|
|microsoft/msclap-2023|89.85|15.21|9.78|10.01|**83.01**|
|laion/larger_clap_music_and_speech|82.85|17.29|9.23|9.03|76.86|
|microsoft/msclap-2022|80.25|13.61|10.09|10.51|76.17|
|laion/clap-htsat-unfused|81.70|13.26|9.58|9.35|74.41|
|laion/clap-htsat-fused|74.00|14.65|11.06|11.76|60.16|
|lyrebird/wav2clip|40.75|11.81|9.74|9.72|34.18|
|microsoft/speecht5_multimodal|1.35|12.99|19.21|18.80|8.30|
|OpenMuQ/MuQ-MuLan-large|2.70|14.72|10.95|10.48|15.33|
|Qwen/Qwen2-Audio-7B|1.00|14.37|9.89|10.38|11.82|
|laion/larger_clap_music|2.00|13.33|10.09|9.72|10.35|



39

<!-- page: 40 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 23._ Reranking results. Models are evaluated on audio-to-audio reranking tasks. Best result per task in bold. 

|Model|ESC50|FSDnoisy18k|GTZAN|UrbanSound8K|VocalSound|
|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|97.58|84.61|78.71|**79.27**|**89.94**|
|LCO-Embedding/LCO-Embedding-Omni-3B|97.16|83.19|75.37|77.93|89.28|
|microsoft/msclap-2023|**97.98**|**85.34**|75.43|76.50|72.12|
|Qwen/Qwen2-Audio-7B|94.34|68.44|80.85|70.66|80.29|
|MIT/ast-fnetuned-audioset-10-10-0.4593|96.89|77.35|77.65|75.91|65.56|
|microsoft/msclap-2022|96.28|79.11|62.91|77.96|59.69|
|laion/clap-htsat-unfused|89.47|71.21|66.45|69.14|52.02|
|laion/larger_clap_general|88.27|67.87|66.78|70.85|53.02|
|laion/larger_clap_music_and_speech|88.26|68.97|65.65|69.84|53.54|
|google/yamnet|83.59|63.59|81.73|70.47|41.89|
|laion/clap-htsat-fused|86.88|67.16|61.30|64.42|49.91|
|google/vggish|75.96|55.66|78.67|61.62|38.20|
|lyrebird/wav2clip|85.52|49.26|68.94|60.26|38.08|
|OpenMuQ/MuQ-MuLan-large|62.63|55.42|**85.41**|50.58|38.62|
|speechbrain/cnn14-esc50|83.09|51.42|53.78|55.89|46.14|
|openai/whisper-medium|72.91|44.26|67.64|52.20|46.70|
|openai/whisper-large-v3|69.53|42.96|63.93|50.58|45.98|
|openai/whisper-base|67.12|44.32|64.98|48.57|44.26|
|microsoft/wavlm-large|63.89|44.73|68.79|49.38|40.43|
|microsoft/wavlm-base-plus-sd|66.80|43.38|66.78|49.34|38.00|
|microsoft/wavlm-base-plus|66.80|43.38|66.78|49.34|38.00|
|microsoft/wavlm-base-plus-sv|66.80|43.38|66.78|49.34|38.00|
|openai/whisper-small|63.91|41.89|64.16|49.89|44.08|
|vitouphy/wav2vec2-xls-r-300m-phoneme|62.59|48.80|58.88|48.51|44.06|
|facebook/hubert-base-ls960|62.52|41.28|66.29|47.72|39.06|
|facebook/seamless-m4t-v2-large|60.11|45.09|50.21|49.41|48.44|
|microsoft/speecht5_multimodal|60.03|45.61|56.50|43.49|45.06|
|openai/whisper-tiny|57.93|42.30|63.40|46.04|40.35|
|facebook/wav2vec2-lv-60-espeak-cv-ft|58.87|47.86|55.64|43.13|41.12|
|facebook/mms-1b-f102|58.63|41.97|57.33|44.84|43.32|
|facebook/mms-1b-all|55.01|40.79|59.50|43.99|43.04|
|microsoft/wavlm-base|61.71|39.18|59.20|46.64|35.57|
|microsoft/wavlm-base-sd|61.71|39.18|59.20|46.64|35.57|
|microsoft/wavlm-base-sv|61.71|39.18|59.20|46.64|35.57|
|facebook/mms-1b-l1107|55.04|39.63|58.79|46.02|38.13|
|facebook/hubert-large-ls960-ft|56.18|41.73|53.48|45.86|39.78|
|facebook/wav2vec2-base|54.83|41.66|58.51|40.30|40.69|
|microsoft/unispeech-sat-base-100h-libri-ft|58.76|41.02|50.25|47.71|37.80|
|facebook/wav2vec2-xls-r-2b|48.89|40.17|62.87|40.73|35.54|
|facebook/data2vec-audio-large-960h|49.83|40.53|56.91|40.31|40.21|
|facebook/wav2vec2-xls-r-300m|55.56|43.39|46.78|47.27|34.36|
|facebook/encodec_24khz|55.96|41.52|51.64|40.62|34.47|
|speechbrain/m-ctc-t-large|42.43|39.69|56.18|36.52|47.16|
|asapp/sew-d-tiny-100k-ft-ls100h|51.82|37.48|51.36|43.02|37.61|
|facebook/wav2vec2-xls-r-1b|50.28|37.24|53.49|39.78|35.95|
|facebook/wav2vec2-base-960h|49.20|36.21|51.67|37.62|41.36|
|facebook/wav2vec2-large|49.90|35.66|52.35|40.56|35.42|
|facebook/data2vec-audio-base-960h|48.39|37.03|48.99|41.58|37.33|
|asapp/sew-d-mid-400k-ft-ls100h|46.83|37.49|47.81|39.77|35.12|
|facebook/wav2vec2-large-xlsr-53|46.35|38.17|46.15|39.33|33.82|
|asapp/sew-d-base-plus-400k-ft-ls100h|46.61|34.66|46.67|36.50|35.76|
|laion/larger_clap_music|40.74|37.10|48.55|38.72|34.36|



40

<!-- page: 41 -->

_Table 24._ English retrieval results (datasets 1–9 of 27). 

|Model|AudioCaps A2T|AudioCaps T2A|AudioSetStrong A2T|AudioSetStrong T2A|CMU Arctic A2T|CMU Arctic T2A|Clotho A2T|Clotho T2A|EmoVDB A2T|
|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|56.74|53.82|44.14|42.19|**99.92**|99.54|37.32|33.89|97.37|
|LCO-Embedding/LCO-Embedding-Omni-3B|54.25|46.95|41.21|34.77|99.70|**99.70**|34.74|32.70|**98.34**|
|laion/larger_clap_general|**76.22**|57.06|**87.30**|**85.55**|21.23|34.12|44.69|33.72|1.80|
|laion/larger_clap_music_and_speech|74.18|**63.14**|46.29|47.07|22.21|44.20|38.95|32.39|3.94|
|microsoft/msclap-2023|53.11|49.94|53.71|53.71|1.82|1.44|54.64|41.93|2.63|
|laion/clap-htsat-unfused|75.20|62.30|46.29|46.29|1.67|0.61|44.69|34.18|1.93|
|microsoft/msclap-2022|64.10|52.73|33.01|25.39|0.91|0.99|**71.77**|**58.73**|2.00|
|laion/clap-htsat-fused|66.03|52.73|39.65|33.59|0.61|0.91|40.57|33.57|1.93|
|OpenMuQ/MuQ-MuLan-large|1.70|2.67|6.25|5.66|0.38|0.61|2.30|2.04|0.55|
|lyrebird/wav2clip|13.14|2.52|14.65|1.76|0.46|0.61|5.55|1.15|0.41|
|microsoft/speecht5_multimodal|0.91|0.66|0.98|1.17|1.36|1.44|0.48|0.48|1.11|
|Qwen/Qwen2-Audio-7B|0.45|0.64|1.56|0.78|0.83|1.21|0.77|0.45|0.48|
|laion/larger_clap_music|0.34|0.57|1.37|0.98|0.38|0.38|0.57|0.47|0.28|



_Table 25._ English retrieval results (datasets 10–18 of 27). 

|Model|EmoVDB T2A|GigaSpeech A2T|GigaSpeech T2A|HiFiTTS A2T|HiFiTTS T2A|JLCorpus A2T JLCorpus T2A LibriTTS A2T|JLCorpus A2T JLCorpus T2A LibriTTS A2T|JLCorpus A2T JLCorpus T2A LibriTTS A2T|LibriTTS T2A|
|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|97.24|**83.11**|**83.29**|**99.67**|**100.00**|69.83|65.33|**99.96**|99.91|
|LCO-Embedding/LCO-Embedding-Omni-3B|**98.69**|82.96|83.13|**99.67**|**100.00**|**70.50**|**70.67**|**99.96**|**99.94**|
|laion/larger_clap_general|2.42|0.13|0.16|4.33|5.33|4.46|6.67|0.21|0.64|
|laion/larger_clap_music_and_speech|4.98|0.15|0.21|3.00|3.67|3.96|9.33|0.13|0.70|
|microsoft/msclap-2023|2.42|0.19|0.10|3.00|3.33|6.42|6.67|0.36|0.36|
|laion/clap-htsat-unfused|2.35|0.09|0.36|4.00|5.33|4.00|6.00|0.45|0.43|
|microsoft/msclap-2022|1.18|0.19|0.16|2.67|3.67|3.71|3.33|0.15|0.21|
|laion/clap-htsat-fused|1.93|0.15|0.18|1.33|5.00|3.21|3.33|0.43|0.45|
|OpenMuQ/MuQ-MuLan-large|0.62|0.12|0.01|2.00|2.33|3.54|3.33|0.08|0.06|
|lyrebird/wav2clip|0.35|0.07|0.09|0.67|2.33|2.04|2.67|0.13|0.17|
|microsoft/speecht5_multimodal|0.62|0.21|0.39|2.67|11.67|6.96|4.67|0.17|1.00|
|Qwen/Qwen2-Audio-7B|0.41|0.06|0.09|3.00|2.33|7.21|10.00|0.21|0.47|
|laion/larger_clap_music|0.35|0.07|0.07|1.67|1.67|3.33|2.67|0.11|0.11|



_Table 26._ English retrieval results (datasets 19–27 of 27). 

|Model|MACS A2T|MACS T2A|MusicCaps A2T MusicCaps T2A SoundDescs A2T SoundDescs T2A SpokenSQuAD T2A UrbanSound8K A2T UrbanSound8K T2A|MusicCaps A2T MusicCaps T2A SoundDescs A2T SoundDescs T2A SpokenSQuAD T2A UrbanSound8K A2T UrbanSound8K T2A|MusicCaps A2T MusicCaps T2A SoundDescs A2T SoundDescs T2A SpokenSQuAD T2A UrbanSound8K A2T UrbanSound8K T2A|MusicCaps A2T MusicCaps T2A SoundDescs A2T SoundDescs T2A SpokenSQuAD T2A UrbanSound8K A2T UrbanSound8K T2A|MusicCaps A2T MusicCaps T2A SoundDescs A2T SoundDescs T2A SpokenSQuAD T2A UrbanSound8K A2T UrbanSound8K T2A|MusicCaps A2T MusicCaps T2A SoundDescs A2T SoundDescs T2A SpokenSQuAD T2A UrbanSound8K A2T UrbanSound8K T2A|MusicCaps A2T MusicCaps T2A SoundDescs A2T SoundDescs T2A SpokenSQuAD T2A UrbanSound8K A2T UrbanSound8K T2A|
|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-7B|16.03|29.77|**20.93**|**24.83**|21.63|32.79|**74.00**|0.79|0.92|
|LCO-Embedding/LCO-Embedding-Omni-3B|18.32|22.65|19.69|18.83|18.82|29.09|72.00|0.67|0.88|
|laion/larger_clap_general|30.03|33.08|17.36|16.49|23.45|22.96|2.00|0.94|0.96|
|laion/larger_clap_music_and_speech|28.24|30.53|13.36|14.02|24.07|24.44|2.00|**1.02**|0.94|
|microsoft/msclap-2023|15.52|27.23|18.48|19.22|**38.14**|**37.28**|0.00|0.90|0.94|
|laion/clap-htsat-unfused|25.19|27.99|11.70|11.89|20.52|21.33|1.33|0.98|**0.98**|
|microsoft/msclap-2022|**40.46**|**41.73**|4.51|3.27|6.71|6.89|2.33|0.90|0.88|
|laion/clap-htsat-fused|29.52|29.26|8.48|7.54|15.99|17.81|2.00|0.59|0.94|
|OpenMuQ/MuQ-MuLan-large|1.02|1.78|12.38|11.19|0.34|0.48|1.00|0.26|0.27|
|lyrebird/wav2clip|1.78|1.78|3.88|0.61|2.85|1.44|0.00|0.37|0.18|
|microsoft/speecht5_multimodal|1.53|1.53|0.12|0.16|0.08|0.00|3.33|0.14|0.18|
|Qwen/Qwen2-Audio-7B|1.27|1.53|0.14|0.21|0.18|0.18|1.00|0.14|0.22|
|laion/larger_clap_music|1.27|1.27|0.12|0.12|0.08|0.12|4.33|0.10|0.10|

<!-- page: 42 -->

_Table 27._ FLEURS A2T retrieval results (languages 1–26 of 102). Best per language in bold. 

|Model|af|am|ar|as|ast|az|be|bg|bn|bs|ca|ceb|ckb|cmn|cs|cy|da|de|el|en|es|et|fa|ff|f|fl|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|88.3|**41.9 **|**98.8 **|**43.1 **|**100.0 **|**82.2 **|**98.3**|91.2|59.8|**87.4 **|**99.9 **|**80.8 **|**47.6 **|**100.0 **|**89.3 **|**35.5**|65.3|**100.0 **|**61.8 **|**100.0 **|**100.0 **|**57.9**|37.3|62.0|**50.4 **|**80.5**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**90.5**|37.0|**98.8**|40.5|**100.0**|75.9|97.9|**92.4 **|**60.8**|84.2|99.6|79.1|39.3|**100.0**|78.1|33.3|**68.4**|99.9|39.5|**100.0 **|**100.0**|54.1|**40.5 **|**65.2**|36.5|77.6|
|Qwen/Qwen2-Audio-7B|1.9|1.7|2.8|1.2|1.9|1.1|0.8|0.9|1.2|1.3|1.7|1.3|1.1|1.7|1.4|0.5|1.2|0.9|2.0|3.4|3.6|1.6|0.9|1.4|1.4|0.8|
|microsoft/speecht5_multimodal|2.3|1.0|2.1|1.1|0.4|0.7|0.6|1.7|0.5|0.4|1.1|1.3|1.1|1.0|1.7|1.1|1.4|0.9|1.1|1.4|0.7|0.7|0.5|1.1|1.5|0.6|
|laion/larger_clap_music_and_speech|2.7|1.4|1.4|0.2|1.9|0.8|0.4|1.5|0.5|0.9|0.7|1.7|0.5|0.5|1.0|0.8|1.2|0.3|0.6|0.9|1.2|0.6|0.6|1.1|0.5|0.3|
|laion/clap-htsat-fused|2.7|0.4|0.9|0.6|0.8|0.3|0.4|0.5|0.4|0.4|0.4|1.1|0.8|1.1|1.1|0.6|0.8|0.6|0.2|0.9|0.4|0.8|0.6|0.6|0.8|0.3|
|laion/larger_clap_general|1.9|0.8|0.9|0.7|1.2|0.8|0.6|1.2|0.5|0.5|0.7|0.6|0.5|0.3|1.4|0.5|1.1|0.7|0.9|1.5|0.9|0.3|0.9|1.1|0.8|0.6|
|laion/clap-htsat-unfused|2.3|1.2|1.2|0.6|1.1|0.3|1.0|0.8|0.5|0.5|0.6|0.7|0.5|0.4|0.8|0.3|1.2|0.5|0.6|0.5|1.4|0.9|0.5|1.4|0.4|0.5|
|lyrebird/wav2clip|1.1|1.2|1.2|0.7|0.7|0.5|0.7|0.8|0.3|0.4|0.6|1.1|0.3|0.5|1.0|0.5|1.1|0.1|1.2|0.5|1.0|0.6|0.5|0.6|0.3|0.6|
|microsoft/msclap-2022|3.0|1.6|0.5|0.5|0.2|0.5|0.8|0.6|0.4|0.4|0.5|1.1|1.0|0.6|0.7|0.6|0.5|0.7|1.2|0.5|0.6|0.7|0.3|0.8|0.1|0.3|
|OpenMuQ/MuQ-MuLan-large|2.3|0.8|0.9|0.4|0.1|0.4|0.5|0.9|0.7|0.4|0.7|1.5|0.7|0.5|0.6|0.5|1.0|0.6|0.6|1.2|0.8|0.8|0.3|0.6|0.4|0.4|
|laion/larger_clap_music|1.9|1.0|0.7|0.5|0.6|0.5|0.5|0.8|0.4|0.5|0.4|0.9|0.5|0.5|0.8|0.5|0.5|0.6|0.8|0.8|0.6|0.6|0.7|0.8|0.3|0.6|
|microsoft/msclap-2023|1.5|1.0|1.4|0.6|0.6|0.7|0.5|0.8|0.5|0.6|0.6|0.9|0.4|0.6|0.4|0.4|0.3|0.7|0.9|0.8|0.3|0.7|0.3|0.8|0.8|0.6|



_Table 28._ FLEURS A2T retrieval results (languages 27–52 of 102). Best per language in bold. 

|Model|fr|ga|gl|gu|ha|he|hi|hr|hu|hy|id|ig|is|it|ja|jv|ka|kam|kea|kk|km|kn|ko|ky|lb|lg|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**100.0 **|**29.2 **|**100.0**|91.8|**40.4 **|**38.3 **|**99.0 **|**95.1 **|**35.5 **|**36.6 **|**99.9 **|**35.5 **|**73.9 **|**100.0 **|**100.0 **|**89.3 **|**52.6**|33.9|98.0|**61.4**|16.0|64.0|**100.0 **|**75.3**|92.7|**44.5**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**100.0**|27.6|**100.0 **|**92.2**|31.1|37.2|98.8|87.7|31.7|27.5|**99.9**|32.7|54.3|**100.0 **|**100.0 **|**89.3**|47.4|**38.0 **|**98.6**|59.3|**16.5 **|**64.9 **|**100.0**|69.7|**93.3**|42.3|
|Qwen/Qwen2-Audio-7B|3.3|1.2|2.9|1.2|0.8|1.0|1.9|1.5|1.2|1.3|2.5|0.8|21.7|3.1|2.9|0.8|1.3|1.0|1.5|1.6|0.4|0.6|2.6|1.4|1.2|1.0|
|microsoft/speecht5_multimodal|0.9|1.1|1.1|1.1|1.1|1.0|2.2|0.3|0.9|0.9|1.6|0.7|15.2|1.2|0.9|1.1|0.9|1.0|0.6|0.6|0.6|0.7|1.3|0.7|1.2|0.7|
|laion/larger_clap_music_and_speech|0.9|0.8|0.9|0.3|1.0|1.1|0.7|0.7|0.7|0.4|1.2|0.5|10.9|1.0|1.1|0.5|0.4|1.0|0.8|0.8|0.4|0.7|2.1|0.4|0.6|1.0|
|laion/clap-htsat-fused|0.4|0.6|0.5|0.6|1.3|0.9|0.7|0.5|1.1|0.8|1.2|0.7|13.0|0.8|0.9|0.5|0.3|0.7|0.5|0.5|1.0|0.2|1.3|0.5|0.9|0.6|
|laion/larger_clap_general|1.0|0.9|0.5|0.5|1.0|0.6|1.2|0.9|1.4|0.6|0.9|0.6|15.2|0.7|0.9|1.0|0.6|0.8|0.7|0.8|0.6|1.1|0.8|0.5|0.5|1.0|
|laion/clap-htsat-unfused|0.9|0.6|0.9|0.8|1.0|0.1|1.0|0.7|0.8|0.5|0.7|0.9|13.0|0.5|0.8|0.4|0.5|1.0|0.6|0.8|0.6|0.7|1.8|0.2|0.1|1.4|
|lyrebird/wav2clip|0.7|0.1|0.4|0.4|0.6|0.6|0.5|0.7|0.6|0.4|0.7|0.5|15.2|0.6|0.8|0.8|0.6|0.6|1.2|0.9|0.3|0.6|1.3|0.6|0.3|0.7|
|microsoft/msclap-2022|0.6|0.4|0.3|0.7|1.1|0.6|1.2|0.5|0.8|0.5|0.7|0.7|6.5|0.5|0.6|0.4|0.5|0.7|0.6|0.5|0.6|0.6|1.6|0.4|0.3|0.8|
|OpenMuQ/MuQ-MuLan-large|0.6|0.6|0.8|0.5|0.6|0.6|1.9|0.9|0.8|0.3|0.4|0.6|8.7|0.7|0.3|0.8|0.4|0.8|0.6|0.7|0.9|1.0|1.6|0.4|0.6|0.6|
|laion/larger_clap_music|0.6|0.4|0.5|0.6|0.8|0.8|1.2|0.5|0.6|0.5|0.7|0.4|8.7|0.6|0.8|0.7|0.5|1.0|0.6|0.7|0.6|0.5|1.3|0.6|0.5|0.8|
|microsoft/msclap-2023|1.0|0.5|0.2|0.4|0.8|0.9|1.4|0.4|0.8|0.4|0.4|0.2|6.5|0.3|0.9|0.8|0.3|0.4|0.6|0.6|0.5|0.4|1.0|0.3|0.4|1.0|



_Table 29._ FLEURS A2T retrieval results (languages 53–78 of 102). Best per language in bold. 

|Model|ln|lo|lt|luo|lv|mi|mk|ml|mn|mr|ms|mt|my|nb|ne|nl|nso|ny|oc|om|or|pa|pl|ps|pt|ro|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|43.1|54.3|51.8|66.0|**57.9 **|**23.8 **|**95.3**|55.8|**31.6**|80.6|99.6|**89.5**|6.2|**92.2**|80.0|**100.0 **|**43.4 **|**43.5**|91.7|**97.6**|64.9|85.0|**98.3**|59.8|**100.0 **|**96.5**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**47.5 **|**74.6 **|**55.2 **|**68.0**|56.5|22.4|92.9|**56.5**|22.1|**82.6 **|**99.9**|89.0|**6.7**|83.8|**80.9**|99.7|43.3|41.5|**93.1**|90.2|**65.3 **|**89.9**|86.3|**60.0 **|**100.0**|93.2|
|Qwen/Qwen2-Audio-7B|0.4|4.0|0.6|1.6|2.1|0.8|1.5|0.8|0.9|0.9|1.6|1.9|0.7|4.5|1.1|2.2|0.6|0.8|1.0|14.6|1.5|1.2|1.1|1.0|3.7|1.2|
|microsoft/speecht5_multimodal|1.7|1.7|0.5|3.1|1.4|0.7|0.8|0.7|0.6|0.9|1.3|0.8|0.6|3.1|0.6|1.9|1.0|1.2|0.7|12.2|0.9|1.2|0.5|2.0|1.0|1.0|
|laion/larger_clap_music_and_speech|1.0|1.7|0.9|3.1|0.7|1.2|0.7|0.5|0.6|0.1|0.8|0.5|0.6|2.5|0.7|2.7|1.1|0.5|0.7|19.5|0.5|0.3|1.2|1.6|0.8|0.8|
|laion/clap-htsat-fused|1.5|1.0|0.4|2.0|0.7|0.4|0.2|0.7|0.4|0.2|0.7|0.5|0.1|1.4|0.4|1.4|1.3|0.8|0.8|19.5|0.6|0.7|0.5|1.0|0.4|0.9|
|laion/larger_clap_general|1.0|1.5|0.7|2.0|0.5|0.7|0.4|0.6|0.1|0.3|0.3|0.5|0.3|1.7|0.4|1.9|1.1|0.8|0.6|7.3|0.6|1.4|1.1|0.2|0.7|0.7|
|laion/clap-htsat-unfused|1.3|1.2|0.4|2.7|0.7|0.5|0.5|0.5|0.6|0.4|0.9|0.4|0.5|2.5|0.8|1.9|0.5|0.5|1.1|9.8|0.8|0.7|0.5|1.4|0.4|0.1|
|lyrebird/wav2clip|1.3|1.7|0.4|1.6|0.7|0.5|0.6|0.4|0.5|0.5|0.5|0.6|0.6|0.8|1.0|1.1|0.8|0.7|0.3|9.8|0.8|0.5|0.5|0.8|0.3|0.8|
|microsoft/msclap-2022|0.8|1.2|0.5|2.0|0.5|0.5|0.6|0.5|0.4|0.6|0.7|0.3|0.6|1.7|0.3|1.4|0.5|0.9|0.6|12.2|0.6|0.9|0.8|1.6|0.5|1.0|
|OpenMuQ/MuQ-MuLan-large|1.5|0.7|0.5|2.3|0.8|0.5|0.3|0.7|0.4|0.8|0.8|0.6|0.6|1.1|0.4|1.4|0.6|1.2|0.4|9.8|0.2|1.4|0.5|1.0|0.5|0.5|
|laion/larger_clap_music|1.0|1.0|0.5|2.0|0.5|0.5|0.4|0.6|0.6|0.4|0.5|0.4|0.5|1.4|0.7|1.1|0.6|0.7|0.5|12.2|0.6|0.7|0.7|0.8|0.5|0.5|
|microsoft/msclap-2023|1.5|1.7|0.5|2.0|0.6|0.3|0.6|0.4|0.4|0.8|0.7|0.6|0.6|1.7|0.7|1.4|0.9|0.7|0.2|12.2|0.5|0.7|0.5|0.6|0.3|0.6|

<!-- page: 43 -->

_Table 30._ FLEURS A2T retrieval results (languages 79–102 of 102). Best per language in bold. 

|Model|ru|sd|sk|sl|sn|so|sr|sv|sw|ta|te|tg|th|tr|uk|umb|ur|uz|vi|wo|xh|yo|yue|zu|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**100.0 **|**77.3 **|**96.6**|78.1|**45.1 **|**37.8 **|**89.7 **|**86.6**|44.1|**32.8 **|**85.8**|66.5|**99.7 **|**99.1**|98.3|**35.1 **|**99.0**|52.9|**99.2 **|**66.8 **|**43.1**|29.5|**100.0 **|**33.0**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**100.0**|76.6|92.2|**83.8**|43.7|33.3|89.6|78.1|**48.9**|26.2|72.0|**69.3 **|**99.7**|97.7|**98.9**|30.1|**99.0 **|**54.5**|99.1|**66.8**|39.4|**31.5 **|**100.0**|27.9|
|Qwen/Qwen2-Audio-7B|2.2|0.7|1.8|1.0|1.0|1.0|1.7|1.6|1.8|0.5|1.9|1.3|0.7|1.5|1.5|0.8|3.0|0.5|1.9|2.2|0.9|0.5|1.3|1.3|
|microsoft/speecht5_multimodal|1.3|0.9|0.6|0.5|0.8|1.2|1.0|1.3|1.8|1.4|1.5|1.2|0.6|0.7|1.2|1.1|2.0|1.3|0.6|1.6|0.7|1.1|1.5|0.7|
|laion/larger_clap_music_and_speech|0.8|0.9|1.3|1.6|0.8|0.5|0.9|1.1|0.8|1.0|1.3|1.5|0.3|0.5|0.7|0.5|2.0|0.5|1.2|1.6|0.8|0.5|0.6|0.8|
|laion/clap-htsat-fused|0.8|0.5|0.8|0.5|0.9|0.4|0.9|0.9|1.0|1.4|1.1|0.7|0.2|0.9|1.2|0.5|1.3|0.5|0.4|0.8|0.3|0.2|0.9|0.7|
|laion/larger_clap_general|0.5|0.7|1.0|0.8|0.4|0.6|0.6|0.7|1.0|1.0|1.7|0.8|0.6|0.9|0.3|1.6|0.7|0.3|1.3|1.1|0.6|0.2|0.5|1.1|
|laion/clap-htsat-unfused|0.8|0.1|0.6|0.6|0.6|0.6|0.9|0.8|1.2|0.5|0.4|1.0|0.3|0.8|0.8|1.6|2.0|0.5|0.5|1.6|0.5|0.8|0.6|0.5|
|lyrebird/wav2clip|0.9|0.6|0.6|0.5|0.5|0.2|0.4|0.7|0.8|1.0|2.1|0.8|0.6|0.7|0.4|0.3|1.0|0.7|0.5|1.3|0.7|0.8|0.4|0.9|
|microsoft/msclap-2022|0.4|0.7|0.9|0.7|0.5|0.3|0.6|0.7|1.6|1.0|1.3|0.8|0.5|0.8|0.5|2.1|2.0|0.5|0.7|1.1|0.8|0.5|0.6|0.5|
|OpenMuQ/MuQ-MuLan-large|0.6|0.6|0.9|0.6|0.4|0.5|0.9|0.9|1.2|0.3|1.1|0.7|0.5|0.4|0.9|1.6|1.0|0.8|0.7|1.6|0.5|0.4|0.9|0.8|
|laion/larger_clap_music|0.5|0.6|0.6|0.6|0.5|0.5|0.9|0.5|0.8|1.2|0.8|0.8|0.5|0.5|0.7|1.6|1.7|0.7|0.4|1.3|0.5|0.5|0.6|0.6|
|microsoft/msclap-2023|0.9|1.1|0.6|0.8|0.6|0.3|0.4|1.1|1.4|1.0|0.8|0.7|0.6|0.7|0.9|1.6|1.3|0.5|0.7|1.3|0.8|0.5|0.2|0.6|



_Table 31._ FLEURS T2A retrieval results (languages 1–26 of 102). Best per language in bold. 

|Model|af|am|ar|as|ast|az|be|bg|bn|bs|ca|ceb|ckb|cmn|cs|cy|da|de|el|en|es|et|fa|ff|f|fl|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**92.4 **|**26.9**|97.9|**47.9 **|**100.0 **|**82.8 **|**98.1**|91.9|**64.5 **|**87.1 **|**99.4 **|**88.4 **|**43.2**|99.9|**90.2**|26.1|**73.7 **|**99.9 **|**69.2 **|**100.0 **|**100.0 **|**66.9 **|**45.0 **|**65.3 **|**58.9 **|**86.5**|
|LCO-Embedding/LCO-Embedding-Omni-7B|90.2|25.6|**98.6**|42.5|99.8|76.4|97.5|**93.0**|62.9|82.2|98.7|81.9|38.0|**100.0**|79.5|**26.3**|68.5|**99.9**|44.5|**100.0 **|**100.0**|55.2|43.7|54.7|39.8|80.8|
|Qwen/Qwen2-Audio-7B|3.4|2.3|1.4|0.6|2.5|0.5|1.2|1.5|0.8|1.0|1.3|1.3|1.0|1.7|1.0|0.4|0.5|1.2|1.7|4.3|2.4|0.9|0.9|1.2|0.5|0.4|
|microsoft/speecht5_multimodal|1.5|1.4|1.2|1.4|0.4|1.1|0.7|1.5|0.9|0.6|0.5|1.1|0.7|1.0|1.4|0.7|1.2|1.2|1.1|3.4|0.6|0.9|0.9|1.1|1.0|0.5|
|laion/larger_clap_music_and_speech|3.0|1.0|1.4|0.5|1.9|0.8|0.6|0.6|0.5|1.7|1.2|0.9|0.7|0.6|1.9|0.8|0.6|1.0|0.9|1.4|1.0|1.1|0.8|0.8|0.9|1.0|
|laion/larger_clap_general|4.2|1.0|0.7|0.5|1.5|0.7|0.4|0.8|0.5|1.1|0.5|0.9|0.4|0.4|1.2|0.5|1.0|1.3|0.6|1.9|0.8|0.8|0.6|1.8|0.9|0.8|
|lyrebird/wav2clip|1.5|1.0|0.5|0.6|0.7|0.4|0.6|0.6|0.4|0.3|0.4|1.3|0.9|0.4|0.6|0.3|0.8|1.2|0.8|1.1|0.2|0.4|0.6|0.5|0.8|0.5|
|laion/clap-htsat-unfused|2.3|1.0|1.2|0.6|0.5|0.8|0.4|0.6|0.5|0.9|0.2|0.7|0.8|0.5|0.7|0.9|0.4|0.5|1.2|2.0|0.6|0.6|0.8|0.8|0.2|0.5|
|microsoft/msclap-2023|2.7|1.2|0.9|0.7|0.6|0.4|0.6|0.8|0.5|0.6|0.4|1.1|0.5|0.3|0.6|0.7|0.4|0.5|0.5|0.5|0.7|0.7|0.3|0.3|0.5|0.5|
|laion/clap-htsat-fused|3.8|0.8|1.2|0.4|0.5|0.3|0.3|0.5|0.7|0.9|1.1|1.3|0.3|0.4|0.8|0.2|0.6|1.0|0.8|0.5|0.2|0.7|0.6|0.5|0.8|0.7|
|OpenMuQ/MuQ-MuLan-large|1.1|1.0|1.6|0.3|0.6|0.4|0.6|0.5|0.4|0.4|0.7|1.5|0.4|0.6|0.7|0.3|0.5|0.5|0.5|1.2|0.7|0.9|0.3|0.8|0.4|0.2|
|microsoft/msclap-2022|2.3|1.0|1.2|0.4|0.5|0.4|0.6|0.9|0.5|0.4|0.6|0.6|0.4|0.5|0.6|0.5|0.6|0.3|1.1|0.8|0.7|0.9|0.3|0.8|0.7|0.4|
|laion/larger_clap_music|1.9|1.0|1.2|0.5|0.5|0.5|0.5|0.8|0.4|0.5|0.4|0.9|0.5|0.5|0.7|0.5|0.5|0.6|0.8|0.8|0.7|0.6|0.6|0.8|0.5|0.5|



_Table 32._ FLEURS T2A retrieval results (languages 27–52 of 102). Best per language in bold. 

|Model|fr|ga|gl|gu|ha|he|hi|hr|hu|hy|id|ig|is|it|ja|jv|ka|kam|kea|kk|km|kn|ko|ky|lb|lg|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|99.9|**29.8**|99.9|91.8|**32.5 **|**43.8**|98.3|**94.6 **|**43.1 **|**39.4 **|**99.9 **|**31.9**|71.7|**100.0**|99.7|91.5|**59.4**|30.8|97.5|**67.8**|15.4|72.1|**100.0 **|**75.1 **|**92.9 **|**38.3**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**100.0**|27.2|**100.0 **|**92.1**|31.4|42.8|**98.6**|87.3|36.2|32.2|**99.9**|23.3|**76.1 **|**100.0 **|**99.8 **|**93.4**|52.8|**33.6 **|**97.9**|59.8|**15.7 **|**73.5 **|**100.0**|70.4|92.2|36.9|
|Qwen/Qwen2-Audio-7B|2.5|0.5|2.3|1.3|0.5|1.3|1.2|1.1|0.4|0.8|1.0|0.5|19.6|2.7|0.9|0.8|0.8|0.6|0.7|0.7|0.9|1.0|2.4|0.5|0.6|1.1|
|microsoft/speecht5_multimodal|1.0|0.7|0.9|1.2|1.0|1.0|1.9|0.9|1.0|1.0|1.6|1.3|13.0|0.7|0.8|1.2|0.6|1.1|0.9|1.1|0.9|0.7|1.6|0.5|0.7|1.1|
|laion/larger_clap_music_and_speech|2.5|1.2|1.3|0.5|0.5|0.5|1.2|1.3|0.8|0.4|1.0|0.4|10.9|2.0|0.5|0.8|0.6|1.0|1.4|0.6|0.6|0.6|1.8|0.6|0.9|1.0|
|laion/larger_clap_general|1.9|1.4|0.4|0.4|0.6|0.6|1.4|1.1|1.2|0.5|0.9|0.8|17.4|1.7|1.2|1.4|0.4|1.0|0.8|0.6|0.6|0.1|1.3|0.1|1.0|1.5|
|lyrebird/wav2clip|0.6|0.6|0.9|0.5|0.8|0.6|1.7|0.3|0.6|0.4|0.6|0.4|15.2|0.3|0.6|0.7|0.4|0.7|0.7|0.5|0.5|0.7|1.6|0.5|0.5|0.4|
|laion/clap-htsat-unfused|0.9|0.8|0.4|0.5|1.0|0.5|1.2|0.4|0.6|0.8|0.3|0.4|13.0|0.8|0.9|0.7|0.5|0.8|0.7|0.7|0.4|0.7|1.3|0.7|0.2|0.3|
|microsoft/msclap-2023|0.4|0.7|0.8|0.5|0.5|0.8|1.2|0.7|0.6|0.4|0.7|0.6|13.0|0.9|0.8|1.1|0.5|0.2|0.9|0.8|0.6|0.5|1.6|0.5|0.3|1.0|
|laion/clap-htsat-fused|1.0|0.2|0.8|0.5|0.8|0.5|1.7|0.7|0.7|0.4|0.9|0.5|6.5|0.5|0.5|0.5|0.7|0.2|0.3|0.5|0.5|0.5|1.6|0.5|0.5|0.8|
|OpenMuQ/MuQ-MuLan-large|0.4|0.1|0.6|0.4|0.8|0.8|1.0|0.5|0.6|0.4|0.7|0.4|13.0|0.5|0.5|1.0|0.9|0.2|1.2|0.4|0.6|1.0|1.0|0.7|0.3|1.4|
|microsoft/msclap-2022|0.6|0.6|0.5|0.5|0.6|0.3|0.7|0.2|0.7|0.5|0.7|0.6|8.7|0.7|0.3|0.7|0.6|0.8|1.0|0.4|0.8|0.6|1.0|0.5|0.5|0.8|
|laion/larger_clap_music|0.7|0.6|0.5|0.5|1.1|0.6|1.2|0.5|0.6|0.5|0.7|0.4|10.9|0.7|0.8|0.7|0.5|0.5|0.6|0.6|0.6|0.6|1.3|0.5|0.5|0.7|

<!-- page: 44 -->

_Table 33._ FLEURS T2A retrieval results (languages 53–78 of 102). Best per language in bold. 

|Model|ln|lo|lt|luo|lv|mi|mk|ml|mn|mr|ms|mt|my|nb|ne|nl|nso|ny|oc|om|or|pa|pl|ps|pt|ro|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**45.6**|40.7|**63.2 **|**73.0 **|**66.4 **|**23.2 **|**95.0 **|**65.2 **|**24.1 **|**82.5 **|**99.6 **|**89.1 **|**5.2 **|**93.8 **|**82.0 **|**99.7 **|**40.0 **|**38.2 **|**91.0 **|**73.2**|62.2|83.4|**97.0 **|**63.5 **|**100.0 **|**95.7**|
|LCO-Embedding/LCO-Embedding-Omni-7B|43.7|**64.0**|58.2|65.2|58.6|21.0|92.2|64.2|21.3|80.5|**99.6**|86.5|4.3|84.9|80.2|**99.7**|31.9|35.5|90.4|70.7|**62.5 **|**87.3**|84.2|59.0|**100.0**|91.4|
|Qwen/Qwen2-Audio-7B|1.0|2.0|0.6|2.3|0.9|0.8|0.3|1.1|0.7|0.3|1.2|0.5|0.9|3.1|1.0|1.4|0.5|0.1|0.5|17.1|1.5|1.0|0.9|1.2|2.5|1.2|
|microsoft/speecht5_multimodal|1.5|1.5|0.8|3.9|0.5|0.5|0.8|0.5|0.4|0.6|0.7|0.6|0.8|2.5|1.5|2.2|1.0|1.2|1.0|19.5|0.7|0.9|1.3|1.2|1.2|0.9|
|laion/larger_clap_music_and_speech|2.1|1.5|1.1|2.3|1.3|1.0|0.6|0.6|0.4|0.6|0.8|1.1|0.5|1.7|1.0|2.5|0.9|0.7|0.6|12.2|0.7|0.5|2.4|1.2|2.6|1.5|
|laion/larger_clap_general|1.9|1.5|1.3|3.1|0.5|1.0|0.5|0.3|0.3|0.3|0.8|1.3|0.7|2.0|0.4|2.2|1.0|1.1|0.7|7.3|0.3|0.5|1.2|0.8|1.3|1.4|
|lyrebird/wav2clip|0.8|1.2|0.3|2.3|0.5|0.3|0.5|0.6|0.6|0.4|0.4|0.4|0.5|1.7|0.7|2.2|0.8|0.7|0.6|17.1|0.6|1.0|0.7|1.0|0.3|0.7|
|laion/clap-htsat-unfused|1.5|1.2|0.0|2.7|0.6|0.3|0.3|0.3|0.6|0.4|1.1|0.5|0.5|2.0|0.8|1.1|1.3|0.9|0.4|12.2|0.7|0.5|0.9|1.2|0.5|0.5|
|microsoft/msclap-2023|1.5|1.0|0.6|2.7|0.2|0.8|0.7|0.5|0.6|0.6|1.6|0.4|0.8|1.1|0.4|1.4|0.9|1.3|0.7|12.2|0.6|0.9|0.9|1.6|0.7|0.5|
|laion/clap-htsat-fused|2.1|1.2|1.0|2.3|0.4|0.5|0.7|0.8|0.5|0.4|1.5|0.3|0.5|1.4|0.6|1.6|0.6|1.2|0.8|17.1|0.8|0.5|0.7|0.6|0.7|0.8|
|OpenMuQ/MuQ-MuLan-large|0.6|1.0|0.7|3.1|0.4|0.7|0.4|0.4|0.5|0.6|0.5|0.6|0.7|1.7|0.8|1.1|0.8|0.5|0.7|12.2|0.5|0.9|0.7|1.6|0.4|1.0|
|microsoft/msclap-2022|1.5|1.7|0.6|2.3|0.9|0.8|0.4|0.5|0.5|0.7|0.7|0.8|0.8|2.5|0.4|1.9|0.6|0.5|1.0|12.2|0.6|0.9|0.8|1.0|1.0|0.9|
|laion/larger_clap_music|1.0|1.2|0.5|1.6|0.6|0.5|0.5|0.5|0.5|0.4|0.7|0.5|0.6|1.1|0.7|1.4|0.5|0.7|0.5|12.2|0.6|0.9|0.7|1.0|0.4|0.6|



_Table 34._ FLEURS T2A retrieval results (languages 79–102 of 102). Best per language in bold. 

|Model|ru|sd|sk|sl|sn|so|sr|sv|sw|ta|te|tg|th|tr|uk|umb|ur|uz|vi|wo|xh|yo|yue|zu|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**100.0 **|**75.8 **|**95.6 **|**83.2 **|**35.5 **|**32.1 **|**91.1 **|**83.7**|45.0|**36.2 **|**85.2**|66.3|99.3|**99.2 **|**98.3 **|**36.7 **|**100.0 **|**56.8**|99.0|**58.0 **|**33.6 **|**33.8**|99.9|**20.8**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**100.0**|74.8|91.5|**83.2**|32.3|21.6|91.0|75.4|**47.4**|33.3|74.8|**68.0 **|**99.7**|98.0|97.5|29.8|99.0|53.8|**99.2**|56.1|28.8|31.9|**100.0**|19.3|
|Qwen/Qwen2-Audio-7B|1.4|0.4|0.9|1.0|0.5|0.7|1.3|0.5|2.1|0.8|1.7|1.2|0.7|1.1|1.6|1.6|2.0|0.5|1.3|1.9|0.8|1.1|0.7|0.8|
|microsoft/speecht5_multimodal|1.2|0.6|1.0|1.2|0.6|0.4|1.7|1.3|1.6|1.4|1.3|1.2|0.5|0.5|0.8|1.8|2.3|0.8|0.5|1.3|1.0|1.0|1.1|1.2|
|laion/larger_clap_music_and_speech|0.8|0.5|1.3|1.0|0.4|0.8|2.4|1.7|1.2|0.7|1.1|0.8|0.5|0.9|1.2|1.3|1.7|0.5|0.8|1.3|0.8|0.7|0.6|0.9|
|laion/larger_clap_general|0.8|0.5|1.6|0.8|0.8|0.5|2.0|1.3|0.8|1.0|0.8|0.8|0.4|1.8|0.8|2.1|1.3|0.8|0.9|1.6|0.8|0.5|0.7|1.1|
|lyrebird/wav2clip|0.6|0.5|0.6|0.6|0.5|0.4|0.7|0.4|1.6|0.8|1.1|0.8|0.7|0.5|0.9|1.1|2.0|0.6|0.6|1.6|0.5|0.6|0.6|0.7|
|laion/clap-htsat-unfused|0.5|0.6|0.6|1.0|0.6|0.8|1.0|0.3|0.8|0.7|1.3|0.8|0.4|1.1|0.7|0.8|1.7|0.7|0.7|1.3|0.9|0.7|0.7|0.6|
|microsoft/msclap-2023|0.4|0.3|0.9|0.6|0.4|0.8|0.6|0.9|1.2|1.0|1.3|0.8|0.4|0.8|0.7|1.1|2.0|0.5|0.4|1.3|0.4|0.7|0.4|0.7|
|laion/clap-htsat-fused|0.8|0.4|0.8|1.0|0.5|0.4|0.3|1.3|1.2|0.3|0.8|1.2|0.2|0.4|0.5|2.4|2.0|0.5|0.6|0.8|0.7|0.7|0.7|0.8|
|OpenMuQ/MuQ-MuLan-large|1.2|0.5|0.5|0.7|0.3|0.7|0.9|0.4|1.0|1.2|1.5|0.8|0.8|1.3|0.7|1.6|2.3|0.6|0.8|1.3|0.6|0.4|0.5|0.6|
|microsoft/msclap-2022|0.5|0.7|0.5|0.8|0.2|0.3|0.4|0.5|0.6|1.0|1.1|0.5|0.4|0.9|0.5|1.3|2.3|0.6|0.2|1.1|0.3|0.5|0.4|0.5|
|laion/larger_clap_music|0.6|0.5|0.6|0.6|0.5|0.4|0.7|0.7|1.0|0.8|1.1|1.0|0.6|0.7|0.7|1.1|1.7|0.6|0.6|1.3|0.5|0.6|0.5|0.6|

<!-- page: 45 -->

## **MAEB: Massive Audio Embedding Benchmark** 

_Table 35._ JamAlt A2T retrieval results (languages 1–4 of 4). Best per language in bold. 

|Model|de|en|es|fr|
|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**73.5 **|**76.4 **|**81.4 **|**72.5**|
|LCO-Embedding/LCO-Embedding-Omni-7B|66.1|71.1|77.2|71.8|
|microsoft/msclap-2023|0.9|0.7|1.8|1.0|
|Qwen/Qwen2-Audio-7B|0.7|0.8|1.1|1.5|
|microsoft/speecht5_multimodal|0.7|1.7|0.8|0.8|
|laion/larger_clap_music_and_speech|1.0|0.5|0.8|1.6|
|laion/larger_clap_music|1.7|0.7|0.6|0.8|
|laion/larger_clap_general|0.9|0.7|0.9|1.1|
|laion/clap-htsat-fused|0.9|1.0|0.4|1.1|
|laion/clap-htsat-unfused|0.6|0.6|1.1|0.9|
|microsoft/msclap-2022|0.5|0.9|0.5|0.8|
|lyrebird/wav2clip|0.4|0.5|0.4|1.0|
|OpenMuQ/MuQ-MuLan-large|0.7|0.8|0.4|0.2|



_Table 36._ JamAlt T2A retrieval results (languages 1–4 of 4). Best per language in bold. 

|Model|de|en|es|fr|
|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**75.6 **|**79.6 **|**81.7**|75.4|
|LCO-Embedding/LCO-Embedding-Omni-7B|72.4|78.6|80.8|**76.0**|
|Qwen/Qwen2-Audio-7B|1.2|2.0|1.7|1.8|
|microsoft/speecht5_multimodal|1.1|3.0|1.0|0.9|
|laion/larger_clap_music_and_speech|1.5|1.3|0.6|1.0|
|laion/clap-htsat-unfused|0.8|1.5|0.9|0.7|
|laion/larger_clap_general|0.8|1.2|0.8|1.0|
|microsoft/msclap-2023|0.8|1.1|0.6|0.7|
|laion/clap-htsat-fused|0.5|0.8|0.7|1.0|
|lyrebird/wav2clip|0.8|0.5|0.7|0.8|
|microsoft/msclap-2022|0.5|0.5|0.8|0.9|
|laion/larger_clap_music|0.7|0.6|0.7|0.7|
|OpenMuQ/MuQ-MuLan-large|0.6|0.7|0.3|0.3|



45

<!-- page: 46 -->

**MAEB: Massive Audio Embedding Benchmark** 

_Table 37._ JamAlt A2A retrieval results (languages 1–4 of 4). Best per language in bold. 

|Model|de|en|es|fr|
|---|---|---|---|---|
|laion/larger_clap_music_and_speech|**99.1**|94.3|**97.9 **|**96.3**|
|OpenMuQ/MuQ-MuLan-large|97.6|**95.2**|97.6|94.7|
|laion/larger_clap_general|97.9|93.2|96.8|94.6|
|MIT/ast-fnetuned-audioset-10-10-0.4593|97.7|90.2|96.2|94.4|
|laion/clap-htsat-unfused|97.0|88.8|94.2|91.8|
|microsoft/msclap-2023|96.5|87.3|94.9|92.5|
|google/vggish|92.4|83.8|90.5|89.2|
|laion/clap-htsat-fused|92.8|82.8|89.0|86.1|
|microsoft/msclap-2022|92.3|82.4|89.4|84.6|
|facebook/encodec_24khz|88.8|75.8|83.0|88.1|
|google/yamnet|84.0|74.5|82.3|78.3|
|microsoft/wavlm-large|82.8|71.8|80.8|78.7|
|lyrebird/wav2clip|86.0|68.9|81.0|78.1|
|speechbrain/cnn14-esc50|85.8|72.3|74.2|75.7|
|facebook/hubert-base-ls960|81.6|70.7|79.5|76.2|
|LCO-Embedding/LCO-Embedding-Omni-7B|76.2|78.2|76.2|69.5|
|openai/whisper-large-v3|76.3|69.1|71.1|74.9|
|microsoft/wavlm-base-plus|77.2|66.0|73.7|71.9|
|microsoft/wavlm-base-plus-sv|77.2|66.0|73.7|71.9|
|microsoft/wavlm-base-plus-sd|77.2|66.0|73.7|71.9|
|openai/whisper-medium|75.7|69.5|70.6|72.6|
|laion/larger_clap_music|82.3|64.5|69.3|69.4|
|openai/whisper-small|73.7|66.2|68.5|71.4|
|openai/whisper-base|75.5|64.5|67.2|71.8|
|openai/whisper-tiny|75.3|62.7|67.5|71.1|
|facebook/wav2vec2-base|74.5|56.9|69.9|70.5|
|LCO-Embedding/LCO-Embedding-Omni-3B|68.7|67.7|69.9|62.3|
|facebook/wav2vec2-large|71.3|52.4|64.1|66.4|
|facebook/wav2vec2-xls-r-300m|72.0|56.0|62.1|62.2|
|microsoft/wavlm-base|64.9|58.6|65.6|61.3|
|microsoft/wavlm-base-sv|64.6|58.7|65.0|61.4|
|microsoft/wavlm-base-sd|64.6|58.7|65.0|61.4|
|facebook/wav2vec2-xls-r-2b|71.3|53.7|62.8|61.1|
|vitouphy/wav2vec2-xls-r-300m-phoneme|65.0|56.2|61.9|63.9|
|microsoft/speecht5_multimodal|62.8|55.6|60.3|59.5|
|facebook/wav2vec2-xls-r-1b|67.2|53.1|60.6|57.4|
|facebook/seamless-m4t-v2-large|60.0|52.4|56.3|54.6|
|facebook/mms-1b-f102|59.7|53.0|54.1|52.9|
|facebook/hubert-large-ls960-ft|57.7|50.5|50.6|50.2|
|facebook/mms-1b-l1107|53.5|50.3|52.7|52.2|
|facebook/mms-1b-all|57.7|48.8|50.8|50.6|
|microsoft/unispeech-sat-base-100h-libri-ft|54.4|49.2|52.0|50.4|
|facebook/wav2vec2-lv-60-espeak-cv-ft|55.4|46.9|52.4|46.7|
|facebook/wav2vec2-large-xlsr-53|53.7|44.1|48.4|46.5|
|asapp/sew-d-tiny-100k-ft-ls100h|49.0|43.8|49.2|45.6|
|speechbrain/m-ctc-t-large|51.0|41.8|47.2|47.4|
|facebook/data2vec-audio-base-960h|48.4|44.6|49.0|44.1|
|facebook/wav2vec2-base-960h|49.1|43.6|46.9|43.1|
|facebook/data2vec-audio-large-960h|43.4|41.5|43.9|41.1|
|asapp/sew-d-mid-400k-ft-ls100h|39.9|38.6|40.9|37.2|
|asapp/sew-d-base-plus-400k-ft-ls100h|38.5|37.0|38.7|36.2|
|Qwen/Qwen2-Audio-7B|42.7|33.9|35.8|37.9|
|facebook/wav2vec2-xls-r-2b-21-to-en|7.6|9.7|4.7|5.8|



46

<!-- page: 47 -->

_Table 38._ CommonVoiceMini17 A2T retrieval results (languages 1–13 of 50). Best per language in bold. 

|Model|ar|ast|be|bg|bn|br|cs|cy|da|de|el|en|es|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|94.4|98.3|**99.4 **|**69.4 **|**58.0 **|**67.4 **|**86.6 **|**38.4 **|**56.6 **|**100.0 **|**46.4 **|**99.8 **|**100.0**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**94.8 **|**99.2 **|**99.4**|66.2|56.6|56.6|70.2|27.2|49.0|99.8|21.6|99.6|**100.0**|
|Qwen/Qwen2-Audio-7B|2.6|8.4|2.8|2.8|2.0|1.4|2.6|1.4|2.2|4.4|2.6|4.2|6.2|
|microsoft/speecht5_multimodal|1.2|8.4|0.2|1.0|0.8|1.6|1.8|1.0|1.6|1.2|0.6|1.0|2.0|
|laion/clap-htsat-unfused|1.0|3.4|1.4|1.4|1.6|2.0|2.2|1.2|1.2|1.2|1.4|1.4|0.4|
|microsoft/msclap-2023|0.8|4.2|1.4|1.0|0.8|0.8|1.8|1.2|1.6|0.8|1.6|1.2|1.6|
|microsoft/msclap-2022|1.2|7.6|1.4|1.2|1.2|1.2|0.8|1.4|1.2|1.0|1.2|1.2|1.4|
|laion/clap-htsat-fused|1.8|5.9|0.6|0.4|0.8|1.4|1.0|1.4|0.8|1.0|1.2|1.4|1.0|
|OpenMuQ/MuQ-MuLan-large|1.0|6.7|1.0|1.0|1.2|0.6|1.6|1.2|0.8|1.4|1.0|0.4|1.0|
|laion/larger_clap_general|0.4|6.7|1.6|1.4|1.4|1.0|0.4|1.6|2.0|0.8|1.0|1.2|0.2|
|lyrebird/wav2clip|1.2|3.4|1.0|1.4|1.2|1.0|0.6|1.2|1.4|0.8|1.4|1.4|1.0|
|laion/larger_clap_music_and_speech|0.8|5.9|0.4|1.0|1.4|1.4|1.8|1.4|1.2|1.8|0.8|1.6|0.4|
|laion/larger_clap_music|1.0|4.2|1.0|1.0|0.8|1.0|1.2|1.0|0.8|1.0|1.0|1.0|1.2|



_Table 39._ CommonVoiceMini17 A2T retrieval results (languages 14–26 of 50). Best per language in bold. 

|Model|et|fa|f|fr|frold|gl|ha|hi|hu|it|ja|ka|ko|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**46.8 **|**25.2 **|**39.0 **|**99.8 **|**99.8 **|**100.0 **|**61.6**|94.0|**50.0 **|**99.6**|95.2|**61.2 **|**99.1**|
|LCO-Embedding/LCO-Embedding-Omni-7B|39.4|21.4|24.2|99.4|99.4|**100.0**|51.2|**94.4**|42.4|**99.6 **|**95.8**|54.4|98.8|
|Qwen/Qwen2-Audio-7B|2.6|2.8|2.0|4.6|4.6|3.6|2.4|1.4|2.0|6.0|3.4|2.8|4.4|
|microsoft/speecht5_multimodal|1.4|1.4|3.2|2.4|2.4|1.8|1.8|0.8|1.8|2.0|1.2|1.2|1.2|
|laion/clap-htsat-unfused|1.0|1.0|1.2|0.6|0.6|1.4|2.1|1.4|1.2|0.8|1.4|1.2|2.4|
|microsoft/msclap-2023|1.0|1.2|2.0|1.6|1.6|1.0|1.5|1.0|1.0|1.4|1.0|1.6|2.1|
|microsoft/msclap-2022|1.2|1.6|1.2|0.4|0.4|0.8|1.2|0.2|0.8|1.0|1.0|1.2|1.8|
|laion/clap-htsat-fused|1.0|1.4|0.6|0.8|0.8|1.8|2.4|1.6|1.8|0.6|2.2|0.8|1.2|
|OpenMuQ/MuQ-MuLan-large|1.4|0.2|1.0|0.6|0.6|0.8|1.5|0.8|1.2|0.6|1.2|0.6|1.8|
|laion/larger_clap_general|0.6|0.4|0.6|1.2|1.2|1.2|1.8|0.6|0.8|1.2|1.0|1.0|1.5|
|lyrebird/wav2clip|1.0|0.8|1.0|0.6|0.6|1.2|1.8|0.8|1.2|1.2|1.0|0.8|2.1|
|laion/larger_clap_music_and_speech|1.0|0.4|1.2|0.8|0.8|0.8|2.4|1.0|1.6|1.8|1.0|1.0|0.6|
|laion/larger_clap_music|1.0|1.0|0.8|1.0|1.0|0.8|1.5|0.8|1.0|1.0|0.8|1.0|1.5|

<!-- page: 48 -->

_Table 40._ CommonVoiceMini17 A2T retrieval results (languages 27–39 of 50). Best per language in bold. 

|Model|lt|lv|mk|ml|mn|mr|nl|oc|pl|pt|ro|ru|sk|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**62.8 **|**43.4 **|**76.2 **|**57.8 **|**20.2**|78.6|**99.0 **|**90.9 **|**88.4 **|**99.0 **|**86.0 **|**99.2 **|**82.2**|
|LCO-Embedding/LCO-Embedding-Omni-7B|59.6|38.8|72.9|40.8|11.8|**81.2 **|**99.0**|90.2|66.2|98.8|78.8|**99.2**|80.2|
|Qwen/Qwen2-Audio-7B|1.8|2.2|3.3|2.2|2.0|2.0|2.8|5.1|2.6|2.4|2.6|3.0|2.4|
|microsoft/speecht5_multimodal|2.4|2.0|1.5|0.8|1.0|2.4|2.2|2.0|1.8|1.6|1.0|0.6|1.8|
|laion/clap-htsat-unfused|0.8|1.4|1.5|1.6|1.2|1.4|0.8|2.4|1.8|0.6|1.0|1.6|1.0|
|microsoft/msclap-2023|1.4|0.4|1.5|1.4|0.8|1.2|1.0|2.4|1.8|1.2|1.4|1.4|0.8|
|microsoft/msclap-2022|1.2|1.2|0.9|1.0|1.2|1.4|0.6|2.0|1.0|1.6|1.2|0.6|0.8|
|laion/clap-htsat-fused|0.8|1.2|2.1|0.8|0.6|1.6|1.4|3.5|1.4|1.4|0.8|1.6|2.0|
|OpenMuQ/MuQ-MuLan-large|1.0|1.0|1.5|1.0|1.2|1.2|1.2|2.4|1.2|0.8|1.2|0.8|1.0|
|laion/larger_clap_general|0.4|1.8|2.4|1.2|1.0|0.8|1.0|3.1|2.0|1.8|1.2|0.6|1.8|
|lyrebird/wav2clip|1.6|1.2|1.5|1.6|1.0|1.4|1.2|1.6|1.4|1.4|1.0|1.2|0.8|
|laion/larger_clap_music_and_speech|1.4|0.8|3.0|0.4|0.8|1.8|0.6|3.1|1.0|0.6|0.8|1.2|0.6|
|laion/larger_clap_music|1.0|0.8|1.2|0.8|1.0|1.0|1.0|2.0|1.0|1.0|1.0|1.0|1.0|



_Table 41._ CommonVoiceMini17 A2T retrieval results (languages 40–50 of 50). Best per language in bold. 

|Model|sl|sr|sv|sw|ta|te|th|tr|uk|ur|vi|
|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|59.8|**58.6 **|**68.2 **|**53.0 **|**42.4 **|**91.8 **|**92.0 **|**72.2 **|**86.6**|82.2|84.4|
|LCO-Embedding/LCO-Embedding-Omni-7B|**63.0**|54.0|46.8|49.4|21.6|59.2|**92.0**|63.4|**86.6 **|**84.6 **|**86.6**|
|Qwen/Qwen2-Audio-7B|1.2|1.8|2.2|2.0|2.0|16.3|1.4|1.2|2.2|1.2|1.6|
|microsoft/speecht5_multimodal|1.0|0.4|1.8|2.4|0.6|6.1|0.8|0.8|1.6|1.0|1.0|
|laion/clap-htsat-unfused|0.8|1.2|1.0|1.4|1.0|12.2|1.2|2.0|1.2|1.2|1.4|
|microsoft/msclap-2023|2.0|2.0|0.4|1.4|1.4|10.2|1.4|1.2|1.2|1.2|0.8|
|microsoft/msclap-2022|0.6|1.8|1.0|1.4|1.2|14.3|1.2|1.6|1.8|1.2|0.6|
|laion/clap-htsat-fused|0.8|1.0|1.2|1.6|0.8|10.2|0.4|1.2|1.4|1.0|1.4|
|OpenMuQ/MuQ-MuLan-large|1.2|0.8|0.2|1.0|2.0|16.3|1.8|1.4|1.2|0.6|1.6|
|laion/larger_clap_general|1.8|0.6|1.6|0.6|1.0|10.2|1.8|1.6|1.6|0.6|0.6|
|lyrebird/wav2clip|1.0|1.4|1.0|1.0|1.0|10.2|0.8|0.8|1.0|1.2|0.6|
|laion/larger_clap_music_and_speech|0.6|0.6|1.8|1.6|0.2|4.1|1.2|1.2|0.8|1.2|1.2|
|laion/larger_clap_music|1.2|0.8|1.0|1.0|1.0|10.2|1.0|1.0|1.0|1.0|1.2|

<!-- page: 49 -->

_Table 42._ CommonVoiceMini17 T2A retrieval results (languages 1–13 of 50). Best per language in bold. 

|Model|ar|ast|be|bg|bn|br|cs|cy|da|de|el|en|es|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|94.2|**99.2**|98.8|**73.2 **|**58.2 **|**51.0 **|**87.4 **|**20.0 **|**63.2 **|**99.8 **|**41.2**|99.6|**100.0**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**95.0**|97.5|**99.0**|65.6|49.4|47.0|72.6|16.2|56.0|99.6|19.4|**99.8 **|**100.0**|
|Qwen/Qwen2-Audio-7B|1.6|8.4|3.0|1.0|2.2|1.8|2.0|2.8|2.2|3.0|1.0|5.2|5.6|
|microsoft/speecht5_multimodal|1.0|9.2|1.0|1.4|1.4|2.0|1.8|1.6|1.4|1.6|0.8|2.0|1.4|
|microsoft/msclap-2023|0.8|10.1|1.0|1.6|1.0|1.6|1.2|1.8|1.6|1.6|1.0|0.8|1.6|
|laion/clap-htsat-fused|0.6|4.2|1.6|1.8|0.6|1.2|1.6|1.6|1.6|1.8|1.2|1.2|0.8|
|laion/clap-htsat-unfused|1.0|6.7|0.8|1.2|1.2|1.4|1.4|1.4|1.4|1.2|1.4|1.0|0.2|
|microsoft/msclap-2022|1.8|6.7|1.2|1.0|1.0|1.0|0.8|0.8|0.6|0.8|0.8|1.6|1.6|
|laion/larger_clap_general|0.8|7.6|1.0|1.0|1.2|1.6|1.2|0.8|1.4|0.6|1.6|1.4|0.6|
|laion/larger_clap_music_and_speech|1.0|4.2|1.2|1.0|0.8|0.8|1.6|0.8|1.6|2.6|0.8|0.8|0.8|
|OpenMuQ/MuQ-MuLan-large|1.6|3.4|1.4|0.8|0.4|1.0|1.2|1.4|1.0|0.8|1.8|0.4|0.6|
|lyrebird/wav2clip|1.4|5.0|0.8|0.8|0.8|1.2|1.0|0.8|1.2|0.8|1.2|0.8|0.6|
|laion/larger_clap_music|1.0|4.2|1.0|1.0|1.0|1.0|1.0|1.0|1.0|1.2|0.8|1.0|1.2|



_Table 43._ CommonVoiceMini17 T2A retrieval results (languages 14–26 of 50). Best per language in bold. 

|Model|et|fa|f|fr|frold|gl|ha|hi|hu|it|ja|ka|ko|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**46.6**|25.6|**41.6 **|**99.6 **|**99.6 **|**99.8 **|**49.7 **|**92.8 **|**48.8**|99.6|95.0|**58.8 **|**98.8**|
|LCO-Embedding/LCO-Embedding-Omni-7B|41.8|**26.0**|22.0|99.4|99.4|**99.8**|40.9|91.8|42.2|**99.8 **|**96.2**|50.6|98.5|
|Qwen/Qwen2-Audio-7B|2.8|1.4|0.8|4.2|4.2|4.4|2.4|1.6|1.2|6.4|1.4|2.4|4.4|
|microsoft/speecht5_multimodal|2.0|1.0|1.8|1.0|1.0|1.4|1.8|1.2|1.4|1.2|1.0|0.8|0.9|
|microsoft/msclap-2023|1.8|1.2|1.4|1.2|1.2|0.8|0.6|1.2|1.4|1.2|0.8|0.8|2.1|
|laion/clap-htsat-fused|1.2|1.0|1.8|1.4|1.4|1.2|1.5|1.0|1.2|0.4|1.2|1.0|1.8|
|laion/clap-htsat-unfused|1.4|0.6|1.4|1.2|1.2|1.6|1.5|1.4|0.8|1.2|2.6|1.0|2.1|
|microsoft/msclap-2022|1.4|1.0|0.6|1.4|1.4|0.6|3.7|0.8|0.6|1.2|1.2|0.6|1.2|
|laion/larger_clap_general|0.6|1.0|0.6|1.0|1.0|0.4|1.2|1.0|1.2|1.2|0.8|1.2|1.2|
|laion/larger_clap_music_and_speech|0.4|1.4|1.8|2.4|2.4|1.0|1.2|1.0|0.6|0.8|1.0|1.0|1.2|
|OpenMuQ/MuQ-MuLan-large|1.2|1.2|1.6|1.0|1.0|0.8|0.9|1.0|0.6|1.0|1.0|1.0|2.1|
|lyrebird/wav2clip|1.0|1.4|1.0|1.0|1.0|0.8|1.8|1.0|1.4|1.2|1.2|1.0|0.9|
|laion/larger_clap_music|1.2|1.0|1.0|1.0|1.0|1.0|1.5|1.0|1.2|1.0|1.0|0.8|1.8|

<!-- page: 50 -->

_Table 44._ CommonVoiceMini17 T2A retrieval results (languages 27–39 of 50). Best per language in bold. 

|Model|lt|lv|mk|ml|mn|mr|nl|oc|pl|pt|ro|ru|sk|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**66.6 **|**47.8 **|**78.3 **|**52.2 **|**11.6 **|**80.8 **|**98.6 **|**91.7 **|**88.6 **|**98.4 **|**84.0 **|**99.2 **|**80.8**|
|LCO-Embedding/LCO-Embedding-Omni-7B|55.6|40.4|75.0|32.4|6.6|77.0|98.4|90.2|66.4|98.0|73.8|99.0|79.0|
|Qwen/Qwen2-Audio-7B|1.8|1.4|3.6|2.8|1.6|1.4|2.8|6.3|1.6|3.4|2.8|2.2|2.8|
|microsoft/speecht5_multimodal|2.0|2.0|1.8|1.0|0.8|0.8|1.6|1.6|1.6|1.4|1.8|1.4|1.0|
|microsoft/msclap-2023|1.6|1.2|2.1|1.4|1.4|1.0|1.2|2.4|2.0|1.0|1.4|1.0|2.0|
|laion/clap-htsat-fused|1.4|2.0|1.8|0.8|1.4|0.8|1.2|2.8|1.4|2.0|0.8|1.8|2.4|
|laion/clap-htsat-unfused|0.6|1.6|1.5|1.2|0.4|1.0|1.2|3.5|1.8|2.2|1.4|1.0|1.8|
|microsoft/msclap-2022|1.6|0.6|1.8|1.8|0.8|1.2|1.0|2.0|0.8|1.0|0.8|1.6|0.6|
|laion/larger_clap_general|1.4|1.0|1.5|0.8|1.4|1.0|0.2|3.1|1.0|1.2|0.6|1.6|2.0|
|laion/larger_clap_music_and_speech|1.4|0.4|2.1|0.6|1.0|1.0|1.2|2.4|0.6|1.2|2.0|1.4|1.8|
|OpenMuQ/MuQ-MuLan-large|0.8|1.2|2.1|1.2|1.0|0.8|1.0|2.0|0.8|0.8|0.8|1.4|1.0|
|lyrebird/wav2clip|1.0|0.8|1.5|1.2|0.8|1.2|1.2|2.8|1.2|1.0|1.0|0.8|1.2|
|laion/larger_clap_music|1.0|0.8|1.5|1.0|1.0|1.0|1.0|2.0|1.0|0.8|0.8|1.0|1.0|



_Table 45._ CommonVoiceMini17 T2A retrieval results (languages 40–50 of 50). Best per language in bold. 

|Model|sl|sr|sv|sw|ta|te|th|tr|uk|ur|vi|
|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|65.2|**49.8 **|**67.0 **|**47.0 **|**29.8 **|**73.5**|90.4|**72.2 **|**91.4**|80.8|81.8|
|LCO-Embedding/LCO-Embedding-Omni-7B|**66.4**|44.6|50.8|41.2|23.2|61.2|**90.8**|63.6|86.6|**83.4 **|**84.6**|
|Qwen/Qwen2-Audio-7B|1.4|1.0|2.4|1.2|0.8|14.3|1.6|1.2|2.2|2.4|1.6|
|microsoft/speecht5_multimodal|1.0|1.2|1.4|1.8|1.0|14.3|0.8|1.4|0.8|0.8|1.2|
|microsoft/msclap-2023|2.0|1.8|1.2|0.6|0.8|8.2|1.4|0.8|1.2|0.8|0.8|
|laion/clap-htsat-fused|0.6|0.2|1.2|1.4|1.6|10.2|1.4|1.8|0.8|1.0|1.8|
|laion/clap-htsat-unfused|0.4|1.4|1.0|1.4|0.8|8.2|1.2|1.6|1.2|1.0|1.4|
|microsoft/msclap-2022|0.4|1.6|1.4|0.6|2.0|12.2|1.0|1.0|1.6|0.6|1.8|
|laion/larger_clap_general|1.6|0.8|1.0|1.0|1.0|12.2|1.2|0.8|1.0|1.0|1.0|
|laion/larger_clap_music_and_speech|1.0|1.0|1.8|0.8|1.2|8.2|0.8|1.0|0.8|1.2|0.8|
|OpenMuQ/MuQ-MuLan-large|1.2|0.8|1.2|1.6|1.2|12.2|1.2|1.8|1.6|0.8|0.8|
|lyrebird/wav2clip|1.0|1.6|0.8|1.0|1.0|10.2|1.0|1.0|1.4|1.4|1.0|
|laion/larger_clap_music|1.0|0.8|1.0|1.0|1.0|10.2|1.0|1.0|1.0|1.0|1.0|

<!-- page: 51 -->

_Table 46._ CommonVoiceMini21 A2T retrieval results (languages 1–29 of 114). Best per language in bold. 

|Model|ab|af|am|ar|as|ast|az|ba|bas|be|bg|bn|br|ca|ckb|cnh|cs|cv|cy|da|dav|de|dv|dyu|el|en|eo|es|et|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**53.5 **|**85.5 **|**55.0**|92.5|**41.5 **|**99.4 **|**83.7 **|**52.0 **|**74.5 **|**99.0 **|**83.0 **|**71.5 **|**79.0 **|**98.0 **|**65.5 **|**66.0 **|**83.5 **|**76.5 **|**45.5 **|**63.0 **|**63.5 **|**99.5 **|**14.0 **|**93.7 **|**51.5 **|**98.5 **|**91.0 **|**100.0 **|**52.0**|
|LCO-Embedding/LCO-Embedding-Omni-7B|44.0|**85.5**|35.5|**93.0**|26.5|97.5|81.5|46.5|69.5|98.0|79.0|64.5|66.5|96.5|47.5|62.5|69.0|65.5|33.5|56.0|53.5|**99.5**|5.5|88.9|30.5|**98.5**|88.5|**100.0**|49.0|
|Qwen/Qwen2-Audio-7B|5.0|7.7|5.0|4.5|7.0|9.6|12.0|3.0|5.5|4.0|4.0|4.0|3.5|6.5|6.0|3.5|5.0|4.5|5.5|5.0|4.0|8.0|4.5|6.3|6.0|8.5|6.5|9.5|5.0|
|microsoft/speecht5_multimodal|2.5|6.0|3.0|3.0|3.5|5.7|7.6|3.0|3.0|3.0|2.0|4.5|2.0|4.0|2.0|4.5|3.0|2.5|5.0|7.0|2.0|4.0|1.5|9.5|1.5|6.0|4.5|2.5|4.0|
|laion/clap-htsat-unfused|2.5|5.1|1.5|3.5|2.0|3.2|4.3|4.5|4.0|1.5|3.5|2.5|6.0|4.5|3.0|3.5|3.5|1.5|3.5|3.0|4.0|3.5|2.5|12.7|3.0|4.0|3.5|3.0|2.5|
|laion/clap-htsat-fused|3.0|5.1|2.0|2.5|1.5|5.1|2.2|3.0|3.5|1.5|3.5|1.5|3.5|4.0|1.5|4.5|3.0|3.5|3.5|2.0|1.5|1.5|1.5|9.5|4.0|3.5|4.0|1.5|2.5|
|OpenMuQ/MuQ-MuLan-large|2.5|4.3|4.0|3.0|2.5|3.8|5.4|2.0|4.5|4.0|2.5|3.0|3.5|2.5|1.5|2.5|3.0|3.0|5.0|3.5|1.5|2.5|1.5|11.1|3.0|1.5|2.0|3.0|2.5|
|microsoft/msclap-2023|2.0|5.1|1.5|1.5|3.0|3.8|7.6|2.0|3.0|2.5|2.5|3.0|3.0|3.0|2.5|3.0|2.0|2.5|4.5|3.0|2.5|3.0|1.5|7.9|2.5|3.5|3.0|1.0|2.0|
|laion/larger_clap_general|3.5|1.7|2.0|2.5|2.0|3.8|6.5|2.0|2.5|3.0|1.0|3.0|3.5|3.0|3.0|3.5|4.0|4.5|2.0|3.5|5.5|2.0|1.5|6.3|2.5|1.0|3.0|5.5|3.5|
|microsoft/msclap-2022|3.0|4.3|2.5|3.0|3.0|3.8|7.6|3.0|3.5|2.0|2.5|2.5|3.0|2.5|3.0|3.5|2.0|1.5|2.5|2.5|2.0|3.5|3.0|11.1|2.5|2.5|3.5|1.5|2.0|
|laion/larger_clap_music|2.5|4.3|2.5|2.5|2.5|3.2|5.4|3.0|2.5|2.5|2.5|2.5|2.5|2.0|2.0|2.5|2.5|2.0|2.5|3.0|2.5|2.5|2.5|7.9|2.0|3.0|2.5|2.5|2.0|
|lyrebird/wav2clip|1.0|4.3|2.0|1.5|0.5|3.2|5.4|4.5|3.0|0.5|3.0|2.5|4.0|2.5|2.5|3.5|1.5|2.5|2.0|2.0|1.5|2.0|3.0|7.9|3.0|2.0|2.5|3.0|3.0|
|laion/larger_clap_music_and_speech|2.0|6.0|1.5|1.5|1.0|3.8|4.3|1.5|3.0|2.0|2.5|1.5|1.5|3.0|2.0|3.0|1.0|2.5|3.0|0.5|1.5|5.5|2.5|12.7|2.0|4.0|2.0|1.5|2.0|



_Table 47._ CommonVoiceMini21 A2T retrieval results (languages 30–58 of 114). Best per language in bold. 

|Model|eu|fa|f|fr|fy|ga|gl|gn|ha|he|hi|hsb|hu|hy|ia|id|it|ja|ka|kab|kk|kln|kmr|ko|ky|lg|lij|lt|ltg|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**75.0 **|**38.5 **|**53.5 **|**100.0 **|**93.0 **|**41.0 **|**100.0 **|**76.0 **|**69.5 **|**40.0 **|**97.0**|72.5|**56.0 **|**49.5 **|**99.5 **|**96.5 **|**100.0**|93.0|**72.0 **|**52.5 **|**51.0 **|**68.0 **|**75.0 **|**99.5 **|**61.0 **|**57.5 **|**95.5**|61.2|**67.0**|
|LCO-Embedding/LCO-Embedding-Omni-7B|69.0|33.5|33.0|**100.0**|92.0|35.5|**100.0**|58.0|55.5|31.5|96.0|**73.5**|45.5|36.5|**99.5**|95.5|**100.0 **|**94.5**|63.0|42.5|31.5|57.0|59.0|98.0|57.0|46.0|92.5|**63.6**|56.4|
|Qwen/Qwen2-Audio-7B|4.0|5.0|5.5|5.0|4.5|3.0|9.5|5.5|3.5|3.0|5.5|7.5|3.5|4.0|6.0|5.5|8.5|5.5|4.0|3.5|4.5|5.0|5.5|3.5|4.0|5.0|6.0|1.8|2.2|
|microsoft/speecht5_multimodal|2.0|2.0|7.0|5.5|6.0|3.5|4.0|4.0|5.5|3.0|2.5|4.5|4.0|3.5|4.5|5.0|4.5|3.0|2.5|2.5|3.0|3.5|4.5|1.5|2.5|3.5|5.0|2.4|1.8|
|laion/clap-htsat-unfused|4.5|1.5|3.0|3.0|3.0|2.5|2.0|4.0|4.0|2.0|1.0|4.5|3.5|2.5|4.0|2.5|2.5|3.5|2.5|4.0|1.5|2.0|2.5|2.5|2.0|3.5|2.5|1.2|2.4|
|laion/clap-htsat-fused|3.0|2.5|3.0|5.0|3.0|1.5|3.0|3.0|2.5|3.0|2.0|1.5|3.5|3.0|5.0|3.5|3.0|4.0|2.5|5.5|2.5|3.5|4.0|2.5|2.0|2.5|2.5|1.4|2.0|
|OpenMuQ/MuQ-MuLan-large|3.0|3.5|2.5|2.5|3.0|2.0|2.0|4.0|2.5|2.0|3.0|3.0|4.5|3.5|3.5|5.5|3.5|2.5|3.5|4.5|3.0|2.5|3.0|1.5|1.5|2.5|3.5|1.8|0.8|
|microsoft/msclap-2023|3.5|2.5|2.0|4.0|2.5|4.0|3.5|3.0|3.0|2.5|2.0|2.5|3.5|3.0|3.5|4.0|1.5|3.0|0.5|3.0|3.5|2.5|3.5|3.5|1.5|3.0|3.0|1.2|1.0|
|laion/larger_clap_general|3.0|1.5|5.0|3.0|1.5|4.0|2.5|3.0|1.5|3.0|3.0|2.5|3.5|1.0|1.5|3.0|1.5|4.5|2.0|1.5|3.5|3.5|3.0|2.0|2.5|3.0|3.0|1.2|0.8|
|microsoft/msclap-2022|3.0|2.0|3.0|4.0|2.5|3.0|3.0|2.0|2.0|2.5|3.0|4.0|3.0|3.0|2.5|3.5|2.0|3.5|2.5|1.0|3.0|2.5|4.0|3.0|3.0|3.0|2.0|0.8|1.6|
|laion/larger_clap_music|2.5|2.5|2.0|2.0|2.5|3.0|3.0|2.5|2.5|3.0|3.0|2.5|3.0|2.5|2.5|3.5|2.5|2.5|2.5|2.0|2.5|2.5|2.5|2.5|3.0|2.0|2.5|1.0|1.0|
|lyrebird/wav2clip|2.5|2.0|3.0|2.0|2.0|3.5|2.5|3.0|3.5|3.0|3.0|3.0|2.0|1.5|3.5|3.5|3.5|1.0|1.5|1.5|3.0|2.5|1.5|2.0|3.0|2.0|3.5|1.2|1.0|
|laion/larger_clap_music_and_speech|2.0|1.0|3.5|2.0|2.5|1.0|2.0|2.0|2.5|2.0|2.5|1.0|1.5|1.0|4.5|1.5|3.0|2.5|2.0|3.0|4.5|3.0|2.5|2.5|2.0|3.5|2.5|1.4|1.0|



_Table 48._ CommonVoiceMini21 A2T retrieval results (languages 59–87 of 114). Best per language in bold. 

|Model|luo|lv|mdf|mhr|mk|ml|mn|mr|mrj|mt|myv|nan|ne|nl|nn|oc|or|os|pa|pl|ps|pt|rm|ro|ru|rw|sah|sat|sc|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**86.0 **|**40.6 **|**92.5 **|**79.0 **|**78.2 **|**55.8 **|**16.2**|78.0|**74.4 **|**80.0 **|**88.0**|59.2|**61.4**|98.8|**66.5 **|**90.9 **|**63.1 **|**66.9**|76.6|**87.0 **|**50.8**|98.6|**98.4 **|**85.0 **|**99.0 **|**46.5 **|**57.5**|4.4|**94.5**|
|LCO-Embedding/LCO-Embedding-Omni-7B|75.0|35.8|86.0|65.4|74.0|39.8|10.4|**79.0**|57.0|77.2|81.3|**61.8**|48.5|**99.4**|53.6|89.4|58.8|59.2|**78.6**|66.6|47.6|**98.8**|97.9|76.6|**99.0**|36.5|43.0|6.2|93.5|
|Qwen/Qwen2-Audio-7B|4.5|2.0|7.5|1.8|1.0|2.6|2.0|1.4|3.0|2.2|3.2|1.2|3.7|1.6|1.7|4.7|3.3|7.7|1.6|2.2|2.0|3.8|3.5|2.4|4.2|4.0|5.0|**8.8**|5.0|
|microsoft/speecht5_multimodal|4.0|1.0|4.7|1.2|0.8|1.0|1.0|0.4|1.4|1.4|1.9|0.6|2.2|2.6|1.9|1.8|0.7|1.5|0.8|1.0|1.0|2.0|3.0|1.2|1.0|3.5|2.0|5.3|7.0|
|laion/clap-htsat-unfused|4.5|1.0|8.4|1.4|0.4|2.0|1.6|1.2|0.6|1.4|0.8|0.8|2.2|0.8|1.5|2.2|1.4|3.8|1.0|1.0|1.4|1.2|2.5|0.8|0.4|4.0|3.0|5.3|2.5|
|laion/clap-htsat-fused|3.5|1.6|5.6|2.2|2.0|1.0|1.8|0.8|1.8|0.8|2.4|1.6|2.9|1.0|1.9|2.2|2.1|5.4|1.6|1.0|1.2|1.6|0.5|1.4|1.0|3.0|4.0|3.5|4.0|
|OpenMuQ/MuQ-MuLan-large|2.5|0.8|3.7|0.4|1.0|1.2|1.0|0.8|0.8|1.4|1.3|0.8|1.5|0.6|1.7|1.8|0.7|6.9|1.2|0.6|1.2|1.0|0.9|1.0|1.2|2.5|3.5|3.5|3.5|
|microsoft/msclap-2023|3.5|1.2|8.4|1.4|1.2|1.0|1.0|1.0|1.6|1.4|0.8|0.6|1.8|1.4|1.2|1.8|1.2|5.4|1.0|1.8|1.4|1.6|0.9|1.6|1.2|2.0|1.5|3.5|2.5|
|laion/larger_clap_general|3.0|1.8|3.7|1.0|1.4|0.8|1.8|0.0|1.4|1.4|1.9|0.6|2.2|1.8|1.0|2.9|1.7|6.2|0.6|1.0|0.6|1.0|1.2|1.4|0.8|2.5|4.5|5.3|3.5|
|microsoft/msclap-2022|2.0|1.2|5.6|0.8|1.4|1.2|0.8|0.8|1.2|1.6|1.9|1.0|2.6|1.0|1.5|1.5|1.4|4.6|1.4|1.4|0.8|1.0|1.6|1.4|1.2|3.0|2.5|4.4|2.5|
|laion/larger_clap_music|3.0|0.8|4.7|1.0|1.0|1.0|0.8|1.0|1.0|0.8|1.6|1.0|1.5|1.0|1.2|1.8|1.0|3.8|0.8|1.0|1.2|1.0|1.2|0.8|1.0|2.5|2.0|6.2|2.5|
|lyrebird/wav2clip|3.0|0.8|6.5|0.8|0.8|1.4|0.8|1.4|0.6|1.4|1.9|1.2|1.5|1.0|0.7|2.2|1.4|3.1|1.4|1.0|1.0|1.2|1.4|1.2|1.0|2.0|3.0|1.8|3.0|
|laion/larger_clap_music_and_speech|3.5|1.2|5.6|2.2|1.4|0.2|0.6|1.0|1.2|0.8|1.3|0.6|1.1|0.2|1.7|2.9|1.4|3.8|0.6|1.2|1.6|1.8|2.1|1.2|1.0|1.5|3.0|3.5|1.0|

<!-- page: 52 -->

_Table 49._ CommonVoiceMini21 A2T retrieval results (languages 88–114 of 114). Best per language in bold. 

|Model|sk|skr|sl|sq|sr|sv|sw|ta|te|th|tig|tk|tn|tok|tr|tt|ug|uk|ur|uz|vi|yi|yo|yue|zgh|zh|zza|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**87.5**|61.0|70.0|**79.0**|62.0|**78.0**|52.0|**56.5 **|**90.2**|93.5|**59.0 **|**55.5 **|**49.5 **|**85.5 **|**88.0 **|**54.0 **|**66.5**|93.0|**91.5 **|**66.5**|86.5|61.5|**49.0**|98.5|62.0|**99.5 **|**69.0**|
|LCO-Embedding/LCO-Embedding-Omni-7B|82.5|**62.0 **|**70.5**|74.0|**63.5**|65.5|**56.0**|28.5|63.9|**95.0**|40.5|41.5|35.5|83.0|75.5|42.0|58.5|**94.0**|90.5|55.0|**89.0 **|**62.2**|33.0|**99.0 **|**66.0 **|**99.5**|58.5|
|Qwen/Qwen2-Audio-7B|6.5|2.5|3.5|5.0|5.5|5.0|5.0|5.0|18.0|5.0|3.0|5.0|4.0|4.5|3.5|4.5|4.0|6.5|6.5|3.5|4.0|4.2|2.0|6.5|3.5|4.5|4.5|
|microsoft/speecht5_multimodal|4.0|2.5|4.5|3.0|3.0|4.0|4.5|4.0|8.2|3.0|3.5|3.0|4.5|3.0|5.0|3.0|1.5|2.5|4.5|2.0|1.0|2.1|1.0|2.5|2.5|2.5|3.0|
|laion/clap-htsat-unfused|3.0|4.0|2.5|4.5|4.5|3.5|2.5|2.5|8.2|2.0|3.0|2.5|1.0|3.0|3.5|2.5|2.0|4.5|1.5|3.5|2.0|4.9|4.0|2.5|2.0|2.5|2.5|
|laion/clap-htsat-fused|3.0|2.5|3.0|3.0|3.0|1.5|2.5|2.0|9.8|1.0|2.5|2.5|3.0|3.5|2.0|3.5|3.0|5.0|3.0|3.5|3.0|2.8|2.0|2.5|2.5|3.0|2.5|
|OpenMuQ/MuQ-MuLan-large|1.5|2.5|2.5|1.0|2.0|2.0|1.5|2.5|9.8|2.5|4.5|2.5|3.5|3.0|4.0|4.0|3.0|2.0|3.5|2.5|2.5|4.2|3.0|3.0|4.0|2.0|2.5|
|microsoft/msclap-2023|2.5|3.0|4.0|3.0|3.5|3.5|2.5|2.0|6.6|2.5|2.5|2.0|2.0|3.0|2.5|4.5|3.0|4.0|1.5|2.0|3.0|3.5|3.5|2.0|2.5|3.0|4.0|
|laion/larger_clap_general|4.5|1.5|3.5|2.0|2.5|2.5|2.5|1.5|6.6|3.5|1.5|3.0|2.5|4.0|3.0|2.0|3.0|1.5|2.5|0.5|3.5|0.7|3.0|5.0|2.5|4.5|1.5|
|microsoft/msclap-2022|2.5|3.0|1.5|3.0|2.5|5.0|2.5|2.0|3.3|2.0|3.0|2.0|2.0|3.0|4.0|3.5|2.5|2.5|2.0|1.0|3.0|2.8|1.5|2.5|2.0|2.5|2.5|
|laion/larger_clap_music|2.5|2.0|2.5|2.5|3.0|2.5|2.5|3.0|8.2|2.5|3.0|2.5|2.5|2.5|2.5|2.5|3.0|2.0|2.5|2.0|2.5|4.9|3.0|2.0|2.5|2.5|2.5|
|lyrebird/wav2clip|1.5|2.0|1.5|3.5|3.0|2.5|3.5|2.5|8.2|4.0|2.5|1.5|2.5|2.0|3.0|4.0|2.5|4.0|2.5|3.0|2.0|2.8|1.5|3.0|2.0|1.5|2.5|
|laion/larger_clap_music_and_speech|2.0|2.5|1.0|3.0|3.5|3.0|3.5|2.0|4.9|2.5|2.5|2.5|4.0|3.0|4.0|2.5|1.5|3.5|3.0|2.0|2.0|2.8|2.5|1.5|2.0|2.0|3.5|



_Table 50._ CommonVoiceMini21 T2A retrieval results (languages 1–29 of 114). Best per language in bold. 

|Model|ab|af|am|ar|as|ast|az|ba|bas|be|bg|bn|br|ca|ckb|cnh|cs|cv|cy|da|dav|de|dv|dyu|el|en|eo|es|et|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**22.0 **|**83.8**|22.0|**92.0 **|**42.0 **|**99.4 **|**83.7**|29.0|**58.0 **|**99.5 **|**84.0 **|**69.0 **|**60.0 **|**99.0 **|**45.5 **|**48.5 **|**86.0 **|**55.0**|23.0|**74.0 **|**51.0 **|**99.5 **|**6.5 **|**87.3 **|**50.5 **|**99.0 **|**96.0 **|**100.0 **|**53.5**|
|LCO-Embedding/LCO-Embedding-Omni-7B|19.5|**83.8 **|**22.5**|91.0|28.0|96.2|79.3|**29.5**|48.0|98.0|75.5|62.0|54.0|98.0|32.5|42.5|72.5|53.5|**28.5**|60.5|33.0|**99.5**|3.5|81.0|23.5|98.5|91.5|**100.0**|52.0|
|Qwen/Qwen2-Audio-7B|5.5|10.3|6.5|4.0|2.5|7.6|5.4|3.5|4.0|5.5|5.5|2.5|5.0|7.5|6.0|4.5|3.0|2.0|2.5|2.5|4.0|6.5|2.0|7.9|4.0|11.0|4.5|8.0|2.5|
|microsoft/speecht5_multimodal|1.5|7.7|2.5|2.5|3.0|5.1|6.5|2.5|3.0|3.5|2.0|2.5|4.5|3.5|1.5|7.5|2.5|2.0|4.5|5.5|3.5|4.0|2.5|14.3|2.5|5.5|3.0|2.5|3.5|
|laion/clap-htsat-unfused|3.0|5.1|2.0|1.5|2.5|3.2|3.3|3.5|3.5|3.0|3.0|3.0|4.5|5.5|3.0|3.0|1.5|2.0|3.5|0.5|4.5|5.0|2.5|11.1|2.5|2.0|3.5|3.5|3.5|
|laion/clap-htsat-fused|2.5|6.0|1.5|1.5|4.0|2.5|2.2|2.0|5.0|1.5|2.0|2.5|3.5|3.5|2.5|3.0|3.5|1.0|1.5|2.5|4.5|1.0|2.0|15.9|3.5|2.0|5.0|1.5|3.5|
|microsoft/msclap-2023|2.0|6.0|2.5|2.0|2.0|3.8|3.3|3.0|3.0|2.5|2.5|3.0|3.5|3.5|2.0|4.0|3.0|2.5|2.5|3.0|1.5|3.5|2.0|12.7|1.5|3.0|3.0|2.5|3.0|
|microsoft/msclap-2022|3.0|6.0|2.5|2.5|2.0|3.8|7.6|3.0|3.0|2.0|4.0|1.5|1.0|3.0|3.0|3.5|4.5|2.5|3.0|2.5|3.5|2.5|4.0|9.5|2.5|2.0|3.0|4.0|2.0|
|OpenMuQ/MuQ-MuLan-large|2.0|6.0|4.0|2.0|2.0|3.8|5.4|1.0|3.0|2.5|1.5|3.0|2.5|2.0|3.5|2.5|2.5|4.5|1.5|2.5|3.5|3.0|2.5|9.5|4.0|3.0|2.0|3.0|5.0|
|laion/larger_clap_general|3.5|2.6|3.0|3.0|2.0|4.5|6.5|1.0|3.0|1.5|2.5|2.0|4.0|2.5|2.5|2.0|3.0|3.5|3.0|2.5|3.5|2.0|1.5|6.3|4.5|2.0|2.0|2.5|4.0|
|laion/larger_clap_music_and_speech|4.5|3.4|2.5|1.5|2.5|5.1|9.8|2.5|2.5|1.0|3.0|2.0|2.0|1.5|2.5|3.0|4.0|2.5|3.0|2.0|1.5|3.5|2.0|4.8|3.5|3.5|1.5|1.5|1.5|
|lyrebird/wav2clip|2.5|4.3|2.5|1.5|2.0|5.1|5.4|3.5|3.0|3.0|3.0|3.0|1.5|4.0|3.0|2.5|2.5|2.5|1.5|3.0|1.5|3.0|3.0|7.9|2.5|2.5|3.0|2.0|1.5|
|laion/larger_clap_music|2.5|3.4|3.0|2.0|2.5|3.2|5.4|2.5|2.5|2.5|2.5|2.5|2.5|2.5|2.5|2.5|3.0|2.5|2.5|3.0|2.5|2.5|2.5|9.5|2.5|2.5|3.0|2.5|2.5|



_Table 51._ CommonVoiceMini21 T2A retrieval results (languages 30–58 of 114). Best per language in bold. 

|Model|eu|fa|f|fr|fy|ga|gl|gn|ha|he|hi|hsb|hu|hy|ia|id|it|ja|ka|kab|kk|kln|kmr|ko|ky|lg|lij|lt|ltg|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**58.5 **|**33.5 **|**54.5 **|**100.0 **|**90.5 **|**34.0**|99.5|**60.0 **|**55.0 **|**45.0 **|**95.5 **|**77.5 **|**53.5 **|**47.0 **|**99.0**|94.5|**100.0 **|**93.5 **|**69.5 **|**38.0 **|**25.5 **|**48.5 **|**53.0 **|**98.5 **|**44.5 **|**31.5 **|**94.5 **|**68.4 **|**57.2**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**58.5**|30.5|33.5|**100.0**|89.5|20.5|**100.0**|47.5|44.5|29.5|94.5|72.0|48.5|27.5|98.5|**96.0 **|**100.0**|92.0|62.0|35.5|20.0|36.5|43.5|97.5|43.0|24.5|90.5|58.4|47.0|
|Qwen/Qwen2-Audio-7B|4.0|2.0|4.5|5.0|4.0|3.5|8.5|4.0|5.0|2.0|5.0|4.0|4.5|3.5|7.0|4.5|8.0|2.0|4.0|3.5|3.5|3.5|4.5|5.0|4.0|7.0|2.5|1.6|1.8|
|microsoft/speecht5_multimodal|3.0|2.5|6.0|4.0|3.5|1.5|3.5|4.0|3.0|3.0|3.0|4.5|2.5|3.0|5.0|5.0|5.0|2.5|3.0|3.5|2.0|4.5|3.0|2.5|3.0|4.5|3.0|1.6|1.2|
|laion/clap-htsat-unfused|2.0|2.5|4.0|3.0|5.0|1.5|2.5|4.5|2.5|3.0|3.0|3.0|3.5|2.5|3.0|2.5|3.5|4.0|2.5|3.5|2.0|2.0|3.5|2.5|2.5|2.5|5.5|1.0|1.4|
|laion/clap-htsat-fused|4.0|2.5|3.5|3.5|4.0|4.0|3.0|5.5|2.0|2.0|1.5|5.5|2.0|2.5|2.5|4.0|3.0|4.0|2.5|4.5|2.5|2.5|2.5|4.0|3.0|1.5|5.0|1.6|1.0|
|microsoft/msclap-2023|3.5|3.5|3.0|3.0|3.5|4.5|1.5|4.0|3.5|2.0|2.5|3.5|2.5|2.0|4.5|3.5|2.0|2.5|3.0|2.0|4.5|2.0|2.5|2.5|1.0|3.0|4.5|1.2|2.2|
|microsoft/msclap-2022|2.5|3.5|3.5|4.0|4.5|1.5|3.5|3.0|3.0|2.5|3.0|1.5|3.0|2.5|4.5|3.0|1.5|2.5|3.5|2.0|3.0|1.5|3.5|3.0|4.0|2.5|4.5|0.8|1.6|
|OpenMuQ/MuQ-MuLan-large|2.0|3.5|3.0|1.5|4.0|2.5|2.0|2.0|1.5|2.5|3.0|3.0|2.5|4.0|2.0|2.0|2.5|3.0|1.5|3.5|1.0|1.5|2.0|4.5|3.0|2.0|3.0|1.2|1.4|
|laion/larger_clap_general|3.5|2.0|3.5|3.5|3.5|2.5|5.0|3.5|2.5|2.5|2.5|3.0|3.0|3.0|3.0|1.5|2.0|2.0|2.0|4.0|3.0|2.0|3.0|2.0|2.5|3.0|2.5|0.8|0.6|
|laion/larger_clap_music_and_speech|1.5|2.0|3.0|1.5|2.5|1.0|2.5|4.0|3.5|3.5|1.5|3.0|3.0|2.0|4.0|3.5|1.5|3.5|2.0|3.0|4.0|4.0|3.0|2.0|2.0|2.5|2.5|1.4|1.4|
|lyrebird/wav2clip|2.0|2.0|3.5|1.5|4.0|2.5|4.0|2.0|3.0|2.5|2.5|2.5|2.5|2.0|2.5|2.0|2.5|3.0|2.5|2.5|2.5|2.0|2.0|2.5|1.5|3.0|3.0|1.2|1.0|
|laion/larger_clap_music|2.5|2.5|2.5|2.5|2.5|2.5|2.5|2.0|2.0|2.5|2.5|2.5|2.5|2.5|2.5|3.0|3.0|2.5|2.5|3.0|2.5|2.5|2.5|2.5|2.0|2.5|2.5|1.0|0.8|

<!-- page: 53 -->

_Table 52._ CommonVoiceMini21 T2A retrieval results (languages 59–87 of 114). Best per language in bold. 

|Model|luo|lv|mdf|mhr|mk|ml|mn|mr|mrj|mt|myv|nan|ne|nl|nn|oc|or|os|pa|pl|ps|pt|rm|ro|ru|rw|sah|sat|sc|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|61.5|**47.0 **|**76.6 **|**50.4 **|**78.2 **|**49.0 **|**8.0 **|**77.6 **|**43.4 **|**74.6 **|**68.3**|60.8|**50.7 **|**98.4 **|**67.5 **|**91.2**|56.0|**45.4**|67.6|**89.4 **|**38.2**|97.8|**96.8 **|**79.2 **|**98.6 **|**26.5**|25.5|3.5|**94.5**|
|LCO-Embedding/LCO-Embedding-Omni-7B|**62.0**|38.8|63.6|41.0|71.8|32.6|5.4|76.0|33.4|70.4|61.1|**63.8**|39.7|98.2|51.9|87.2|**61.9**|33.1|**70.0**|63.8|35.6|**98.2**|95.4|69.8|98.4|23.0|**27.0**|3.5|94.0|
|Qwen/Qwen2-Audio-7B|4.0|1.6|4.7|1.6|1.6|3.4|1.8|0.8|1.8|1.4|2.1|1.6|3.7|2.6|2.4|4.0|1.4|5.4|1.8|1.0|1.4|4.4|3.0|2.6|4.0|3.5|3.0|5.3|3.5|
|microsoft/speecht5_multimodal|6.5|1.6|4.7|0.8|1.2|1.0|0.4|1.0|1.0|1.8|1.1|1.6|1.8|2.4|1.9|1.1|1.0|2.3|1.0|1.2|1.2|1.2|1.8|1.8|1.0|4.0|1.5|4.4|4.0|
|laion/clap-htsat-unfused|3.5|1.2|8.4|0.6|1.4|1.2|0.6|1.4|1.4|1.2|1.9|1.6|1.8|1.2|2.2|2.6|1.2|4.6|0.8|2.4|1.4|2.2|1.2|1.6|1.0|3.5|2.0|**7.1**|5.0|
|laion/clap-htsat-fused|3.5|2.0|5.6|1.2|1.0|0.6|1.4|1.0|2.0|2.0|1.3|1.8|2.6|0.8|2.7|1.8|0.7|10.0|1.4|1.0|0.6|1.6|1.4|1.4|1.6|2.0|2.0|2.7|4.5|
|microsoft/msclap-2023|3.5|2.0|6.5|1.4|0.8|1.0|0.8|1.0|0.6|1.2|1.6|1.4|1.8|1.2|1.5|2.2|1.7|2.3|1.0|1.2|0.8|1.4|1.6|2.0|1.0|6.5|2.5|3.5|2.5|
|microsoft/msclap-2022|3.5|1.6|6.5|0.4|1.0|1.6|1.0|1.4|1.2|1.2|1.6|0.8|2.2|1.6|1.2|1.5|1.7|4.6|0.8|0.8|0.8|2.2|1.8|1.6|1.4|1.5|0.5|4.4|2.5|
|OpenMuQ/MuQ-MuLan-large|1.5|1.0|7.5|0.4|0.2|0.8|0.8|0.8|0.2|1.0|1.1|1.2|2.6|0.8|2.4|2.6|1.0|3.8|1.2|1.4|1.2|1.2|1.4|1.4|2.0|3.5|3.5|4.4|1.5|
|laion/larger_clap_general|2.5|1.4|2.8|1.0|1.2|1.0|0.8|1.0|1.2|1.2|1.3|1.2|1.5|0.8|1.2|1.5|1.4|3.1|1.0|0.4|1.4|1.0|1.4|1.0|1.2|4.0|2.0|4.4|3.0|
|laion/larger_clap_music_and_speech|1.5|1.2|5.6|1.4|1.2|0.8|0.8|0.8|0.8|1.6|2.1|1.0|1.8|1.2|1.7|1.5|1.0|3.8|1.0|1.4|1.2|1.2|0.9|1.0|1.0|3.5|2.5|5.3|2.0|
|lyrebird/wav2clip|2.5|0.8|4.7|1.2|1.0|1.0|1.0|0.8|0.6|1.4|1.3|1.0|1.8|1.4|1.2|2.9|1.2|4.6|1.2|1.2|1.4|1.0|1.4|1.0|1.2|3.0|2.5|4.4|3.0|
|laion/larger_clap_music|2.0|1.0|4.7|1.0|1.0|1.0|1.0|1.0|1.0|1.0|1.3|1.0|1.8|1.0|1.2|1.8|1.2|3.1|1.2|1.0|1.2|0.8|1.2|1.0|1.0|2.5|2.5|4.4|2.5|



_Table 53._ CommonVoiceMini21 T2A retrieval results (languages 88–114 of 114). Best per language in bold. 

|Model|sk|skr|sl|sq|sr|sv|sw|ta|te|th|tig|tk|tn|tok|tr|tt|ug|uk|ur|uz|vi|yi|yo|yue|zgh|zh|zza|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|LCO-Embedding/LCO-Embedding-Omni-3B|**89.0 **|**48.0**|74.5|**72.0 **|**55.5 **|**74.0 **|**52.5 **|**40.5 **|**77.0**|92.5|**30.0 **|**38.0 **|**29.5 **|**67.5 **|**83.0**|32.5|**45.0 **|**93.5**|88.5|**50.0**|86.0|44.8|**25.5 **|**99.5**|26.5|**99.0 **|**63.0**|
|LCO-Embedding/LCO-Embedding-Omni-7B|84.5|45.0|**75.0**|71.0|52.5|63.0|51.5|24.0|57.4|**94.5**|24.0|32.5|20.5|62.5|76.5|**33.0**|41.5|92.5|**90.5**|43.5|**87.5 **|**52.4**|10.5|**99.5 **|**30.0 **|**99.0**|50.5|
|Qwen/Qwen2-Audio-7B|2.5|3.5|4.5|5.0|6.5|4.5|5.0|2.5|14.8|3.0|3.5|3.5|5.0|4.5|3.5|2.0|5.0|6.0|4.5|4.0|5.0|4.9|1.5|6.0|3.5|6.5|2.5|
|microsoft/speecht5_multimodal|2.5|2.5|2.0|3.5|2.0|3.0|3.5|1.0|8.2|2.5|3.0|4.5|5.5|7.5|4.5|1.5|3.0|1.0|4.0|2.5|3.5|4.2|2.5|2.5|1.5|2.5|5.5|
|laion/clap-htsat-unfused|3.0|2.5|5.0|5.0|1.5|4.5|2.5|2.0|6.6|2.5|2.5|3.0|2.5|4.5|3.5|4.5|2.5|4.5|3.0|1.0|2.0|4.2|2.0|7.5|2.5|3.0|2.0|
|laion/clap-htsat-fused|3.0|3.0|4.0|3.0|4.5|2.0|2.5|1.0|8.2|2.5|1.5|2.5|3.5|5.0|4.5|1.5|3.0|2.5|1.5|4.0|2.5|5.6|2.0|2.5|1.5|2.5|5.5|
|microsoft/msclap-2023|3.5|2.0|5.5|3.0|3.0|3.5|3.0|2.5|6.6|2.0|2.5|6.5|3.0|4.5|2.5|4.0|3.5|2.5|2.5|3.0|2.5|1.4|2.0|2.5|2.0|3.0|2.5|
|microsoft/msclap-2022|2.0|2.5|2.5|2.0|3.5|4.0|0.5|2.5|6.6|2.0|2.0|1.5|2.0|2.0|5.5|1.5|4.5|2.5|1.5|2.0|1.5|4.2|1.0|3.0|3.0|2.0|2.5|
|OpenMuQ/MuQ-MuLan-large|2.0|1.5|2.0|4.0|3.5|3.0|2.5|3.5|8.2|4.0|1.0|4.5|2.0|3.0|3.0|2.5|5.0|2.5|3.5|3.0|1.0|4.2|2.0|1.5|2.5|3.5|3.5|
|laion/larger_clap_general|4.5|2.0|2.0|4.0|2.5|4.5|2.0|2.5|11.5|3.0|2.5|2.0|2.5|3.0|1.5|3.0|3.0|3.0|2.0|2.5|2.0|1.4|2.0|2.5|1.0|1.5|2.5|
|laion/larger_clap_music_and_speech|2.5|2.0|2.5|3.0|1.5|3.5|2.5|2.5|8.2|3.0|2.5|2.5|2.0|4.0|2.0|3.0|3.0|2.0|2.5|0.5|3.0|2.1|3.0|2.5|2.5|1.5|3.0|
|lyrebird/wav2clip|2.5|2.5|4.0|2.0|3.5|2.5|2.0|2.0|8.2|3.0|1.5|2.0|2.5|2.0|3.0|2.5|2.5|2.0|2.5|2.0|2.0|4.2|3.0|1.5|2.5|2.0|2.5|
|laion/larger_clap_music|2.5|2.5|2.5|2.5|2.5|2.5|2.5|2.5|8.2|3.0|2.5|2.5|2.5|2.5|2.5|2.5|2.5|2.5|2.0|2.5|2.5|3.5|2.5|2.5|2.5|2.0|2.0|
