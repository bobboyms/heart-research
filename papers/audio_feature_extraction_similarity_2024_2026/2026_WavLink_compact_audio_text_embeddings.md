<!-- page: 1 -->

# **WAVLINK: COMPACT AUDIO-TEXT EMBEDDINGS WITH A GLOBAL WHISPER TOKEN** 

_Gokul Karthik Kumar Ludovick Lepauloux Hakim Hacid_ 

Technology Innovation Institute, Abu Dhabi, UAE 

## **ABSTRACT** 

Whisper has become the de-facto encoder for extracting general-purpose audio features in large audio–language models, where a 30-second clip is typically represented by 1500 frame features projected into an LLM. In contrast, audio–text embedding models like CLAP-based models have largely relied on alternative audio encoders (e.g., HTS-AT, PaSST), and have not leveraged Whisper effectively. We present WavLink, a compact audio–text embedding model that augments Whisper encoder with a learnable global token, trained jointly with a text encoder. Through a systematic study of design choices, including pretrained text encoders, loss functions, training modes, and data mixtures, we identify configurations that yield state-of-the-art retrieval performance. Our two-stage training recipe across three model sizes, combined with Matryoshka-style supervision, improves scalability, enabling 8× smaller embeddings with minimal performance drop. WavLink also demonstrates competitive performance on AIR-Bench with MCQs and zero-shot classification. 

## **1. INTRODUCTION** 

Learning joint audio–text representations has become a central problem in speech, sound, and music understanding. Recent progress has been driven by the release of large-scale audio–text datasets, including AudioCaps [1], Clotho [2], and VGGSound [3], as well as synthetically generated corpora such as Auto-ACD [4] and AudioSetCaps [5]. These resources have enabled large-scale training of audio–text embedding models such as LAION-CLAP [6], MS-CLAP [7], MGA-CLAP [8], ReCLAP [9], and AF-CLAP [10], which achieve strong results in retrieval and zero-shot classification. In parallel, the rise of audio-understanding Large Language Models (audio-LLMs) e.g. Qwen2-Audio [11], Falcon3Audio [12], SALMONN [13], Audio Flamingo 3 [14], and Voxtral [15] demonstrates the effectiveness of projecting audio features into LLMs for instruction-following on audio inputs. Despite these advances, there remains a methodological divide between audio-LLMs and embedding models. Audio 

© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works. 

**Table 1** . WavLink model specifications. 

|**Size**|**Parameters (Audio + Text)**|**Supported Dimensions**|
|---|---|---|
|Large|761M (637 + 123)|768, 384, 192, 96|
|Small|152M (88 + 63)|512, 256, 128, 64|
|Base|84M (20 + 63)|512, 256, 128, 64|



LLMs typically adopt _Whisper_ [16] as their audio encoder: a 30s clip yields 1500 frame-level features, which are then projected into the LLM for instruction following. Whisper has also been adapted for audio tagging in Whisper-AT [17]. By contrast, embedding models—tasked with retrieval, captioning, and zero-shot classification—have almost exclusively relied on specialized encoders such as HTS-AT [18] or PaSST [19]. Thus, Whisper has become ubiquitous for LLM-based audio reasoning, but has been largely unused for compact audio–text embeddings which motivates our work. 

We introduce _**WavLink**_ , a compact audio–text embedding model that leverages Whisper. We augment Whisper with a _learnable global token_ , trained jointly with a text encoder. Instead of 1500 frame-level tokens, _WavLink_ produces a single representation, yielding savings in storage and similarity search cost. Through design sweeps, we compare CLIP [20] and ModernBERT [21] text encoders, CLIP versus SigLIP [22] losses, and different adaptation regimes (projector-only, LoRA [23], full finetuning), across both audio-only and joint tower updates. The best configuration is then scaled in a two-stage training recipe, with Matryoshka [24] supervision for multi-resolution embeddings. 

We evaluate _WavLink_ across **retrieval** with AudioCaps and Clotho, **zero-shot classification** with VGGSound, ESC50 [25] and US8K [26], and **MCQ** using AIR-Bench [27]. Results show that _WavLink_ not only surpasses prior CLAP variants in retrieval, but also achieves competitive accuracy on multiple-choice QA benchmark compared to much larger audio-LLMs such as Qwen2-Audio and Falcon3-Audio. To the best of our knowledge, _WavLink_ is the first to show that sub-100 dimensional embeddings, enabled by Matryoshka supervision, can retain competitive performance. This demonstrates that Whisper’s ASR-pretrained features can be adapted effectively for general audio–text embeddings, and that a single global token can bridge the efficiency gap between framelevel audio-LLMs and compact embedding models.

<!-- page: 2 -->

## **2. METHOD** 

## **2.1. Model Architecture** 

Our goal is to obtain a compact audio embedding from Whisper, which otherwise produces _∼_ 1500 frame tokens for a 30second clip. We augment Whisper’s encoder with a **learnable** _**global token**_ that serves as a content-adaptive aggregator. Given log-Mel features _X ∈_ R _[B][×][F][ ×][T]_ , the convolutional front-end produces hidden states _H_[˜] 0 _∈_ R _[B][×][T][ ′][×][D]_ . We append a parameter vector **a** cls _∈_ R[1] _[×][D]_ to each sequence and propagate the extended sequence through Whisper’s Transformer stack. The final state of this token is used as the pooled audio representation: 

**==> picture [168 x 13] intentionally omitted <==**

Text inputs are encoded using either the _CLIP text encoder_ or _ModernBERT_ , yielding pooled text features **z** _t_ from its respective _CLS_ token. Both modalities are mapped to a shared embedding space via lightweight linear projectors, followed by _ℓ_ 2 normalization: 

**==> picture [161 x 24] intentionally omitted <==**

**Fig. 1** . Design sweep based on Recall@1 retrieval performance on the AudioCaps and Clotho benchmarks, identifying a fully trained CLIP-BERT model with CLIP’s contrastive loss as the optimal configuration. 

## **2.5. Matryoshka Loss Adaptation** 

## **2.2. Training Objectives** 

We study two objectives. _CLIP loss (InfoNCE):_ 

**==> picture [94 x 13] intentionally omitted <==**

where _τ_ is a learnable temperature. Cross-entropy is applied over rows (audio _→_ text) and columns (text _→_ audio). _SigLIP loss:_ a sigmoid-based variant applying Binary Cross Entropy to all pairs, labeling only diagonal pairs as positives. 

## **2.3. Training Strategies** 

We evaluate three adaptation regimes for the encoders: (i) _projector-only_ (both encoders frozen), (ii) _LoRA_ adapters in Transformer layers, and (iii) _full finetuning_ . In all cases, the global audio token is learned from scratch. 

We also consider two update scopes: (a) _audio-only_ , where the text tower is frozen, and (b) _both towers_ , where audio and text are updated jointly. 

## **2.4. Design Sweep Configuration** 

Combining the factors of text encoder (CLIP vs. ModernBERT), loss function (CLIP vs. SigLIP), adaptation regime (3), and update scope (2), we obtain a total of 2 _×_ 2 _×_ 3 _×_ 2 = 24 configurations. These settings are explored in the design sweep to study how different choices affect performance. 

To enable dimensional scalability, we adopt Matryoshka supervision. The idea is to train embeddings that remain useful when truncated to smaller dimensions. Let _d_ 1 _> d_ 2 _> · · · > dK_ be target sizes (e.g., 768 _→_ 384 _→_ 192 _→_ 96), and let slice( _·, d_ ) select the first _d_ channels. At each level _k_ , contrastive loss is applied on the sliced embeddings: 

**==> picture [182 x 14] intentionally omitted <==**

The overall loss is the mean across all levels: 

**==> picture [132 x 30] intentionally omitted <==**

This produces a single model capable of emitting nested embeddings at multiple scales. 

## **3. EXPERIMENTS** 

## **3.1. Experimental Setup** 

**Datasets.** The design sweep is trained on _∼_ 2M audio–text pairs from Auto-ACD (AudioSet + VGGSound derived). Scaled training uses two stages: Stage-1 with additional _∼_ 6M captions from AudioSetCaps (AudioSet + VGGSound + YouTube-8M derived), and Stage-2 with _∼_ 0.1M captions from AudioCaps v2 and Clotho training splits. 

**Models.** _Audio encoders Initialization:_ Whisper-Large(v3) for _Large_ , Whisper-Small(en) for _Small_ , Whisper-Base(en)

<!-- page: 3 -->

**Table 2** . Retrieval performance with Recall@K on the AudioCaps and Clotho benchmarks. Shaded sub-rows (∆) show the performance change from using only the first 1/8 Matryoshka dimensions, with changes greater than 0.5 in magnitude marked green for gains and red for losses. **Top scores** are bolded; second-best are underlined. 

|**Model**|**AudioCaps**<br>**Text-To-Audio**<br>**Audio-To-Text**<br>**R@1**<br>**R@5**<br>**R@10**<br>**R@1**<br>**R@5**<br>**R@10**|**AudioCaps**<br>**Text-To-Audio**<br>**Audio-To-Text**<br>**R@1**<br>**R@5**<br>**R@10**<br>**R@1**<br>**R@5**<br>**R@10**|**AudioCaps**<br>**Text-To-Audio**<br>**Audio-To-Text**<br>**R@1**<br>**R@5**<br>**R@10**<br>**R@1**<br>**R@5**<br>**R@10**|**Clotho**<br>**Text-To-Audio**<br>**Audio-To-Text**<br>**R@1**<br>**R@5**<br>**R@10**<br>**R@1**<br>**R@5**<br>**R@10**|**Clotho**<br>**Text-To-Audio**<br>**Audio-To-Text**<br>**R@1**<br>**R@5**<br>**R@10**<br>**R@1**<br>**R@5**<br>**R@10**|
|---|---|---|---|---|---|
||**Text-To-Audio**<br>**R@1**<br>**R@5**<br>**R@10**|||**Text-To-Audio**<br>**R@1**<br>**R@5**<br>**R@10**||
||**R@1**|||||
|WavLink-Large|**46.7**|**80.2**<br>**89.5**|**60.0**<br>**85.3**<br>**92.6**|**22.4**<br>**48.7**<br>**62.6**|**27.4**<br>**52.3**<br>**66.0**|
|∆_M-1/8_|-0.3|+0.1<br>+0.1|-1.4<br>-0.3<br>0.0|-0.6<br>-0.2<br>-0.3|0.0<br>+0.1<br>+0.6|
|WavLink-Small|44.5|79.0<br>88.0|54.3<br>84.1<br>92.3|21.2<br>46.8<br>60.1|25.3<br>49.4<br>64.0|
|∆_M-1/8_|-0.6|-0.3<br>-0.1|+0.2<br>+0.4<br>+0.4|+0.2<br>-0.7<br>+0.1|+0.4<br>+0.1<br>-1.1|
|WavLink-Base|39.7|74.5<br>85.3|50.5<br>79.4<br>90.3|17.6<br>41.3<br>56.0|21.1<br>45.8<br>58.8|
|∆_M-1/8_|-0.1|-0.6<br>+0.1|-0.6<br>+0.5<br>-0.6|-0.5<br>0.0<br>-0.2|-1.0<br>-0.6<br>-0.4|
|||**Reported Performance From Prior Studies**||||
|LAION-CLAP<br>MGA-CLAP<br>ReCLAP<br>AF-CLAP|36.1<br>41.8<br>37.1<br>37.3|71.8<br>83.9<br>46.8<br>82.9<br>90.7<br>16.1<br>38.3<br>51.1<br>76.1<br>-<br>54.4<br>83.6<br>-<br>20.4<br>46.0<br>-<br>73.2<br>85.0<br>48.0<br>80.4<br>90.8<br>18.9<br>44.7<br>59.0<br>72.9<br>84.0<br>46.9<br>84.1<br>91.9<br>17.3<br>43.9<br>56.8|||22.7<br>48.5<br>60.8<br>25.3<br>51.2<br>-<br>20.5<br>45.7<br>58.9<br>23.2<br>51.2<br>63.5|



**Table 3** . Zero-shot classification performance with Accuracy (%) on the VGG-Sound, US8K, and ESC-50 benchmarks. 

|**Model**|**VGG-Sound**|**US8K**|**ESC-50**|
|---|---|---|---|
|WavLink-Large|**31.7**|74.5|83.0|
|WavLink-Small|**31.8**|75.0|80.3|
|WavLink-Base|27.7|69.9|75.4|
|**Reported Performance From**||**Prior Studies**||
|LAION-CLAP|29.1|73.2|89.1|
|MGA-CLAP|**31.8**|83.7|**94.9**|
|ReCLAP|29.2|**95.2**|92.8|
|AF-CLAP|24.1|92.3|91.3|



for _Base_ . _Text encoders Initialization:_ CLIP-ViT-L/14 or ModernBERT-Large for _Large_ , CLIP-ViT-B/32 for _Small_ and _Base_ . _Projectors_ : Single linear layer to match the projector dimension, with pretrained weights used for CLIP based text projectors. The global token is trained from scratch always. More model specifications are shown in Table 1. 

**Training.** _Framework:_ PyTorch Lightning. _Common configuration:_ DDP strategy; BF16 precision; AdamW optimizer; 1e-4 learning rate; cosine scheduler with 5% warmup; Embeddings are gathered across GPUs before computing CLIP loss. _Design sweep configuration:_ 8 _×_ H100 80GB GPUs; 80 batch size; 10 epochs; _Large_ variant; LoRA rank 8. _Scaled runs configuration:_ 64 _×_ H100 80GB GPUs; 768 batch size; 3 epochs per stage; Matryoshka supervision with _K_ = 4 and dimensions _d, d/_ 2 _, d/_ 4 _,_ and _d/_ 8. 

## **3.2. Design Sweep Results** 

We evaluate 24 setups (2 encoders _×_ 2 losses _×_ 3 regimes _×_ 2 scopes). Figure 1 shows R@1 on AudioCaps and Clotho. 

The best setting is _CLIP text, CLIP loss, full finetuning, both towers updated_ , adopted for scaled Stage-1/2 training. Interestingly, ModernBERT underperformed CLIP despite its stronger text benchmarks, suggesting that CLIP’s alignment priors transfer better to audio–text retrieval. 

## **3.3. Retrieval Performance** 

Table 2 reports results on AudioCaps and Clotho. _WavLinkLarge_ improves over CLAP baselines by _∼_ 2–6 points across R@K. _WavLink-Small_ trails by only 1–2 points while using _∼_ 20% of parameters. _WavLink-Base_ (<100M parameters) is competitive with existing CLAP based models showing that Whisper-derived embeddings can rival models explicitly trained for audio–text alignment. 

## **3.4. Generalization Beyond Retrieval: ZSC** 

_WavLink_ achieves top zero-shot classification accuracy (Table 3) on VGGSound. On ESC-50 and US8K it lags behind prior models, potentially due to relatively more dense training captions. Task-specific fine-tuning could close this gap. 

## **3.5. Generalization Beyond Retrieval: MCQ** 

We reframe multiple-choice QA as zero-shot classification and show the performance on AirBench Foundational. We combine the question with each candidate choice, and pick the option whose joint text embedding is most similar to that of the audio. As shown in Table 4, _WavLink-Base_ (84M, 1 token) achieves 42.0%, +6 over LAION-CLAP and comparable to Falcon3-Audio-3B, while trailing Qwen2-Audio Instruct by only 2 points despite being 43–100 _×_ smaller. Performance is strong on classification tasks across speech, sound, and music,

<!-- page: 4 -->

**Table 4** . Multiple-choice QA performance with Accuracy (%) on the AirBench Foundational benchmark. Results for the Audio Encoder-LLM Decoder models are from [12]. LAION-CLAP is taken from https://huggingface.co/laion/ larger_clap_general. **Top scores** are bolded; second-best are underlined. 

|**Task**|**Dual Encoder**<br>**WavLink-Base**<br>**LAION-CLAP**|**Audio Encoder-LLM Decoder**<br>**Qwen2-Audio Instruct**<br>**Falcon3-Audio 3B**|
|---|---|---|
|_∼_parameter count in M (relative size)<br>Number of audio tokens|84<br>193 (2x)<br>1<br>1|8400 (100x)<br>3600 (43x)<br>750<br>750|
|**Total Average**|42.0<br>35.8|**44.0**<br>42.0|
|**Sound Average**|48.3<br>42.6|49.8<br>**53.4**|
|Audio grounding<br>Vocal sound classifcation<br>Acoustic scene classifcation<br>Sound question answering|26.0<br>20.8<br>71.3<br>55.9<br>44.2<br>44.8<br>50.8<br>43.5|17.8<br>60.0<br>71.1<br>74.9<br>40.5<br>40.8<br>62.8<br>52.4|
|**Music Average**|**47.9**<br>46.2|46.1<br>42.2|
|Music instruments classifcation<br>Music genre classifcation<br>Music note analysis-pitch<br>Music note analysis-velocity<br>Music question answering<br>Music emotion detection|62.5<br>56.2<br>64.0<br>64.5<br>26.0<br>31.4<br>27.8<br>25.6<br>33.1<br>29.5<br>40.7<br>38.6|49.6<br>46.9<br>63.9<br>59.9<br>24.3<br>19.6<br>24.7<br>22.8<br>56.0<br>41.8<br>38.7<br>39.9|
|**Speech Average**|34.4<br>24.7|**43.5**<br>35.1|
|Speech grounding<br>Spoken language identifcation<br>Speaker gender recognition<br>Speaker emotion recognition<br>Speaker age prediction<br>Speaker entity recognition<br>Speaker intent classifcation<br>Speaker number verifcation<br>Synthesized voice detection|29.2<br>25.4<br>27.1<br>19.3<br>53.3<br>35.9<br>37.3<br>19.7<br>13.5<br>25.3<br>25.3<br>22.5<br>31.4<br>27.0<br>54.8<br>21.4<br>16.2<br>20.1|26.3<br>20.3<br>38.1<br>4.9<br>52.5<br>23.9<br>35.4<br>60.3<br>22.3<br>26.4<br>48.3<br>26.2<br>78.0<br>51.4<br>38.0<br>39.2<br>51.9<br>49.3|



but weaker on grounding and fine-grained analysis. This pattern is intuitive: speech-heavy tasks benefit from Whisper’s ASR pretraining, while grounding requires finer token-level alignment that frame-based LLM methods capture better. 

limitations with longer audio segments ( _>_ 10s, common in Clotho), underscoring Whisper’s robustness as a backbone for general-purpose audio–text embeddings. 

## **4. CONCLUSION** 

## **3.6. Scalability** 

Embeddings compressed to 1/8 dimension maintain accuracy within _<_ 1 point on average (Table 2), reducing storage and similarity compute by 8 _×_ . Larger models degrade less under compression, suggesting redundancy. This property is especially valuable for web-scale retrieval, where both storage and similarity search cost are dominant bottlenecks. 

## **3.7. Ablations** 

Replacing the Whisper encoder with a pretrained HTSAT from LAION-CLAP in the _large_ setup yielded R@1 (T2A/A2T) scores of 45.8/56.4 on AudioCaps and 14.0/14.6 on Clotho. While lower than _WavLink-Large_ on AudioCaps, the drop was more severe on Clotho, where it underperformed even _WavLink-Base_ . This indicates HTS-AT’s 

We introduced _WavLink_ , a compact audio–text embedding model that augments Whisper with a learnable global token. Through systematic design sweeps, scaled two-stage training, and Matryoshka supervision, WavLink achieves state-of-theart retrieval performance, strong zero-shot classification on VGGSound, and competitive results on AIR-Bench while using a single global token instead of 1500 frame-level features. These findings highlight the underexplored potential of Whisper beyond speech recognition for efficient representation learning. 

Future directions include extending WavLink to multilingual audio–text alignment and leveraging the global token mechanism for audio–LLMs, where compact and adaptive embeddings can reduce compute cost while improving crosstask generalization.

<!-- page: 5 -->

## **5. REFERENCES** 

- [1] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, et al., “AudioCaps: Generating captions for audios in the wild,” in _Proc. NAACL-HLT_ , 2019. 

- [2] Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen, “Clotho: An audio captioning dataset,” in _Proc. ICASSP_ , 2020. 

- [3] Honglie Chen, Weidi Xie, Andrea Vedaldi, et al., “Vggsound: A large-scale audio-visual dataset,” in _Proc. ICASSP_ , 2020. 

- [4] Luoyi Sun, Xuenan Xu, Mengyue Wu, et al., “Auto-acd: A large-scale dataset for audio-language representation learning,” in _Proc. ACM Multimedia_ , 2024. 

- [5] Jisheng Bai, Haohe Liu, Mou Wang, et al., “Audiosetcaps: An enriched audio-caption dataset using automated generation pipeline with large audio and language models,” _IEEE/ACM TASLP_ , 2025. 

- [6] Yusong Wu, Ke Chen, Tianyu Zhang, et al., “Largescale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in _Proc. ICASSP_ , 2023. 

- [7] Benjamin Elizalde, Soham Deshmukh, and Huaming Wang, “Natural language supervision for generalpurpose audio representations,” _arXiv:2309.05767_ , 2023. 

- [8] Yiming Li, Zhifang Guo, Xiangdong Wang, et al., “Advancing multi-grained alignment for contrastive language-audio pre-training,” in _Proc. ACM Multimedia_ , 2024. 

- [9] Sreyan Ghosh, Sonal Kumar, Chandra Kiran Reddy Evuru, et al., “Reclap: Improving zero shot audio classification by describing sounds,” in _Proc. ICASSP_ , 2025. 

- [10] Sreyan Ghosh, Zhifeng Kong, Sonal Kumar, et al., “Audio flamingo 2: An audio-language model with longaudio understanding and expert reasoning abilities,” _arXiv:2503.03983_ , 2025. 

- [11] Yunfei Chu, Jin Xu, Qian Yang, et al., “Qwen2-audio technical report,” _arXiv:2407.10759_ , 2024. 

- [12] Gokul Karthik Kumar, Rishabh Saraf, Ludovick Lepauloux, et al., “Competitive audio-language models with data-efficient single-stage training on public data,” _arXiv:2509.07526_ , 2025. 

- [13] Changli Tang, Wenyi Yu, Guangzhi Sun, et al., “Salmonn: Towards generic hearing abilities for large language models,” in _Proc. ICLR_ , 2024. 

- [14] Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, et al., “Audio flamingo 3: Advancing audio intelligence with fully open large audio language models,” _arXiv preprint arXiv:2507.08128_ , 2025. 

- [15] Alexander H Liu, Andy Ehrenberg, Andy Lo, et al., “Voxtral,” _arXiv preprint arXiv:2507.13264_ , 2025. 

- [16] Alec Radford, Jong Wook Kim, Tao Xu, et al., “Robust speech recognition via large-scale weak supervision,” in _Proc. ICML_ , 2023. 

- [17] Yuan Gong, Sameer Khurana, Leonid Karlinsky, and James Glass, “Whisper-at: Noise-robust automatic speech recognizers are also strong general audio event taggers,” _arXiv preprint arXiv:2307.03183_ , 2023. 

- [18] Ke Chen, Xingjian Du, Bilei Zhu, et al., “Hts-at: A hierarchical token-semantic audio transformer for sound classification and detection,” in _Proc. ICASSP_ , 2022. 

- [19] Khaled Koutini, Jan Schlüter, Hamid Eghbal-Zadeh, et al., “Efficient training of audio transformers with patchout,” _arXiv:2110.05069_ , 2021. 

- [20] Alec Radford, Jong Wook Kim, Chris Hallacy, et al., “Learning transferable visual models from natural language supervision,” in _Proc. ICML_ , 2021. 

- [21] Benjamin Warner, Antoine Chaffin, Benjamin Clavié, et al., “Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference,” _arXiv:2412.13663_ , 2024. 

- [22] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, et al., “Sigmoid loss for language image pre-training,” in _Proc. ICCV_ , 2023. 

- [23] Edward J Hu, Yelong Shen, Phillip Wallis, et al., “Lora: Low-rank adaptation of large language models,” _Proc. ICLR_ , 2022. 

- [24] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, et al., “Matryoshka representation learning,” _Proc. NeurIPS_ , 2022. 

- [25] Karol J. Piczak, “Esc: Dataset for environmental sound classification,” in _Proc. ACM Multimedia_ , 2015. 

- [26] Justin Salamon, Christopher Jacoby, and Juan Pablo Bello, “A dataset and taxonomy for urban sound research,” in _Proc. ACM Multimedia_ , 2014. 

- [27] Qian Yang, Jin Xu, Wenrui Liu, et al., “AIR-bench: Benchmarking large audio-language models via generative comprehension,” in _Proc. ACL_ , 2024.
