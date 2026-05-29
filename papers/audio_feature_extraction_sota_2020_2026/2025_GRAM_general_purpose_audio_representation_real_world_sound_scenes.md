<!-- page: 1 -->

# **GRAM: Spatial general-purpose audio representations for real-world environments** 

**Goksenin Yuksel**[1] **Marcel van Gerven**[1] **Kiki van der Heijden**[1 2] 

## **Abstract** 

## **1. Introduction** 

Audio foundation models learn general-purpose audio representations that facilitate a wide range of downstream tasks. While the performance of these models has greatly increased for conventional single-channel, dry audio clips, their success in real-world acoustic environments with reverberation and noise is limited. Furthermore, most audio foundation models ignore the spatial dimension of real-world acoustic environments, ruling out tasks involving sound localization. To address these limitations, we propose GRAM: a general-purpose real-world audio model that employs a multi-channel masked autoencoder to efficiently learn spatial audio representations. We evaluated GRAM and other audio foundation models in a standardized manner on high-quality simulations of naturalistic, spatial acoustic environments as well as recordings of real-world environments and release these two complementary benchmark task suites: NatHEAR and RealSELD. Our results demonstrate that GRAM outperforms all state-of-the-art self-supervised audio foundation models on NatHEAR and the clean, single-channel version HEAR, while using only a fraction of the training data. GRAM also shows state-of-the-art localization performance in simulated environments and generalizes efficiently to real-world recordings in RealSELD. Taken together, GRAM presents a significant advance toward robust spatial audio foundation models for real-world environments.[1] 

> 1Donders Institute, Radboud University, Nijmegen, The Netherlands[2] Mortimer B Zuckerman Institute, Columbia University, New York, United States. Correspondence to: Goksenin Yuksel _<_ goksenin.yuksel@donders.ru.nl _>_ . 

_Preprint. February 5, 2026._ 

> 1All the code and data is available on https:// github.com/labhamlet and https://huggingface. co/labhamlet 

Despite the complexity and diversity of everyday sound scenes, human listeners effortlessly interact with their acoustic environment in myriad ways. Audio foundation models that perform a similar, human-like range of tasks have received widespread attention (Turian et al., 2022; Wang et al., 2022a; Yang et al., 2021). While these models demonstrate strong performance on audio benchmarks with minimal finetuning (e.g., (Chen et al., 2023; Baevski et al., 2020; Yadav et al., 2024)), they overlook inherent aspects of real-world sound scenes: the spatial dimension, reverberation, and background noise. Specifically, audio foundation models typically lack effectiveness in naturalistic, complex acoustic environments with background noise and reverberation because they are primarily trained on dry, large-scale sound datasets such as AudioSet (Gemmeke et al., 2017) and Librispeech (Panayotov et al., 2015). 

Crucially, the lack of spatial information in audio embeddings precludes sound localization and the use of spatial sound features to improve performance on complex listening tasks, such as audio scene analysis. Audio scene analysis refers to the separation of overlapping sound waves in complex multi-source sound scenes and the subsequent grouping of the frequency components into distinct auditory objects (Bregman, 1984; Bizley & Cohen, 2013). In humans, such audio scene analysis is aided by spatial cues (Bizley & Cohen, 2013; van der Heijden et al., 2019). And, incorporating spatial knowledge into universal audio embedding models is also expected to benefit downstream tasks that require ambient intelligence and acoustic awareness, such as acoustic scene understanding. 

To address these limitations of audio foundation models for real-world environments, we present GRAM (Generalpurpose, Real-world Audio Model). GRAM is a selfsupervised, multi-channel masked auto-encoder model that efficiently learns spatial general-purpose audio representations from multi-channel audio clips. To train GRAM, we developed a custom pipeline that uses the Soundspace 2.0 platform (Chen et al., 2022a) to simulate high-quality real-world sound scenes. Further, to promote the systematic evaluation of audio foundation models on naturalistic sound scenes, we introduce two complementary benchmark suites: 

1

<!-- page: 2 -->

**Spatial general purpose audio representation learning** 

## NatHEAR and RealSELD. 

NatHEAR is an extension of the HEAR benchmark suite that includes simulated, real-world versions of the downstream tasks. Additionally, it includes two sound localization tasks and reverberation time (T60) estimation tasks. NatHEAR plays a crucial role in assessing the robustness of audio foundation models to controlled noisy and reverberant conditions. RealSELD comprises real-life datasets for sound event localization and detection (SELD) tasks collected from previous DCASE challenges (Mesaros et al., 2025). We embed these evaluation tasks into the standardized HEAR setup, therefore unifying the evaluation of SELD performance of audio foundation models in challenging real-world scenarios. 

Empirical results demonstrate that GRAM efficiently learns robust, general-purpose spatial audio representations, outperforming all state-of-the-art audio foundation models and speech models on HEAR and NatHEAR. GRAM excels at complex tasks, such as audio scene analysis, and achieves strong sound localization performance, outperforming even supervised models trained with auxiliary spatial features. Finally, GRAM demonstrates robust transfer to recordings of real-world sound scenes, as evidenced by RealSELD performance, thereby overcoming the need for extensive domain-specific adaptations. Taken together, our key contributions can be summarized as: 

**General-Purpose Audio Foundation Model (GRAM):** We present GRAM, a multi-channel masked auto-encoder that shows state-of-the-art performance on a human-like range of tasks in naturalistic sound scenes, including sound localization. GRAM is the first audio foundation model available in both binaural and four-channel Ambisonics audio formats. 

**A large-scale dataset for high-quality simulations of realworld sound scenes** : We release the complete set of binaural room impulse responses (BRIRs) and ambisonics room impulse responses (ARIRs) corresponding to 85,000 naturalistic sound scenes that we used for our naturalistic training pipeline. 

**NatHEAR and RealSELD:** To enable the systematic evaluation of audio foundation models in complex acoustic environments, we introduce the NatHEAR and RealSELD. NatHEAR extends the HEAR framework with simulated naturalistic scenes and novel spatial tasks, while RealSELD establishes the first benchmark for assessing pre-trained audio embeddings on real-world SELD tasks. 

AST (Gong et al., 2021a), PaSST (Koutini et al., 2022) and HTS-AT (Chen et al., 2022b) have Transformer-based architectures as a backbone, for example ViT (Dosovitskiy et al., 2021) and Swin Transformer (Liu et al., 2021). To mitigate the need for large annotated datasets, some of these approaches rely on pre-trained image models (e.g., PSLA (Gong et al., 2021b)). Question-and-answer models constitute a more recent category of supervised approaches that integrate audio representation learning with large language models (for example, Spatial-AST (Zheng et al., 2024) and Qwen-Audio (Chu et al., 2023))res large-scale annotated datasets and is sub-optimal for learning generalpurpose audio representations that generalize across tasks. 

**Self-supervised audio representation learning:** Selfsupervised audio representation learning approaches aim to learn robust audio representations that generalize across a wide range of tasks. Masking-based approaches utilizing transformer backbones to reconstruct masked patches of input spectrograms currently constitute the predominant approach, including (SSAST (Gong et al., 2022)), MSMMAE (Niizumi et al., 2022), MaskSpec (Chong et al., 2023), MAE-AST (Baade et al., 2022), and Audio-MAE (Huang et al., 2022) achieves state-of-the-art performance on the HEAR benchmark by using multi-window local-global attention in the decoder. Recently, SSAM (Yadav & Tan, 2024) utilized a Mamba (Gu & Dao, 2023) architecture in their encoder and achieved similar performance as MWMAE. In contrast to masked auto-encoders, BEATS (Chen et al., 2023) employs a masking-based approach based on latent embeddings extracted by an acoustic tokenizer. Finally, successful self-supervised approaches that do not rely on masking at all include contrastive learning frameworks such as COLA (Saeed et al., 2021). 

Another category of self-supervised audio representation models focuses specifically on speech representations, using generative, predictive, or contrastive learning (Mohamed et al., 2022). These speech models are typically trained on datasets such as Librispeech (Panayotov et al., 2015) or LibriLight (Kahn et al., 2020) and include state-of-the-art models such as Wav2Vec2 (Baevski et al., 2020), HuBERT (Hsu et al., 2021), and WavLM (Chen et al., 2021). However, while these models excel at speech-based tasks, they do not necessarily generalize well to non-speech sounds and non-speech tasks (Turian et al., 2022). Crucially, none of the existing self-supervised approaches for audio or speech representation learning optimize for performance in realworld sound scenes that are spatial, reverberant, and noisy. 

## **2. Related Work** 

**Supervised audio representation learning:** Supervised methods for audio representation learning have achieved notable success in recent years. Approaches such as 

## **3. Materials and Methods** 

**Simulating real-world acoustic scenes:** A room impulse response (RIR) captures room-specific acoustic properties 

2

<!-- page: 3 -->

**Spatial general purpose audio representation learning** 

**==> picture [483 x 197] intentionally omitted <==**

**----- Start of picture text -----**<br>
A        Naturalistic scene generation in a       B            Pre-training methodology of GRAM-Binaural<br>            simulated MatterPort3D house<br>Binaural  Patch and mask Encode non-masked  Decode padded Reconstruct<br>Noise  spectrograms patches outputs<br>source 3 Ambisonics<br>M M<br>| ® a E O er<br>») in Watt Binaural M | E O || M<br>Mn-1 En On Mn-1<br>eee, * a Mn Uom|/ Mn m<br>En On Learn spatial cues<br>Target sound source Noise  Noise  o) Capture spatial  ©) from reconstruction<br>source 1 source 2 and semantic attributes<br>Direct Sounds<br>Reflections<br>Encoder Decoder<br>Patch Extractor<br>**----- End of picture text -----**<br>


_Figure 1._ **Proposed self-supervised approach for training GRAMs on naturalistic binaural scenes** . (A) We generate binaural and ambisonics naturalistic scenes using the SoundSpace2.0 simulator (Chen et al., 2022a) in MatterPort3D houses. (B) MAE approach for learning audio representations with spatial attributes. For the ambisonics spectrograms, the methodology remains the same, except that the inputs now include 4-channel mel-spectrograms and intensity vectors (IVs). 

such as reverberation. We utilized high-resolution, detailed 3D meshes of houses with various architectural characteristics from Matterport3D (Chang et al., 2017) to simulate RIRs for many different rooms in each house with the Monte Carlo ray tracing RIR simulator provided by SoundSpaces 2.0 (Chen et al., 2022a). SoundSpaces 2.0 combines simulated RIRs with a head-related transfer function (HRTF) (Algazi et al., 2001) to generate a binaural RIR (BRIR) or, with an ambisonics microphone configuration, an ambisonics RIR (ARIR). BRIRs capture both room acoustic properties and human spatial hearing characteristics introduced by the shape of the ears, head and torso, while ARIRs capture room acoustic properties as well as the spatial cues encoded in first-order Ambisonics. 

**Components of simulated real-world scenes:** Matterport3D contains scans of 90 houses. We discarded five houses for which the meshes were not of sufficient quality. For each of the remaining 85 houses, we simulated 1,000 real-world scenes. Each scene comprised a randomly sampled listener location, a sound source location, and a noise source location within the room. For BRIRs (binaural), we randomly sampled head orientation from a range [0°, 360°]). We placed the sound source at a randomly sampled position relative to the listener or microphone (distance range [1.5 m, 5 m]; azimuth range [0°, 360°]; elevation range [-90°, +90°]). Noise was either localized (50% of the scenes) or diffuse (50% of the scenes). For localized noise, we randomly sampled a single location within the room. For diffuse noise, we randomly sampled three, four, or five locations in the room. We then rendered a set of RIRs to describe all components in the naturalistic scene. Given sound source location 

_s_ , listener (microphone) location _r_ , and receiver head orientation _θ_ , we rendered RIRs describing the sound path from the source to the listener (microphone) as BRIR( _s, r, θ_ ) and as ARIR( _s, r, θ_ ). Given a number of noise sources _ni_ at noise source location _ϕi_ , listener location _r_ , and receiver head orientation _θ_ , we rendered the RIR describing the path from the noise source(s) to the listener as BRIR _i_ ( _ϕi, r, θ_ ) and as ARIR( _ϕi, r, θ_ ). This procedure resulted in a total of 85,000 sets of BRIRs as well as 85,000 sets of ARIRs (see Appendix L for all parameters). 

## **3.1. GRAM framework** 

GRAM learns spatial audio representations by reconstructing masked multi-channel binaural and ambisonics spectrogram patches. Importantly, GRAM reconstructs crucial localization cues, such as interaural level differences (ILDs) for binaural scenes and intersity vectors (IVs) for ambisonics scenes, thereby learning to encode the necessary spatial information. We refer to the binaural version as GRAM-Binaural, and the ambisonics version as GRAMAmbisonics. First, a patch extractor consisting of a single convolutional layer with convolutional filters divides each multi-channel spectrogram into _n_ non-overlapping patches _P_ 1 _, . . . , Pn_ with _Pi ∈_ R _[C][×][T][ ×][F]_ , and embeds each patch into a linear patch embedding _Ei ∈_ R[768] (Figure 1). Nonmasked patch embeddings are input to the encoder, for which we selected the 12-layer ViT-Base (ViT-B) Transformer (Dosovitskiy et al., 2021) similar to Huang et al. (2022); Yadav et al. (2024). The encoder outputs patch representations _Oi ∈_ R[768] for _i_ = 1 _, . . . , n_ , where _n_ is the number of unmasked patches. Finally, a Transformer 

3

<!-- page: 4 -->

**Spatial general purpose audio representation learning** 

decoder with local-global attention (Yadav et al., 2024) followed by a linear head takes all patch representations _O_ 1 _, . . . , On_ as well as all masked patches _M_ 1 _, . . . , Mn_ to reconstruct the multi-channel spectrogram from last layer embeddings. 

## **3.2. Pre-training** 

**Online mixing of naturalistic sound scenes:** The 85,000 naturalistic scenes were split into a train set of 70,000 scenes (corresponding to 70 Matterport3D houses), and a test set of 15,000 scenes (15 Matterport3D houses) for downstream evaluation (see Section 4). We used the 70,000 naturalistic scenes in the train set to generate naturalistic scenes for all audio clips in the unbalanced AudioSet training set (10-second soundtracks from 1.74 million YouTube videos (Gemmeke et al., 2017)). Specifically, during training, we randomly paired an AudioSet clip with a noise sound clip from the WHAMR! background noise database (Maciejewski et al., 2020). WHAMR! Noise clips longer than 10 s were trimmed to 10 s duration, and a linear fade-in/fadeout of 200 ms was applied to every noise clip before mixing of the sound scene. 

To create a naturalistic sound scene, we then convolved the AudioSet clip with either BRIR( _s, r, θ_ ) for GRAMBinaural or ARIR( _s, r, θ_ ) for GRAM-Ambisonics to obtain _T_ . Similarly, we convolved the WHAMR! noise clip with the BRIR( _ϕi, r, θ_ ) to obtain _Ni_ . In naturalistic scenes with diffuse background noise, the diffuse noise field _D_ was generated by summing the noise clips _D_ =[�] _[M] i_ =1 _[N][i]_[where] _Ni_ are individual noise clips and _M_ is the total number of noise clips. The naturalistic sound scene _S_ was then computed as _S_ = _T_ + _bN_ for scenes with localized noise and as _S_ = _T_ + _bD_ for scenes with a diffuse noise field. Here, _b_ is a scaling parameter that mixes target and noise clips at a signal-to-noise ratio (SNR) ranging from +5 dB to +40 dB. 

**Input features:** We transformed the channels of each sound scene into log-scale mel spectrograms using 128 mel filters in the frequency range of 50-16000 Hz with a 25 ms Hanning window and 10 ms hop length, resulting in spectrograms of dimension 1001 _×_ 128. We added zero padding to achieve a 1024 _×_ 128 dimension. For GRAM-Ambisonics, following Devnani et al. (2024); Wang et al. (2022b), we extracted intensity vectors utilizing the equation below: 

**==> picture [209 x 36] intentionally omitted <==**

Where _An,m_ are the _n_[th] and _m_[th] order and mode of the ambisonics signal corresponding to its omnidirectional ( _W_ ) and three dipole ( _Z, Y, X_ ) components, and ( _·_ ) _[∗]_ denotes complex conjugation. IVs are scaled to unit norm. We concatenated mel spectrograms and intensity vectors, resulting 

## in input _x_ = [ _x_ mel _, IV s_ ]. 

**In-batch sampling:** As the online mixing of naturalistic acoustic scenes is computationally expensive due to multiple long convolutions, we used a random in-batch sampling procedure to increase the effective batch size in a computationally efficient manner. We randomly sampled 16 partially overlapping segments of 2 seconds to create 16 samples of dimension 200 _×_ 128. This increases the original batch size of 96 to an effective batch size of 1536. 

**Patch extraction and masking:** For pre-training, we divided the binaural spectrogram into _Pi ∈_ R[2] _[×]_[8] _[×]_[16] , and ambisonics spectrograms into _Pi ∈_ R[7] _[×]_[8] _[×]_[16] patches. We used an adapted version of the mask-based framework of MWMAE (Yadav et al., 2024), randomly selecting a subset of _n_ patches _M_ 1 _, . . . , Mn_ for _i_ = 1 _, . . . , n_ for masking (masking ratio = 0.8) and replacing their embedding with a learnable mask token. Finally, we added fixed sinusoidal positional embeddings to all embedded patches. 

**Decoder with local-global attention:** The decoder takes as input both the unmasked patches _O_ 1 _, . . . , On_ with _Oi ∈_ R[768] , and the masked patches _M_ 1 _, . . . , Mn_ with _Mi ∈_ R[768] as well as fixed sinusoidal positional embeddings for each patch (Figure 1). To implement local-global attention (Yadav et al., 2024), we selected window sizes of [2, 5, 10, 25, 50, 100, 0, 0]. Here, 0 signifies global plain attention. 

## **3.3. Ablations** 

To identify key factors for successful learning of spatial general-purpose audio representations, we conducted a series of ablation experiments. For GRAM-Binaural, we swapped the Transformer encoder with a Mamba encoder (Gu & Dao, 2023; Yadav & Tan, 2024). To ensure that computational overhead and model capacity were comparable between the Transformer and Mamba encoders, we used similar parameter counts. We also tested the impact of mask type for GRAM-Binaural, comparing patch-based masking and time-based masking. For time-based masking, patches were defined as _Pi ∈_ R[2] _[×]_[2] _[×]_[128] , spanning the entire frequency range. For time-based masking, we used window sizes [2, 5, 10, 25, 50, 0, 0, 0] to implement local-global attention in the decoder. Furthermore, for both GRAM-Binaural and GRAM-Ambisonics, we assessed the optimal ratio ( _λ_ ) between simulated real-world sound scenes and clean, dry sound clips in pretraining data for _λ_ = 0 _._ 0 _,_ 0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 75 _,_ 1 _._ 0. Finally, we examined various masking ratios [0.4, 0.6, 0.8, 0.9] and in-batch sampling factors [4, 8, 16] for GRAM-Binaural. For all ablations, GRAM-Binaural and, if applicable, GRAM-Ambisonics were trained with the same parameters specified above, except for the masking-ratio ablation, where we reduced the effective batch size from 1536 to 384 to alleviate out-ofmemory errors. 

4

<!-- page: 5 -->

**Spatial general purpose audio representation learning** 

**==> picture [464 x 181] intentionally omitted <==**

**----- Start of picture text -----**<br>
A Training efficiency of Audioset models  B Impact of simulated real - world scenes<br>NatHEAR HEAR<br>100 60<br>60 30<br>0<br>20 models<br>10 [7] 10 [8] 10 [9] 10 [10] 10 [11] 10 [7] 10 [8] 10 [9] 10 [10] 10 [11]<br>#training samples #training samples<br>Ours SSL-Audioset SSL-Speech Supervised<br>GRAM-Binaural MAE MWMAE Wav2Vec2.0 PASST<br>GRAM-Ambisonics SSAST SSAM HuBERT Spatial-AST<br>GRAM-Clean BEATs WavLM<br>s(m)<br>HEAR- Nat-HEAR<br>**----- End of picture text -----**<br>


_Figure 2._ **Downstream model performance** . (A) NatHEAR and HEAR performance as a function of training data. (B) Difference in performance on HEAR and NatHEAR (excluding the DCASE-2016 task). Box limits reflect first and third quartile, center line the median. 

## **4. Downstream evaluation** 

**HEAR** : This benchmark task suite (Turian et al., 2022) includes a wide range of tasks to evaluate the downstream performance of audio representation models (Turian et al., 2022). We selected the same subset of HEAR tasks as previously used in (Yadav et al., 2024), but added HEAR’s time-stamp-based sound event detection task DCASE-2016 Task 2 (Mesaros et al., 2018) to enable in-depth evaluation of audio scene analysis capabilities. 

and moving sound sources, varying in level of complexity. The datasets included are TUT-2018 (Adavanne et al., 2019a), TAU-2019 (Adavanne et al., 2019b), TAU-NIGENS2020 (Politis et al., 2021), TAU-NIGENS-2021 (Politis et al., 2021) and STARSS23 (Shimada et al., 2023). More information regarding the datasets can be found in Appendix Table 5. By embedding these evaluation tasks into the standardized HEAR setup, we present a standardized evaluation of SELD performance for audio foundation models in challenging real-world scenarios. 

**NatHEAR - Simulated naturalistic sound scenes** : To test 

performance in a wide range of noisy, spatial, naturalistic scenes, we generated NatHEAR. This benchmark task suite contains the same tasks as the original HEAR, but the sound clips are converted to high-quality simulations of naturalistic scenes rather than clean, single-channel scenes. NatHEAR exists in two audio formats: a two-channel binaural format and a four-channel first-order Ambisonics format. We furthermore included sound localization and T60 estimation tasks in NatHEAR across two audio domains. Specifically, we used the SC-5 as an example of the speech domain and ESC-50 as an example of the environmental sound domain. These localization tasks are modeled as a multi-output regression task in which model outputs represent the estimated 3D Cartesian coordinates [ _x, y, z_ ] on the unit sphere (Adavanne et al., 2018). Further, T60 tasks are modeled as regression tasks with continuous outputs. 

**RealSELD - recordings of real-world sound scenes** : To evaluate performance in real-world scenes in a standardized manner, we created RealSELD. This benchmark task suite comprises real-life datasets for sound event localization and detection (SELD) tasks collected from previous DCASE challenges (Mesaros et al., 2025) that include both static 

## **4.1. Evaluation methodology** 

**Downstream task evaluation:** Following the HEAR protocol (Turian et al., 2022) for downstream task evaluation, we extracted embeddings from the frozen pretrained models and subsequently trained a shallow downstream classifier on these embeddings to assess the extent to which the learned representations generalize across a broad range of tasks. We applied this set-up to all benchmark suites - that is, HEAR, NatHEAR and RealSELD - to create a uniform evaluation set-up (for an overview of all tasks, see Appendix D). The procedure for the GRAM embedding extraction is described in Appendix C. 

To test downstream SELD performance on the real-life recordings in benchmark task suite RealSELD (resampled to 32 kHz), we formulated the localization and detection task following the commonly used Activity-Coupled Cartesian Direction of Arrival (ACCDOA) framework (Shimada et al., 2021). This framework jointly models sound event detection and localization across target sound classes. Further, to adapt the RealSELD datasets to the HEAR pipeline, we processed the ground truth timestamp labels into fixedduration segments. For static sources, we extracted start and 

5

<!-- page: 6 -->

**Spatial general purpose audio representation learning** 

end times alongside fixed azimuth and elevation coordinates. For moving sources, we extracted active segments and their corresponding time-varying directions. In frames where no sources are active, the ACCDOA target is defined as a zero vector. Crucially, the datasets in RealSELD utilize 100 ms segments while GRAM operates on 80 ms segments (Appendix C). To temporally align GRAM with the RealSELD labels, we applied average pooling to the model representations. We then extract the time-stamp embeddings and their corresponding ground truth labels using the HEAR evaluation kit, and linear probe the extracted information using the HEAR evaluation protocol. 

**Impact of real-life recordings during training:** In addition to the evaluation of the direct transfer of GRAMAmbisonics to real-life recordings described above, we tested the impact of fine-tuning GRAM-Ambisonics on real-life recordings, as well as training GRAM-Ambisonics from scratch on real-world recordings. To this end, we used the most challenging task in RealSELD, namely the STARSS23 dataset (an extended version with additional training data,similar to Politis et al. (2022)). To fine-tune GRAM-Ambisonics, we initialized the model using pretrained weights. Both fine-tuning and training from scratch were implemented with a batch size of 512 and 100 training epochs. All other experimental settings followed the SELD baseline model (Politis et al., 2022). 

**Matching audio formats:** The input audio format varies across audio foundation models and benchmark task suites. Hence, to adapt the audio format of the multi-channel task suites NatHEAR and RealSELD to the single-channel models, we selected the omnidirectional channel _W_ of the firstorder Ambisonics (Zotter & Frank, 2019) as their model input. Further, to adapt the audio format of single-channel task suite HEAR to the multi-channel models GRAMBinaural and GRAM-Ambisonics, we duplicated the original HEAR single-channel spectrograms to generate two- or four-channel audio inputs. 

## **4.2. Performance metrics** 

**HEAR and NatHEAR** : For each model _m_ , we use two metrics to assess overall performance across for each of these task suites: The average performance across all tasks and the metric _s_ ( _m_ ), which reflects a model’s improvement relative to the maximum improvement over the baseline achieved by the current state-of-the-art model, averaged across all tasks in the benchmark. This metric effectively ranks model improvements relative to the baseline as a function of the current maximum improvement (see Appendix E). We use the HEAR-Naive baseline based on mel-spectrograms as baseline here (Turian et al., 2022). 

**Evaluation of SELD tasks on real-life recordings:** For dynamic scenes, we used four joint localization and detec- 

tion metrics for the RealSELD dataset (Politis et al., 2020b), which are widely-used in audio-only SELD tasks (Politis et al., 2022). These include two metrics for location-aware detection (error rate ER20 _◦_ and F-score F20 _◦_ ), and two metrics for class-aware localization (localization error LE _CD_ and localization recall LR _CD_ ). For static scenes, inline with the reported baseline results (Adavanne et al., 2018), we used the localization independent version of these metrics. See Appendix H for details on evaluation metrics and the ACCDOA training framework. 

## **5. Results** 

**Downstream tasks in NatHEAR and HEAR:** Table 1 demonstrates that GRAM-Binaural and GRAM-Ambisonics learn robust general-purpose audio representations, outperforming all other self-supervised audio representation models on NatHEAR. Moreover, (Figure 2 A shows that GRAM requires substantially less training data to achieve this performance level). Further, all GRAMs surpass all other self-supervised audio representation models on the dry, non-spatial and clean sound scenes in HEAR (Table 6). GRAM-Clean achieved state-of-the art performance, followed by GRAM-Binaural and GRAM-Ambisonics. Comparing the performance of the GRAM models on NatHEAR and HEAR highlights two key findings: (1) The success of GRAM-Binaural and GRAM-Ambisonics on NatHEAR demonstrates the impact of the naturalistic training pipeline for downstream task performance in real-world scenes (Figure 2B); and (2) The superior performance of GRAMBinaural and GRAM-Ambisonics on HEAR relative to other audio representation models indicates that the naturalistic training pipeline does not degrade downstream performance on clean, dry sound scenes. 

**Sound localization and T60 estimation in NatHEAR:** Figure 3A shows that GRAMs exhibit excellent localization performance in simulated real-world sound scenes despite the presence of reverberation and background noise . GRAM-Ambisonics has the lowest localization error, substantially outperforming the supervised localization model Spatial-AST. For GRAM-Binaural, time-based masking is more successful than patch-based masking. Figure 3B shows that GRAMs with spatial attributes (i.e. GRAMBinaural and -Ambisonics) estimated T60s statistically significantly better than GRAM-Clean and Spatial-AST. 

**SELD task in real-life recordings:** RealSELD contains two tasks with real-life recordings of static scenes: TUT2018 Real (Adavanne et al., 2019a) and TAU-2019 (Adavanne et al., 2019b). Table 2 demonstrates that GRAMAmbisonics generalizes well to both datasets, achieving lower localization errors than both supervised models trained on in-domain data as well as self-supervised models. Furthermore, Table 3 shows that GRAM-Ambisonics 

6

<!-- page: 7 -->

**Spatial general purpose audio representation learning** 

_Table 1._ Performance on NatHEAR. Reported values reflect the average performance _±_ standard deviation, calculated using _n_ -fold cross-validation as specified by the HEAR. Bold numbers indicate the best performing model on the specific task. Grayed-out rows indicate supervised models. Tasks are specified in Appendix D. **LS** : LibriSpeech 960 h, **AS** : AudioSet 

|**Model**<br>**Corpus**|**Acoustic Events and Scene Analysis**<br>**DCASE FSD50K**<br>**LC**<br>**ESC-50**|**Speech**<br>**CD**<br>**VL**<br>**SC-5**|**Music**<br>**NS**<br>**BO**<br>**Mri-S**<br>**Mri-T**|**s(m) Avg.**|
|---|---|---|---|---|
|HEAR-Naive<br>-|26.5<br>8.7<br>27_._4_±_1_._6 17_._2_±_2_._2|32_._3_±_2_._2 11_._7_±_2_._2<br>12.0|75.6 84_._3_±_4_._5 68_._6_±_1_._3<br>60_._5_±_1_._3|0.0<br>38.6|
|Wav2Vec2<br>LS<br>HuBERT<br>LS<br>WavLM<br>LS|32.0<br>23.0<br>54_._6_±_1_._9 36_._4_±_2_._9<br>57.6<br>26.6<br>52_._5_±_2_._2 49_._5_±_2_._2<br>25.3<br>20.5<br>52_._1_±_0_._6 41_._4_±_2_._1|48_._6_±_0_._6 27_._2_±_1_._6<br>78.9<br>57_._4_±_1_._1 46_._8_±_3_._4<br>89.2<br>52_._3_±_1_._5<br>**47.9**_±_4_._6 89.9|15.2 71_._2_±_6_._4 75_._7_±_0_._5<br>45_._9_±_0_._6<br>16.0 77_._1_±_6_._0 78_._2_±_0_._7<br>52_._4_±_1_._6<br>11.2 61_._4_±_7_._2 69_._3_±_0_._9<br>39_._0_±_2_._0|32.5 46.2<br>45.2 54.8<br>37.8 46.4|
|MAE<br>AS<br>SSAST<br>AS + LS<br>BEATs<br>AS<br>MWMAE<br>AS<br>SSAM<br>AS|–<br>27.9<br>53_._2_±_1_._0 65_._7_±_1_._2<br>–<br>15.6<br>41_._6_±_2_._4 44_._8_±_1_._0<br>–<br>46.5<br>63_._7_±_1_._2 72_._6_±_3_._9<br>83.8<br>44.3<br>64_._8_±_1_._1 69_._7_±_5_._6<br>70.0<br>46.0<br>63_._2_±_1_._1 73_._1_±_2_._4|48_._5_±_1_._3 19_._0_±_1_._5<br>57.4<br>39_._7_±_2_._9 12_._7_±_1_._3<br>19.9<br>54_._8_±_1_._6 27_._5_±_4_._3<br>83.5<br>59_._3_±_1_._0 31_._8_±_1_._8<br>86.7<br>62_._3_±_1_._0 38_._8_±_2_._6<br>86.2|53.4 79_._2_±_7_._8 81_._0_±_4_._9 56_._5_±_12_._3<br>52.0 81_._8_±_3_._6 76_._5_±_3_._6<br>64_._6_±_1_._5<br>54.2 70_._3_±_6_._2 83_._2_±_1_._0<br>71_._0_±_1_._4<br>59.2 77_._1_±_3_._6 90_._1_±_0_._8<br>73_._9_±_0_._6<br>65.4 84_._3_±_7_._0 92_._6_±_0_._4<br>76_._8_±_1_._0|34.5 54.2<br>17.5 44.9<br>55.7 62.7<br>62.5 67.3<br>68.4 68.9|
|GRAM-Binaural<br>AS|**93.0**<br>**52.8**<br>**72.3**_±_0_._7<br>**82.6**_±_3_._2|**63.3**_±_1_._3<br>35_._1_±_3_._8<br>**91.0**|**67.6**<br>**85.6**_±_5_._1<br>**91.7**_±_0_._9<br>**78.3**_±_1_._3|**74.8**<br>**73.9**|
|GRAM-Ambisonics<br>AS|90.2<br>49.5<br>68_._8_±_0_._9<br>79_._4_±_2_._7|61_._4_±_0_._9<br>36_._4_±_4_._2<br>87.2|64.6<br>83_._4_±_4_._7<br>91_._3_±_0_._6<br>78_._1_±_1_._4|70.5<br>71.8|
|GRAM-Clean<br>AS|90.9<br>50.5<br>66_._4_±_0_._8<br>80_._0_±_2_._4|62_._0_±_1_._3<br>32_._2_±_2_._3<br>87.3|65.2<br>82_._2_±_5_._6<br>90_._2_±_0_._8<br>75_._1_±_0_._7|67.3<br>71.1|
|PASST<br>AS<br>Spatial-AST<br>AS|–<br>56.9<br>52_._1_±_1_._9 89_._7_±_2_._1<br>–<br>40.0<br>49_._9_±_1_._5 70_._1_±_3_._3|49_._9_±_1_._0 18_._4_±_2_._3<br>61.1<br>41_._6_±_0_._5 11_._7_±_2_._7<br>54.8|16.0 93_._6_±_4_._0 85_._5_±_1_._7<br>55_._6_±_3_._0<br>50.2 77_._1_±_2_._8 77_._7_±_0_._9<br>55_._0_±_1_._6|56.2 57.9<br>30.9 52.8|



**==> picture [485 x 117] intentionally omitted <==**

**----- Start of picture text -----**<br>
A Localization performance B T60 Estimation Results GRAM-Bin. Patch<br>GRAM-Bin. Time<br>SC-5 ESC-50 SC-5 ESC-50<br>180 0.15 GRAM-Ambisonics<br>GRAM-Clean<br>SpatialAST(supervised)<br>120 0.10<br>   60 0.05<br>      0 0.00<br>Models Models Models Models<br>)DoA error (°<br>Absolute errorr (s)<br>**----- End of picture text -----**<br>


_Figure 3._ **Sound localization and T60 estimation in simulated real-world sound scenes** . (A) Direction of arrival (DoA) error. (B) T60 estimation absolute error. Boxes indicate first and third quartile; center line: median; whiskers: 1.5 times the interquartile range. 

_Table 2._ Performance on real-life recordings of static sound scenes (two tasks in RealSELD). Scores reflect direct transfer without fine-tuning (HEAR pipeline). 

|||**TUT-2018**|**TUT-2018**||**TAU-2019**|
|---|---|---|---|---|---|
||**1 source**|**2 sources**||**3 sources**|**2 sources**|
|_Detection scores (Error Rate ↓/ F-score ↑)_||||||
|SELDnet|0.40 / 60.3|0.49|/ 53.1|0.53 / 51.1|0.34 / 79.9|
|MSEDnet|0.35 / 66.2|0.38|/ 61.6|**0.41**/ 59.5|–|
|SEDnet|0.38 / 64.6|0.42|/ 61.5|0.43 / 57.2|–|
|GRAM-Ambisonics|**0.30**/**80.3**|**0.38**|/**75.0**|0.43 /**70.8**|**0.23**/**86.4**|
|_Localization scores (Localization Error◦↓/ Frame_||||_Recall ↑)_||
|SELDnet|26.6 / 64.9|33.7|/ 41.5|36.1 / 24.6|28.5 /**85.4**|
|DOAnet|**6.3**/ 46.5|20.1|/ 11.5|25.8 / 2.9|–|
|Spatial Librispeech|12.4 / –||–|–|–|
|ELSA|15.0 / –||–|–|–|
|GRAM-Ambisonics|11.7 /**82.1**|**18.6**|/**49.9**|**23.0**/**28.4**|**12.6**/ 79.1|



also achieves competitive performance on the datasets comprising real-life recordings of dynamic sound scenes with moving sound sources (TAU-NIGENS-2020 and TAUNIGENS-2021). GRAM-Ambisonics surpasses the supervised baseline on the TAU-NIGENS-2020 dataset (Politis et al., 2020a) as presented in the DCASE challenge. Although GRAM-Ambisonics did not outperform the baseline 

on TAU-NIGENS-2021 (Politis et al., 2021) (likely due to increased polyphony and the high-intensity directional noise interferers), our results show that even without any fine-tuning or specialized data augmentation during training, GRAM-Ambisonics generalizes well to moving-sound scenes in very challenging real-world conditions. 

**Impact of real-life recordings during training:** Figure 5 shows that fine-tuning a pre-trained GRAM-Ambisonics model generalizes better and faster to real-life recordings (STARSS23) than training a GRAM-Ambisonics model from scratch on real-world recordings. This highlights that pre-training on simulated naturalistic scenes provides a strong starting point for efficient generalization to real-life recordings through fine-tuning. 

**Mixing clean and naturalistic scenes in pre-training:** We investigated to what extent pretraining on a mixture of clean and simulated naturalistic sound scenes rather than naturalistic sound scenes only affected the performance of GRAM-Binaural and GRAM-Ambisonics on HEAR and NatHEAR. Figure 4 (left panel) shows that performance 

7

<!-- page: 8 -->

**Spatial general purpose audio representation learning** 

**==> picture [441 x 138] intentionally omitted <==**

_Figure 4._ **Ablation studies** . Scores reflect downstream task performance across all tasks of the benchmark task suite (y-axis, _s_ ( _m_ )). From left to right: (1) Impact of mixing clean and naturalistic scenes during pre-training of GRAM-Binaural. (2) Impact of ratio _λ_ of clean and naturalistic scenes during pre-training of GRAM-Ambisonics. (3) Effect of masking strategy for GRAM-Binaural (4) Comparing Mamba and Transformer encoders for binaural audio. 

_Table 3._ Performance on real-life recordings of dynamic sound scenes (two tasks in RealSELD). Scores reflect direct transfer without fine-tuning (HEAR pipeline). 

**==> picture [235 x 243] intentionally omitted <==**

**----- Start of picture text -----**<br>
TAU-NIGENS-2020 TAU-NIGENS-2021<br>Location-aware detection scores (ER 20 ◦ / F 20 ◦ )<br>Baseline 0.72 / 37.4 0.73  /  30.7<br>GRAM-Ambisonics 0.57  /  47.4 0.74 / 21.4<br>Localization scores (LECD [◦] [↓] [/][ LR][CD] [↑][)]<br>Baseline 22.8 / 60.7 24.5  /  44.8<br>GRAM-Ambisonics 22.3  /  63.1 37.5 / 49.2<br>0.6 Scratch lr=1e-4<br>Scratch lr=2e-4<br>0.4 Scratch lr=5e-4<br>Scratch lr=1e-3<br>0.2 Fine-tune lr=1e-4<br>Direct Transfer<br>0.0 lr=1e-4<br>150<br>100<br>50<br>0 20 40 60 80 100<br>Localization Recall<br>Localization Error<br>**----- End of picture text -----**<br>


_Figure 5._ **Impact of real-life data during training.** Comparing GRAM-Ambisonics validation scores on STARSS23 for training from scratch, fine-tuning a pre-trained model, and direct transfer of the pre-trained model using HEAR pipeline. 

of GRAM-Binaural on NatHEAR increases with more naturalistic training data (i.e., lower _λ_ ), while performance on HEAR is optimal with a mixture of clean and naturalistic scenes. Figure 4 (second panel from the left) shows that GRAM-Ambisonics performed best on both NatHEAR and HEAR with a mixture of clean and naturalistic scenes, consistent with Devnani et al. (2024). 

**Masking strategy:** Figure 4 (second panel from the right) illustrates that patch-based masking results in better downstream performance on both HEAR and NatHEAR, although masking strategy had less impact on HEAR performance than NatHEAR performance. 

**Encoder architecture:** As shown in Figure 4, the Transformer backbone consistently performed better than the Mamba backbone both on clean downstream tasks and on synthetic naturalistic downstream tasks. 

## **6. Discussion and conclusion** 

We present GRAM, a general-purpose, robust spatial audio representation model based on multi-channel MAE. GRAM demonstrates remarkable performance on HEAR and NatHEAR, achieving state-of-the-art results for a self-supervised spectrogram-based audio foundation model while requiring only a fraction of the training data. Moreover, GRAM is the first audio foundation model to perform SELD tasks in real-life recordings. Our experiments demonstrated that GRAM successfully encoded spatial generalpurpose audio representations, both in simulated naturalistic scenes and recordings of real-world scenes. Additionally, we release NatHEAR and RealSELD, two new benchmark task suites that provide a standardized manner to evaluate the performance of audio foundation models on simulated naturalistic sound scenes and recordings of real-life sound scenes. In sum, GRAM is a new state-of-the-art audio representation model that incorporates spatial learning and exhibits robust performance in real-world sound scenes, representing a crucial step towards successful applications of audio foundation models in real-world environments. 

**Limitations and future work** : The resolution of melspectrograms for binaural inputs was not adequate for learning interaural time differences, which may have hindered the localization performance of the GRAM-Binaural. For 

8

<!-- page: 9 -->

**Spatial general purpose audio representation learning** 

future work, GRAM opens the way to multimodal spatial learning. It can serve as a basis for downstream applications such as audio-visual scene representation learning (Mahmud & Marculescu, 2023), robotics (Ledder et al., 2025), and audio-language representation learning (Zheng et al., 2024; Chu et al., 2023). 

## **Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

## **Acknowledgements** 

This project received funding from the NWO Talent Program (VI.Veni.202.184; KH). This work used the Dutch national e-infrastructure with the support of the SURF Cooperative, using grant no. EINF-12218. We would like to thank Robert Jan Schlimbach from the Snellius team for helpful discussions and their help with high-performance cluster utilization. 

## **References** 

- Adavanne, S., Politis, A., Nikunen, J., and Virtanen, T. Sound event localization and detection of overlapping sources using convolutional recurrent neural networks. _IEEE Journal of Selected Topics in Signal Processing_ , 13 (1):34–48, 2018. 

- Adavanne, S., Politis, A., Nikunen, J., and Virtanen, T. Sound event localization and detection of overlapping sources using convolutional recurrent neural networks. _IEEE Journal of Selected Topics in Signal Processing_ , 13 (1):34–48, 2019a. doi: 10.1109/JSTSP.2018.2885636. 

- Adavanne, S., Politis, A., and Virtanen, T. A multi-room reverberant dataset for sound event localization and detection. In _Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019)_ , pp. 10–14, New York University, NY, USA, October 2019b. URL https://dcase. community/workshop2019/proceedings. 

- Algazi, V., Duda, R., Thompson, D., and Avendano, C. The cipic hrtf database. In _Proceedings of the 2001 IEEE Workshop on the Applications of Signal Processing to Audio and Acoustics (Cat. No.01TH8575)_ , pp. 99–102, 2001. doi: 10.1109/ASPAA.2001.969552. 

- Anantapadmanabhan, A., Bellur, A., and Murthy, H. A. Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization. In _2013 IEEE International Conference on Acoustics, Speech and_ 

_Signal Processing_ , pp. 181–185, 2013. doi: 10.1109/ ICASSP.2013.6637633. 

- Baade, A., Peng, P., and Harwath, D. Mae-ast: Masked autoencoding audio spectrogram transformer. In _Interspeech 2022_ , pp. 2438–2442, 2022. doi: 10.21437/ Interspeech.2022-10961. 

- Baevski, A., Zhou, H., Mohamed, A., and Auli, M. wav2vec 2.0: a framework for self-supervised learning of speech representations. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546. 

- Bizley, J. K. and Cohen, Y. E. The what, where and how of auditory-object perception. _Nature Reviews Neuroscience_ , 14(10):693–707, 2013. 

- Bregman, A. S. Auditory scene analysis. In _Proceedings of the 7th International Conference on Pattern Recognition_ , pp. 168–175. Citeseer, 1984. 

- Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., and Verma, R. CREMA-D: Crowd-sourced emotional multimodal actors dataset. _IEEE Trans Affect Comput_ , 5(4):377–390, October 2014. 

- Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner, M., Savva, M., Song, S., Zeng, A., and Zhang, Y. Matterport3d: Learning from rgb-d data in indoor environments. _International Conference on 3D Vision (3DV)_ , 2017. 

- Chen, C., Schissler, C., Garg, S., Kobernik, P., Clegg, A., Calamia, P., Batra, D., Robinson, P. W., and Grauman, K. Soundspaces 2.0: A simulation platform for visualacoustic learning. In _NeurIPS 2022 Datasets and Benchmarks Track_ , 2022a. 

- Chen, K., Du, X., Zhu, B., Ma, Z., Berg-Kirkpatrick, T., and Dubnov, S. Hts-at: A hierarchical token-semantic audio transformer for sound classification and detection. In _ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 646–650, 2022b. doi: 10.1109/ICASSP43922.2022. 9746312. 

- Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian, Y., Qian, Y., Wu, J., Zeng, M., and Wei, F. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. _CoRR_ , abs/2110.13900, 2021. URL http://dblp.uni-trier.de/db/journals/ corr/corr2110.html#abs-2110-13900. 

9

<!-- page: 10 -->

**Spatial general purpose audio representation learning** 

- Chen, S., Wu, Y., Wang, C., Liu, S., Tompkins, D., Chen, Z., Che, W., Yu, X., and Wei, F. BEATs: Audio pretraining with acoustic tokenizers. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 5178–5193. PMLR, 23– 29 Jul 2023. URL https://proceedings.mlr. press/v202/chen23ag.html. 

- Chong, D., Wang, H., Zhou, P., and Zeng, Q. Masked spectrogram prediction for self-supervised audio pre-training. In _ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 1–5, 2023. doi: 10.1109/ICASSP49357.2023.10095691. 

- Chu, Y., Xu, J., Zhou, X., Yang, Q., Zhang, S., Yan, Z., Zhou, C., and Zhou, J. Qwen-audio: Advancing universal audio understanding via unified large-scale audiolanguage models. _arXiv preprint arXiv:2311.07919_ , 2023. 

- Devnani, B., Seto, S., Aldeneh, Z., Toso, A., Menyaylenko, E., Theobald, B.-J., Sheaffer, J., and Sarabia, M. Learning spatially-aware language and audio embeddings. In Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C. (eds.), _Advances in Neural Information Processing Systems_ , volume 37, pp. 33505–33537. Curran Associates, Inc., 2024. doi: 10.52202/079017-1056. 

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. _ICLR_ , 2021. 

- Engel, J., Resnick, C., Roberts, A., Dieleman, S., Eck, D., Simonyan, K., and Norouzi, M. Neural audio synthesis of musical notes with wavenet autoencoders, 2017. 

- Fonseca, E., Favory, X., Pons, J., Font, F., and Serra, X. Fsd50k: An open dataset of human-labeled sound events. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 30:829–852, 2022. doi: 10.1109/TASLP.2021.3133208. 

- Gemmeke, J. F., Ellis, D. P. W., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., Plakal, M., and Ritter, M. Audio set: An ontology and human-labeled dataset for audio events. In _2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 776–780, 2017. doi: 10.1109/ICASSP.2017.7952261. 

- Gong, Y., Chung, Y.-A., and Glass, J. AST: Audio Spectrogram Transformer. In _Proc. Interspeech 2021_ , pp. 571–575, 2021a. doi: 10.21437/Interspeech.2021-698. 

- Gong, Y., Chung, Y.-A., and Glass, J. Psla: Improving audio tagging with pretraining, sampling, labeling, and aggregation. _IEEE/ACM Trans. Audio, Speech and Lang. Proc._ , 29:3292–3306, October 2021b. ISSN 2329-9290. doi: 10.1109/TASLP.2021.3120633. URL https:// doi.org/10.1109/TASLP.2021.3120633. 

- Gong, Y., Lai, C.-I., Chung, Y.-A., and Glass, J. Ssast: Selfsupervised audio spectrogram transformer. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pp. 10699–10709, 2022. 

- Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_ , 2023. 

- Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., and Mohamed, A. Hubert: Selfsupervised speech representation learning by masked prediction of hidden units. _IEEE/ACM Trans. Audio, Speech and Lang. Proc._ , 29:3451–3460, October 2021. ISSN 2329-9290. doi: 10.1109/TASLP.2021. 3122291. URL https://doi.org/10.1109/ TASLP.2021.3122291. 

- Huang, P.-Y., Xu, H., Li, J., Baevski, A., Auli, M., Galuba, W., Metze, F., and Feichtenhofer, C. Masked autoencoders that listen. In _NeurIPS_ , 2022. 

- Kahn, J., Riviere, M., Zheng, W., Kharitonov, E., Xu, Q., Mazare,´ P.-E., Karadayi, J., Liptchinsky, V., Collobert, R., Fuegen, C., et al. Libri-light: A benchmark for asr with limited or no supervision. In _ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 7669–7673. IEEE, 2020. 

- Koutini, K., Schluter,¨ J., Eghbal-zadeh, H., and Widmer, G. Efficient training of audio transformers with patchout. In _Interspeech 2022, 23rd Annual Conference of the International Speech Communication Association, Incheon, Korea, 18-22 September 2022_ , pp. 2753–2757. ISCA, 2022. doi: 10.21437/Interspeech. 2022-227. URL https://doi.org/10.21437/ Interspeech.2022-227. 

- Ledder, W., Qin, Y., and van der Heijden, K. Audio-driven reinforcement learning for head-orientation in naturalistic environments. In _ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 1–5. IEEE, 2025. 

- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ , 2021. 

10

<!-- page: 11 -->

**Spatial general purpose audio representation learning** 

- Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_ , 2017. 

- Maciejewski, M., Wichern, G., and Le Roux, J. Whamr!: Noisy and reverberant single-channel speech separation. In _Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , May 2020. 

- Mahmud, T. and Marculescu, D. AVE-CLIP: AudioCLIPbased Multi-window Temporal Transformer for Audio Visual Event Localization . In _2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ , pp. 5147–5156, Los Alamitos, CA, USA, January 2023. IEEE Computer Society. doi: 10.1109/WACV56688.2023.00513. URL https://doi.ieeecomputersociety.org/ 10.1109/WACV56688.2023.00513. 

- Mesaros, A., Heittola, T., Benetos, E., Foster, P., Lagrange, M., Virtanen, T., and Plumbley, M. D. Detection and classification of acoustic scenes and events: Outcome of the DCASE 2016 challenge. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 26(2):379–393, Feb 2018. ISSN 2329-9290. doi: 10.1109/TASLP.2017. 2778423. 

- Mesaros, A., Serizel, R., Heittola, T., Virtanen, T., and Plumbley, M. D. A decade of dcase: Achievements, practices, evaluations and future challenges. In _ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 1–5, 2025. doi: 10.1109/ICASSP49660.2025.10887673. 

- Mohamed, A., Lee, H.-y., Borgholt, L., Havtorn, J. D., Edin, J., Igel, C., Kirchhoff, K., Li, S.-W., Livescu, K., Maaløe, L., et al. Self-supervised speech representation learning: A review. _IEEE Journal of Selected Topics in Signal Processing_ , 16(6):1179–1210, 2022. 

- Niizumi, D., Takeuchi, D., Ohishi, Y., Harada, N., and Kashino, K. Masked spectrogram modeling using masked autoencoders for learning general-purpose audio representation. In Turian, J., Schuller, B. W., Herremans, D., Kirchoff, K., Perera, P. G., and Esling, P. (eds.), _HEAR: Holistic Evaluation of Audio Representations (NeurIPS 2021 Competition)_ , volume 166 of _Proceedings of Machine Learning Research_ , pp. 1–24. PMLR, 13–14 Dec 2022. URL https://proceedings.mlr.press/ v166/niizumi22a.html. 

- Panayotov, V., Chen, G., Povey, D., and Khudanpur, S. Librispeech: An asr corpus based on public domain audio books. In _2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 5206–5210, 2015. doi: 10.1109/ICASSP.2015.7178964. 

- Piczak, K. J. ESC: Dataset for Environmental Sound Classification. In _Proceedings of the 23rd Annual ACM Conference on Multimedia_ , pp. 1015–1018. ACM Press. ISBN 978-1-4503-3459-4. doi: 10.1145/2733373. 2806390. URL http://dl.acm.org/citation. cfm?doid=2733373.2806390. 

- Politis, A., Adavanne, S., and Virtanen, T. A dataset of reverberant spatial sound scenes with moving sources for sound event localization and detection. In _Proceedings of the Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020)_ , pp. 165–169, Tokyo, Japan, November 2020a. URL https://dcase.community/ workshop2020/proceedings. 

- Politis, A., Mesaros, A., Adavanne, S., Heittola, T., and Virtanen, T. Overview and evaluation of sound event localization and detection in dcase 2019. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 29:684–698, 2020b. URL https://ieeexplore. ieee.org/abstract/document/9306885. 

- Politis, A., Adavanne, S., Krause, D., Deleforge, A., Srivastava, P., and Virtanen, T. A dataset of dynamic reverberant sound scenes with directional interferers for sound event localization and detection. In _Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)_ , pp. 125– 129, Barcelona, Spain, November 2021. ISBN 978-8409-36072-7. URL https://dcase.community/ workshop2021/proceedings. 

- Politis, A., Shimada, K., Sudarsanam, P., Adavanne, S., Krause, D., Koyama, Y., Takahashi, N., Takahashi, S., Mitsufuji, Y., and Virtanen, T. Starss22: A dataset of spatial recordings of real scenes with spatiotemporal annotations of sound events, 2022. URL https: //arxiv.org/abs/2206.01948. 

- Saeed, A., Grangier, D., and Zeghidour, N. Contrastive learning of general-purpose audio representations. In _ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 3875–3879, 2021. doi: 10.1109/ICASSP39728.2021. 9413528. 

- Scheibler, R., Bezzam, E., and Dokmanic,´ I. Pyroomacoustics: A python package for audio room simulation and array processing algorithms. In _2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 351–355, 2018. doi: 10.1109/ICASSP.2018.8461310. 

- Schroeder, M. R. New method of measuring reverberation time. _The Journal of the Acoustical Society of America_ , 37 (6 ~~S~~ upplement):1187–1188, 06 1965. ISSN 0001-4966. 

11

<!-- page: 12 -->

**Spatial general purpose audio representation learning** 

doi: 10.1121/1.1939454. URL https://doi.org/ 10.1121/1.1939454. 

- Shimada, K., Koyama, Y., Takahashi, N., Takahashi, S., and Mitsufuji, Y. Accdoa: Activity-coupled cartesian direction of arrival representation for sound event localization and detection, 2021. URL https://arxiv. org/abs/2010.15306. 

- Shimada, K., Politis, A., Sudarsanam, P., Krause, D., Uchida, K., Adavanne, S., Hakala, A., Koyama, Y., Takahashi, N., Takahashi, S., Virtanen, T., and Mitsufuji, Y. Starss23: an audio-visual dataset of spatial recordings of real scenes with spatiotemporal annotations of sound events. In _Proceedings of the 37th International Conference on Neural Information Processing Systems_ , NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc. 

- Stoter,¨ F.-R., Chakrabarty, S., Habets, E., and Edler, B. Libricount, a dataset for speaker count estimation, April 2018. URL https://doi.org/10.5281/ zenodo.1216072. 

- Tian, M., Srinivasamurthy, A., Sandler, M., and Serra, X. A study of instrument-wise onset detection in beijing opera percussion ensembles. In _2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 2159–2163, 2014. doi: 10.1109/ICASSP. 2014.6853981. 

- Turian, J., Shier, J., Khan, H. R., Raj, B., Schuller, B. W., Steinmetz, C. J., Malloy, C., Tzanetakis, G., Velarde, G., McNally, K., Henry, M., Pinto, N., Noufi, C., Clough, C., Herremans, D., Fonseca, E., Engel, J., Salamon, J., Esling, P., Manocha, P., Watanabe, S., Jin, Z., and Bisk, Y. HEAR: Holistic Evaluation of Audio Representations. In Kiela, D., Ciccone, M., and Caputo, B. (eds.), _Proceedings of the NeurIPS 2021 Competitions and Demonstrations Track_ , volume 176 of _Proceedings of Machine Learning Research_ , pp. 125–145. PMLR, 06–14 Dec 2022. URL https://proceedings.mlr.press/ v176/turian22a.html. 

- Valk, J. and Alumae, T.¨ Voxlingua107: A dataset for spoken language recognition. In _2021 IEEE Spoken Language Technology Workshop (SLT)_ , pp. 652–658, 2021. doi: 10.1109/SLT48900.2021.9383459. 

   - _International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 4593–4597, 2022a. doi: 10.1109/ICASSP43922.2022.9746790. 

   - Wang, S., Politis, A., Mesaros, A., and Virtanen, T. Selfsupervised learning of audio representations from audiovisual data using spatial alignment. _IEEE Journal of Selected Topics in Signal Processing_ , 16(6):1467–1479, 2022b. doi: 10.1109/JSTSP.2022.3180592. 

   - Warden, P. Speech commands: A dataset for limitedvocabulary speech recognition, 2018. URL https: //arxiv.org/abs/1804.03209. 

   - Yadav, S. and Tan, Z.-H. Audio mamba: Selective state spaces for self-supervised audio representations. In _Interspeech 2024_ , pp. 552–556, 2024. doi: 10.21437/ Interspeech.2024-1274. 

   - Yadav, S., Theodoridis, S., Hansen, L. K., and Tan, Z.-H. Masked autoencoders with multi-window local-global attention are better audio learners. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum? id=Q53QLftNkA. 

   - Yang, S., Chi, P.-H., Chuang, Y.-S., Lai, C.-I. J., Lakhotia, K., Lin, Y. Y., Liu, A. T., Shi, J., Chang, X., Lin, G.T., Huang, T.-H., Tseng, W.-C., tik Lee, K., Liu, D.-R., Huang, Z., Dong, S., Li, S.-W., Watanabe, S., Mohamed, A., and yi Lee, H. Superb: Speech processing universal performance benchmark. In _Interspeech 2021_ , pp. 1194– 1198, 2021. doi: 10.21437/Interspeech.2021-1775. 

   - Zheng, Z., Peng, P., Ma, Z., Chen, X., Choi, E., and Harwath, D. Bat: learning to reason about spatial sounds with large language models. In _Proceedings of the 41st International Conference on Machine Learning_ , ICML’24. JMLR.org, 2024. 

   - Zotter, F. and Frank, M. _XY, MS, and First-Order Ambisonics_ , pp. 1–22. Springer International Publishing, Cham, 2019. ISBN 978-3-030-17207-7. doi: 10.1007/ 978-3-030-17207-7 ~~1~~ . URL https://doi.org/10. 1007/978-3-030-17207-7_1. 

- van der Heijden, K., Rauschecker, J. P., de Gelder, B., and Formisano, E. Cortical mechanisms of spatial hearing. _Nature Reviews Neuroscience_ , 20(10):609–623, 2019. 

- Wang, L., Luc, P., Wu, Y., Recasens, A., Smaira, L., Brock, A., Jaegle, A., Alayrac, J.-B., Dieleman, S., Carreira, J., and van den Oord, A. Towards learning universal audio representations. In _ICASSP 2022 - 2022 IEEE_ 

12

<!-- page: 13 -->

**Spatial general purpose audio representation learning** 

## **Appendix** 

## **A. Pre-training specification for GRAMs** 

We trained all GRAMs for 500K steps on an H100 92GB GPU machine with 16 CPU cores. We used the AdamW optimizer (Loshchilov & Hutter, 2017) with weight decay rate of 0.01, gradient clipping, and a cosine learning rate scheduler with 10 K steps warm-up. The initial learning rate was set to 0.0002, and decayed to 0. We minimize the mean squared error (MSE) between the predicted masked patches and their corresponding input spectrogram patches. GRAM training converges in only 40 GPU hours. 

## **B. Evaluated models** 

We compare the performance and efficiency of GRAM-Binaural, GRAM-Ambisonics on downstream tasks with state-of-theart self-supervised audio representation models with a similar number of parameters as GRAM (90 M): MAE-16x16 (Huang et al., 2022), SSAST-patch (Gong et al., 2022), BEATs-iter3 (Chen et al., 2023), MWMAE-B-200-4x16 (Yadav et al., 2024), SSAM (Yadav & Tan, 2024); self-supervised speech representation models Wav2Vec 2.0 Base (Baevski et al., 2020), HuBERT Base (Hsu et al., 2021), WavLM Base (Chen et al., 2021). To quantify the impact of pre-training with naturalistic sound scenes, we further train GRAM-Clean. GRAM-Clean follows the same experimental setup as GRAM-Binaural and GRAM-Ambisonics, but only uses dry audio clips from the AudioSet. Furthermore, we included Spatial-AST (Zheng et al., 2024) because it is the only model trained on spatial sound scenes. 

## **C. Extracting GRAM embeddings for downstream tasks** 

We extracted GRAM embeddings for downstream evaluations by encoding embeddings for all patches _P_ 1 _, . . . , Pn_ using the GRAM encoder. We used the exact patch aggregation process as in (Niizumi et al., 2022). Audio clips were split into non-overlapping 2-second chunks and the embedded patches concatenated over time. Later, we took the mean over the time axis to generate scene embeddings independent of the input audio duration. Finally, to evaluate GRAMs on the NatHEAR on localization tasks, we used [CLS] embeddings of the 2-second samples, and averaged them to create scene embeddings for localization tasks. 

## **D. HEAR and NatHEAR and RealSELD Tasks** 

Tables 4 and 5 illustrate the abbreviations, task descriptions, and the types we have used to benchmark our models. 

_Table 4._ Overview of the HEAR and NatHEAR tasks. 

|**Abbreviation**|**Task Name**|**Description**|**Type**|
|---|---|---|---|
|DCASE|DCASE-2016 Task 2 (Mesaros et al.,2018)|Event detection of overlapping offce sounds in synthetic mixtures|Scene Analysis|
|FS50K|FSD50k (Fonseca et al.,2022)|Multilabel, large scale audio tagging|Scene Analysis|
|LC|LibriCount (St¨oter et al.,2018)|Speaker Count Identifcation, Simulated Cocktail Party|Scene Analysis|
|ESC-50|ESC-50 (Piczak)|Environmental Sound Classifcation|Environmental Sound Classifcation|
|CD|Crema-D (Cao et al.,2014)|Emotion Recognition|Speech Analysis|
|VL|VoxLingua107 Top10 (Valk & Alum¨ae,2021)|Spoken language identifcation|Speech Analysis|
|SC-5|Speech Command 5h (Warden,2018)|Keyword Spotting, reduced training subset|Speech Analysis|
|NS|NSynth Pitch 5h (Engel et al.,2017)|Pitch Classifcation, reduced training subset|Pitch Classifcation|
|BO|Beijing Opera (Tian et al.,2014)|Classifying percussion instruments|Percussion|
|Mri-S|Mridangam Stroke (Anantapadmanabhan et al.,2013)|Stroke classifcation in pitched percussion instruments|Percussion|
|Mri-T|Mridangam Tonic (Anantapadmanabhan et al.,2013)|Tonic classifcation in pitched percussion instruments|Percussion|



## **E. Downstream performance metric** 

Similar to the procedure in SUPERB (Yang et al., 2021), let _st_ be the metric for task _t_ . We then calculate the generalizability metric HEAR _s_ ( _m_ ), and NatHEAR _s_ ( _m_ ) for model _m_ as: 

13

<!-- page: 14 -->

**Spatial general purpose audio representation learning** 

_Table 5._ Characteristics of the evaluation datasets for RealSELD. 

|**Name**|**Motion**|**Impulse Response**|**Max Overlap**|**Noise**|
|---|---|---|---|---|
|TUT-2018 Real (ov1, 2, 3) (Adavanne et al.,2019a)|Static|Real|1, 2, 3|Ambient 30dB|
|TAU-2019 (Adavanne et al.,2019b)|Static|Real|2|Ambient 30dB|
|TAU-NIGENS-2020 (Politis et al.,2020a)|Dynamic|Real|2|Ambient [30-6dB]|
|TAU-NIGENS-2021 (Politis et al.,2021)|Dynamic|Real|3|Directional interference + Ambient|
|STARSS23 (Shimada et al.,2023)|Dynamic|–|6|Recorded|



**==> picture [182 x 29] intentionally omitted <==**

Intuitively, this metric ranks the improvement of models over the baseline as a function of the maximum improvement over the baseline obtained by the current state-of-the-art. Note that we replace _st_ ( _m_ ) for task _t_ of model _m_ with 0 when the model scores below baseline performance for task _t_ . Similarly, when _st_ ( _SOTA_ ) is lower than baseline for task _t_ , we set for all models _st_ for this task to 0. In this way, all values are restricted to a range of improvement between 0% and 100%. 

## **F. Additional Ablation Studies** 

Firstly, we further investigated the masking ratio, and in batch sampling as a function of HEAR and NatHEAR performance. Secondly, we investigated the localization performance as a function of the mixture of naturalistic and clean audio ( _λ_ ). Thirdly, we investigated the localization performance in terms of noise levels in the NatHEAR-Synethic benchmark, which is low [20-40dB], medium [10-20dB] and high [5-10]dB. Lastly, we examined the effect of in-batch sampling when the effective batch size is held constant. For this experiment, we used gradient accumulation over 16 batches. Consequently, the number of in-batch samples was set to 16, yielding an effective batch size of 512 for both models. 

**==> picture [303 x 116] intentionally omitted <==**

_Figure 6._ Additional ablation studies. Effect of hyperparameters on HEAR and NatHEAR Performance. From left to right; (1) GRAMBinaural downstream performance as a function of the number of in batch samples. (2) The effect of masking ratio for GRAM-Binaural. Important to note that GRAM-Binaural depicted in (2) was trained on reduced number of samples (16 _→_ 4). 

**In-batch sampling:** Figure 6 1 depicts that in-batch sampling helped immensely with the downstream performance on both HEAR and NatHEAR downstream. Increasing the number of in-batch samples leads to higher batch sizes with minimal computational constraints. Furthermore, Figure 7 shows that in-batch sampling does not result in a drop in downstream performance or model convergence. 

**Masking ratio:** Figure 6 2 depicts that optimal masking ratio is 0.6 for HEAR and NatHEAR performance, and higher masking ratios, such as 0.9 harms the performance. 

## **G. Results on original HEAR benchmark suite** 

We evaluated our models on the dry, non anechoic, and non-spatialized HEAR Benchmark suite. Table 6 depicts the achieved results on the HEAR sub tasks. 

14

<!-- page: 15 -->

**==> picture [207 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
Spatial general purpose audio representation learning<br>**----- End of picture text -----**<br>


**==> picture [392 x 221] intentionally omitted <==**

**----- Start of picture text -----**<br>
CD ESC-50 LC VL<br>With Sampling<br>Without Sampling<br>Training steps 1e [4] Training steps 1e [4] Training steps 1e [4] Training steps 1e [4]<br>CD ESC-50 LC VL<br>Training steps 1e [4] Training steps 1e [4] Training steps 1e [4] Training steps 1e [4]<br>Score<br>Score<br>**----- End of picture text -----**<br>


_Figure 7._ Additional ablation studies. Effect of in batch sampling on HEAR and NatHEAR performance when the effective batch size is kept the same. From top to bottom; (1) GRAM-Binaural downstream performance on HEAR as a function of the in-batch sampling (2) GRAM-Binaural downstream performance on NatHEAR as a function of the in-batch sampling 

_Table 6._ Performance comparison of audio representation models across HEAR tasks. All values represent the HEAR scores with standard deviation where available. Bold numbers indicate the best performing model on the specific task. SSAST* is trained on both AudioSet and Librispeech. 

|**Model**|**Acoustic Events and Scene Analysis**<br>**DCASE FSD50K**<br>**LC**<br>**ESC-50**|**Speech**<br>**CD**<br>**VL**<br>**SC-5**|**Music**<br>**NS**<br>**BO**<br>**Mri-S**<br>**Mri-T**|**s(m) Avg.**|
|---|---|---|---|---|
|**Baseline**|||||
|HEAR-Naive|8.8<br>13.2<br>43_._5_±_1_._6 28_._6_±_3_._1|38_._0_±_2_._3 14_._8_±_3_._0 13.3|87.6 **98.7**_±_1_._9 94_._1_±_0_._5<br>87_._6_±_6_._4|0.0<br>48.0|
|**Speech SSL**|||||
|Wav2Vec 2.0<br>HuBERT<br>WavLM|23.5<br>29.4<br>69_._9_±_2_._1 46_._4_±_1_._8<br>78.3<br>32.8<br>63_._3_±_1_._2 58_._6_±_2_._8<br>27.0<br>25.7<br>61_._3_±_2_._3 49_._5_±_3_._8|57_._3_±_1_._1 34_._9_±_2_._4 85.3<br>71_._2_±_1_._2 **65.2**_±_2_._9 **94.0**<br>64_._3_±_1_._3 60_._1_±_3_._2 93.8|17.4 81_._4_±_4_._8 90_._7_±_0_._8<br>77_._0_±_0_._9<br>19.8 93_._2_±_5_._9 94_._6_±_0_._4<br>85_._0_±_2_._5<br>18.2 84_._3_±_6_._3 88_._8_±_1_._0<br>76_._8_±_0_._5|30.7 55.7<br>43.6 68.7<br>36.1 59.1|
|**AudioSet SSL**|||||
|MAE<br>SSAST*<br>BEATs<br>MWMAE<br>SSAM<br>GRAM-Binaural<br>GRAM-Ambisonics<br>GRAM-Clean|–<br>33.4<br>62_._3_±_1_._1 72_._9_±_2_._1<br>–<br>21.4<br>57_._8_±_3_._3 58_._3_±_2_._6<br>–<br>54.1<br>77_._8_±_1_._2 85_._8_±_2_._9<br>94.2<br>51.8<br>80_._3_±_1_._9 82_._2_±_3_._2<br>87.3<br>53.5<br>75_._5_±_1_._4 82_._9_±_3_._6<br>**95.6**<br>56.1<br>81_._0_±_1_._1 86_._7_±_2_._4<br>94.3<br>53.0<br>79_._4_±_1_._5 85_._9_±_1_._5<br>95.3<br>56.8<br>**81.3**_±_1_._8 **87.5**_±_2_._3|60_._8_±_1_._8 21_._3_±_5_._8 66.6<br>48_._0_±_2_._1 15_._4_±_2_._6 22.0<br>66_._9_±_2_._5 39_._7_±_4_._3 86.9<br>74_._4_±_1_._5 45_._5_±_1_._7 91.6<br>70_._2_±_0_._4 56_._4_±_5_._2 89.3<br>75_._0_±_1_._4 53_._2_±_3_._0 92.5<br>71_._9_±_1_._9 53_._7_±_1_._2 89.6<br>**75.1**_±_0_._6 57_._3_±_3_._4 93.5|63.6 94_._5_±_5_._6 94_._8_±_0_._6 85_._1_±_10_._4<br>64.2 95_._8_±_4_._3 90_._2_±_5_._9<br>89_._1_±_8_._0<br>68.6 94_._1_±_3_._5 95_._5_±_0_._4<br>96_._6_±_0_._5<br>69.4 95_._8_±_4_._3 97_._5_±_0_._4<br>97_._6_±_0_._6<br>72.6 93_._2_±_3_._5 **97.8**_±_0_._5<br>96_._9_±_0_._5<br>**77.0** 94_._9_±_3_._2 97_._3_±_0_._3<br>98_._1_±_0_._2<br>73.8 94_._9_±_4_._9 97_._6_±_0_._5<br>**98.5**_±_0_._4<br>75.8 95_._8_±_3_._7 97_._4_±_0_._3<br>98_._0_±_0_._2|31.3 65.5<br>15<br>56.2<br>59.2 76.6<br>68.9 80.8<br>69.0 79.6<br>72.3 82.5<br>71.3 81.1<br>**73.8 83.1**|
|**Supervised**<br>PASST<br>Spatial-AST|–<br>**64.1**<br>60_._7_±_3_._7 **94.8**_±_0_._3<br>61_._8_±_1_._1 25_._9_±_2_._6 68.7<br>24.2 **96.6**_±_3_._2 96_._4_±_0_._7<br>87_._8_±_1_._2<br>46.2 68.1<br>–<br>54.7<br>72_._6_±_1_._5 90_._3_±_1_._7<br>62_._2_±_1_._3 29_._1_±_1_._9 80.6<br>69.8 96_._2_±_5_._3 96_._2_±_0_._4<br>94_._6_±_0_._6<br>54.6 74.6||||



## **H. SELD Training with ACCDOA** 

We extend the HEAR Benchmark with ACCDOA (Shimada et al., 2021) framework to solve newly introduced real world SELD tasks. This framework jointly models sound event detection and localization across target sound classes. A class is considered active at frame _t_ when the predicted Cartesian coordinate magnitude _∥_ **c** _t∥ >_ 0 _._ 5, where **c** _t ∈_ R[3] represents the unit direction vector. We do not perform any post-processing steps unlike the HEAR protocol. 

15

<!-- page: 16 -->

**Spatial general purpose audio representation learning** 

To adapt the datasets to the HEAR format, we processed the ground truth timestamp labels into fixed-duration segments. For static sources, we extracted start and end times alongside fixed azimuth and elevation coordinates. For moving sources, we extracted active segments and their corresponding time-varying directions. In frames where no sources are active, the ACCDOA target is defined as a zero vector. Crucially, the dynamic datasets in RealSELD utilize 100ms segments (10 Hz resolution). To align the output resolution of the audio embedding model with these labels, we applied average pooling to the model representations. This ensures the GRAM model output is temporally aligned with the 100ms ground truth segments. 

After the mapping, we extract the time-stamp embeddings and their corresponding ground truth labels using the HEAR evaluation kit, and linear probe the extracted information using the HEAR evaluation protocol. 

## **I. Evaluation Metrics** 

We evaluate the sound localization performance on the newly generated sound localization tasks in NatHEAR by calculating the Direction of Arrival (DoA) error _θ_ between the [ _x, y, z_ ] coordinates of the target sound source on the unit sphere using the arc cosine of the dot product of the unit vectors: _θ_ = arccos( _v ·_ ˆ _v_ ). 

**SELD metrics on RealSELD Dynamic Motion:** We used four joint localization and detection metrics (Politis et al., 2020b), which are widely-used in audio-only SELDtasks (Politis et al., 2022). Two metrics, referred to as location-aware detection, are the error rate (ER20 _◦_ ) and F-score (F20 _◦_ ) for one-second non-overlapping segments. We consider a prediction to be a true positive (TP) if the prediction and the reference class are the same and the angle difference is less than 20 _[◦]_ . F20 _◦_ is calculated from location-aware precision and recall, whereas ER20 _◦_ is the sum of insertion, deletion, and substitution errors, divided by the total number of references. The other two metrics, referred to as class-aware localization, are localization error (LE _CD_ ) in degrees and localization recall (LR _CD_ ) in one-second non-overlapping segments, where the subscript denotes classification-dependent. Unlike location-aware detection, we do not use a threshold; instead, we estimate the difference between the correct prediction and the reference. LE _CD_ expresses the average angular difference between the same class’s predictions and references. LR _CD_ reports the true positive rate as the number of localization estimates detected in a class out of the total number of class instances. We used the macro mode of computation, which does not apply to ER20 _◦_ because it includes substitution errors between two classes. We first computed the metrics for each class and then averaged them for the other three metrics to obtain the final system performance. 

**SELD metrics on RealSELD Static Motion:** For testing the model on the TUT-2018 REAL and TAU-2019, we used the non-localization-dependent counterparts of the aforementioned metrics. ER and F-score calculated in segments of one second with no overlap, as proposed. The segment-wise results are obtained from the classifier’s frame-level predictions, treating a sound event as active across the entire segment if it is active in any frame. Similarly, we obtain labels for one-second segments of the reference from its framewise annotation and calculate segment-wise ER and F-scores. Additionally, in order to account for time frames where the number of estimated and reference DOAs are unequal, we report the frame recall, calculated as TP (TP + FN) in percentage, where true positives TP is the total number of time frames in which the number of DOAs predicted is equal to reference, and false negatives FN is the total number of frames where the predicted and reference DOA are unequal. 

## **J. T60 estimation tasks** 

NatHEAR includes two T60 estimation tasks (ESC-50 and SC-5) in addition to the direction-of-arrival estimation tasks. For synthesizing these tasks, we did not add additional localized/diffused noise. Specifically, we convolved ESC-50 and SC-5 clips with BRIR( _s, r, θ_ ) for NatHEAR Binaural, or with ARIR( _s, r, θ_ ) for NatHEAR Ambisonics. We estimated the ground truth RT60s using the first channel of the ARIR. To estimate the T60s, we utilized the Schroeder method (Schroeder, 1965) from Pyroomacoustics package (Scheibler et al., 2018). Explicitly, we measure the RT30 and extrapolate to T60 using the decay curve. 

Figure 8 depicts the RT60 distributions of ESC-50 and SC-5 datasets. Furthermore Table 7 presents the median absolue errors that we got with GRAMs on ESC-50 and SC-5 tasks. 

16

<!-- page: 17 -->

**Spatial general purpose audio representation learning** 

**==> picture [213 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
ESC-50 SC-5<br>**----- End of picture text -----**<br>


**==> picture [361 x 139] intentionally omitted <==**

_Figure 8._ Distribution of the estimated T60s for ESC-50 and SC-5 datasets. 

_Table 7._ Absolute median error comparison on T60 estimation tasks. 

_(a)_ SC-5 _(b)_ ESC-50 

|**Model**<br>**Median Error**<br>GRAM-T-Clean<br>0.0225<br>GRAM-T-Ambisonics<br>0.0169<br>GRAM-T-Binaural (Patch)<br>0.0146<br>GRAM-T-Binaural (Time)<br>0.0179<br>Spatial-AST<br>0.0299|**Model**<br>**Median Error**|
|---|---|
||GRAM-T-Clean<br>0.0461<br>GRAM-T-Ambisonics<br>0.0421<br>GRAM-T-Binaural (Patch)<br>0.0397<br>GRAM-T-Binaural (Time)<br>0.0418<br>Spatial-AST<br>0.0468|



## **K. Evaluating training efficiency** 

For all models trained solely on AudioSet, we calculated the number of seconds seen during the training as: batch size _×_ steps per epoch _×_ epochs _×_ input length. This comparison accounts for the number of 10-second AudioSet sound clips processed by each model. 

_Table 8._ Training details of the recent audio foundation models. We retrieve the numbers from the references where possible. Various works utilized various sizes of AudioSet. Therefore, we used the dataset size reported by the references to calculate the steps per epoch. For MWMAE and SSAM we retrieved their dataset size from their corresponding code repository. 

|**Model**|**Batch Size**|**Epochs**|**Steps per Epoch**|**Total Steps**|**Total Samples Seen**|
|---|---|---|---|---|---|
|MWMAE (Yadav et al.,2024)|1024|100|1985|198500|_∼_200M|
|GRAMs|96|N/A|N/A|500000|_∼_48M|
|Audio-MAE (Huang et al.,2022)|512|32|3829|122528|_∼_48M|
|BEATs (Chen et al.,2023)|5600|N/A|N/A|1.2M|_∼_6.7B|
|SSAM (Yadav & Tan,2024)|1024|100|2003|200300|_∼_207M|



17

<!-- page: 18 -->

**Spatial general purpose audio representation learning** 

## **L. SoundSpaces 2.0 specifications** 

We generate our BRIRs using the simulator provided by SoundSpaces 2.0 (Chen et al., 2022a). Our hyperparameters for the simulator is depicted in Table 9. 

_Table 9._ Acoustic configuration parameters utilized in SoundSpaces 2.0 to generate our BRIRs. 

|**Parameter**|**Value**|**Parameter**|**Value**|
|---|---|---|---|
|directSHOrder|3|indirectSHOrder|3|
|sampleRate|32000|frequencyBands|8|
|maxDiffractionOrder|10|transmission|True|
|indirect|True|indirectRayCount|15000|
|indirectRayDepth|400|sourceRayCount|200|
|sourceRayDepth|20|threadCount|16|
|agentHeigth|1.5m|||



18
