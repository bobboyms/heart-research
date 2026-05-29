<!-- page: 1 -->

## WAVJEPA: SEMANTIC LEARNING UNLOCKS ROBUST AUDIO FOUNDATION MODELS FOR RAW WAVEFORMS 

## **Goksenin Yuksel** 

Donders Institute, Radboud University Nijmegen, The Netherlands Goksenin.yuksel@donders.ru.nl 

## **Pierre Guetschel** 

Donders Institute, Radboud University Nijmegen, The Netherlands pierre.guetschel@donders.ru.nl 

## **Michael Tangermann** 

Donders Institute, Radboud University Nijmegen, The Netherlands michael.tangermann@donders.ru.nl 

## **Marcel van Gerven** 

Donders Institute, Radboud University Nijmegen, The Netherlands marcel.vangerven@donders.ru.nl 

## **Kiki van der Heijden** 

Donders Institute, Radboud University, Nijmegen, The Netherlands Mortimer B Zuckerman Institute, Columbia University, New York, United States kiki.vanderheijden@donders.ru.nl 

## ABSTRACT 

Learning audio representations from raw waveforms overcomes key limitations of spectrogram-based audio representation learning, such as the long latency of spectrogram computation and the loss of phase information. Yet, while self-supervised speech representation learning from raw waveforms has been remarkably successful, these approaches have not achieved similar feats for general-purpose audio representation learning from waveforms. Here, we propose WavJEPA, a waveform-based version of the Joint-Embedding Predictive Architecture. WavJEPA leverages high-level semantic representation learning to tackle the shortcomings of representation learning at the speech unit or token level. We show that this approach substantially outperforms state-of-the-art time-domain audio foundation models across a wide variety of downstream benchmark tasks, while requiring considerably fewer computational resources. Additionally, to overcome the performance drop that time-domain models typically exhibit in noisy and reverberant real-world acoustic environments, we present WavJEPA-Nat. WavJEPA-Nat is a multi-channel extension of the WavJEPA architecture trained on simulated naturalistic scenes. We find that WavJEPA-Nat is highly robust to reverberation and noise. These results highlight the feasibility and computational efficiency of general-purpose audio representation learning from raw waveforms, showcasing the potential for low-latency, robust time-domain audio foundation models for real-world applications.[1] 

## 1 INTRODUCTION 

State-of-the-art approaches for self-supervised general-purpose audio representation learning predominantly operate on spectrograms, that is, time-frequency representations of sound clips (Turian et al., 2022; Yadav et al., 2024; Chen et al., 2023; Gong et al., 2022; Yadav & Tan, 2024). However, these approaches suffer from two fundamental limitations: The latency introduced by the shorttime Fourier transform (STFT) required for spectrogram computation impedes real-time deployment (Luo & Mesgarani, 2019), and (2) the loss of phase information reduces the performance on generative audio tasks (Luo & Mesgarani, 2019; Li et al., 2025). In contrast, time-domain models, 

> 1All code and materials are available on https://github.com/labhamlet, and https:// huggingface.co/labhamlet 

1

<!-- page: 2 -->

which learn directly from raw audio waveforms, achieved remarkable success in speech representation learning (Baevski et al., 2020; Hsu et al., 2021; Chen et al., 2021). Crucially, end-to-end audio representation learning from raw waveforms overcomes the key limitations of spectrogram-based audio representation learning (long latency and loss of phase information) (Luo & Mesgarani, 2019). Yet, when state-of-the-art approaches for speech representation learning are trained for generalpurpose audio representation learning, their performance is less strong (La Quatra et al., 2024). Furthermore, existing time-domain models exhibit significant degradation in noisy and reverberant acoustic environments compared to their spectrogram-based counterparts, limiting their effectiveness for real-world applications (Yuksel et al., 2025). 

To improve these shortcomings, we propose WavJEPA, a novel framework for end-to-end generalpurpose audio representation learning from raw waveforms. The idea behind WavJEPA is that the semantic learning capabilities of joint-embedding predictive architectures (JEPAs) (Bardes et al., 2024; Assran et al., 2023; LeCun, 2022) can be leveraged to overcome the limitations of learning representations at the token or speech unit level, which is the typical approach of audio foundation models operating on raw waveforms. Instead, WavJEPA learns semantic representations by predicting the latent representations of training targets from a temporally distributed context representation of the same sound wave. 

WavJEPA is the first framework applying semantic learning to general-purpose audio representations in the time domain, surpassing state-of-the-art time-domain approaches on the HEAR (Turian et al., 2022) and ARCH (La Quatra et al., 2024) benchmark suites while requiring only a fraction of the computational resources. Additionally, we address the degraded performance of time-domain models in real-world sound scenes with WavJEPA-Nat, a multi-channel extension of the WavJEPA framework trained on simulated real-world sound scenes. Evaluation on Nat-HEAR (Yuksel et al., 2025), a naturalistic version of the HEAR benchmark suite, demonstrates that WavJEPA-Nat exceeds the robustness of other time-domain foundation models to noise and reverberation. We furthermore elucidate the critical factors for semantic representation learning from raw waveforms through extensive ablation studies, targeting context-target sampling, top- _K_ averaging and the optimal ratio between real-world scenes and dry sound clips. In sum, WavJEPA and WavJEPA-NAT demonstrate that robust time-domain approaches for audio representation learning are feasible and efficient, opening the door to low-latency audio foundation models for real-world applications. 

## 2 RELATED WORK 

**Spectrogram-based audio representation learning:** These approaches aim to learn generalpurpose representations from the time-frequency representation of a sound clip (spectrogram) calculated with a short-time Fourier transform. Masked auto-encoder (MAE) approaches achieve stateof-the-art performance on benchmark suites, learning rich audio representations by reconstructing masked spectrogram patches (He et al., 2022; Yadav et al., 2024; Gong et al., 2022; Chong et al., 2023; Baade et al., 2022; Huang et al., 2022; Yadav & Tan, 2024). Other approaches – inspired by the success in the visual domain (Grill et al., 2020; Bardes et al., 2024; Assran et al., 2023) – avoid reconstructing the original spectrogram input space, instead predicting targets in the latent space (Niizumi et al., 2023; Fei et al., 2024; Chen et al., 2023; 2024). 

**Waveform-based audio representation learning:** Representation learning from raw waveforms is based on predictive or contrastive self-supervised learning strategies at the token or speech unit level (Baevski et al., 2020; Hsu et al., 2021; Chen et al., 2021). More recently, Data2Vec (Baevski et al., 2022) introduced a modality-agnostic framework for training across speech, vision, and text domains, leveraging a teacher-student approach. They demonstrated that the proposed latent prediction framework achieves state-of-the art on speech recognition with minimal fine-tuning. While these frameworks have proven extremely fruitful for speech representation learning, they have been less successful in learning general-purpose audio representations (Yadav et al., 2024; Turian et al., 2022; Yuksel et al., 2025; La Quatra et al., 2024). 

**Representation learning with joint embedding predictive architectures (JEPAs):** Recent work demonstrated that JEPA models efficiently learn semantic image representations by predicting latent representations of parts of the input image (that is, training targets) from a context representation of other parts of that same image (Assran et al., 2023; Bardes et al., 2024). Based on this success, 

2

<!-- page: 3 -->

**==> picture [393 x 211] intentionally omitted <==**

**----- Start of picture text -----**<br>
input waveform waveform sampled context context predictor predictions<br>waveform encoder embedding context encoder block<br>EMA<br>latent representation<br>learned mask<br>context indices<br>target indices<br>target latent target training<br>positional embeddings encoder representation targets<br>sampling indices<br>**----- End of picture text -----**<br>


Figure 1: **Semantic representation learning from raw waveforms.** WavJEPA predicts latent target representations at specific locations from a context representation. The weights of the target encoder are not trained but updated using the exponential moving average (EMA) of the weights of the contextencoder. 

others applied JEPA models to spectrograms (Fei et al., 2024), EEG signals (Guetschel et al., 2024) and fMRI measurements (Dong et al., 2024), highlighting the versatility of the JEPA framework. 

## 3 METHODOLOGY 

## 3.1 THE WAVJEPA FRAMEWORK 

Our proposed architecture and approach for learning general-purpose audio representations from raw waveforms are illustrated in Figure 1. The WavJEPA framework comprises a waveform encoder, context encoder, target encoder and a predictor. WavJEPA’s objective is to predict latent representation of various targets blocks based on a single context block extracted from the same sound wave. As waveform encoder, we use the feature encoder of Wav2Vec 2.0, which is composed of stacked temporal convolution layers (Baevski et al., 2020). Similar to the original I-JEPA architecture (Assran et al., 2023), a Vision Transformer (ViT) (Dosovitskiy et al., 2021) is used for the target encoder, context encoder and predictor. Detailed specifications of the framework components can be found in Appendix A. In the following, we describe the main components of the WavJEPA framework. 

**Waveform encoder:** A sound wave _x ∈_ R _[T][ ×]_[1] is transformed into an embedding _w ∈_ R _[N][×]_[768] by the waveform encoder _w_ := _W_ ( _x_ ). To obtain a more fine-grained embedding, we removed the last convolutional layer of the Wav2Vec2.0 feature encoder. 

**Sampling the context block and target blocks:** A temporally distributed context and _Ktarget_ target blocks are sampled from the _N_ indices in the waveform embedding _w_ in an iterative procedure. We first randomly sample starting indices for the context block with uniform probability _pcontext_ over the range [1 _. . . N_ ]. For each starting index, we then include the subsequent _Mcontext_ -many indices in our context block. Then, for each target block _k ∈_ [1 _. . . Ktarget_ ], we randomly sample a starting index and select the subsequent _Mtarget_ indices as training targets. Context indices that overlap with training targets are removed. We repeat this procedure until at least 10% of indices in [1 _. . . N_ ] are designated as the context. Ultimately, we obtain _n_ non-overlapping context indices _c_ 1 _, . . . , cn ∈_ [1 _. . . N_ ], and, for each target block _k ∈_ [1 _. . . Ktarget_ ], we obtain _Mtarget_ target indices _t[k]_ 1 _[, . . . , t][k] Mtarget[∈]_[[1] _[ . . . N]_[]][.] 

3

<!-- page: 4 -->

**Context encoder:** To obtain a latent context representation _z_ = _{z_ 1 _, . . . , zn}_ , the context encoder _C_ ( _·_ ) converts the context waveform embedding _w[c]_ = _{wc_ 1 _, . . . , wcn}_ into a latent representation _z_ := _C_ ( _w[c]_ ). Attention masking is used to ensure that the context encoder operates only on the context indices _c_ 1 _, . . . , cn_ for the generation of _z_ . 

**Predictor:** For each target _k ∈_ [1 _. . . Ktarget_ ], we concatenate the latent context representation _z_ with learnable mask embeddings and additive positional embedding in order to replace the target indices: _z_ ˜ _[k]_ = _{z_ 1 _, . . . , zn, mtk_ 1 _[, . . . , m][t][k] Mtarget[}]_[.][The][predictor] _[P]_[(] _[·]_[)][then][takes][this][augmented] ˜ ˆ ˆ latent representation _z[k]_ to predict the latent target representations _y[k]_ = _{y_ 1 _[k][, ...,]_[ ˆ] _[y] M[k] target[}]_[ such that] _y_ ˆ _[k]_ := _P_ (˜ _z[k]_ ). The predictor is thus applied _Ktarget_ times. 

**Target encoder and learning objective:** In this waveform-based approach, latent representations of the sound wave embeddings constitute the targets. The target encoder _E_ ( _·_ ) converts the whole waveform embedding _w_ into a latent target representation. Similar to Baevski et al. (2022; 2020), the outputs of the top _K_ layers are instance-normalized (Ulyanov et al., 2017) and averaged. For each time step _i ∈_ [1 _. . . N_ ], we obtain a target embedding _yi ∈_ R[768] . For each target block _k ∈_ [1 _. . . Ktarget_ ], we select the tokens _y[k]_ = _{ytk_ 1 _[, ..., y][t][k] Mtarget[}]_[ corresponding to the target block] indices _t[k]_ 1 _[, . . . , t][k] Mtarget_[and compute the] _[ L]_[2][ distance between the predicted target representation] _[y]_[ˆ] _[k]_ and the actual training target _y[k]_ . The final loss corresponds to the average error across targets. 

**Target encoder parametrization:** The parameters ∆ of the target encoder are not trained, but instead updated on every iteration by an exponential moving average (EMA) of context encoder parameters _θ_ according to ∆ _← τ_ ∆+ (1 _− τ_ ) _θ_ . Here, _τ_ linearly increased over the first _τn_ updates from _τ_ 0 to target _τe_ , after which it was kept constant for the remainder of training. 

## 3.2 EXPERIMENTAL SET-UP WAVJEPA 

**Data and sound wave embeddings:** We train WavJEPA on the unbalanced training set of AudioSet, which consists of 1.74 million 10-second sound clips scraped from YouTube (Gemmeke et al., 2017). Each sound clip was resampled to 16 kHz and mean centered to enforce equal loudness across sound clips. We then randomly sampled 8 sections of 2 s from each sound clip, effectively increasing the batch size by a factor of 8 in a computationally efficient manner. Finally, each instance is instance normalized (Ulyanov et al., 2017). The waveform encoder converts each 2 s instance into an embedding _w_[200] _[×]_[768] , effectively resampling the audio to 100 Hz with a stride of 10 ms and a receptive field size of 12.5 ms. 

**Pre-training:** We sampled starting indices for the context block with _p_ = 0.065 and for target blocks with _p_ = 0.025. We set _M_ to 10 for both context block and target block . To update the target encoder parameters ∆, we linearly increased _τ_ from _τ_ 0 = 0 _._ 999 to _τe_ = 0 _._ 99999 over the first 100,000 steps, after which _τ_ was kept constant. We used _K_ = 8 for the top _K_ averaging. 

We trained WavJEPA for 375,000 steps using a batch size of 32 on two NVIDIA H100 94 GB GPUs. Given our in-batch sampling factor of 8, we boost our effective batch size to 256. We use the AdamW optimizer (Loshchilov & Hutter, 2019) with a weight decay coefficient _λw_ = 0.04. The learning rate schedule follows a cosine decay with linear warm-up over 100,000 steps, reaching a peak learning rate of 2 _×_ 10 _[−]_[4] before decaying to zero. 

## 3.3 THE WAVJEPA-NAT FRAMEWORK 

The proposed WavJEPA-Nat is a multi-channel extension of WavJEPA (illustrated in Appendix D). While the overall approach is similar, WavJEPA-Nat is equipped with two waveform encoders and utilizes a 2D instead of a 1D positional embedding to ensure capturing both intra- and inter-channel information. As before, WavJEPA-Nat’s objective is to predict the latent representation of target blocks from latent representation of the context block. Crucially, for WavJEPA-Nat, target blocks and the context block indices are shared across _both_ channels of the embedded waveform _w_ . 

**Data and sound wave embeddings:** We use the pipeline of Yuksel et al. (2025) to transform AudioSet sound clips into naturalistic, spatialized sound scenes with reverberation and noise. In brief, we simulate naturalistic, spatialized scenes by using the room impulse response (RIR) simulator and binaural renderer provided by Soundspaces 2.0 (Chen et al., 2022), resulting in two-channel sound 

4

<!-- page: 5 -->

clips containing naturalistic spatial cues. To each sound scene, we add similarly spatialized noise clips from the WHAMR! database (Maciejewski et al., 2020). A full description of the sound scene generation can be found in Appendix D. 

Each two-channel sound wave _x_ ( _t_ ) _∈_ R _[T][ ×]_[2] corresponding to a naturalistic scene is transformed by two independent waveform encoders into embeddings _w_ 1 _∈_ R _[N][×]_[768] and _w_ 2 _∈_ R _[N][×]_[768] . The hyperparameters of the waveform encoders are identical to those of WavJEPA. The embedded waveforms _w_ 1 and _w_ 2 are subsequently concatenated to form _w_[2] _[N][×]_[768] . 

**Learning inter-channel dependencies:** Instead of adding 1D fixed positional embeddings to _w_ as in the original WavJEPA framework, we now add 2D sinusoidal positional embeddings that explicitly encode both inter-channel and intra-channel positional information. The sampling procedure for obtaining a context block and target blocks is similar to WavJEPA, but shared along the channels. This procedure forces WavJEPA-Nat to predict the latent embedding of the same time step for both channels . 

**Pre-training:** As for WavJEPA, we also update the target encoder parameters ∆ for WavJEPANat with the exponential moving average (EMA) of context encoder parameters _θ_ using a similar schedule for _τ_ . Similarly, we used _K_ = 8 for the top _K_ averaging. As the dimensions of _w_ , _cw_ and _zw_ are twice as large for WavJEPA-Nat, we trained the model with a smaller batch size to avoid out-of-memory errors. Specifically, we used an in-batch sampling factor of 8 and a batch size of 16, resulting in an effective batch size of 128. In agreement with WavJEPA, we trained WavJEPA-Nat for 375 K steps on the same _L_ 2 objective. The optimization hyper-parameters were kept the same as for WavJEPA. 

## 3.4 DOWNSTREAM EVALUATION 

**Downstream tasks:** We evaluated WavJEPA and WavJEPA-Nat on two large benchmark task suites for the evaluation of general-purpose audio foundation models: HEAR (Turian et al., 2022) and ARCH (La Quatra et al., 2024). We use the same subset of HEAR benchmark tasks as previously used in Yadav et al. (2024) but added DCASE2016 Task 2 (Mesaros et al., 2018b) as a time stampbased task to evaluate the audio scene analysis capabilities of the models more in-depth. HEAR and ARCH contain a selection of complementary tasks and datasets for acoustic events and scene analysis, speech, and music. For more detailed description of tasks please see Appendix C. 

We additionally evaluated models on NatHEAR (Yuksel et al., 2025), a naturalistic version of the HEAR benchmark suite comprising high-quality simulations of real-world sound scenes with reverberation and noise, spatialized in two formats (either binaural and ambisonics). To accommodate the input format of single-channel models, we utilized the first channel (that is, the omndirectional microphone) of NatHEAR in the Ambisonics format (Zotter & Frank, 2019). For the dual waveform encoder approach of WavJEPA-Nat, we used both channels of NatHEAR in a binaural format. 

**Model fine-tuning for downstream evaluation:** For the downstream evaluation on HEAR and ARCH benchmark tasks, we trained a shallow downstream classifier on representations that were extracted after self-supervised pre-training, following the exact fine-tuning procedures detailed by HEAR (Turian et al., 2022) and ARCH (La Quatra et al., 2024). Model weights were frozen after pre-training. Note that the difference between the fine-tuning approaches in HEAR and ARCH causes the differences in performance for tasks that are in both suites, for example, ESC50. Further, to evaluate WavJEPA-Nat on HEAR, we duplicated the single-channel audio recordings of the original HEAR to make the input compatible with the dual waveform encoder architecture of WavJEPA-Nat. 

**Down stream evaluation metric** _s_ ( _m_ ) **:** As the tasks in HEAR and ARCH vary considerably in terms of evaluation criteria and difficulty level, we calculate for each model _m_ a generalizability metric _s_ ( _m_ ) to give an impression of the overall performance of a model, similar to Yang et al. (2021).This metric effectively ranks models as a function of the maximum improvement they obtain over the baseline model, normalized by the difference in scores between SOTA and the baseline for the specific task (see Appendix B). The baseline used here is HEAR-Naive, consisting of melspectrogram representations. For calculating this score, we included all the evaluated methods in all upcoming sections, including ablations. 

5

<!-- page: 6 -->

**Model comparison:** We compare the performance of WavJEPA to state-of-the-art self-supervised models using transformer architectures for representation learning from raw waveforms. We include Wav2Vec2.0 (Baevski et al., 2020), HuBERT (Hsu et al., 2021), WavLM (Chen et al., 2021), Data2Vec (Baevski et al., 2022), all pre-trained on large quantities of speech data. We furthermore include the recently released versions of Wav2Vec2.0 and HuBERT pre-trained on AudioSet (La Quatra et al., 2024) to assess their ability to learn general-purpose audio representations. For all models, we include both the _Base_ (approximately 90 m parameters) and the _Large_ version (300 m parameters). In comparison, WavJEPA has 90 m parameters. 

## 3.5 ABLATIONS: 

To identify the critical parameters for a successful learning of general-purpose audio representations with the WavJEPA framework, we conducted comprehensive ablation studies on the pre-training parameters. Specifically, we examined the effect of sampling parameters for target and context blocks ( _ptarget_ , _Mcontext_ and _Mtarget_ ) and the effectiveness of top- _K_ layer averaging for training targets. For WavJEPA-Nat, we systematically assessed the impact of the ratio between clean and naturalistic sound scenes in the pre-training data. For all ablation studies, pre-training and downstream evaluation settings were similar to those of WavJEPA and WavJEPA-Nat. 

## 4 RESULTS 

## 4.1 PERFORMANCE ON DOWNSTREAM TASKS 

As shown in Table 1 and Table 2, WavJEPA surpasses all state-of-the-art models on HEAR ( _s_ ( _m_ ) = 66.0) and ARCH ( _s_ ( _m_ ) = 92.3). Base models pre-trained on speech score low on both HEAR and ARCH, but improve slightly when pre-trained on AudioSet. This demonstrates that, besides a lack of generalization to out-of-distribution downstream tasks when pre-trained on speech data, these models fail to learn robust general-audio representations from AudioSet pre-training. Among the Large models, WavLM generalizes best to HEAR. It is conceivable that this is a consequence of the size and diversity of the large-scale speech dataset that WavLM _Large_ was pre-trained on Chen et al. (2021). HuBERT Large obtained the best score on ARCH when pre-trained on AudioSet. 

Table 1: Performance on HEAR benchmark suite. Values represent either the primary score (in case no cross-validation scheme was specified) or the mean _±_ standard deviation calculated with the _k_ -fold cross-validation scheme specified by HEAR. For each task, the best performance per pretraining dataset is highlighted in bold. The best overall performance for a given task (i.e., across pre-training datasets) is highlighted with a light-blue background. _Base_ and _Large_ refers to the total model parameters, _∼_ 90 m and _∼_ 300 m respectively. 

|**Model**<br>**Size**|**Acoustic Events and Scene Analysis**<br>**DCASE FSD50K**<br>**LC**<br>**ESC-50**|**Speech**<br>**CD**<br>**VL**<br>**SC-5**|**Music**<br>**NS**<br>**BO**<br>**Mri-S**<br>**Mri-T**|**s(m)**|
|---|---|---|---|---|
|**Baseline**|||||
|HEAR-Naive N/A|7.6<br>12.5<br>40_._3_±_1_._2 27_._4_±_3_._3|36_._7_±_2_._5 16_._0_±_3_._4 13.3|**89.2**<br>**97.1**_±_3_._2 94_._2_±_1_._1<br>**93.7**_±_0_._3|0.0|
|**Speech pre-training**|||||
|Wav2Vec2.0<br>B<br>HuBERT<br>B<br>WavLM<br>B<br>Data2Vec<br>B<br>Wav2Vec2.0<br>L<br>HuBERT<br>L<br>WavLM<br>L<br>Data2Vec<br>L|23.5<br>29.4<br>**69.9**_±_2_._1 46_._4_±_1_._8|57_._3_±_1_._1 34_._9_±_2_._4 85.3|17.4<br>81_._4_±_4_._8 90_._7_±_0_._8 77_._0_±_0_._9<br>19.8<br>93_._2_±_5_._9 94_._6_±_0_._4 85_._0_±_2_._5<br>16.0<br>84_._3_±_6_._3 88_._8_±_1_._0 76_._8_±_0_._5<br>14.0<br>78_._4_±_4_._1 85_._1_±_0_._7 70_._5_±_3_._3<br>40.6<br>**93.6**_±_2_._6 94_._8_±_0_._5 82_._4_±_3_._0<br>20.4<br>93_._6_±_3_._0 95_._3_±_0_._8 82_._5_±_2_._0<br>18.2<br>93_._6_±_5_._4 **95.8**_±_0_._8 **90.1**_±_1_._0<br>14.4<br>80_._1_±_8_._5 84_._7_±_2_._6 65_._6_±_3_._1|30.9<br>47.3<br>35.1<br>23.6<br>42.5<br>44.3<br>**58.1**<br>29.0|
||**78.0**<br>32.8<br>63_._3_±_1_._2 58_._6_±_2_._8<br><br><br>|71_._2_±_1_._2 65_._2_±_2_._9<br>**94.0**<br>|||
||27.0<br>25.7<br>61_._3_±_2_._3 49_._5_±_3_._8<br>46.5<br>15.2<br>47_._9_±_1_._2 28_._0_±_2_._8<br>66.0<br>34.8<br>64_._6_±_1_._9 59_._8_±_1_._5<br>34.8<br>31.4<br>63_._8_±_1_._3 60_._4_±_3_._0|64_._3_±_1_._3 60_._1_±_3_._2 93.6<br>55_._7_±_1_._0 44_._9_±_3_._1 88.5<br>65_._7_±_0_._8 53_._3_±_6_._3 75.8<br>71_._0_±_1_._2 69_._0_±_2_._8 84.8<br>**76.3**_±_2_._2<br>**79.2**_±_3_._9 93.8<br>62_._8_±_1_._6 60_._0_±_4_._9 86.1|||
||77.4<br>**40.1**<br>69_._4_±_2_._1 **66.6**_±_2_._5<br><br><br>||||
||40.8<br>18.7<br>50_._9_±_1_._7 34_._4_±_2_._5||||
|**AudioSet pre-training**|||||
|Wav2Vec2.0<br>B<br>HuBERT<br>B<br>Wav2Vec2.0<br>L<br>HuBERT<br>L<br>WavJEPA<br>B|52.0<br>34.7<br>60_._4_±_1_._7 58_._9_±_1_._9<br>86.2<br>41.1<br>63_._5_±_3_._4 69_._1_±_1_._6<br>82.6<br>47.8<br>73_._6_±_1_._2 72_._6_±_2_._1|56_._3_±_1_._3 27_._9_±_4_._6 72.1<br>69_._5_±_1_._2 **53.3**_±_3_._1<br>83.5<br>68_._2_±_1_._7 42_._2_±_6_._0 83.9|**42.0**<br>86_._0_±_9_._6 92_._9_±_1_._4 77_._3_±_0_._5<br>38.8<br>91_._5_±_8_._8 95_._6_±_0_._5 **90.4**_±_0_._8<br>30.8<br>91_._5_±_5_._0 96_._5_±_0_._3 88_._7_±_2_._5<br>38.6<br>**91.6**_±_9_._6<br>**97.3**_±_0_._5 89_._6_±_2_._3|31.9<br>51.1<br>55.9<br>57.7|
||86.2<br>45.4<br>75_._2_±_1_._4 66_._3_±_4_._6|70_._1_±_0_._8 39_._6_±_3_._6 85.7|||
||**93.9**<br>**54.4**<br>**76.7**_±_2_._4<br>**86.5**_±_3_._3|**71.0**_±_0_._8 49_._8_±_3_._4 **90.0**|34.4<br>89_._4_±_5_._4<br>**97.3**_±_0_._4 88_._5_±_0_._5|**66.0**|



**Audio scene analysis and acoustic events:** Inspecting performances at the task level demonstrates that WavJEPA performs exceptionally well on acoustic events and audio scene analysis. On tasks such as sound event detection (DCASE 2016 Task 2), WavJEPA improves the SOTA by 8.9 %, and on audio event multi-labeling task FSD50K - a very challenging task - WavJEPA increases the SOTA 

6

<!-- page: 7 -->

Table 2: Performance on ARCH benchmark suite. Values and colors as in Table 1. 

|Model<br>Size|**Acoustic Events and Scene Analysis**<br>ESC-50<br>US8K<br>FSD50K<br>VIVAE|**Music**<br>FMA<br>MTT<br>IRMAS<br>MS-DB|**Speech**<br>RAVDESS<br>AM<br>SLURP<br>EMOVO|s(m)|
|---|---|---|---|---|
|**Baseline**|||||
|HEAR-Naive<br>N/A|13.0<br>36.0<br>2.2<br>22.0|39.0<br>9.9<br>19.9<br>35.2|22.6<br>45.7<br>5.4<br>18.4|0.0|
|**Speech pre-training**|||||
|Wav2Vec2.0<br>B<br>WavLM<br>B<br>HuBERT<br>B<br>Data2Vec<br>B<br>Wav2Vec2.0<br>L<br>WavLM<br>L<br>HuBERT<br>L<br>Data2Vec<br>L|45.7<br>55.5<br>19.4<br>31.5<br>49.9<br>61.8<br>17.6<br>36.3<br>58.9<br>67.3<br>24.5<br>40.5<br>23.6<br>45.6<br>10.1<br>30.2<br>13.1<br>42.7<br>5.8<br>22.0<br>**67.2**<br>**70.9**<br>**32.2**<br>**42.5**|50.5<br>37.6<br>35.1<br>66.1<br>48.7<br>34.9<br>32.6<br>54.2<br>54.6<br>38.8<br>36.7<br>58.5<br>40.6<br>27.6<br>25.9<br>50.7<br>41.7<br>21.0<br>19.9<br>50.2<br>**61.1**<br>**41.3**<br>**42.5**<br>**68.0**|55.3<br>86.4<br>14.4<br>31.8<br>67.9<br>99.5<br>31.0<br>43.1<br>65.3<br>99.6<br>33.8<br>40.5<br>48.0<br>99.1<br>43.6<br>27.3<br>11.6<br>45.7<br>7.3<br>19.3<br>71.8<br>99.8<br>42.3<br>**45.3**<br>**72.6**<br>**99.9**<br>**45.3**<br>43.8<br>45.1<br>99.2<br>28.6<br>23.1|49.7<br>68.0<br>59.7<br>38.8<br>8.6<br>75.8<br>**81.5**<br>35.1|
||64.0<br>70.0<br>29.5<br>41.0<br><br><br><br>|54.8<br>38.4<br>36.8<br>64.1<br><br><br><br>|||
||25.4<br>49.2<br>10.8<br>30.6|43.5<br>28.5<br>27.1<br>44.2|||
|**AudioSet pre-training**|||||
|W2V2<br>B<br>HuBERT<br>B<br>Wav2Vec 2.0<br>L<br>HuBERT<br>L<br>WavJEPA<br>B|52.6<br>70.5<br>21.3<br>31.3<br>68.8<br>79.1<br>31.1<br>40.1<br>74.4<br>79.0<br>37.6<br>39.7|59.5<br>37.9<br>35.9<br>64.6<br>65.9<br>43.4<br>47.7<br>67.8<br>66.6<br>44.5<br>49.9<br>76.9|45.9<br>88.1<br>11.0<br>30.8<br>63.5<br>98.8<br>20.5<br>33.4<br>59.5<br>99.4<br>17.7<br>38.2<br>**73.3**<br>**99.6**<br>20.5<br>38.6|53.8<br>75.5<br>80.0<br>83.9|
||71.5<br>75.6<br>37.4<br>**44.3**|67.5<br>43.4<br>50.5<br>77.8|||
||**83.9**<br>**83.5**<br>**48.0**<br>44.06|**68.2**<br>**46.0**<br>**59.0**<br>**79.5**|62.5<br>99.5<br>**23.3**<br>**46.6**|**92.3**|



by 13.8 %. For environmental sound classification, WavJEPA’s accuracy is 19.1 % higher than the next best performing model (WavLM Large). 

**Speech:** The tasks covered by the pre-training data has, as expected, a large impact on the speechrelated downstream tasks. In particular, WavLM Large pre-trained on speech data obtains the highest performance on HEAR speech tasks, while HuBERT Large scores best on ARCH speech tasks (followed by WavLM Large). However, among the Base models pre-trained on AudioSet, WavJEPA performs best on several of the HEAR speech tasks, including spoken command classification (SC5) and emotion recognition (CD), as well as on most of the ARCH speech tasks, including spoken digit recognition (AudioMNIST, AM), intent classification (SLURP) and emotion recognition (EMOVO). Moreover, WavJEPA outperforms several Base models pre-trained on speech, both on HEAR and on ARCH speech tasks, illustrating the generalization of WavJEPA to speech data. 

**Music:** WavJEPA obtains the highest performance on all music tasks in the ARCH benchmark suite. However, we find that models pre-trained on AudioSet do not unequivocally perform better on HEAR music tasks as well. This may be related to the type of music tasks. That is, while ARCH includes music tasks of a general nature (genre classification, tagging and instrument recognition (La Quatra et al., 2024)), HEAR includes niche music tasks including pitch classification and percussion classification. These types of tasks appear less suitable for WavJEPA representations, as WavJEPA obtains SOTA performance on just one of the HEAR music tasks. 

**Model efficiency:** Crucially, Figure 2 demonstrates that WavJEPA requires only a fraction of the pre-training data to surpass other time-domain models on HEAR and ARCH, despite the small model size of only 90 m parameters. Furthermore, we find that WavJEPA’s performance scales with the amount of pre-training data (Figure 2). 

## 4.2 EVALUATION ON NATURALISTIC SCENES 

**Transferability to naturalistic scenes:** Table 3 shows that the performance of all models is lower in naturalistic scenes. However, we find that, even when trained on non-naturalistic data, WavJEPA generalizes best to naturalistic scenes ( _s_ = 62.1) and performs almost similarly on NatHEAR as on HEAR (∆ _s_ = _−_ 3 _._ 9). This demonstrates that the high-level semantic representation learning approach of the JEPA architecture can successfully learn robust representations which generalize to noisy and reverberant environments. Further, WavJEPA excels specifically on tasks related to audio scene analysis and acoustic events on Nat-HEAR Table 3. WavJEPA also surpasses other Base and Large models trained on AudioSet on most speech and music tasks in NatHEAR, but not the WavLM Large model on Nat-HEAR speech tasks. 

**Impact of pre-training on naturalistic scenes:** We find that pre-training on naturalistic scenes improves the downstream performance on HEAR as well as NatHear. In particular, Table 3 shows that WavJEPA-Nat performs better than WavJEPA on both HEAR and NatHEAR on almost all tasks. Moreover, WavJEPA-Nat exhibits superior performance compared to all other models on both HEAR ( _s_ = 60.0, compare to Table 1) and NatHEAR ( _s_ = 61.2, compare to Table 3), even 

7

<!-- page: 8 -->

**==> picture [210 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
Model Performance vs. Pre-training Data Volume<br>**----- End of picture text -----**<br>


**==> picture [336 x 120] intentionally omitted <==**

**----- Start of picture text -----**<br>
HEAR ARCH<br>Model<br>80 WavJEPA<br>Wav2Vec2.0<br>HuBERT<br>60<br>Parameters<br>40 90M<br>300M<br>10 [7] 10 [8] 10 [7] 10 [8]<br>Training data (samples seen) Training data (samples seen)<br>)<br>(<br>s(m)<br>**----- End of picture text -----**<br>


Figure 2: **Downstream task performance** _s_ ( _m_ ) **vs. pre-training data (AudioSet).** Symbols depict performance _s_ for HEAR (left panel) and for ARCH (right panel) as a function of number of samples seen during pre-training. Symbol size reflects the number of model parameters. For WavJEPA, we depict performance after 50 k, 100 k, 200 k and 375 k training steps. 

Table 3: Generalization to naturalistic scenes (NatHEAR benchmark suite). Values and colors as in Table 1. 

|**Model**<br>**Size**|**Acoustic Events and Scene Analysis**<br>**DCASE FSD50K**<br>**LC**<br>**ESC-50**|**Speech**<br>**CD**<br>**VL**<br>**SC-5**|**Music**<br>**NS**<br>**BO**<br>**Mri-S**<br>**Mri-T**|**s(m)**|
|---|---|---|---|---|
|**Baseline**|||||
|HEAR-Naive N/A|0.7<br>8.7<br>26_._9_±_1_._9 16_._1_±_2_._0|28_._8_±_2_._6 12_._7_±_3_._6 12.3|**78.6**<br>**88.6**_±_6_._0 80_._5_±_0_._7<br>**75.0**_±_4_._0|0.0|
|**Speech pre-training**|||||
|W2V2<br>B<br>HuBERT<br>B<br>WavLM<br>B<br>D2V<br>B<br>W2V2<br>L<br>HuBERT<br>L<br>WavLM<br>L<br>D2V<br>L|32.0<br>23.0<br>54_._6_±_1_._9 36_._4_±_2_._9<br>57.6<br>26.6<br>52_._5_±_2_._2 49_._5_±_2_._2<br>25.3<br>20.5<br>52_._1_±_0_._6 41_._4_±_2_._1<br>15.5<br>12.0<br>39_._1_±_1_._1 19_._1_±_1_._5<br>52.7<br>26.6<br>53_._0_±_0_._9 42_._5_±_3_._5<br>16.7<br>23.4<br>52_._3_±_0_._3 48_._7_±_0_._7|48_._6_±_0_._6 27_._2_±_1_._6 78.9<br>57_._4_±_1_._1 46_._8_±_3_._4 89.2<br>52_._3_±_1_._5 47_._9_±_4_._6 89.9<br>42_._8_±_0_._9 30_._5_±_1_._5 71.9<br>50_._9_±_1_._0 33_._2_±_5_._0 58.7<br>50_._5_±_1_._2 42_._9_±_3_._9 69.9|15.2 71_._2_±_6_._4 75_._7_±_0_._5 45_._9_±_0_._6<br>16.0 77_._1_±_6_._0 78_._2_±_0_._7 52_._4_±_1_._6<br>11.2 61_._4_±_7_._2 69_._3_±_0_._9 39_._0_±_2_._0<br>4.6<br>58_._5_±_3_._2 55_._5_±_1_._7 36_._1_±_1_._2<br>**30.6** 69_._5_±_5_._7 77_._4_±_0_._8 54_._8_±_2_._7<br>14.6 75_._0_±_5_._7 84_._4_±_1_._4 54_._8_±_1_._4<br>14.6 76_._6_±_7_._6 82_._7_±_0_._6 54_._9_±_1_._4<br>10.4 63_._1_±_6_._6 59_._0_±_5_._2 33_._2_±_3_._1|32.7<br>44.6<br>37.3<br>19.7<br>35.6<br>38.6<br>58.5<br>30.1|
||75.6<br>34.1<br>58_._7_±_1_._0 56_._5_±_2_._8<br><br><br>|**63.7**_±_1_._6<br>**64.5**_±_2_._7<br>**92.6**<br>|||
||40.6<br>15.0<br>43_._5_±_0_._5 22_._9_±_2_._8|53_._7_±_1_._5 43_._1_±_4_._6 73.5|||
|**AudioSet pre-training**|||||
|W2V2<br>B<br>HuBERT<br>B<br>W2V2<br>L<br>HuBERT<br>L<br>WavJEPA<br>B|33.1<br>27.7<br>51_._0_±_1_._2 48_._1_±_2_._1<br>69.8<br>34.7<br>53_._0_±_1_._0 56_._6_±_2_._5<br>65.2<br>39.8<br>57_._6_±_1_._5 56_._1_±_2_._4|43_._9_±_2_._2 22_._3_±_1_._5 60.1<br>48_._9_±_1_._6 **40.6**_±_2_._0 76.3<br>52_._4_±_1_._0 26_._2_±_5_._1 74.2|21.2 75_._8_±_6_._0 74_._4_±_1_._6 45_._2_±_1_._5<br>**29.8** 80_._1_±_5_._8 79_._3_±_1_._1 52_._8_±_1_._2<br>17.8 74_._1_±_6_._2 81_._3_±_0_._9 52_._5_±_2_._5<br>26.2 77_._9_±_7_._2<br>**87.2**_±_1_._2 59_._9_±_2_._0<br>25.0 **82.2**_±_4_._4 87_._1_±_0_._7 57_._0_±_1_._2|30.5<br>44.3<br>45.2<br>52.4<br>**62.1**|
||68.1<br>37.8<br>58_._1_±_1_._9 55_._3_±_4_._1|54_._1_±_0_._5 29_._5_±_2_._6 77.6|||
||**83.1**<br>**47.0**<br>**59.7**_±_1_._8<br>**76.0**_±_2_._8|**57.6**_±_0_._4 35_._0_±_3_._0 **82.2**|||



though pre-trained with only half the batch size. This suggests that WavJEPA-Nat could benefit from further upscaling. 

Table 4: Impact of naturalistic pre-training on HEAR and NatHEAR performance. Note that WavJEPA-Nat was pre-trained with a lower batch size than the original WavJEPA. For comparison, we depict the results of WavJEPA pre-trained with a similar batch size as WavJEPA-Nat subsection 3.4. We indicate the best performing model per benchmark in **bold** . 

|**Model**<br>**Size**|**Acoustic Events and Scene Analysis**<br>**DCASE FSD50K**<br>**LC**<br>**ESC-50**|**Speech**<br>**CD**<br>**VL**<br>**SC-5**|**Music**<br>**NS**<br>**BO**<br>**Mri-S**<br>**Mri-T**|**s(m)**|
|---|---|---|---|---|
|**Performance on HEAR**|||||
|WavJEPA<br>B<br>WavJEPA-Nat<br>B|**92.3**<br>**51.2**<br>69.5_±_2_._4<br>78.7_±_2_._7<br>91.6<br>48.7<br>**72.4**_±_1_._8<br>**80.2**_±_1_._7|64.5_±_1_._3<br>**43.5**_±_3_._0<br>**89.2**<br>**65.9**_±_0_._7<br>39.7_±_2_._4<br>87.4|25.8 89.8_±_6_._6<br>96.8_±_0_._4<br>86.2_±_0_._5<br>**33.4 96.2**_±_5_._3<br>**97.4**_±_0_._5<br>**90.4**_±_0_._8|58.3<br>**60.0**|
|**Performance on Nat-HEAR**|||||
|WavJEPA<br>B<br>WavJEPA-Nat<br>B|80.6<br>**43.0**<br>56_._1_±_2_._9 68_._4_±_3_._1<br>**86.0**<br>42.4<br>**59.2**_±_1_._6 **72.6**_±_2_._5|52_._2_±_1_._8 **28.5**_±_2_._6 81_._5<br>**56.3**_±_1_._2 27_._9_±_3_._3 **81.9**|17_._0 79_._6_±_6_._2 86_._9_±_0_._8 58_._2_±_1_._0<br>**26.8 87.7**_±_3_._6 **89.3**_±_0_._4 **63.5**_±_0_._9|55.8<br>**61.2**|



## 4.3 ABLATION STUDIES 

**Ratio of clean versus naturalistic pre-training data:** Prior work on spectrogram-based representation learning showed that downstream task performance in scenes with reverberation benefits 

8

<!-- page: 9 -->

from pre-training on a mix of naturalistic, reverberant sounds and clean sounds in comparison to pre-training exclusively on naturalistic, reverberant scenes (Devnani et al., 2024). We investigated to what extent pre-training on a mixture of clean and naturalistic sound scenes affected the performance of WavJEPA-Nat on HEAR and NatHEAR. Figure 3 (left panel) shows that the higher the ratio of clean data ( _λ_ ), the lower the performance of WavJEPA-Nat on both HEAR and NatHEAR. This demonstrates that WavJEPA-Nat learns more robust and generalizable representations from naturalistic scenes and, importantly, that pre-training on naturalistic scenes boosts performance on downstream tasks comprising only clean sounds as well. These results demonstrate that combining the high-level semantic representation learning of the JEPA architecture with a dual waveform encoder as in WavJEPA-Nat can learn robust audio representations from noisy and reverberant data, enhancing performance on both clean sounds as well as noisy and reverberant scenes. 

**Top-K averaging:** We assessed whether averaging training targets over the top- _K_ layers improved the quality and robustness of WavJEPA’s learned representations for _K_ = 1, 4, 8, and 12 (i.e., all layers) (Baevski et al., 2020). The results show that top-K averaging indeed improves downstream performance on all HEAR tasks, although the range of improvement varied across tasks, see Figure 3 (middle panel). Moreover, for some scene analysis and speech tasks (LibriCount, ESC50, and Crema-D), performance peaked at _K_ = 8 and decreased again for _K_ = 12, while other tasks did not exhibit a difference in performance between _K_ = 8 and _K_ = 12. These findings indicate that top- _K_ layer averaging substantially improves downstream performance, but that an optimal value of _K_ is task-dependent. 

**Target length and context length:** The length of segments sampled for the training targets ( _Mtarget_ ) and segments sampled for the context block ( _Mcontext_ ) impacts their degree of distribution. A small value of _M_ leads to a more distributed context block or training target, while a large _M_ results in a less distributed context block or training target. We found that _Mcontext_ had little impact on the downstream task performance ( _s_ ( _m_ ) = [66.2, 66.0, 64.0] for _M_ = [5, 10, 15], see Appendix E). In contrast, we found that highly distributed training targets were consistently suboptimal for scene analysis and speech tasks, see Figure 3 (right panel). 

**Target probability:** A higher sampling probability for target indices ( _ptarget_ ) results in larger training targets and a smaller context block (as the proportion of _w_ sampled as target indices goes up, while the proportion of _w_ sampled as context indices goes down, see Appendix F). Ablating _ptarget_ revealed some variation in downstream performance, although not substantially: _ptarget_ = [0.15, 0.20, 0.25, 0.30] resulted in _s_ ( _m_ ) = [64.8, 65.9, 66.0, 63.0], see Appendix E. These findings suggest that sampling target indices with a probability between 0.15 and 0.25 is optimal, whereas a higher sampling probability reduces WavJEPA’s representation learning capacity. 

**==> picture [391 x 99] intentionally omitted <==**

**----- Start of picture text -----**<br>
Ratio clean to naturalistic Top- K averaging Target length (     ) M<br>70 100 100<br>60 80<br>re 60<br>*g 60<br>50 - HEAR  Wa)<br>NatHEAR 20 1 4 8 12 5 10 15<br>40<br>0 0.5 1.0<br>Clean data fraction (    ) A<br>(%)<br>(%)<br>DCASE FSD50K LC ESC50 CD VL SC5 MRI-S DCASE FSD50K LC ESC50 CD VL SC5 MRI-S<br>**----- End of picture text -----**<br>


Figure 3: **Ablation studies.** The left panel compares the performances on HEAR and NatHEAR for the WavJEPA-Nat architecture as a function of the ratio ( _λ_ ) between clean and naturalistic scenes in the pre-training data. The middle panel depicts the impact of the top- _K_ averaging parameter per HEAR task for WavJEPA. The right panel compares the impact of target length ( _Mtarget_ ) per task. The middle and right panels include only HEAR tasks for which WavJEPA performed better than baseline for ease of visualization. 

## 5 DISCUSSION AND CONCLUSION 

We presented WavJEPA, a state-of-the-art audio foundation model that leverages self-supervised semantic learning to obtain robust general-purpose audio representations from raw waveforms. Wav- 

9

<!-- page: 10 -->

JEPA’s results highlight the superior performance of semantic audio representation learning in comparison with representation learning at the speech unit or token level, as is common in existing time-domain speech representation learning approaches. Moreover, WavJEPA is highly efficient, requiring only a fraction of the training data in comparison to other time-domain models. Furthermore, our results demonstrate that WavJEPA is robust to noise and reverberation, emphasizing the suitability of semantic learning for deriving representations that generalize across acoustic environments. As WavJEPA’s speech representation learning could still be improved in comparison to Large speech models, we plan to investigate the benefit of pretraining WavJEPA on a combination of sound databases such as AudioSet and speech databases. Taken together, WavJEPA unlocks generalpurpose audio representation learning in the time domain, opening up avenues towards real-time audio foundation models. and high-quality audio generation audio foundation models. WavJEPA also highlights the potential of time-domain audio foundation models for high-quality speech stream generation in speech separation and speech denoising applications, as well other generative audio tasks. 

## ACKNOWLEDGMENTS 

This project received funding from the NWO Talent Program (VI.Veni.202.184; KH). This work used the Dutch national e-infrastructure with the support of the SURF Cooperative using grant no. EINF-14624. We would like to thank Robert Jan Schlimbach from the Snellius team for helpful discussions and their help with high performance cluster utilization. 

## REFERENCES 

- Akshay Anantapadmanabhan, Ashwin Bellur, and Hema A Murthy. Modal analysis and transcription of strokes of the mridangam using non-negative matrix factorization. In _2013 IEEE International Conference on Acoustics, Speech and Signal Processing_ , pp. 181–185, 2013. doi: 10.1109/ICASSP.2013.6637633. 

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. _arXiv preprint arXiv:2301.08243_ , 2023. 

- Alan Baade, Puyuan Peng, and David Harwath. Mae-ast: Masked autoencoding audio spectrogram transformer. In _Interspeech 2022_ , pp. 2438–2442, 2022. doi: 10.21437/Interspeech.2022-10961. 

- Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: a framework for self-supervised learning of speech representations. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546. 

- Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. data2vec: A general framework for self-supervised learning in speech, vision and language. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 1298–1312. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/baevski22a.html. 

- Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video, 2024. URL https://arxiv.org/abs/2404.08471. 

- Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, and Verena Rieser. SLURP: A spoken language understanding resource package. In _EMNLP_ . ACM, November 2020. doi: 10.18653/v1/ 2020.emnlp-main.588. URL https://aclanthology.org/2020.emnlp-main.588. 

- S¨oren Becker, Johanna Vielhaben, Marcel Ackermann, Klaus-Robert M¨uller, Sebastian Lapuschkin, and Wojciech Samek. AudioMNIST: Exploring explainable artificial intelligence for audio analysis on a simple benchmark. _Journal of the Franklin Institute_ , 2024. 

10

<!-- page: 11 -->

- Juan J Bosch, Jordi Janer, Ferdinand Fuhrmann, and Perfecto Herrera. A comparison of sound segregation techniques for predominant instrument recognition in musical audio signals. In _ISMIR_ , 2012. 

- Houwei Cao, David G Cooper, Michael K Keutmann, Ruben C Gur, Ani Nenkova, and Ragini Verma. CREMA-D: Crowd-sourced emotional multimodal actors dataset. _IEEE Trans Affect Comput_ , 5(4):377–390, October 2014. 

- Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. _International Conference on 3D Vision (3DV)_ , 2017. 

- Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vicenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu, Philip Robinson, and Kristen Grauman. Soundspaces: Audio-visual navigaton in 3d environments. In _ECCV_ , 2020. 

- Changan Chen, Carl Schissler, Sanchit Garg, Philip Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra, Philip W Robinson, and Kristen Grauman. Soundspaces 2.0: A simulation platform for visual-acoustic learning. In _NeurIPS 2022 Datasets and Benchmarks Track_ , 2022. 

- Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, and Furu Wei. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. _CoRR_ , abs/2110.13900, 2021. URL http://dblp.uni-trier. de/db/journals/corr/corr2110.html#abs-2110-13900. 

- Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, Wanxiang Che, Xiangzhan Yu, and Furu Wei. BEATs: Audio pre-training with acoustic tokenizers. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pp. 5178–5193. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/chen23ag.html. 

- Wenxi Chen, Yuzhe Liang, Ziyang Ma, Zhisheng Zheng, and Xie Chen. EAT: Self-supervised pretraining with efficient audio transformer. In Kate Larson (ed.), _Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24_ , pp. 3807–3815. International Joint Conferences on Artificial Intelligence Organization, 8 2024. doi: 10.24963/ijcai.2024/421. URL https://doi.org/10.24963/ijcai.2024/421. Main Track. 

- Dading Chong, Helin Wang, Peilin Zhou, and Qingcheng Zeng. Masked spectrogram prediction for self-supervised audio pre-training. In _ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 1–5, 2023. doi: 10.1109/ICASSP49357. 2023.10095691. 

- Giovanni Costantini, Iacopo Iaderola, Andrea Paoloni, and Massimiliano Todisco. EMOVO corpus: an Italian emotional speech database. In _LREC_ . European Language Resources Association (ELRA), 2014. 

- Micha¨el Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. Fma: A dataset for music analysis. In _ISMIR_ , 2017. 

- Bhavika Devnani, Skyler Seto, Zakaria Aldeneh, Alessandro Toso, Elena Menyaylenko, Barry-John Theobald, Jonathan Sheaffer, and Miguel Sarabia. Learning spatially-aware language and audio embeddings. _Advances in Neural Information Processing Systems_ , 37:33505–33537, 2024. 

- Zijian Dong, Ruilin Li, Yilei Wu, Thuan Tinh Nguyen, Joanna Su Xian Chong, Fang Ji, Nathanael Ren Jie Tong, Christopher Li Hsian Chen, and Juan Helen Zhou. Brain-JEPA: Brain dynamics foundation model with gradient positioning and spatiotemporal masking. _NeurIPS 2024_ , 2024. 

- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. _ICLR_ , 2021. 

11

<!-- page: 12 -->

Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck, Karen Simonyan, and Mohammad Norouzi. Neural audio synthesis of musical notes with wavenet autoencoders, 2017. 

- Zhengcong Fei, Mingyuan Fan, and Junshi Huang. A-JEPA: Joint-embedding predictive architecture can listen, 2024. URL https://arxiv.org/abs/2311.15830. 

- Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, and Xavier Serra. Fsd50k: An open dataset of human-labeled sound events. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 30:829–852, 2022. doi: 10.1109/TASLP.2021.3133208. 

- Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In _2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 776–780, 2017. doi: 10.1109/ICASSP.2017.7952261. 

- Yuan Gong, Cheng-I Lai, Yu-An Chung, and James Glass. Ssast: Self-supervised audio spectrogram transformer. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 36, pp. 10699–10709, 2022. 

- Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, R´emi Munos, and Michal Valko. Bootstrap your own latent a new approach to self-supervised learning. In _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546. 

- Pierre Guetschel, Thomas Moreau, and Michael Tangermann. S-JEPA: towards seamless crossdataset transfer through dynamic spatial attention. In _9th Graz Brain-Computer Interface Conference_ , Graz, Austria, September 2024. doi: 10.3217/978-3-99161-014-4-003. URL https: //arxiv.org/abs/2403.11772. 

- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In _2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 15979–15988, 2022. doi: 10.1109/CVPR52688.2022.01553. 

- Natalie Holz, Pauline Larrouy-Maestri, and David Poeppel. The variably intense vocalizations of affect and emotion (vivae) corpus prompts new perspective on nonspeech perception. _Emotion_ , 2022. 

- Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. _IEEE/ACM Trans. Audio, Speech and Lang. Proc._ , 29:3451–3460, October 2021. ISSN 2329-9290. doi: 10.1109/TASLP.2021.3122291. URL https://doi. org/10.1109/TASLP.2021.3122291. 

- Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. In _NeurIPS_ , 2022. 

- Moreno La Quatra, Alkis Koudounas, Lorenzo Vaiani, Elena Baralis, Luca Cagliero, Paolo Garza, and Sabato Marco Siniscalchi. Benchmarking representations for speech, music, and acoustic events. In _2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)_ , pp. 505–509, 2024. doi: 10.1109/ICASSPW62465.2024.10625960. 

- Edith Law, Kris West, Michael I Mandel, Mert Bay, and J Stephen Downie. Evaluation of algorithms using games: The case of music tagging. In _ISMIR_ , 2009. 

- Yann LeCun. A path towards autonomous machine intelligence version 0.9.2, 2022-06-27. 2022. 

- Kai Li, Guo Chen, Wendi Sang, Yi Luo, Zhuo Chen, Shuai Wang, Shulin He, Zhong-Qiu Wang, Andong Li, Zhiyong Wu, et al. Advances in speech separation: Techniques, challenges, and future trends. _arXiv preprint arXiv:2508.10830_ , 2025. 

12

<!-- page: 13 -->

- Steven R. Livingstone and Frank A. Russo. The ryerson audio-visual database of emotional speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english. _PloS one_ , 2018. 

- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL https: //arxiv.org/abs/1711.05101. 

- Vincent Lostanlen and Carmine-Emanuele Cella. Deep convolutional networks on the pitch spiral for musical instrument recognition. In _ISMIR_ , 2016. 

- Yi Luo and Nima Mesgarani. Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 27(8): 1256–1266, 2019. doi: 10.1109/TASLP.2019.2915167. 

- Matthew Maciejewski, Gordon Wichern, and Jonathan Le Roux. Whamr!: Noisy and reverberant single-channel speech separation. In _Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , May 2020. 

- A. Mesaros, T. Heittola, E. Benetos, P. Foster, M. Lagrange, T. Virtanen, and M. D. Plumbley. Detection and classification of acoustic scenes and events: Outcome of the DCASE 2016 challenge. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 26(2):379–393, Feb 2018a. ISSN 2329-9290. doi: 10.1109/TASLP.2017.2778423. 

- A. Mesaros, T. Heittola, E. Benetos, P. Foster, M. Lagrange, T. Virtanen, and M. D. Plumbley. Detection and classification of acoustic scenes and events: Outcome of the DCASE 2016 challenge. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 26(2):379–393, Feb 2018b. ISSN 2329-9290. doi: 10.1109/TASLP.2017.2778423. 

- Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino. Byol for audio: Exploring pre-trained general-purpose audio representations. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 31:137–151, 2023. doi: 10.1109/TASLP.2022. 3221007. 

- Karol J. Piczak. ESC: Dataset for Environmental Sound Classification. In _Proceedings of the 23rd Annual ACM Conference on Multimedia_ , pp. 1015–1018. ACM Press. ISBN 978-1-4503-34594. doi: 10.1145/2733373.2806390. URL http://dl.acm.org/citation.cfm?doid= 2733373.2806390. 

- Justin Salamon, Christopher Jacoby, and Juan Pablo Bello. A dataset and taxonomy for urban sound research. In _ACM Multimedia_ , MM ’14, New York, NY, USA, 2014. ACM. ISBN 9781450330633. doi: 10.1145/2647868.2655045. URL https://doi.org/10.1145/ 2647868.2655045. 

- Fabian-Robert St¨oter, Soumitro Chakrabarty, Emanu¨el Habets, and Bernd Edler. Libricount, a dataset for speaker count estimation, April 2018. URL https://doi.org/10.5281/ zenodo.1216072. 

- Mi Tian, Ajay Srinivasamurthy, Mark Sandler, and Xavier Serra. A study of instrument-wise onset detection in Beijing opera percussion ensembles. In _2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 2159–2163, 2014. doi: 10.1109/ICASSP. 2014.6853981. 

- Joseph Turian, Jordie Shier, Humair Raj Khan, Bhiksha Raj, Bj¨orn W. Schuller, Christian J. Steinmetz, Colin Malloy, George Tzanetakis, Gissel Velarde, Kirk McNally, Max Henry, Nicolas Pinto, Camille Noufi, Christian Clough, Dorien Herremans, Eduardo Fonseca, Jesse Engel, Justin Salamon, Philippe Esling, Pranay Manocha, Shinji Watanabe, Zeyu Jin, and Yonatan Bisk. HEAR: Holistic Evaluation of Audio Representations. In Douwe Kiela, Marco Ciccone, and Barbara Caputo (eds.), _Proceedings of the NeurIPS 2021 Competitions and Demonstrations Track_ , volume 176 of _Proceedings of Machine Learning Research_ , pp. 125–145. PMLR, 06–14 Dec 2022. URL https://proceedings.mlr.press/v176/turian22a.html. 

- Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance normalization: The missing ingredient for fast stylization, 2017. URL https://arxiv.org/abs/1607.08022. 

13

<!-- page: 14 -->

- J¨orgen Valk and Tanel Alum¨ae. Voxlingua107: A dataset for spoken language recognition. In _2021 IEEE Spoken Language Technology Workshop (SLT)_ , pp. 652–658, 2021. doi: 10.1109/ SLT48900.2021.9383459. 

- Pete Warden. Speech commands: A dataset for limited-vocabulary speech recognition, 2018. URL https://arxiv.org/abs/1804.03209. 

- Sarthak Yadav and Zheng-Hua Tan. Audio mamba: Selective state spaces for self-supervised audio representations. In _Interspeech 2024_ , pp. 552–556, 2024. doi: 10.21437/Interspeech.2024-1274. 

- Sarthak Yadav, Sergios Theodoridis, Lars Kai Hansen, and Zheng-Hua Tan. Masked autoencoders with multi-window local-global attention are better audio learners. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum? id=Q53QLftNkA. 

- Shuwen Yang, Po-Han Chi, Yung-Sung Chuang, Cheng-I Jeff Lai, Kushal Lakhotia, Yist Y. Lin, Andy T. Liu, Jiatong Shi, Xuankai Chang, Guan-Ting Lin, Tzu-Hsien Huang, Wei-Cheng Tseng, Ko tik Lee, Da-Rong Liu, Zili Huang, Shuyan Dong, Shang-Wen Li, Shinji Watanabe, Abdelrahman Mohamed, and Hung yi Lee. Superb: Speech processing universal performance benchmark. In _Interspeech 2021_ , pp. 1194–1198, 2021. doi: 10.21437/Interspeech.2021-1775. 

- Goksenin Yuksel, Marcel van Gerven, and Kiki van der Heijden. General-purpose audio representation learning for real-world sound scenes, 2025. URL https://arxiv.org/abs/2506. 00934. 

- Franz Zotter and Matthias Frank. _XY, MS, and First-Order Ambisonics_ , pp. 1–22. Springer International Publishing, Cham, 2019. ISBN 978-3-030-17207-7. doi: 10.1007/978-3-030-17207-7 ~~1~~ . URL https://doi.org/10.1007/978-3-030-17207-7_1. 

14

<!-- page: 15 -->

## APPENDIX 

## A DETAILED TRAINING SPECIFICATIONS 

Table 5: **Pre-training specifications** 

|Confguration|Pre-training|
|---|---|
|Optimizer<br>Optimizer momentum<br>Weight decay<br>Base learning rate<br>Learning rate schedule<br>Minimum learning rate<br>Dropout<br>Warm-up steps<br>Total steps<br>Early Stopping<br>Batch size<br>Accelerators<br>Target-Encoder & Context Encoder<br>Predictor<br>Target-Encoder & Context-Encoder Parameters<br>Predictor Parameters<br>Waveform Encoder<br>Waveform Encoder Parameters|AdamW<br>_β_1 = 0_._9,_β_2 = 0_._98<br>0.04<br>0.0004<br>linear-warmup + cosine decay<br>0.0<br>0.<br>100,000<br>375,000<br>N/A<br>32<br>2 x GPU H100 92 GB<br>ViT-B<br>ViT-S<br>86 M<br>22 M<br>Convolutions with 512 channels, strides (5,2,2,2,2,2) and kernel widths (10,3,3,3,3,2)<br>4 M|



## B DOWNSTREAM EVALUATION METRIC 

Similar to the procedure in SUPERB (Yang et al., 2021), let _st_ be the metric for task _t_ . We then calculate the generalizability metric HEAR _s_ ( _m_ ), ARCH _s_ ( _m_ ) and Nat-HEAR _s_ ( _m_ ) for model _m_ as: 

**==> picture [181 x 30] intentionally omitted <==**

Intuitively, this metric ranks the improvement of models over the baseline as a function of the maximum improvement over the baseline obtained by the current state-of-the-art. Note that we replace _st_ ( _m_ ) for task _t_ of model _m_ with 0 when the model scores below baseline performance for task _t_ . Similarly, when _st_ ( _SOTA_ ) is lower than baseline for task _t_ , we set for all models _st_ for this task to 0. In this way, all values are restricted to a range of improvement between 0 % and 100 %. 

## C HEAR, NAT-HEAR AND ARCH TASKS 

Table 6 illustrates the abbreviations, task description, and the type that we have utilized to benchmark our models. Furthermore, Table 2 demonstrates the specification of ARCH tasks. 

Table 6: Overview of the HEAR and Nat-HEAR tasks. 

|**Abbreviation**|**Task Name**|**Description**|**Type**|
|---|---|---|---|
|DCASE|DCASE-2016 Task 2 (Mesaros et al., 2018a)|Event detection of overlapping offce sounds in synthetic mixtures|Scene Analysis|
|FS50K|FSD50k (Fonseca et al., 2022)|Multilabel, large scale audio tagging|Environmental Sound Classifcation|
|LC|LibriCount (St¨oter et al., 2018)|Speaker Count Identifcation, Simulated Cocktail Party|Scene Analysis|
|ESC-50|ESC-50 (Piczak)|Environmental Sound Classifcation|Environmental Sound Classifcation|
|CD|Crema-D (Cao et al., 2014)|Emotion Recognition|Speech Analysis|
|VL|VoxLingua107 Top10 (Valk & Alum¨ae, 2021)|Spoken language identifcation|Speech Analysis|
|SC-5|Speech Command 5h (Warden, 2018)|Keyword Spotting, reduced training subset|Speech Analysis|
|NS|NSynth Pitch 5h (Engel et al., 2017)|Pitch Classifcation, reduced training subset|Music|
|BO|Beijing Opera (Tian et al., 2014)|Classifying percussion instruments|Music|
|Mri-S|Mridangam Stroke (Anantapadmanabhan et al., 2013)|Stroke classifcation in pitched percussion instruments|Music|
|Mri-T|Mridangam Tonic (Anantapadmanabhan et al., 2013)|Tonic classifcation in pitched percussion instruments|Music|



15

<!-- page: 16 -->

Table 7: Datasets included in ARCH with their corresponding domain, classification task types (single S or multi-label M), number of samples, average duration, and number of classes. 

|Dataset|Domain|Task|Samples|Avg duration|Classes|
|---|---|---|---|---|---|
|ESC-50 (Piczak)|Environmental Sound Classifcation|S|2000|5.0 s|50|
|US8K (Salamon et al., 2014)|Environmental Sound Classifcation|S|8732|3.61 s|10|
|FSD50K (Fonseca et al., 2022)|Environmental Sound Classifcation|M|51197|7.64 s|200|
|VIVAE (Holz et al., 2022)|Environmental Sound Classifcation|S|1085|0.90 s|6|
|FMA (Defferrard et al., 2017)|Music|S|8000|29.98 s|8|
|MTT (Law et al., 2009)|Music|M|21108|29.12 s|50|
|IRMAS (Bosch et al., 2012)|Music|M|8278|5.73 s|11|
|MS-DB (Lostanlen & Cella, 2016)|Music|S|21571|2.97 s|8|
|RAVDESS (Livingstone & Russo, 2018)|Speech Analysis|S|1440|3.70 s|8|
|AM (Becker et al., 2024)|Speech Analysis|S|30000|0.64 s|10|
|SLURP (Bastianelli et al., 2020)|Speech Analysis|S|72396|2.85 s|77|
|EMOVO (Costantini et al., 2014)|Emotion Recognition|S|588|3.12 s|7|



## D WAVJEPA-NAT FRAMEWORK 

To train WavJEPA-Nat on naturalistic scenes, we make use of the natural scenes introduced by (Yuksel et al., 2025). In particular, (Yuksel et al., 2025) provide a set of 85,000 binaural room impulse responses (BRIRs) for rendering two-channel sound scenes consisting of a sound source sampled from AudioSet and a noise source from WHAMR! (either localized or diffuse). A brief description of BRIRs and naturalistic sound scenes is provided here, a full description can be found in the original paper. 

The BRIRs encompass 85 houses from MatterPort3D (Chang et al., 2017). Room Impulse Responses (RIRs) are simulated for the different rooms in the houses with the Monte Carlo ray tracing simulator of SoundSpaces2.0 (Chen et al., 2020; 2022). Naturalistic scenes are generated by randomly positioning a listener, sound source and noise source in a room (1,000 for each house). Noise sources were either added as localized or as a diffuse noise field. The SoundSpaces2.0 simulator combined the simulated RIRs for each scene with a head-related imulse response (HRIR) to render a binaural RIR (BRIR). The BRIR captures the characteristics of both the room acoustics and binaural hearing. In total, the set consists of 85,000 BRIRs corresponding to 85,000 naturalistic sound scenes with _RT_ 60 ( reverberation strength) ranging between 0.2 and 0.5. 

**Simulating naturalistic sound scenes:** We used the naturalistic sound scene generation pipeline introduced by Yuksel et al. (2025). A brief description of the pipeline is included here, a full description can be found in the original paper. 

The pipeline makes use of the high-resolution 3D meshes of 85 houses from MatterPort3D [REF] to simulate room impulse responses (RIRs) for many different rooms with the Monte Carlo ray tracing simulator of SoundSpaces2.0 [REF]. A naturalistic scene (1,000 for each house) is subsequently generated by randomly positioning a listener, sound source and noise source in a room. Noise sources were either added as localized or as a diffuse noise field. The SoundSpaces2.0 simulator combines the simulated RIRs for each scene with a head-related imulse response (HRIR) to render a binaural RIR (BRIR). The BRIR captures the characteristics of both the room acoustics and binaural hearing. In this way, we generated Here, we used the state-of-art Monte Carlo ray tracing RIR simulator provided by SoundSpaces to simulate RIRs for a wide variety of rooms. We extracted high-resolution, detailed 3D meshes of houses with various architectural characteristics from Matterport3D as input for the SoundSpaces2.0 simulator. SoundSpaces combines the simulated RIRs with a head-related transfer function (HRTF) to generate a binaural RIR (BRIR), which captures both room specific acoustic properties and binaural hearing properties. Matterport3D contains scans of 90 houses. We discarded five houses for which meshes were not of sufficient quality. For each of the remaining 85 houses, we generated 1,000 naturalistic scenes. 

We generated a naturalistic scene by randomly sampling a listener location, a sound source location and a noise source location in the room. Listeners were placed within the room with a randomly sampled head orientation (range [0°, 360°]). We placed the sound source location at a randomly 

16

<!-- page: 17 -->

**==> picture [393 x 278] intentionally omitted <==**

**----- Start of picture text -----**<br>
input waveform waveform sampled context context predictor predictions<br>waveforms encoders embeddings context encoder block<br>€R™! wy, € RAx8<br>= we RR” We Cw Cw = HE.<br>©<br>60<br>+<br>€ R™™! wy € RY*78 o ee<br>8 EMA ™<br>>> ee = m n:<br>= | |<br>. training target aoe |<br>e learned mask Y © :<br>context indices rat Stee L ot2 :<br>target indices<br>2D positional embeddings target latent target training<br>encoder representation targets<br>sampling indices<br>**----- End of picture text -----**<br>


Figure 4: **Robust representation learning from naturalistic sound scenes including noise and reverberation.** WavJEPA-Nat is a multi-channel extension of WavJEPA which uses a dual waveform encoder to learn inter- and intra-channel characteristics and predicts 2D latent target representations from a 2D context block. The weights of the target encoder are not trained but updated using the exponential moving average (EMA) of the weights of the context encoder. 

sampled location with respect to the listener (distance range [1.5 m, 5 m]; azimuth range [0°, 360°]; elevation range [-90°, +90°]). Noise could either be localized (50 % of the scenes) or diffuse (50 % of the scenes). For localized noise, we randomly sampled one location in the room. For diffuse noise, we randomly sampled three, four or five locations in the room. We then rendered a set of BRIRs to describe the naturalistic scene. Given sound source location s, listener location r, and receiver head orientation _θ_ , we rendered the BRIR between the listener and the source as BRIR(s, r, _θ_ ). Given a number of noise sources _ni_ with noise source location _ϕi_ , listener location r, and receiver head orientation _θ_ , we rendered the BRIR between the listener and each noise source as BRIRi( _ϕi_ 140 , r, _θ_ ). This procedure resulted in a total of 85,000 sets of BRIRs with _RT_ 60 ( reverberation strength) ranging between 0.2 and 0.5. 

**Training on naturalistic scenes:** Similar to Yuksel et al. (2025), we divided the 85,000 BRIRs for the naturalistic scenes into a train set (70,000 scenes) and a test set (15,000 scenes) for down-stream evaluation (see section experiments). We used the 70,000 naturalistic scenes in the train set to generate a naturalistic version of the unbalanced training set of AudioSet. Specifically, during training we randomly paired every AudioSet clip with a noise sound clip from the WHAMR! background noise database. WHAMR! noise clips longer than 10 s were trimmed to 10 s duration and a linear fadein/fade-out of 200 ms was added to every WHAMR! noise clip prior to mixing of the sound scene. To create the naturalistic sound scene, we then convolved the sound source BRIR with the AudioSet clip to obtain S, and the noise source BRIR(s) with the WHAM! clip to obtain _Ni_ . In naturalistic scenes with diffuse background noise, the diffuse noise field was generated by summing the noise clips N = P i Ni 183. The naturalistic sound scene S was then calculated as S = T + bN, where b is 184, a scaling parameter introduced to mix target and noise sound clips at a given signal-to-noise ratio of 185 (SNR) ranging between +5 dB and +40 dB. 

17

<!-- page: 18 -->

## E DETAILED RESULTS ABLATION STUDIES 

Table 8: **Ablations for context and training target sampling procedure** . Downstream performance on HEAR benchmark. _Italics_ denote modifications with respect to the baseline. 

||WavJEPA|_Mcontext_|_Mtarget_|_ptarget_|
|---|---|---|---|---|
|_Mcontext_<br>10<br>_5_<br>_15_<br>10<br>10<br>10<br>10<br>10<br>_Mtarget_<br>10<br>10<br>10<br>_5_<br>_15_<br>10<br>10<br>10<br>_ptarget_<br>0.25<br>0.25<br>0.25<br>0.25<br>0.25<br>_0.15_<br>_0.20_<br>_0.30_|||||
|_s_(_m_)<br>66.0<br>66.2<br>64.0<br>66.9<br>62.9<br>64.8<br>65.9<br>63.0|||||



## F DISTRIBUTION OF TARGET AND CONTEXT SAMPLING 

Table 9: **Proportion of sound wave embedding** _w_ **sampled as context block and as training targets** . Values indicate average and 95 % confidence interval. Note that each sound wave embedding _w_ contains on average 4 training targets. 

|_Mcontext_|_Mtarget_|_ptarget_|Context block indices(%)|Trainingtarget indices(%)|
|---|---|---|---|---|
|_Baseline_|||||
|10|10|0.25|19.6 [11.5, 30.0]|22.7 [17.5, 25.0]|
|_Target Length_|||||
|10<br>10|5<br>15|0.25<br>0.25|18.8 [11.5, 26.5]<br>19.9 [11.0, 31.5]|22.8 [19.5, 25.0]<br>22.8 [15.5, 30.0]|
|_Context Length_|||||
|5<br>15|10<br>10|0.25<br>0.25|18.8 [11.5, 27.5]<br>19.7 [11.0, 30.5]|22.7 [17.0, 25.0]<br>22.7 [17.5, 25.0]|
|_Target Probability_|||||
|10<br>10<br>10|10<br>10<br>10|0.15<br>0.20<br>0.30|28.1 [18.0, 39.0]<br>23.2 [13.5, 34.0]<br>16.7 [10.5, 26.5]|14.3 [10.5, 15.0]<br>18.7 [14.0, 20.0]<br>26.6 [21.0, 30.0]|



18
