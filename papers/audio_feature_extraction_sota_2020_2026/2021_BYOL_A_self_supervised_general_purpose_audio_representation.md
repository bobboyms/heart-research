<!-- page: 1 -->

# BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation 

Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino _NTT Corporation_ , Japan daisuke.niizumi.dt@hco.ntt.co.jp 

_**Abstract**_ **—Inspired by the recent progress in self-supervised learning for computer vision that generates supervision using data augmentations, we explore a new general-purpose audio representation learning approach. We propose learning generalpurpose audio representation from a single audio segment without expecting relationships between different time segments of audio samples. To implement this principle, we introduce Bootstrap Your Own Latent (BYOL) for Audio (BYOL-A, pronounced ”viola”), an audio self-supervised learning method based on BYOL for learning general-purpose audio representation. Unlike most previous audio self-supervised learning methods that rely on agreement of vicinity audio segments or disagreement of remote ones, BYOL-A creates contrasts in an augmented audio segment pair derived from a single audio segment. With a combination of normalization and augmentation techniques, BYOL-A achieves state-of-the-art results in various downstream tasks. Extensive ablation studies also clarified the contribution of each component and their combinations.** 

_**Index Terms**_ **—self-supervised learning, general-purpose audio representation, audio data augmentation, mixup, random resize crop, BYOL** 

## I. INTRODUCTION 

The recent progress in unsupervised learning in natural language processing and computer vision domains has had a significant impact [1] [2], showing a substantial possibility of exploiting massive data without labels. For these successes, self-supervised learning methods that generate pseudo labels as supervision have played a central role [3]. In the computer vision domain, contrastive learning, which leverages the instance discrimination pretext task, has become dominant in self-supervised learning [4]. It achieves competitive performance compared to conventional supervised learning and even outperforms it in some downstream tasks such as object detection [5] [6]. 

In the contrastive learning setting, training is driven by comparison among positive and negative samples. The positive samples are augmented copies, or views, from the same input, and the negative samples are augmented views from different inputs. In the training process, representations of positive samples in embedding space are mapped closer together, whereas those of positive samples and negative samples are pushed away. However, for achieving better performance, this contrastive learning requires a large number of negative samples to compare [3]. To mitigate this problem, SimCLR [7] uses a significant number of batch samples, and MoCo [6] operates a large queue to accommodate a larger number of negative samples. 

**==> picture [202 x 54] intentionally omitted <==**

**----- Start of picture text -----**<br>
online<br>loss<br>target<br>Networks<br>Input space Embedding space<br>**----- End of picture text -----**<br>


Fig. 1. BYOL [8] for audio representation learning scenario. A single input audio _xi_ branches in two directions, or views, _vi_ and _vi[′]_[by][mixing][audio] _xj_ and _xk_ and making pitch/time modifications. _vi_ , _vi[′]_[are][projected][through] networks, and then loss is minimized on the projected embeddings. BYOL updates its online network weights with calculated loss, while it updates target network weights as an exponential moving average of the online counterpart. 

On the other hand, Bootstrap Your Own Latent (BYOL) [8] takes a different approach in that it does not use negative samples. Instead, it directly minimizes the mean squared error of embeddings originating from the same input with contrasts created by data augmentations. This representation learning setting may lead to collapsed representations [3], but the system architecture and training algorithm can avoid this problem. On the contrary, it claims state-of-the-art (SOTA) performance. 

In the audio domain, various self-supervised audio representation learning methods have been proposed [9] [10] [11] [12] [13]. In particular, COLA [14] learns general-purpose representations and outperforms previous methods. Many of these methods utilize the time-series aspect of audio signals: audio segments cropped closer in time are expected to have closer representations, whereas those far away in time are expected to have distanced representations. This is conceived to be a rational expectation, but contradictory use cases can be found easily. For example, repetitive sounds like music could have similar contents in the remote time segments because music compositions, by their nature, repeat motifs. On the other hand, short acoustic events (e.g., a single knock, a gunshot) can occur in a short duration; thus, even adjacent segments (e.g., a knock followed by a footstep) can make differences in contents for acoustic events. 

We think these are the fundamental problems caused by expecting relationships among multiple segments. In addition, similar problems can also happen when we use contrastive learning [11] [14] or triplet loss [12] [13] because the comparison of multiple samples is the core of their loss calculation. 

We address these problems by having general-purpose audio representations learned from a single audio segment instead

<!-- page: 2 -->

of from a comparison of multiple segments. In addition, the use of contrastive or triplet loss has to be avoided. This consequently requires the use of BYOL with effective audio data augmentations. For augmentations, we focus on learning 1) the foreground acoustic event sound (e.g., dog barking, gunshot) as a dominant sound representation, and 2) the sound texture details for describing general-purpose representation. 

- The better foreground sound representation is supposed to be consistent regardless of the background sound contents. Then it can be better learned from samples with random background variations while the foreground is kept unchanged. Considering a natural sound is a mixture of sounds from various sources, mixing small amount of sounds can approximate making variations on the background. Therefore, we adopt mixup [15], which mixes samples within a dataset. 

- Sounds from an acoustic scene or a sound texture can vary their pitch/speed/time, while the details can be consistent. This suggests that the details can be learned under the random variations of pitch/speed/time shifts; thus, we use approximation of audio pitch shifting and time stretching techniques [16] for this purpose. We expect learned representation to compress useful information of sound details in order to serve various general tasks. 

These techniques were used in a similar way in [11], but sources from different time segments were cropped. We have clear purpose regarding what information is to be learned, and we create changes on a pair of segments originating from exactly the same segment, not from multiple segments. 

The contributions of this paper are as follows: 

- We propose learning general-purpose audio representations from a single audio segment without expecting relationships between different time segments of audio samples. 

- We propose a self-supervised learning method named BYOL for Audio (BYOL-A, pronounced ”viola”). The method learns representations from a single audio segment input with a dedicated audio augmentation module that focuses on foreground and content details. It outperforms previous methods that learn from contrast of segments derived from different times. 

- We propose to learn foreground sound by combining pre-normalization and mixup, while learning content details through approximation of pitch shifting and time stretching. An extra post-normalization is also applied to compensate for statistical drift caused by augmentations. 

- We conduct extensive ablation studies that make clear the contributions of each block in the BYOL-A augmentation module. 

Fig. 1 illustrates the entire scenario of self-supervised learning from a single segment input. An input audio segment is once normalized. Then, two augmented copies, or views, are created by adding another sample as background sound and modifying pitch and time. These views are processed through parallel networks (i.e., online and target networks). 

Then, the online network is updated with loss calculated from projected embeddings, and the target network is updated as an exponential moving average of the online network. The system gradually learns to produce better representations by repeating these training processes. 

## II. RELATED WORK 

## _A. Self-supervised learning schemes_ 

In the image domain, self-supervised learning variants that make use of data augmentation techniques have been proposed. Contrastive learning methods such as SimCLR [7] use a big batch size, while MoCo [5] [6] has a FIFO queue to accommodate negative samples. Both claim SOTA performance. On the other hand, BYOL [8] also claims SOTA performance, though they use no negative samples. Among these methods, BYOL meets our needs for learning from a single input without the use of contrastive loss. 

Methods that combine self-supervised learning and mixup have also been proposed. Domain-agnostic contrastive learning (DACL) [17] proposes a mixup variant _Mixup-noise_ for contrastive learning setting. i-MIX [18] is a contrastive learning method that follows more of the original concept of applying mixup to both the features and its loss calculation. Fonseca et al. [11] proposed a contrastive learning approach for sound event representations and adopted a mixup variant _mix-back_ to add background noise to input log-mel spectrogram audio. All these methods are based on contrastive loss that compares positive and negative samples, and this is a fundamental difference from our approach. Experiments on i-MIX have also been conducted using BYOL and audio, as one of the multimodal applications of its domain-agnostic approach. However, audio has not been tested extensively, and the basic concepts are different from the present work; for instance, its usage of mixup is not concerned with audio content. 

In the audio domain, many methods rely on the relationships between segments cropped from different times. CPC [9] uses contrastive learning with future representations predicted from a series of past segments. Jansen et al. [12] uses triplet loss for learning with augmentation techniques, including adding random noise, time/frequency translation, example mixing, and temporal proximity. TRILL [13] learns representation by using triplet loss based on [12] so that segments that are closer in time are also closer in the embedding space. COLA [14] is a contrastive learning method for general-purpose audio representation that handles segments from the same clip as positives and segments from others as negatives. Among conventional methods for downstream tasks we tested, TRILL and COLA show SOTA performance. 

Cross-modal methods that input audio and other modalities are also related. COALA [19] uses tags (labels) accompanied with audio and uses contrastive loss to maximize co-alignment of both. _L_[3] -Net [20] (or OpenL3) inputs video along with audio and learns representations from the correspondence between the audio and video. These approaches show reference performance when non-audio data is used.

<!-- page: 3 -->

**==> picture [444 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
Views Representations Projections Prediction<br>«) fol) aol) online<br>single loss<br>input minimization<br>[ea bog pet target<br>Original BYOL Image Augmentation Encoding Projection Prediction exponential<br>moving<br>BYOL for Audio Audio Normalization &   average<br>Segment Augmentation<br>**----- End of picture text -----**<br>


Fig. 2. BYOL and BYOL-A system overview. 

**==> picture [157 x 164] intentionally omitted <==**

**----- Start of picture text -----**<br>
Pre- Random  Post-<br>Normalization Mixup Resize  Normalization<br>Crop<br>ucAAS<br>Fig. 3. Audio augmentation module of BYOL-A.<br>a) Crop crop area<br>virtual<br>crop<br>boundary<br>b) Resize<br>**----- End of picture text -----**<br>


Fig. 3. Audio augmentation module of BYOL-A. 

Fig. 4. Random resize crop of spectrogram. a) Randomly chosen crop area content is b) resized to the same size as the input. 

## _B. Bootstrap Your Own Latent (BYOL)_ 

BYOL is an algorithm for self-supervised learning of image representations. As shown in Fig. 2, it consists of two neural networks, referred to as online and target networks. The online network is defined by a set of weights _θ_ , and the target network has the same architecture as the online network but uses a different set of weights _ξ_ . First, BYOL produces two augmented views, _v_ ≜ _t_ ( _x_ ) and _v[′]_ ≜ _t[′]_ ( _x_ ), from an image _x_ by applying respectively image augmentations _t ∼T_ and _t[′] ∼T[′]_ , where _T_ and _T[′]_ denote the two distributions of the image augmentations. Then, the online network outputs a representation _yθ_ , a projection _zθ_ , and a prediction _qθ_ ( _zθ_ ) from the first view _v_ . On the other hand, the target network outputs _yξ[′]_[and][the][target][projection] _[z] ξ[′]_[from][the][second][view] _v[′]_ . Finally, the following mean squared error between the L2-normalized predictions _qθ_ ( _zθ_ ) and target projections _z_ ~~_[′]_~~ _ξ_[is] calculated: 

**==> picture [240 x 28] intentionally omitted <==**

where _⟨·, ·⟩_ denotes the inner product. To symmetrize the loss _Lθ,ξ_ , _L[′] θ,ξ_[is computed by feeding] _[ v][′]_[to the online network and] _v_ to the target network. The final loss is defined as _L_[BYOL] _θ,ξ_ = 

_Lθ,ξ_ + _L[′] θ,ξ_[.][At][each][training][step,][BYOL][minimizes][this][loss] function with respect to _θ_ only, but _ξ_ is a slowly moving exponential average of _θ_ : _ξ ← τξ_ + (1 _− τ_ ) _θ_ , where _τ_ is a target decay rate. 

It has been empirically shown that the combination of adding the predictor to the online network and using the moving average of the online network parameters as the target network encourages encoding more and more information within the online projection and avoids collapsed solutions such as constant representations. 

## III. BYOL FOR AUDIO (BYOL-A) 

We propose learning general-purpose audio representations from a single audio segment without expecting relationships between different time segments of audio samples. To implement this principle, we introduce BYOL-A. 

As shown in Fig. 2, we extend BYOL [8] for generalpurpose audio representation learning. In BYOL-A, we input audio preprocessed as a log-scaled mel-spectrogram, a timefrequency feature, because a typical encoder convolutional neural network accepts time-frequency features and converts them into representation embeddings. In addition, we replace the augmentation module in BYOL with ours so that the learning system can handle audio and create contrasts in augmented views for learning general-purpose audio representations. 

As shown in Fig. 3, the BYOL-A augmentation module consists of four blocks. First, the Pre-Normalization block normalizes a single input audio segment so that following augmentation processes, especially mixup, become stable. The normalized input is duplicated into two copies and fed to the following Mixup block. Then the Mixup block creates two outputs that are mixes of normalized inputs and randomly chosen past normalized inputs. The following Random Resize Crop (RRC) block resizes and crops the outputs randomly, and finally the Post-Normalization block adjusts statistical drifts caused by the former augmentations. 

For learning general-purpose audio representations, we focus on foreground acoustic events and all content details. The Mixup block is designed to create contrast for learning foreground acoustic event representations, and it is combined with the Pre-Normalization block for stable performance gain. The RRC block approximates pitch shifting and time stretching in

<!-- page: 4 -->

time-frequency features for learning generic representations of content details. 

## _A. Pre-Normalization_ 

Input data _x_ is normalized to _x_ ˜ = _x−σ µ_[,][where] _[µ]_[and] _[σ]_ are the average and standard deviation of training samples respectively. 

This normalization stabilizes computations in the system in two ways. One is by mitigating augmentation parameter sensitivity, which enables following blocks to assume that input range virtually follows _N_ (0 _,_ 1). The other is by normalizing statistical differences between training datasets. 

## _B. Mixup for foreground acoustic event_ 

Using normalized log-mel spectrogram audio as input, the Mixup block mixes past randomly selected input audio in a small ratio. As a result, added audio becomes a part of the background sound in the mixed audio. This produces contrast in the background sound in the pair of Mixup outputs, which in turn encourages learning representations of invariant foreground acoustic event sounds. This is similar to _mixback_ [11], which adds a random sample from a dataset as background sound, but the purpose of the _mix-back_ is to create a set of positive samples sharing less information in the contrastive learning setting. 

We use the basic mixup calculation as an augmentation technique. While mixup was originally designed for mixing both features and labels, we apply it to audio features only (because of the absence of labels). In addition, as audio is log-scaled, we convert input to a linear scale before the mixup calculation and convert it back to a log-scale again. In this paper, we refer to these operations as log-mixup-exp, from the analogy to the log-sum-exp [21] calculation. Log-mixup-exp of _i_ th input _xi_ is 

**==> picture [208 x 11] intentionally omitted <==**

where _xk_ is a mixing counterpart, and mixing ratio _λ_ is sampled from uniform distribution _U_ (0 _._ 0 _, α_ ) instead of from a beta distribution in the original mixup. In addition, _α_ is a mixing ratio hyper-parameter that controls the degree of contrast between the resulting two mixed outputs. We observed that the evaluation result improves with smaller _α_ , 0 _._ 4 for example, where _x_ ˜ _i_ keeps the original contents _xi_ more than its counterpart _xk_ , as we found in preliminary experiments. 

_xk_ is randomly chosen from a memory bank, a FIFO queue, storing past inputs. As input is randomly sampled from the training dataset, queued samples in the memory bank form random subsets of the dataset. We store 2 _,_ 048 samples in the memory bank, which is larger than the batch size and big enough to maintain randomness. 

## _C. RRC for all content details_ 

RRC is an image augmentation technique we use as an approximation of pitch shifting and time stretching of input audio log-mel spectrograms for learning a representation of all content details. We expect details spread over a spectrogram 

to be learned regardless of the created differences in pitch and time among outputs. 

Fig. 4 shows the random crop procedure. The unit size of the input spectrogram consists of a number of frequency bins, _F_ , and a number of time frames, _T_ . First, we sample the random crop area from the virtual crop boundary, which has longer time frames than the input, 1 _._ 5 _× T_ for example. The size of the crop area is randomly sampled as 

**==> picture [139 x 29] intentionally omitted <==**

where _FC_ and _TC_ are the number of frequency bins and number of time frames of random crop size, respectively, _h_ 1 and _h_ 2 form a frequency bin range [ _h_ 1 _, h_ 2], _w_ 1 and _w_ 2 form a time frame range [ _w_ 1 _, w_ 2], _⌊·⌋_ is a floor function, and min( _·, ·_ ) is a minimum function. Contents in the crop area are then resized to the size of the input with bicubic interpolation. The virtual crop boundary is wider than the input and we use [0 _._ 6 _,_ 1 _._ 5] for both frequency bin and time frame ranges in this paper, so the crop area can contain the outside of the input. This area is filled with zeros. Note that we do not crop the outside of the frequency bin, which is restricted by the min() function in the _FC_ calculation above. 

## _D. Post-Normalization for statistical drift adjustment_ 

Augmentation operations before this block can cause statistical drift in their outputs. This block adjusts the drift so that final output views of the BYOL-A augmentation module become _∼ N_ (0 _,_ 1). The calculation is done in the same manner as pre-normalization, but uses average and standard deviations calculated from batch samples. 

## IV. EXPERIMENTS 

We assessed the representations learned by BYOL-A by conducting experiments with six audio downstream tasks under a linear evaluation protocol. The downstream tasks have both audio samples and labels, and the evaluation result is the accuracy of a linear model trained in a supervised setting with representation feature embeddings as input. These feature embeddings were converted from audio samples using a frozen encoder network pretrained by BYOL-A. 

In all experiments, BYOL-A was pretrained on AudioSet [22], a large scale dataset commonly used in previous studies. 

## _A. Experimental Setup_ 

We repeated the cycle of pretraining and evaluation and averaged the results. The number of cycles was three for pretraining on the full AudioSet; it was five for pretraining on a 1/10 subset of AudioSet or FSD50K [23]. 

_1) Audio data format and conversion parameters:_ We converted all sound clips to a log-scaled mel spectrogram with a sampling frequency of 16,000 Hz, window size of 64 ms, hop size of 10 ms, and mel-spaced frequency bins _F_ = 64 in the range 60–7,800 Hz. The number of frames, _T_ , in one segment was 96 in pretraining, which corresponds to 1,014 ms. A segment of shape _F × T_ was randomly cropped from

<!-- page: 5 -->

each audio clip and used in pretraining. For the downstream tasks, the number of frames, _T_ , in one segment was determined by the average duration of each dataset (e.g., 400 frames for NSynth with average duration of 4.0 s.). A segment of shape _F × T_ was randomly cropped from each audio clip and encoded for linear evaluation in the downstream tasks. Shorter clips were padded with zeros at both the head and tail. 

_2) Encoder network:_ We used a simple CNN based on a network used in a solution of Task 6 (Automated Audio Captioning) of the Detection and Classification of Acoustic Scenes and Events (DCASE) 2020 Challenge [24] [25]. Though this network is simpler and smaller than networks from the image domain (e.g., ResNet [26], EfficientNet-B0 [27]) used in previous studies [13] [14], we think this is a realistic design choice because smaller networks have been practically used in audio machine learning studies [24] [28] [29]. In addition, the experimental results in this paper show that the performance of our CNN is good enough to score SOTA. 

The CNN has a variable hyper-parameter, namely the dimension size of representation embedding, that is also used as the size of the linear layers. We varied this hyper-parameter in the experiments to see the performance changes. 

The details of the network are described in appendix A. 

_3) BYOL-A pretraining details:_ Projection and prediction in BYOL-A networks are the same multilayer perceptrons (MLPs) in the original BYOL, i.e., a linear layer with output size of 4,096 followed by batch normalization [30], rectified linear units (ReLU), and a linear layer to output embeddings with 256 dimensions. We used the Adam optimizer with a learning rate of 0 _._ 0003, target decay rate parameter _τ_ = 0 _._ 99, and batch size of 256, and it was trained for 100 epochs. 

For augmentation blocks, we used mixup _α_ = 0 _._ 4. The size of the virtual crop boundary has the number of time frames to 1 _._ 5 times the input and the number of frequency bins to the same as the input. The crop range is [0 _._ 6 _,_ 1 _._ 5] for both frequency bins and time frames. All these values were found in a preliminary parameter search using Optuna [31]. We used a single set of augmentation _T_ , i.e., _t, t[′] ∼T_ . 

The number of pretraining samples collected from balanced train segments and unbalanced train segments data splits in the AudioSet [22] dataset was 1,963,807 in total. No labels were used in pretraining. In the ablation studies, a 1 _/_ 10 subset of AudioSet (210,315 samples in total) was used in pretraining. 

_4) Downstream tasks:_ Below are the tested tasks. These tasks were also used in previous studies [13] [14] [19] [20], so evaluation results can be compared. 

- NSynth (NS) [32] dataset as musical instrument family classification with 11 family name classes for 305,979 samples and average duration of 4.0 s. 

- UrbanSound8K (US8K) [33] dataset as sound classification with ten acoustic scene classes for 8,732 samples and average duration of 4.0 s. This task dataset has predefined splits with ten folds. We followed leave-one-out cross validation of these ten folds to get the average accuracy. 

- VoxCeleb1 (VC1) [34] dataset as speaker identification task with 1,211 speaker ID classes for 153,514 samples and average duration of 8.2 s. 

- VoxForge (VF) [35] dataset as language identification task with six language ID classes for 176,438 samples and average duration of 5.8 s. 

- Speech Commands V2 (SPCV2) [36] dataset as command classification task with 35 command word classes for 105,829 examples and duration of 1.0 s. 

- The same Speech Commands V2 dataset, but with 12 command word classes (SPCV2/12). Unlike SPCV2, classes consist of ten basic word selected from 35 words, as well as new classes _silence_ and _others_ . The _silence_ audio samples were created from background noise sounds, which are not used as a class in SPCV2, and _others_ contains all other 25 class samples from SPCV2. This setup results in a highly imbalanced number of class samples and additional complexity compared to SPCV2 in spite of the smaller number of classes. 

_5) Linear evaluation details:_ In the linear evaluation protocol, a single linear layer is trained to fit the downstream dataset. We trained the linear model using Adam optimizer with a learning rate of 0 _._ 001, for a 200 epoch maximum with early stopping of ten epochs. We ran each evaluation ten times and averaged the accuracy results. 

_6) Comparison with COLA [14]:_ COLA was specifically reproduced with our implementation, denoted as COLA’. COLA takes the opposite approach from ours: It maximizes agreement of two random cropped segments of the same sound clip, uses contrastive loss for comparison with negative samples, and includes no data augmentations. 

COLA’ has an extra normalization module, the same as BYOL-A, and the same encoder network as BYOL-A. This enables us to compare results between a single-segment input and two-segment input and evaluate the effectiveness of BYOLA augmentation blocks under the two-segment input setup of COLA’. We used a version of COLA with bilinear similarity [14] that has better performance. Pretraining parameters were the same as with BYOL-A pretraining, except batch size was 1 _,_ 024, four times larger than with BYOL-A, for ensuring better performance under contrastive learning. 

## _B. Experimental results: Comparison with previous methods_ 

Table I shows the results for previous methods (TRILL and COLA), reference methods (OpenL3 [20], pretrained with audio+video, and COALA [19], pretrained with audio+tag), COLA’ (our implementation of COLA), and our proposed BYOL-A. For BYOL-A and COLA’, we varied the dimensions of representation embeddings as 512, 1,024, and 2,048, which also increases encoder network capacity. 

As shown in the Table I, BYOL-A with 2,048 dimensions outperforms other methods in all tasks. It shows the average result of 77 _._ 8%, which surpasses the 68 _._ 7% for COLA’ with 2,048 dimensions. While BYOL-A with 2,048 dimensions exhibits the best results, embeddings with 512 dimensions

<!-- page: 6 -->

TABLE I 

|TABLE I|TABLE I|TABLE I|
|---|---|---|
|PERFORMANCE COMPARISON RESULTS FOR DOWNSTREAM TASKS|||
|Method<br>Dim.<br>Remarks|NS<br>US8K<br>VC1<br>VF<br>SPCV2/12<br>SPCV2|Average|
|TRILL [13]<br>conventional<br>COLA [14]<br>conventional<br>OpenL3 [20] 1<br>reference<br>COALA [19] 2<br>reference|N/A<br>N/A<br>17.9%<br>88.1%<br>74.9%<br>N/A<br>63.4%<br>N/A<br>29.9%<br>71.3%<br>71.7%<br>62.4%<br>N/A<br>78.2%<br>N/A<br>N/A<br>N/A<br>N/A<br>73.1%<br>72.7%<br>N/A<br>N/A<br>N/A<br>N/A|N/A<br>N/A<br>N/A<br>N/A|
|COLA’<br>512-d<br>our impl.<br>COLA’<br>1024-d<br>our impl.<br>COLA’<br>2048-d<br>our impl.|65.4%<br>76.3%<br>25.9%<br>73.5%<br>59.1%<br>63.1%<br>69.3%<br>77.1%<br>31.2%<br>76.7%<br>71.9%<br>71.0%<br>70.2%<br>78.5%<br>30.4%<br>79.5%<br>76.7%<br>76.8%|60.5%<br>66.2%<br>68.7%|
|BYOL-A<br>512-d<br>**proposed**<br>BYOL-A<br>1024-d<br>**proposed**<br>BYOL-A<br>2048-d<br>**proposed**<br>|69.1%<br>78.2%<br>33.4%<br>83.5%<br>86.5%<br>88.9%<br>72.7%<br>78.2%<br>38.0%<br>88.5%<br>90.1%<br>91.4%<br>**74.1%**<br>**79.1%**<br>**40.1%**<br>**90.2%**<br>**91.0%**<br>**92.2%**<br>|73.3%<br>76.5%<br>**77.8%**|



> 1 Reference results pretrained with audio+video and trained with MLP instead of linear layer. 

> 2 Reference results pretrained with audio+tag and trained with MLP instead of linear layer. 

TABLE II 

ABLATIONS OF BYOL-A AUGMENTATION MODULE WITH ACCURACY RESULTS, PRETRAINED WITH 1/10 AUDIOSET 

|Augmentation blocks used|NS<br>US8K<br>VC1<br>VF<br>SPCV2/12<br>SPCV2|Average<br>Degradation|
|---|---|---|
|Mixup+RRC (BYOL-A)|**71.2%**<br>77.0%<br>31.0%<br>83.1%<br>**84.5%**<br>87.2%|**72.3%**|
|Mixup+Gaussian+RRC<br>Gaussian+RRC<br>RRC<br>Mixup<br>Gaussian|69.5%<br>74.3%<br>25.2%<br>**84.0%**<br>82.8%<br>**87.4%**<br>69.7%<br>73.1%<br>29.2%<br>83.1%<br>78.0%<br>83.1%<br>69.4%<br>**77.1%**<br>**34.5%**<br>80.3%<br>71.4%<br>77.4%<br>55.6%<br>69.4%<br>22.3%<br>78.3%<br>75.8%<br>82.0%<br>29.5%<br>31.2%<br>0.9%<br>57.9%<br>9.4%<br>10.3%|70.5%<br>BYOL-A -1.8<br>69.3%<br>BYOL-A -3.0<br>68.4%<br>BYOL-A -3.9<br>63.9%<br>BYOL-A -8.4<br>23.2%<br>BYOL-A -49.1|



## TABLE III 

also shows competitive performance, especially in the speech command tasks. 

## _C. Ablation study: Contribution of data augmentations_ 

In this experiment, we tested various combinations of data augmentation blocks with BYOL-A with 512 dimensions, which was pretrained with 1/10 AudioSet. We kept the Preand Post-Normalization blocks, and replaced augmentation blocks between them. We also used a Gaussian-noise augmentation block that interpolates training input with random data points sampled from the normal distribution. This was done for comparison with mixup that interpolates within dataset samples. The Gaussian-noise block sampled from _N_ (0 _,_ 0 _._ 4), the best parameter in a preliminary test, and followed the logmixup-exp calculation. 

Table II shows the results for combining augmentations. 

_1) Contribution of mixup compared with Gaussian-noise:_ If we focus on the average result for the Gaussian-noise block, it improves 0 _._ 9 from the RRC’s 68 _._ 4% to Gaussian+RRC’s 69 _._ 3%. On the other hand, mixup improves 3 _._ 9 from RRC’s 68 _._ 4% to Mixup+RRC (BYOL-A)’s 72 _._ 3%. However, if we add Gaussian-noise on top of Mixup+RRC, Mixup+Gaussian+RRC degrades average performance 1 _._ 8 down to 70 _._ 5%. This empirically shows that mixup’s interpolating with samples within dataset works effectively in the BYOL-A setting, whereas interpolating with random data points is not effective. 

_2) Contribution of mixup, RRC, and their combination:_ When only Gaussian noise was used, the average result was 23 _._ 2%; representations cannot achieve sufficient performance. With mixup only, it improves to 63 _._ 9%, especially with a larger performance gain on speech command SPCV2. The use of mixup consistently gains performance for SPCV2 tasks in other results also, indicating that mixup is effective for learning foreground sound representation, because the SPCV2 

ABLATIONS OF NORMALIZATION BLOCKS WITH AVERAGE ACCURACY RESULTS, PRETRAINED ON 1/10 AUDIOSET 

|Method|Average|Degradation|
|---|---|---|
|BYOL-A|**72.3%**||
|w/o Post-Norm<br>w/o Pre-Norm (mixup _α_= 0_._05)<br>w/o Pre-Norm (mixup _α_= 0_._1)<br>w/o Pre-Norm (mixup _α_= 0_._4)|72.1%<br>70.5%<br>70.3%<br>68.9%|BYOL-A -0.2<br>BYOL-A -1.8<br>BYOL-A -2.0<br>BYOL-A -3.4|



dataset is dominant with clear word utterances. With RRC only, the average result improves up to 68 _._ 4% and shows competitive performance among all tasks, indicating that RRC is effective for learning general-purpose representations. Finally, the average result for Mixup+RRC (BYOL-A) is 72 _._ 3%, the best average performance among all combinations. This shows that mixup and RRC work complementarily on average, except for a performance drop in the VC1 task. The VC1 task is speaker identification from real-world random utterances of 1,211 celebrities; the class label is more related to the speaker’s voice characteristics than to the utterance content. The recordings have relatively low background noise. Thus, we think the details of textures like consonant or breathing sounds is more important for the VC1 task. Therefore, just applying RRC was better than the use of both mixup and RRC; mixup can add noise to the details. 

## _D. Ablation study: Contribution of normalization blocks_ 

In this experiment, we assessed the contribution of Preand Post-Normalization blocks by removing one of them. We avoided removing both to keep the basic configuration of the machine learning setup. Table III shows that removing pre-normalization degrades performance, ranging from _−_ 1 _._ 8 to _−_ 3 _._ 4, which is a larger impact than removing postnormalization (degradation of _−_ 0 _._ 2). 

The reason for the larger degradation observed with the removal of pre-normalization is related to the log-mixup-

<!-- page: 7 -->

exp calculation. In this calculation, the log-mel spectrogram is once expanded to a linear scale by exp( _·_ ), mixed with other samples for random degree _λ ∼ U_ (0 _._ 0 _, α_ ), and then compressed again with log( _·_ ). Then the effect of mixup and its hyper-parameter _α_ depends on the range of the log-mel spectrogram, because the output of mixup is compressed by _·_ following log( ). Pre-normalization helps to stabilize the effect of log-mixup-exp by making the range of the spectrogram constant. The mixup _α_ = 0 _._ 4 was the sweetest spot found in the preliminary parameter search, and ”w/o Pre-Norm” results show that the sweet spot of mixup _α_ drifts, and it does not recover the best performance of 72 _._ 3% even when we set _α_ down to 0 _._ 05. 

The post-normalization helps prevent degradation caused by statistical drift caused by augmentations. 

In summary, the combination of normalizations and augmentations contributes to both performance gain and recovery. 

## V. CONCLUSION 

In this paper, we proposed a self-supervised learning method called BYOL for Audio (BYOL-A), a version of BYOL extended to the audio domain, which learns representations from a single segment audio, and showed its state-of-the-art performance. The augmentation module of BYOL-A consists of Normalization, Mixup and Random Resize Crop (RRC) blocks. The mixup is effective for learning representations of foreground acoustic event sounds, while RRC works effectively for general-purpose audio representation, and applying both works complementarily. The Pre- and Post-Normalization blocks work for performance gain of mixup and the recovery from statistical drift. As a result, all these modules work together as a whole augmentation module in BYOL-A to surpass the previous state-of-the-art results. 

The expectation of agreement or disagreement of multiple audio segments was shown to be effective in previous studies, but in this study, it was found that representation learning from a single segment is possible, and it even outperforms former methods. 

## APPENDIX 

In this appendix, we explain the details of experiments and additional experiments with analysis. In appendix A, we explain the details of the encoder network and parameter settings. In appendix B, we assess the effectiveness of the BYOL-A augmentation module by applying it to COLA’. Finally, in appendix C, we show the performance of BYOL-A pretrained on the FSD50K dataset and compare results. 

## _A. Details of encoder network_ 

Table IV shows the architecture of the encoder convolutional neural network based on a network used in a solution of Task 6 (Automated Audio Captioning) at DCASE 2020 Challenge [24] [25], where input shape is [ _B,_ 1 _,_ 64 _,_ 96], _B_ is batch size; a single channel, 64 frequency bins, and 96 time frames. 

Input is compressed by three sets of convolutional blocks of all 64 channels with stride of 2. Then, it is reshaped to 

[ _B,_ 12 _,_ 512], where time frames 12 = 2 _·_ 962 _·_ 2[and][feature][di-] mensions 512 = 64 _·_ 2 _·_ 642 _·_ 2[.][The][following linear][layers][upscale] the size of feature dimensions to the hyper-parameter _d_ , 2,048 dimensions for example. Then, in the last layer, final output embedding _y_ is calculated as _y_ = max( _x,_ 1) _⊕_ mean( _x,_ 1), where _x_ is input to this calculation, max( _x,_ 1) is max operation along the time axis, mean( _x,_ 1) is averaging along the time axis, and _⊕_ is an element-wise sum. 

This network outputs representation embeddings with a fixed shape ( _d,_ ). The total numbers of network parameters are 600,192 (512-d), 1,649,792 (1024-d) and 5,321,856 (2048-d). 

_B. Experiments on BYOL-A augmentation blocks with COLA’+_ 

We assessed the effectiveness of data augmentation blocks in the BYOL-A augmentation module by adding augmentation blocks to COLA’, named COLA’+; the original COLA does not use data augmentations. Table V shows that improvement of COLA’+Mixup is 5 _._ 8, larger than Gaussian-noise’s 1 _._ 8. Mixup is also effective for making useful contrast with two differently random cropped input segments. 

However, RRC was not as effective as when it was used with BYOL-A. Adding RRC to COLA’+Mixup improves performance to 68 _._ 2 _−_ 66 _._ 7 = 1 _._ 5, which is less performance gain compared to when RRC is applied to BYOL-A (Mixup only), where the improvement is 72 _._ 3 _−_ 63 _._ 9 = 8 _._ 4. The explanation for this is that additional random resizing and cropping to segments that have already been randomly cropped is less 

TABLE IV 

ENCODER NETWORK ARCHITECTURE (2048-D) 

|Layer-#<br>Layer prms.|Output shape|Parameters|
|---|---|---|
|Conv2D-1<br>3x3@64<br>BatchNorm2D-2<br>ReLU-3<br>MaxPool2D-4<br>2x2,stride=2<br>Conv2D-5<br>3x3@64<br>BatchNorm2D-6<br>ReLU-7<br>MaxPool2D-8<br>2x2,stride=2<br>Conv2D-9<br>3x3@64<br>BatchNorm2D-10<br>ReLU-11<br>MaxPool2D-12<br>2x2,stride=2<br>Reshape-13<br>Linear-14<br>out=2048<br>ReLU-15<br>Dropout-16<br>0.3<br>Linear-17<br>out=2048<br>ReLU-18<br>max(_·_)_⊕_mean(_·_)-19|[B, 64, 64, 96]<br>[B, 64, 64, 96]<br>[B, 64, 64, 96]<br>[B, 64, 32, 48]<br>[B, 64, 32, 48]<br>[B, 64, 32, 48]<br>[B, 64, 32, 48]<br>[B, 64, 16, 24]<br>[B, 64, 16, 24]<br>[B, 64, 16, 24]<br>[B, 64, 16, 24]<br>[B, 64, 8, 12]<br>[B, 12, 512]<br>[B, 12, 2048]<br>[B, 12, 2048]<br>[B, 12, 2048]<br>[B, 12, 2048]<br>[B, 12, 2048]<br>[B, 2048]|640<br>128<br>0<br>0<br>36,928<br>128<br>0<br>0<br>36,928<br>128<br>0<br>0<br>0<br>1,050,624<br>0<br>0<br>4,196,352<br>0<br>0|



## TABLE V 

AVERAGE ACCURACY RESULTS OF AUGMENTATION BLOCKS ON COLA’+ AND BYOL-A, PRETRAINED WITH 1/10 AUDIOSET 

|Method|Average|Improvement|
|---|---|---|
|COLA’|60.9%||
|COLA’+Gaussian<br>COLA’+Mixup<br>COLA’+Mixup+RRC|62.7%<br>66.7%<br>68.2%|COLA’ +1.8<br>COLA’ +5.8<br>COLA’ +7.3|
|BYOL-A (Mixup+RRC)|**72.3%**||
|BYOL-A (Mixup only)<br>BYOL-A (RRC only)|63.9%<br>68.4%|BYOL-A -8.4<br>BYOL-A -3.9|

<!-- page: 8 -->

TABLE VI 

AVERAGE PERFORMANCE OF BYOL-A WITH PRETRAINING DATASETS 

|Pretraining dataset<br>Size|Average|Difference|
|---|---|---|
|AudioSet (1/10 subset)<br>210K|**72.3%**||
|FSD50K<br>40K|70.1%|AudioSet -2.2|



effective; and the RRC-only setting of BYOL-A performs better. 

In summary, the BYOL-A augmentation module is also effective with COLA’+, but more effective with BYOL, a single segment input setting. 

## _C. Experiment for pretraining on FSD50K_ 

In addition to AudioSet, we conducted pretraining on the FSD50K [23] dataset and compared performance with pretraining on AudioSet. 

All the training hyper-parameters and the setup were the same as in the former experiments, except the number of pretraining epochs was set to 500 on FSD50K, so that the total number of data samples consumed during training would be closer to the experiments on the AudioSet 1/10 subset. In addition, we used the development subset of the FSD50K, which has 40,966 samples, five times less than the AudioSet 1/10 subset (210,315 samples in total). 

As shown in Table VI, the average performance for FSD50K is 70.1%; the difference from AudioSet is -2.2. We think this degradation is caused by the smaller data size. Though the performance is lower than that for AudioSet pretraining, it still outperforms conventional methods shown in the Table I. 

## REFERENCES 

- [1] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. HerbertVoss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “Language models are few-shot learners,” in _NeurIPS_ , 2020. 

- [2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” _arXiv preprint arXiv:2010.11929_ , 2020. 

- [3] X. Liu, F. Zhang, Z. Hou, Z. Wang, L. Mian, J. Zhang, and J. Tang, “Self-supervised learning: Generative or contrastive,” _arXiv preprint arXiv:2006.08218_ , 2020. 

- [4] P. H. Le-Khac, G. Healy, and A. F. Smeaton, “Contrastive representation learning: A framework and review,” _IEEE Access_ , vol. 8, pp. 193 907– 193 934, 2020. 

- [5] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in _CVPR_ , 2020, pp. 9726–9735. 

- [6] X. Chen, H. Fan, R. Girshick, and K. He, “Improved baselines with momentum contrastive learning,” _arXiv preprint arXiv:2003.04297_ , 2020. 

- [7] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in _ICML_ , 2020, pp. 1597–1607. 

- [8] J.-B. Grill, F. Strub, F. Altch´e, C. Tallec, P. H. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar, B. Piot, K. Kavukcuoglu, R. Munos, and M. Valko, “Bootstrap your own latent - a new approach to self-supervised learning,” in _NeurIPS_ , 2020. [Online]. Available: http://arxiv.org/abs/2006.07733 

- [9] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” _arXiv preprint arXiv:1807.03748_ , 2019. 

- [10] M. Ravanelli, J. Zhong, S. Pascual, P. Swietojanski, J. Monteiro, J. Trmal, and Y. Bengio, “Multi-task self-supervised learning for robust speech recognition,” in _ICASSP_ , 2020, pp. 6989–6993. 

- [11] E. Fonseca, D. Ortego, K. McGuinness, N. E. O’Connor, and X. Serra, “Unsupervised Contrastive Learning of Sound Event Representations,” _arXiv preprint arXiv::2011.07616_ , 2020. 

- [12] A. Jansen, M. Plakal, R. Pandya, D. P. W. Ellis, S. Hershey, J. Liu, R. C. Moore, and R. A. Saurous, “Unsupervised learning of semantic audio representations,” in _ICASSP_ , 2018, pp. 126–130. 

- [13] J. Shor, A. Jansen, R. Maor, O. Lang, O. Tuval, F. de C. Quitry, M. Tagliasacchi, I. Shavitt, D. Emanuel, and Y. Haviv, “Towards learning a universal non-semantic representation of speech,” _arXiv preprint arXiv::2002.12764_ , 2020. 

- [14] A. Saeed, D. Grangier, and N. Zeghidour, “Contrastive learning of general-purpose audio representations,” _arXiv preprint arXiv::2010.10915_ , 2020. 

- [15] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “mixup: Beyond empirical risk minimization,” in _ICLR_ , 2018. [Online]. Available: https://openreview.net/forum?id=r1Ddp1-Rb 

- [16] U. Z¨olzer, _DAFX: Digital Audio Effects_ . John Wiley & Sons, 2011. 

- [17] V. Verma, M.-T. Luong, K. Kawaguchi, H. Pham, and Q. V. Le, “Towards domain-agnostic contrastive learning,” _arXiv preprint arXiv::2011.04419_ , 2020. 

- [18] K. Lee, Y. Zhu, K. Sohn, C.-L. Li, J. Shin, and H. Lee, “i-mix: A strategy for regularizing contrastive representation learning,” _arXiv preprint arXiv:2010.08887_ , 2020. 

- [19] X. Favory, K. Drossos, T. Virtanen, and X. Serra, “Coala: Co-aligned autoencoders for learning semantically enriched audio representations.” 

- [20] J. Cramer, H.-H. Wu, J. Salamon, and J. P. Bello, “Look, listen and learn more: Design choices for deep audio embeddings,” in _ICASSP_ , Brighton, UK, May 2019, pp. 3852––3 856. 

- [21] G. C. Calafiore and L. El Ghaoui, _Optimization Models_ . Cambridge University Press, 2014. 

- [22] J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in _ICASSP_ , 2017, pp. 776–780. 

- [23] E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra, “Fsd50k: an open dataset of human-labeled sound events,” _arXiv preprint arXiv:2010.00475_ , 2020. 

- [24] Y. Koizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “The NTT DCASE2020 challenge task 6 system: Automated audio captioning with keywords and sentence length estimation,” DCASE2020 Challenge, Tech. Rep., 2020. 

- [25] D. Takeuchi, Y. Koizumi, Y. Ohishi, N. Harada, and K. Kashino, “Effects of word-frequency based pre- and post- processings for audio captioning,” in _DCASE2020_ , 2020, pp. 190–194. 

- [26] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _CVPR_ , 2016, pp. 770–778. 

- [27] M. Tan and Q. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in _ICML_ , 2019, pp. 6105–6114. 

- [28] H. Kameoka, T. Kaneko, K. Tanaka, N. Hojo, and S. Seki, “Voicegrad: Non-parallel any-to-many voice conversion with annealed langevin dynamics,” _arXiv preprint arXiv:2010.02977_ , 2020. 

- [29] Z. Zhang, S. Xu, S. Cao, and S. Zhang, “Deep convolutional neural network with mixup for environmental sound classification,” in _PRCV_ , 2018, pp. 356–367. 

- [30] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in _ICML_ , 2015, pp. 448–456. 

- [31] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in _SIGKDD_ , 2019. 

- [32] J. Engel, C. Resnick, A. Roberts, S. Dieleman, M. Norouzi, D. Eck, and K. Simonyan, “Neural audio synthesis of musical notes with WaveNet autoencoders,” in _ICML_ , 2017, pp. 1068–1077. 

- [33] J. Salamon, C. Jacoby, and J. P. Bello, “A dataset and taxonomy for urban sound research,” in _ACM-MM’14_ , Orlando, FL, USA, Nov. 2014, pp. 1041–1044. 

- [34] A. Nagrani, J. S. Chung, and A. Zisserman, “Voxceleb: A large-scale speaker identification dataset,” in _Proc. Interspeech 2017_ , 2017, pp. 2616–2620. 

- [35] K. MacLean, “Voxforge,” 2018. [Online]. Available: http://www. voxforge.org/home 

- [36] P. Warden, “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition,” _arXiv preprint arXiv::1804.03209_ , Apr. 2018.
