<!-- page: 1 -->

1 

# Temporal Pooling Strategies for Training-Free Anomalous Sound Detection with Self-Supervised Audio Embeddings 

Kevin Wilkinghoff, Sarthak Yadav, and Zheng-Hua Tan 

_**Abstract**_ **—Training-free anomalous sound detection (ASD) based on pre-trained audio embedding models has recently garnered significant attention, as it enables the detection of anomalous sounds using only normal reference data while offering improved robustness under domain shifts. However, existing embedding-based approaches almost exclusively rely on temporal mean pooling, while alternative pooling strategies have so far only been explored for spectrogram-based representations. Consequently, the role of temporal pooling in trainingfree ASD with pre-trained embeddings remains insufficiently understood. In this paper, we present a systematic evaluation of temporal pooling strategies across multiple state-of-the-art audio embedding models. We propose relative deviation pooling (RDP), an adaptive pooling method that emphasizes informative temporal deviations, and introduce a hybrid pooling strategy that combines RDP with generalized mean pooling. Experiments on five benchmark datasets demonstrate that the proposed methods consistently outperform mean pooling and achieve state-of-theart performance for training-free ASD, including results that surpass all previously reported trained systems and ensembles on the DCASE2025 ASD dataset.** 

_**Index Terms**_ **—anomalous sound detection, temporal pooling, training-free, self-supervised learning, domain generalization** 

## I. INTRODUCTION 

EMI-supervised anomalous sound detection (ASD) is the **S** task of distinguishing between normal and anomalous recordings of an acoustic phenomenon, while only having access to normal reference samples representing the meaning of the term _normal_ for a particular application. State-of-theart ASD systems typically rely on projecting acoustic signals into a latent embedding space and computing distances to the reference samples, which serve as anomaly scores. 

In challenging acoustic conditions, off-the-shelf pre-trained audio embedding models often underperform discriminative methods that are trained using metadata or labeled auxiliary tasks [1], [2]. The reason for this is that discriminative systems can implicitly suppress irrelevant signal components, such as background noise, by focusing on features that are predictive of the target labels, thereby improving robustness in noisy environments [3]. In contrast, training-free approaches must rely solely on the structure of the acoustic representations themselves, and thus are equally sensitive to both irrelevant signal components as well as possibly subtle target signal components indicating anomalies. 

Recently, however, training-free ASD methods based on large-scale pre-trained audio embedding models have attracted 

The authors are with the Department of Electronic Systems, Aalborg University, Aalborg, Denmark and Pioneer Centre for Artificial Intelligence, Denmark (e- mail: kevin.wilkinghoff@ieee.org, sarthaky@es.aau.dk, zt@es.aau.dk. 

growing interest [4]–[8]. Such approaches offer several advantages: They reduce reliance on domain-specific metadata, generalize better under domain shifts [9], and can be readily applied for pseudo-labeling or bootstrapping discriminative systems when metadata is scarce [10], [11]. These properties make training-free ASD particularly attractive for rapidly deployable and scalable monitoring systems. 

Most pre-trained audio embedding models produce sequences of frame-level embeddings whose length depends on the input duration. Direct comparison of such variablelength sequences is computationally expensive and memoryintensive, as it requires storing and matching full embedding trajectories. To enable efficient similarity computation, temporal pooling is therefore used to aggregate sequences into fixed-dimensional representations that can be compared using simple distance measures such as Euclidean or cosine distance. 

Despite the central role of temporal pooling in embeddingbased training-free ASD, its impact has remained largely unexamined. To the best of our knowledge, existing approaches based on pre-trained audio embeddings uniformly rely on simple temporal mean pooling. While alternative aggregation mechanisms have been explored for spectrogram-based features [12], [13], no prior work has systematically studied temporal pooling in embedding-based training-free ASD. As a result, a fundamental architectural component of these systems has effectively been treated as a fixed design choice. This is particularly critical in anomaly detection, where rare and localized deviations, rather than global averages, often carry the most discriminative information. Since strictly training-free pipelines fix the embedding model and avoid supervised finetuning, temporal pooling represents one of the few remaining design variables that can be modified without introducing supervision, making its systematic investigation essential. 

In this work, we argue that temporal pooling should explicitly account for the distribution of temporal deviations within an embedding sequence. Based on this insight, we propose relative deviation pooling (RDP), a novel training-free pooling strategy that emphasizes informative temporal variations while suppressing irrelevant background components. Furthermore, we introduce a hybrid pooling framework that combines RDP with generalized mean (GeM) pooling [14], leveraging the complementary strengths of both approaches. 

The main contributions of this paper are as follows: 

- We provide the first systematic investigation of temporal pooling as an independent design variable in embeddingbased training-free ASD, isolating its effect across multiple state-of-the-art embeddings and benchmark datasets. 

- We propose RDP and a hybrid RDP+GeM pooling framework that introduce adaptive and non-linear aggregation

<!-- page: 2 -->

2 

mechanisms tailored to training-free ASD. 

- Through extensive experiments on five benchmark datasets, we demonstrate that revisiting temporal pooling alone yields consistent and statistically significant performance gains, achieving state-of-the-art results for training-free ASD and surpassing previously reported trained systems on the DCASE2025 dataset. 

The remaining parts of this article are organized as follows. In Section II, related literature for training-free anomalous sound detection with self-supervised audio embeddings is discussed. In Section III, existing and novel temporal pooling approaches are presented. The effectiveness of the proposed normalization scheme is experimentally evaluated in Section V using the setup from Section IV with several state-of-the-art embeddings on multiple datasets. In addition, a few ablation studies are carried out. The paper is concluded with a summary and possible extensions for future work in Section VI. 

## II. RELATED WORK 

Temporal pooling is a fundamental component in many audio and speech processing tasks, where variable-length sequences must be aggregated into fixed-dimensional representations. In speaker recognition, x-vector systems [15] aggregate frame-level features using simple statistics pooling, such as temporal mean and standard deviation, to obtain utterance-level embeddings that have been shown to be highly discriminative. Similarly, in weakly labeled sound event detection, where only clip-level annotations are available, framelevel representations or predictions are commonly aggregated using simple pooling mechanisms, including mean or max pooling, to infer clip-level decisions [16]. These approaches demonstrate that effective temporal aggregation can compensate for the absence of frame-level labels and highlight the importance of pooling strategies when operating on framelevel representations. 

In the context of ASD, early studies have shown that relatively simple temporal pooling strategies can already achieve strong detection performance. In particular, applying temporal mean or maximum pooling to short-time Fourier transform (STFT)-based feature representations has been demonstrated to be effective for detecting anomalous machine operating sounds, even in the absence of explicit temporal modeling [12]. Subsequent work introduced generalized weighted rank pooling (global weighted ranking pooling (GWRP)) [17] as a temporal aggregation strategy for ASD, showing that it can outperform fixed pooling methods when applied to spectrogram-based features [13]. However, achieving these improvements requires optimizing the decay parameter individually for each machine type, which depends on access to machine-specific labels and implicitly leverages anomalous test data. 

More recently, training-free ASD based on pre-trained audio embedding models has gained increasing attention. In contrast to spectrogram-based systems, existing embedding-based ASD approaches almost exclusively rely on temporal mean pooling to aggregate frame-level embeddings [2], [5], [18], [19]. While effective, this design choice has largely remained 

unquestioned, and alternative pooling strategies have not been systematically explored in this setting. As a result, the impact of temporal pooling on training-free ASD with pre-trained embeddings remains insufficiently understood. 

A related but distinct line of work incorporates learnable pooling mechanisms during fine-tuning. The AnoPatch framework [20], [21], for instance, employs an attentive statistics pooling layer originally proposed for speaker recognition [22]. Other works utilize a weighted mean with trainable weights [23]. While such approaches can improve performance, they rely on supervised or semi-supervised training and therefore fall outside the scope of strictly training-free ASD. In contrast, the present work focuses on temporal pooling strategies that can be directly applied to pre-trained embeddings without any additional training or adaptation. 

## III. TEMPORAL POOLING STRATEGIES 

In this section, we discuss several temporal pooling strategies. Let **X** = _{_ **x** _t}[T] t_ =1 _[⊂]_[R] _[D]_[denote][a][sequence][of] _[T][∈]_[N] frame-level feature vectors extracted from an audio segment, for example using a pre-trained audio embedding model or time–frequency representations such as the STFT. The goal of temporal pooling is to aggregate **X** into a vector-sized representation **x** pooled _∈_ R _[D]_ that can be used to distinguish between normal and anomalous samples. 

## _A. Mean pooling_ 

The most commonly used pooling strategy is to compute the temporal mean of the sequence, i.e., 

**==> picture [143 x 30] intentionally omitted <==**

Mean pooling averages the features over time, producing a representation that primarily reflects the typical, steady-state sound of normal machine operation. This averaging helps to suppress background noise and random fluctuations, but it can also smooth out short or subtle anomalous events. 

## _B. Max pooling_ 

A complementary alternative to mean pooling is temporal max pooling, i.e., 

**==> picture [163 x 68] intentionally omitted <==**

In contrast to mean pooling, max pooling keeps only the strongest response observed over time, which helps short, unusual sounds stand out, but can also make the result sensitive to random noise or brief spikes.

<!-- page: 3 -->

3 

## _C. Global weighted ranking pooling_ 

GWRP [17] provides a smooth transition between mean and max pooling by weighting embeddings according to their strength, so that larger responses contribute more to the pooled representation while information from the entire sequence is retained. Formally, GWRP first rank-orders the values within each feature dimension of **X** in descending order. It is parameterized by a decay parameter _r ∈_ [0 _,_ 1] that controls the selectivity of the weighting and is defined as 

**==> picture [222 x 55] intentionally omitted <==**

where **x** ( _t_ ) _j_ denotes the _t_ -th largest value of dimension _j_ in the sequence **X** . For _r_ = 1, this pooling strategy resembles mean pooling; for _r_ = 0, this corresponds to max pooling. 

## _D. Generalized mean pooling_ 

GeM pooling [14] is an alternative generalization of mean and max pooling defined as 

**==> picture [210 x 31] intentionally omitted <==**

with _p ∈_ R _>_ 0. For _p_ = 1, this pooling strategy resembles mean pooling applied to non-negative entries. The higher the parameter _p_ , the more emphasis is placed on large values, and in the limit _p →∞_ , this converges to max pooling. Note that negative entries are removed to preserve monotonicity for even integer values of _p_ and to avoid obtaining complexvalued embeddings for non-integer _p_ . We also experimented with taking the absolute value instead of setting all negative entries to zero, but this did not improve the performance. 

## _E. Relative deviation pooling_ 

As one of the main contributions of this paper, we propose RDP, a representative pooling method inspired by deviation pooling [24]. RDP assigns higher weights to embeddings that differ most from the typical sound pattern observed over time. Unlike classical deviation pooling, which computes relative deviations to the entire feature sequence, RDP uses these deviations to form a weighted temporal average. By doing so, RDP emphasizes unusual or potentially anomalous frames while preserving the overall temporal context. Such weighted aggregation is conceptually related to attention-based pooling strategies with learnable weights [25], but in contrast, RDP operates in a fully training-free manner. 

As a first step of RDP, sample-wise deviations from the temporal mean are calculated as 

**==> picture [151 x 11] intentionally omitted <==**

and then normalized by 

**==> picture [123 x 24] intentionally omitted <==**

Based on these deviations, weights that indicate how much individual embeddings deviate relatively to the entire sequence are computed as 

**==> picture [159 x 29] intentionally omitted <==**

with _γ ∈_ R _≥_ 0. Using these weights, the pooled representation is a weighted mean given by 

**==> picture [157 x 29] intentionally omitted <==**

For _γ_ = 0, RDP corresponds to mean pooling. The higher the value of _γ_ , the more emphasis is placed onto the embeddings strongly deviating from the mean. 

## _F. Hybrid pooling strategies_ 

Building on the idea of weighting embeddings according to their relative importance, GeM pooling can be extended to a weighted formulation by introducing non-uniform positive weights _wt ∈_ R _>_ 0 [26], 

**==> picture [211 x 34] intentionally omitted <==**

In this work, we exploit this formulation to construct a hybrid pooling strategy by directly using the weights derived from RDP, i.e., by setting _wt_ := _wt_[RDP] ( _γ_ ). This combination allows the selective weighting behavior of RDP to be integrated with the non-linear aggregation characteristics of GeM pooling. 

## IV. EXPERIMENTAL SETUP 

## _A. Datasets_ 

We conduct experiments on five benchmark datasets from the DCASE challenge series. These include the DCASE2020 ASD dataset [27], which is based on the MIMII corpus [28] and ToyADMOS [29]; the DCASE2022 ASD dataset [30], built from MIMII-DG [31] and ToyADMOS2 [32]; the DCASE2023 ASD dataset [30], extending MIMII-DG with ToyADMOS2+ [33]; the DCASE2024 ASD dataset [34], which combines MIMII-DG, ToyADMOS2# [35], and additional recordings collected using the IMAD-DS setup [36]; and the DCASE2025 ASD dataset [37], consisting of MIMII-DG, ToyADMOS2025 [38], and further samples recorded under the same IMAD-DS conditions [36]. 

All datasets address semi-supervised acoustic anomaly detection for machine condition monitoring in realistic and noisy environments and include multiple machine types. Each dataset follows the official DCASE protocol and is divided into a development split and a separate evaluation split. In both splits, only recordings of normal operation are provided as reference data, while the corresponding test recordings contain both normal and anomalous samples. Except for DCASE2020, which contains data from a single domain, the datasets are designed to study domain generalization. To this end, the reference data consist of 990 samples from a source domain and 10 samples from a target domain. In the test data, machine

<!-- page: 4 -->

4 

types are known and the domain distributions are balanced, but explicit domain labels are not provided. No model parameters are learned from the reference data; they are solely used for distance-based anomaly scoring. 

The goal of the ASD system is to assign a continuous anomaly score to each test recording, where higher scores indicate a higher likelihood of abnormal behavior. All experiments follow the official evaluation protocols of the respective datasets. For DCASE2020, performance is measured using the arithmetic mean of the area under the ROC curve (AUC) and the partial AUC (pAUC) [39] with _p_ = 0 _._ 1. For the remaining datasets, we report the metrics specified in their evaluation guidelines, namely the harmonic mean of the domain-specific AUCs and the domain-agnostic pAUC. Further details about the datasets are provided in the cited references. 

## _B. Audio Embedding Models_ 

A wide range of self-supervised audio embedding models has been proposed in recent years for various downstream audio tasks [40], [41]. In this work, we focus on four embedding models that are widely used and have demonstrated strong performance in ASD, namely OpenL3 [42], BEATs [43], efficient audio transformer (EAT) [44], and Dasheng [45]. These models represent different design philosophies and temporal resolutions, making them well suited for a comprehensive evaluation of temporal pooling strategies. 

Below, we describe the specific configurations and implementation details of the embedding models used in the experimental evaluation. 

**OpenL3** We employ OpenL3 embeddings [42], which are based on the Look, Listen and Learn framework [46], [47]. Input waveforms are segmented using a sliding window of 1 s duration with a hop size of 0 _._ 1 s, resulting in a temporal sequence of frame-level embeddings. Performing temporal segmentation prior to pooling has been shown to yield improved performance compared to using a single clip-level embedding [1]. For each segment, a 128-bin mel spectrogram is computed and passed through the OpenL3 model pre-trained on the environmental sound subset, producing embeddings of dimensionality 512. 

**BEATs:** BEATs [43] is one of the strongest and most widely adopted embedding models for ASD [2], [48], achieving performance comparable to or exceeding that of ASD-specific foundation models such as ECHO [7] and FISHER [8]. In this work, we use the official BEATs model pre-trained for three iterations on AudioSet [49], without any additional fine-tuning. All experiments are conducted using this frozen pre-trained model. 

**EAT:** For EAT [44], we use the official large-model checkpoint pre-trained for 20 epochs on AudioSet [49]. We found that appropriate pre-processing of the extracted embeddings is crucial for obtaining competitive performance with EAT. Specifically, hard thresholding of low-valued components (threshold set to 0.1) combined with suppression of large activation spikes using a hyperbolic tangent nonlinearity (applied to values above 0.5) proved essential. These hyperparameters were optimized based on the performance on the 

TABLE I 

AVERAGE PERFORMANCE OBTAINED WITH EAT ACROSS THE DEVELOPMENT AND EVALUATION SETS OF THE DCASE2020, DCASE2022, DCASE2023, DCASE2024, AND DCASE2025 ASD DATASETS. ∆ DENOTES IMPROVEMENT OVER BASELINE. CIS ARE 95% PAIRED-BOOTSTRAP INTERVALS. BEST PERFORMANCES IN EACH COLUMN ARE IN BOLD. ALL RESULTS ARE DETERMINISTIC. 

|min clamp<br>spike supp.|**Mean Pooling**<br>average<br>∆(CI)|**Max Pooling**<br>average<br>∆(CI)|
|---|---|---|
|✓<br>✓<br>✓<br>✓|62.87%<br>baseline<br>64.35%<br>+1.48 [1.22, 3.33]<br>65.58%<br>+2.71 [2.35, 4.80]<br>**65.63%**<br>+2.76 [2.19, 4.88]|63.33%<br>baseline<br>64.17%<br>+0.83 [0.41, 1.35]<br>63.35%<br>+0.02 [0.01, 0.04]<br>**64.30%**<br>+0.97 [0.52, 1.48]|



development sets. The impact of this pre-processing step is quantitatively analyzed in Table I and constitutes an important practical insight for the use of EAT embeddings in ASD. Note that the proposed pre-processing has no significant effect on openL3, BEATs, or Dasheng. This indicates that EAT embeddings exhibit higher dynamic range and less calibrated activation statistics, likely due to their contrastive training objective, which does not explicitly constrain embedding scale or tail behavior. The pre-processing effectively regularizes the embedding distribution, suppressing spurious low-magnitude noise and extreme activation spikes that would otherwise dominate distance-based anomaly scores. In contrast, the other embeddings are already well-normalized by design or training objective, rendering the same pre-processing largely redundant. 

**Dasheng:** Dasheng [45] is a recently proposed audio foundation model designed for general-purpose audio representation learning. In our experiments, we use the official Dasheng base model without additional fine-tuning. 

In addition to these models, we also evaluated several other self-supervised audio representations, including data2vec 2.0 [50], WavLM [51], and Self-Supervised Audio Mamba (SSAM) [52]. Preliminary experiments indicated that applying more sophisticated pooling strategies also led to performance improvements over temporal mean pooling for these embeddings. However, we refrain from including a detailed analysis in this work, as their overall performance as off-theshelf representations for ASD remained substantially weaker and well below the state of the art. A likely explanation is that these representations emphasize very short-term details, which may limit their ability to capture the longer-term acoustic context required to distinguish context-dependent anomalies from normal machine sounds. 

## _C. Anomaly Score Calculation_ 

Let Pool : R _[T][ ×][D] →_ R _[D]_ denote a temporal pooling operator. Further, let _X_ test _⊂_ R _[T][ ×][D]_ denote the set of test samples and _X_ ref _⊂_ R _[T][ ×][D]_ denote a reference set of normal training samples. Then, anomaly scores are computed as the Euclidean distance between the temporally pooled embeddings of a test sample **X** _∈X_ test and its closest normal reference sample 

**==> picture [243 x 35] intentionally omitted <==**

<!-- page: 5 -->

5 

TABLE II 

AVERAGE PERFORMANCE OBTAINED WITH DIFFERENT POOLING STRATEGIES AND EMBEDDING MODELS ACROSS THE DEVELOPMENT AND EVALUATION SETS OF THE DCASE2020, DCASE2022, DCASE2023, DCASE2024, AND DCASE2025 ASD DATASETS. ∆ DENOTES IMPROVEMENT OVER BASELINE. CIS ARE 95% PAIRED-BOOTSTRAP INTERVALS. HIGHEST NUMBERS IN EACH COLUMN ARE IN BOLD. ALL RESULTS ARE BASED ON THE OPTIMAL HYPERPARAMETERS DETERMINED ON THE DEVELOPMENT SETS AND ARE DETERMINISTIC. 

|**Pooling**|||**Embedding Model**|**Embedding Model**||
|---|---|---|---|---|---|
|||OpenL3<br>average<br>∆(CI)|BEATs<br>average<br>∆(CI)|EAT<br>average<br>∆(CI)|Dasheng<br>average<br>∆(CI)|
|mean<br>max<br>GWRP<br>GeM<br>RDP<br>RDP + GeM||64.65%<br>baseline<br>64.29%<br>-0.37 [-1.52, 0.76]<br>65.34%<br>+0.69 [0.04, 1,37]<br>**65.40%**<br>+0.75 [0.32, 1.21]<br>64.80%<br>+0.15 [-0.62, 0.84]<br>65.26%<br>+0.61 [0.11, 1.10]|67.01%<br>baseline<br>68.12%<br>+1.11 [0.24, 1.90]<br>68.38%<br>+1.38 [0.67, 2.06]<br>68.18%<br>+1.18 [0.56, 1.82]<br>**68.72%**<br>+1.71 [0.79, 2.70]<br>68.71%<br>+1.71 [0.87, 2.60]|65.63%<br>baseline<br>64.30%<br>-1.33 [-2.37, -0.28]<br>65.61%<br>-0.01 [-0.59, 0.50]<br>**65.69%**<br>+0.07 [-0.11, 0.27]<br>65.62%<br>-0.00 [-0.06, 0.05]<br>**65.69%**<br>+0.06 [-0.14, 0.30]|63.10%<br>baseline<br>62.87%<br>-0.23 [-1.10, 0.67]<br>63.54%<br>+0.44 [-0.34, 1.16]<br>63.72%<br>+0.62 [0.02, 1.14]<br>**64.64%**<br>+1.53 [0.85, 2.27]<br>64.59%<br>+1.49 [0.67, 2.34]|



## TABLE III 

BEST PERFORMING PARAMETER SETTINGS ACROSS THE DEVELOPMENT SETS OF THE DCASE2020, DCASE2022, DCASE2023, DCASE2024, AND DCASE2025 ASD DATASETS. 

|**Pooling**|**Embedding Model**<br>OpenL3<br>BEATs<br>EAT<br>Dasheng<br>All|
|---|---|
|mean<br>max<br>GWRP<br>GeM<br>RDP<br>RDP + GeM (p=3)|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_r_ = 0_._9<br>_r_ = 0_._4<br>_r_ = 0_._9<br>_r_ = 0_._6<br>_r_ = 0_._7<br>_p_= 9<br>_p_= 10<br>_p_= 3<br>_p_= 6<br>_p_= 6<br>_α_= 10<br>_α_= 19<br>_α_= 1<br>_α_= 20<br>_α_= 10<br>_α_= 8<br>_α_= 16<br>_α_= 1<br>_α_= 20<br>_α_= 9|



Following best practices, all available training samples are used as reference samples. 

To reduce performance degradations caused by domain shifts, we applied local density-based anomaly score normalization [19], [53] with _K_ = 1 and variance-minimization [54] in log-space. Formally, this corresponds to calculating anomaly scores of the form 

**==> picture [259 x 33] intentionally omitted <==**

where **Y** 1 = **Y** denotes the nearest neighbor in _X_ ref to **Y** , and 

**==> picture [233 x 18] intentionally omitted <==**

This particular scoring backend does not require any labels and does not make any assumptions about the data, which are important properties of a training-free ASD approach. Moreover, since the normalization constants depend only on the reference samples, they can be pre-computed without introducing any additional computational overhead during inference. 

## V. RESULTS AND DISCUSSION 

## _A. Comparison of Pooling Strategies_ 

As an initial experiment, we compare the temporal pooling strategies introduced in Section III across different audio embedding models (see Section IV-B). The results are reported in Table II. For all experiments, hyperparameters were fixed to the values specified in Table III, which were selected by maximizing performance on the development splits. Further details on the hyperparameter selection procedure are provided 

in Section V-B. Since temporal mean pooling constitutes the de facto standard aggregation mechanism in embedding-based training-free ASD, it serves as the primary baseline throughout our analysis. 

The results indicate that both the optimal choice of a pooling strategy and its effectiveness are strongly dependent on the underlying embedding model. A consistent trend emerges when comparing simple pooling baselines to more advanced methods: When the performance of maximum pooling is comparable to or exceeds mean pooling, advanced pooling strategies tend to yield substantial gains. Conversely, for embeddings such as EAT, where maximum pooling performs considerably worse than mean pooling, none of the advanced pooling approaches provides statistically significant improvements. A plausible explanation for the limited gains observed with advanced temporal pooling for EAT lies in the preprocessing applied to the embeddings. After normalization, EAT embeddings exhibit reduced temporal variance and a more uniform distribution of anomaly-relevant information across time. Consequently, mean pooling is already nearoptimal, and more sophisticated pooling strategies, which primarily exploit temporal sparsity or extreme activations, offer little additional benefit. In contrast, embeddings such as OpenL3, BEATs, and Dasheng retain higher temporal heterogeneity even after pre-processing, allowing advanced pooling methods to significantly improve performance. 

The effectiveness of individual pooling strategies varies across embeddings. RDP is particularly effective for BEATs and Dasheng embeddings, where it yields the largest performance improvements among all evaluated methods. In contrast, RDP is less effective for OpenL3, for which GeM pooling achieves the highest performance. Similarly, for EAT embeddings, GeM pooling consistently outperforms the alternative strategies. Although GWRP has previously been explored for spectrogram-based representations, its effectiveness in embedding-based training-free ASD has not been systematically assessed. In this setting, it provides moderate improvements but generally falls short of GeM pooling and RDP. Across all evaluated embeddings, the proposed hybrid RDP+GeM strategy achieves performance comparable to the best embedding-specific method, indicating that it offers a robust and reliable choice when an embedding-agnostic pooling strategy is required or no prior knowledge is available.

<!-- page: 6 -->

6 

**==> picture [510 x 175] intentionally omitted <==**

**----- Start of picture text -----**<br>
OpenL3 BEATs EAT Dasheng average<br>GWRP – dev. set RDP – dev. set GeM pooling – dev. set RDP + GeM pooling ( p  = 3) – dev. set<br>2 2 2 2<br>0 0 0 0<br>− 2 − 2 − 2 − 2<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 2 4 6 8 10 12 14 16 18 20 2 4 6 8 10 2 4 6 8 10 12 14 16 18 20<br>r γ p γ<br>GWRP – eval. set RDP – eval. set GeM pooling – eval. set RDP + GeM pooling ( p  = 3) – eval. set<br>2 2 2 2<br>0 0 0 0<br>− 2 − 2 − 2 − 2<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 2 4 6 8 10 12 14 16 18 20 2 4 6 8 10 2 4 6 8 10 12 14 16 18 20<br>r γ p γ<br>change<br>performance<br>change<br>performance<br>**----- End of picture text -----**<br>


Fig. 1. Sensitivity analysis of the performance with respect to the hyperparameters of different pooling strategies. Performance ratios compared to mean pooling are depicted. All values are geometric means across the development and evaluation sets of the DCASE2020, DCASE2022, DCASE2023, DCASE2024, and DCASE2025 datasets. 

## TABLE IV 

AVERAGE PERFORMANCE OBTAINED WITH DIFFERENT POOLING STRATEGIES FOR THE EMBEDDING MODELS OPENL3, BEATS, EAT, AND DASHENG. RESULTS ARE AVERAGES ACROSS THE DEVELOPMENT AND EVALUATION SETS OF THE DCASE2020, DCASE2022, DCASE2023, DCASE2024, AND DCASE2025 ASD DATASETS. ∆ DENOTES IMPROVEMENT OVER BASELINE. CIS ARE 95% PAIRED-BOOTSTRAP INTERVALS. HIGHEST NUMBERS IN EACH COLUMN ARE IN BOLD. ALL RESULTS ARE BASED ON THE OPTIMAL HYPERPARAMETERS DETERMINED ON THE DEVELOPMENT SETS AND ARE DETERMINISTIC. 

|**Pooling**|**Hyperparameter Settings**<br>embedding-agnostic<br>embedding-specifc<br>average<br>∆(CI)<br>average<br>∆(CI)|**Hyperparameter Settings**<br>embedding-agnostic<br>embedding-specifc<br>average<br>∆(CI)<br>average<br>∆(CI)|
|---|---|---|
||embedding-agnostic<br>average<br>∆(CI)||
|mean<br>max<br>GWRP<br>GeM<br>RDP<br>RDP + GeM|65.10%<br>baseline<br>64.89%<br>-0.21 [-0.77, 0.36]<br>65.48%<br>+0.38 [-0.03, 0.79]<br>65.73%<br>+0.63 [0.35, 0.92]<br>65.50%<br>+0.40 [0.03, 0.75]<br>**65.76%**<br>+0.66 [0.24, 1.07]|65.10%<br>baseline<br>64.89%<br>-0.21 [-0.77, 0.36]<br>65.72%<br>+0.62 [0.26, 0.99]<br>65.75%<br>+0.65 [0.38, 0.93]<br>65.94%<br>+0.84 [0.43, 1.28]<br>**66.06%**<br>+0.96 [0.59, 1.37]|



pooling even without embedding-specific tuning, indicating that the benefits of advanced temporal aggregation are not merely a consequence of hyperparameter adaptation. Among these methods, GeM exhibits particularly stable performance across both regimes, whereas RDP achieves substantially larger improvements when embedding-specific hyperparameters are employed, suggesting that it benefits from adaptation to the characteristics of individual embedding models. Across all evaluated strategies, the hybrid RDP+GeM pooling approach achieves the best overall performance, with its gains similarly amplified under embedding-specific settings, indicating that its effectiveness is primarily driven by the adaptive weighting mechanism introduced by RDP. 

## _B. Hyperparameter Sensitivity Analysis_ 

Importantly, all observed improvements are achieved without modifying the embedding model, anomaly scoring formulation, or evaluation protocol. This suggests that embeddingbased training-free ASD systems are constrained not only by representation quality but also by suboptimal temporal aggregation. In other words, temporal pooling represents a previously under-examined design choice with measurable impact in the ASD pipeline. Revisiting this single architectural component yields gains comparable in magnitude to differences observed between commonly used embedding models. 

To assess whether the observed improvements depend on embedding-specific tuning, we compare embedding-agnostic and embedding-specific hyperparameter settings, as summarized in Table IV. Overall, max pooling performs slightly worse than temporal mean pooling, although the difference is not statistically significant, while GWRP provides moderate improvements that lose statistical significance under embedding-agnostic settings. In contrast, both GeM and RDP yield consistent and statistically significant gains over mean 

Next, we analyze the performance sensitivity with respect to the pooling hyperparameters and justify the choice of the embedding-specific parameter values. The corresponding results are shown in Figure 1. Overall, with the exception of GeM pooling, most pooling strategies exhibit a pronounced sensitivity to the chosen hyperparameters. Consistent with previous observations, the resulting performance improvements are strongly embedding-dependent but only weakly datasetdependent. This behavior is particularly evident for GWRP, which exhibits a sharp performance peak around _r_ = 0 _._ 9 for OpenL3 embeddings on the development set and for several embedding models on the evaluation set, while such a peak is absent for others, most notably BEATs. At the same time, for the other pooling approaches the overall trends and relative ordering of performance remain consistent between development and evaluation sets, indicating that the observed improvements are not driven by dataset-specific overfitting. These findings indicate that embedding-specific tuning of pooling hyperparameters is highly beneficial, whereas datasetspecific tuning is largely unnecessary, making the proposed pooling strategies practical for real-world deployment.

<!-- page: 7 -->

7 

TABLE V 

OFFICIAL PERFORMANCE METRICS AND DATASET-WISE HARMONIC MEANS FOR DIFFERENT ASD SYSTEMS. ALL PROPOSED SYSTEMS EMPLOY THE HYBRID RDP+GEM POOLING STRATEGY WITH EMBEDDING-SPECIFIC, DATASET-INDEPENDENT PARAMETER SETTINGS. ALL VALUES ARE REPORTED IN PERCENTAGES. THE BEST RESULT IN EACH COLUMN IS SHOWN IN BOLD, WHILE THE BEST TRAINING-FREE RESULT IS UNDERLINED. 

|ASD system<br>training-free|DCASE2020 [27]<br>dev.<br>eval.<br>mean|DCASE2022 [30]<br>dev.<br>eval.<br>mean|DCASE2023 [55]<br>dev.<br>eval.<br>mean|DCASE2024 [34]<br>dev.<br>eval.<br>mean|DCASE2025 [37]<br>dev.<br>eval.<br>mean|
|---|---|---|---|---|---|
|openL3<br>✓<br>BEATs<br>✓<br>EAT<br>✓<br>Dasheng<br>✓|80_._5<br>81_._8<br>81_._2<br>86_._8<br>87_._4<br>87_._1<br>79_._1<br>82_._2<br>80_._6<br>80_._9<br>81_._6<br>81_._2|61_._9<br>64_._0<br>62_._9<br>66_._1<br>65_._1<br>65_._6<br>63_._2<br>62_._5<br>62_._8<br>60_._5<br>61_._8<br>61_._1|60_._3<br>66_._2<br>63_._1<br>65_._7<br>70_._8<br>68_._2<br>62_._9<br>65_._6<br>64_._2<br>62_._4<br>64_._0<br>63_._2|57_._9<br>60_._8<br>59_._3<br>58_._7<br>60_._1<br>59_._4<br>58_._9<br>60_._9<br>59_._9<br>57_._1<br>58_._5<br>57_._8|60_._6<br>58_._6<br>59_._6<br>64_._0<br>**62****_._4**<br>**63****_._2**<br>63_._0<br>58_._5<br>60_._7<br>61_._3<br>57_._9<br>59_._6|
|Wilkinghoff (STFT) [12]<br>✓<br>Saengthong et al. (BEATs) [5]<br>✓<br>Fujimura et al. (BEATs) [2]<br>✓<br>Wilkinghoff et al. (BEATs) [19]<br>✓<br>Fan et al. (FISHER) [8]<br>✓<br>Zhang et al. (ECHO) [7]<br>✓|73_._6<br>75_._4<br>74_._5<br>74_._7a<br>_−_<br>_−_<br>76_._9<br>77_._5<br>77_._2<br>81_._5<br>82_._2<br>81_._8<br>_−_<br>_−_<br>71_._0<br>_−_<br>_−_<br>72_._2|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>61_._5<br>60_._3<br>60_._9<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>59_._6<br>_−_<br>_−_<br>60_._0|_−_<br>_−_<br>_−_<br>_−_<br>73_._8<br>a<br>_−_<br>62_._3<br>62_._6<br>62_._4<br>64_._8<br>67_._6<br>66_._2<br>_−_<br>_−_<br>62_._6<br>_−_<br>_−_<br>63_._7|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>57_._7<br>55_._7<br>56_._7<br>58_._1<br>62_._4<br>60_._2<br>_−_<br>_−_<br>55_._6<br>_−_<br>_−_<br>57_._9|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>58_._7|
|Koizumi et al. [27]<br>Wilkinghoff [12]<br>Liu et al. [56]<br>Harada et al. [57]<br>Wilkinghoff [58]<br>Hou et al. [59]<br>Wilkinghoff [60]<br>Han et al. [23]<br>Zhang et al. [61]<br>Jiang et al. [20], [21]<br>Wilkinghoff [62]<br>Yin et al. [63]<br>Fujimura et al. [10]<br>Yin et al. [64]<br>Jiang et al. [21]<br>Fujimura et al. [2]<br>Matsumoto et al. [54]<br>Wilkinghoff et al. [19]<br>Fujimura et al. [11]<br>Han et al. [48]|66_._6<br>70_._0<br>68_._3<br>90_._7<br>92_._8<br>91_._7<br>89_._4<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>88_._8<br>92_._0<br>90_._4<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>90_._9<br>**94****_._3**<br>92_._6<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>90_._4<br>93_._5<br>91_._9<br>_−_<br>_−_<br>_−_<br>**94****_._2**<br>93_._3<br>**93****_._7**<br>_−_<br>_−_<br>_−_<br>91_._1<br>93_._1<br>92_._1|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>59_._0<br>_−_<br>62_._8<br>63_._0<br>62_._9<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>73_._1<br>67_._1<br>70_._0<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>**73****_._9**<br>69_._9<br>**71****_._8**<br>71_._0<br>68_._8<br>69_._9<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>72_._4<br>66_._6<br>69_._4|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>61_._1<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>64_._2<br>66_._6<br>65_._4<br>64_._3<br>_−_<br>_−_<br>_−_<br>71_._3<br>_−_<br>64_._2<br>**74****_._2**<br>68_._8<br>62_._9<br>64_._5<br>63_._7<br>68_._1<br>_−_<br>_−_<br>67_._2<br>68_._8<br>68_._0<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>64_._0<br>72_._0<br>67_._8<br>66_._5<br>69_._0<br>67_._7<br>**71****_._3**<br>72_._4<br>**71****_._8**<br>_−_<br>_−_<br>_−_<br>64_._3<br>68_._7<br>66_._4|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>56_._5<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>62_._5<br>65_._6<br>64_._0<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>67_._6<br>62_._0<br>64_._7<br>_−_<br>**67****_._1**<br>_−_<br>64_._1<br>66_._0<br>65_._0<br>59_._9<br>61_._5<br>60_._7<br>62_._7<br>57_._1<br>59_._8<br>65_._2<br>56_._5<br>60_._5<br>_−_<br>_−_<br>_−_<br>64_._1<br>65_._5<br>64_._8|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>56_._5<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>**64****_._9**<br>60_._0<br>62_._4<br>_−_<br>_−_<br>_−_|
|DCASE2020 Challenge winner [65]<br>DCASE2022 Challenge winner [66]<br>DCASE2023 Challenge winner [67]<br>DCASE2024 Challenge winner [68]<br>DCASE2025 Challenge winner [69]|_−_<br>89_._8<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>**71****_._0**<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>67_._0<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>**67****_._8**<br>66_._2<br>**67****_._0**<br>_−_<br>_−_<br>_−_|_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>_−_<br>60_._9<br>61_._6<br>61_._2|



> a Obtained by using a domain-wise standardization of the test scores, which requires domain labels and destroys independence between test samples. 

## _C. Comparison to the State-of-the-Art_ 

Having established that improved pooling yields consistent gains across embeddings and hyperparameter settings, we finally evaluate whether these improvements translate into competitive performance against previously reported state-ofthe-art ASD systems, including both training-free approaches and methods requiring supervised or semi-supervised training. For completeness, we also include the winning systems of the corresponding DCASE challenges for each dataset. The comparative results are summarized in Table V. Across the evaluated datasets, the choice of embedding model strongly influences overall performance. In particular, BEATs yields substantially higher scores than the other embedding models on most datasets. On the DCASE2024 dataset, however, BEATs performs comparably to OpenL3 and slightly below EAT. This observation underscores the importance of representation quality for training-free ASD. 

On average, the proposed approach outperforms existing training-free methods across the majority of evaluated datasets, despite not relying on domain labels, machine-specific tuning, or additional constraints on the evaluation protocol. Although a performance gap between training-free and fully trained approaches generally remains, this gap is substantially reduced by the combination of the proposed pooling strategy and the chosen normalization approach. In several cases, 

the proposed method matches or exceeds previously reported trained systems, including challenge-winning approaches on the DCASE2023 and DCASE2025 datasets. The remaining performance difference on the DCASE2023 evaluation set is largely attributable to domain-wise score standardization employed in [5], which assumes access to domain labels and introduces dependencies between test samples that are not permitted under the strictly training-free evaluation protocol adopted here. Notably, on the DCASE2025 dataset, the proposed approach achieves a new state-of-the-art result, outperforming all previously reported systems, including trained methods and ensemble-based approaches. These findings suggest that the commonly assumed performance gap between training-free and trained ASD systems is to a considerable extent a consequence of suboptimal temporal aggregation rather than an inherent limitation of trainingfree approaches. Importantly, all reported results are obtained without supervised training, test-set adaptation, or domainwise standardization. 

## VI. CONCLUSION 

This paper revisited temporal pooling in embedding-based training-free ASD, a component that has largely remained fixed to simple mean aggregation. Through a systematic evaluation across four state-of-the-art embedding models and

<!-- page: 8 -->

8 

five benchmark datasets, we showed that temporal pooling constitutes a significant limiting factor in current embeddingbased pipelines. By introducing RDP and a hybrid RDP+GeM strategy, we achieved consistent improvements over mean pooling without modifying the embedding models or anomaly scoring backend. The proposed methods attained state-of-theart performance for training-free ASD and, on DCASE2025, surpassed previously reported trained systems and ensembles. 

These findings demonstrate that temporal pooling is a decisive design choice in training-free ASD pipelines. Revisiting this previously overlooked component yields gains that are comparable in magnitude to those observed when switching between different embedding models, highlighting the importance of carefully analyzing aggregation mechanisms in embedding-based anomaly detection. 

Future work will investigate integrating the proposed pooling strategies into fine-tuning frameworks, as well as comparing them with learnable attention-based aggregation methods. Further directions include extending the approach to more complex acoustic scenarios and studying its interaction with alternative distance metrics and normalization schemes. Beyond ASD, deviation-aware pooling may also be relevant for other distance-based embedding comparison tasks, such as nearest-neighbor retrieval or prototype-based recognition. More generally, our findings highlight that temporal aggregation can materially influence performance in embedding-based pipelines and therefore merits careful consideration. 

## VII. GENERATIVE AI DISCLOSURE 

Generative AI tools were used for language editing and polishing of the manuscript. All scientific content, interpretations, and conclusions are the responsibility of the authors. 

## REFERENCES 

- [1] K. Wilkinghoff and F. Fritz, “On using pre-trained embeddings for detecting anomalous sounds with limited training data,” in _Proc. EUSIPCO_ , 2023. 

- [2] T. Fujimura, K. Wilkinghoff, K. Imoto, and T. Toda, “ASDKit: A toolkit for comprehensive evaluation of anomalous sound detection methods,” in _Proc. DCASE_ , 2025. 

- [3] K. Wilkinghoff and F. Kurth, “Why do angular margin losses work well for semi-supervised anomalous sound detection?” _IEEE/ACM Trans. Audio, Speech, Lang. Process._ , vol. 32, 2024. 

- [4] A. I. Mezza, G. Zanetti, M. Cobos, and F. Antonacci, “Zero-shot anomalous sound detection in domestic environments using large-scale pretrained audio pattern recognition models,” in _Proc. ICASSP_ , 2023. 

- [5] P. Saengthong and T. Shinozaki, “Deep generic representations for domain-generalized anomalous sound detection,” in _Proc. ICASSP_ , 2025. 

- [6] H.-H. Wu, W.-C. Lin, A. Kumar, L. Bondi, S. Ghaffarzadegan, and J. P. Bello, “Towards few-shot training-free anomaly sound detection,” in _Proc. Interspeech_ , 2025. 

- [7] Y. Zhang, J. Liu, and M. Li, “ECHO: Frequency-aware hierarchical encoding for variable-length signal,” _arXiv preprint arXiv:2508.14689_ , 2025. 

- [8] P. Fan _et al._ , “FISHER: A foundation model for multi-modal industrial signal comprehensive representation,” _arXiv preprint arXiv:2507.16696_ , 2025. 

- [9] K. Wilkinghoff, T. Fujimura, K. Imoto, J. Le Roux, Z.-H. Tan, and T. Toda, “Handling domain shifts for anomalous sound detection: A review of DCASE-related work,” in _Proc. DCASE_ , 2025. 

- [10] T. Fujimura, I. Kuroyanagi, and T. Toda, “Improvements of discriminative feature space training for anomalous sound detection in unlabeled conditions,” in _Proc. ICASSP_ , 2025. 

- [11] ——, “Discriminative anomalous sound detection using pseudo labels, target signal enhancement, and ensemble feature extractors,” in _Proc. DCASE_ , 2025. 

- [12] K. Wilkinghoff, “Sub-cluster AdaCos: Learning representations for anomalous sound detection,” in _Proc. IJCNN_ , 2021. 

- [13] J. Guan, Y. Liu, Q. Zhu, T. Zheng, J. Han, and W. Wang, “Timeweighted frequency domain audio representation with GMM estimator for anomalous sound detection,” in _Proc. ICASSP_ , 2023. 

- [14] F. Radenovic, G. Tolias, and O. Chum, “Fine-tuning CNN image retrieval with no human annotation,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 41, 2019. 

- [15] D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudanpur, “X-vectors: Robust DNN embeddings for speaker recognition,” in _Proc. ICASSP_ , 2018. 

- [16] Q. Kong, Y. Xu, W. Wang, and M. D. Plumbley, “Sound event detection of weakly labelled data with CNN-transformer and automatic threshold optimization,” _IEEE/ACM Trans. Audio, Speech, Lang. Process._ , vol. 28, pp. 2450–2460, 2020. 

- [17] A. Kolesnikov and C. H. Lampert, “Seed, expand and constrain: Three principles for weakly-supervised image segmentation,” in _Proc. ECCV_ , 2016. 

- [18] P. Saengthong, T. Nishida, K. Dohi, N. Yamashita, and Y. Kawaguchi, “Retaining mixture representations for domain generalized anomalous sound detection,” _arXiv:2510.25182_ , 2025. 

- [19] K. Wilkinghoff, H. Yang, J. Ebbers, F. G. Germain, G. Wichern, and J. L. Roux, “Local density-based anomaly score normalization for domain generalization,” _IEEE Trans. Audio, Speech, Lang. Process._ , vol. 33, 2025. 

- [20] A. Jiang _et al._ , “AnoPatch: Towards better consistency in machine anomalous sound detection,” in _Proc. Interspeech_ , 2024. 

- [21] ——, “Adaptive prototype learning for anomalous sound detection with partially known attributes,” in _Proc. ICASSP_ , 2025. 

- [22] N. Dawalatabad, M. Ravanelli, F. Grondin, J. Thienpondt, B. Desplanques, and H. Na, “ECAPA-TDNN embeddings for speaker diarization,” in _Proc. Interspeech_ , 2021. 

- [23] B. Han _et al._ , “Exploring large scale pre-trained models for robust machine anomalous sound detection,” in _Proc. ICASSP_ , 2024. 

- [24] H. Z. Nafchi, R. Hedjam, A. Shahkolaei, and M. Cheriet, “Deviation based pooling strategies for full reference image quality assessment,” _arXiv:1504.06786_ , 2015. 

- [25] K. Okabe, T. Koshinaka, and K. Shinoda, “Attentive statistics pooling for deep speaker embedding,” in _Proc. Interspeech_ , 2018. 

- [26] X. Wang, Y. Hua, E. Kodirov, and N. M. Robertson, “Ranked list loss for deep metric learning,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 44, no. 9, 2022. 

- [27] Y. Koizumi _et al._ , “Description and discussion on DCASE2020 Challenge Task2: Unsupervised anomalous sound detection for machine condition monitoring,” in _Proc. DCASE_ , 2020. 

- [28] H. Purohit _et al._ , “MIMII dataset: Sound dataset for malfunctioning industrial machine investigation and inspection,” in _Proc. DCASE_ , 2019. 

- [29] Y. Koizumi, S. Saito, H. Uematsu, N. Harada, and K. Imoto, “ToyADMOS: A dataset of miniature-machine operating sounds for anomalous sound detection,” in _Proc. WASPAA_ , 2019. 

- [30] K. Dohi _et al._ , “Description and discussion on DCASE 2022 Challenge Task 2: Unsupervised anomalous sound detection for machine condition monitoring applying domain generalization techniques,” in _Proc. DCASE_ , 2022. 

- [31] ——, “MIMII DG: Sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task,” in _Proc. DCASE_ , 2022. 

- [32] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, M. Yasuda, and S. Saito, “ToyADMOS2: Another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions,” in _Proc. DCASE_ , 2021. 

- [33] N. Harada, D. Niizumi, D. Takeuchi, Y. Ohishi, and M. Yasuda, “ToyADMOS2+: New Toyadmos data and benchmark results of the firstshot anomalous sound event detection baseline,” in _Proc. DCASE_ , 2023. 

- [34] T. Nishida _et al._ , “Description and discussion on DCASE 2024 Challenge Task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring,” in _Proc. DCASE_ , 2024. 

- [35] D. Niizumi, N. Harada, Y. Ohishi, D. Takeuchi, and M. Yasuda, “ToyADMOS2#: Yet another dataset for the DCASE2024 challenge task 2 first-shot anomalous sound detection,” in _Proc. DCASE_ , 2024. 

- [36] D. Albertini, F. Augusti, K. Esmer, A. Bernardini, and R. Sannino, “IMAD-DS: A dataset for industrial multi-sensor anomaly detection under domain shift conditions,” in _Proc. DCASE_ , 2024.

<!-- page: 9 -->

9 

- [37] T. Nishida _et al._ , “Description and discussion on DCASE 2025 challenge task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring,” in _Proc. DCASE_ , 2025. 

- [38] N. Harada, D. Niizumi, Y. Ohishi, D. Takeuchi, and M. Yasuda, “ToyADMOS2025: The evaluation dataset for the DCASE2025T2 firstshot unsupervised anomalous sound detection for machine condition monitoring,” in _Proc. DCASE_ , 2025. 

- [39] D. K. McClish, “Analyzing a portion of the ROC curve,” _Medical decision making_ , vol. 9, no. 3, 1989. 

- [40] S. Liu _et al._ , “Audio self-supervised learning: A survey,” _Patterns_ , vol. 3, no. 12, 2022. 

   - [66] Y. Zeng, H. Liu, L. Xu, Y. Zhou, and L. Gan, “Robust abomaly sound detection framework for machine condition monitoring,” DCASE2022 Challenge, Tech. Rep., 2022. 

   - [67] W. Junjie, W. Jiajun, C. Shengbing, S. Yong, and L. Mengyuan, “Anomalous sound detection based on self-supervised learning,” DCASE2023 Challenge, Tech. Rep., 2023. 

   - [68] Z. Lv _et al._ , “AITHU system for first-shot unsupervised anomalous sound detection,” DCASE2024 Challenge, Tech. Rep., 2024. 

   - [69] L. Wang, “Pre-trained model enhanced anomalous sound detection system for DCASE2025 task2,” DCASE2025 Challenge, Tech. Rep., 2025. 

- [41] S. Yadav, S. Theodoridis, and Z.-H. Tan, “An overview of neural architectures for self-supervised audio representation learning from masked spectrograms,” _arXiv:2509.18691_ , 2025. 

- [42] A. Cramer, H. Wu, J. Salamon, and J. P. Bello, “Look, listen, and learn more: Design choices for deep audio embeddings,” in _Proc. ICASSP_ , 2019. 

- [43] S. Chen _et al._ , “BEATs: Audio pre-training with acoustic tokenizers,” in _Proc. ICML_ , 2023. 

- [44] W. Chen, Y. Liang, Z. Ma, Z. Zheng, and X. Chen, “EAT: self-supervised pre-training with efficient audio transformer,” in _Proc. IJCAI_ , 2024. 

- [45] H. Dinkel, Z. Yan, Y. Wang, J. Zhang, Y. Wang, and B. Wang, “Scaling up masked audio encoder learning for general audio classification,” in _Proc. Interspeech_ , 2024. 

- [46] R. Arandjelovic and A. Zisserman, “Look, listen and learn,” in _Proc. ICCV_ , 2017. 

- [47] ——, “Objects that sound,” in _Proc. ECCV_ , 2018. 

- [48] B. Han, A. Jiang, X. Zheng, W.-Q. Zhang, J. Liu, P. Fan, and Y. Qian, “Exploring self-supervised audio models for generalized anomalous sound detection,” _IEEE Trans. Audio, Speech, Lang. Process._ , vol. 33, 2025. 

- [49] J. F. Gemmeke _et al._ , “Audio set: An ontology and human-labeled dataset for audio events,” in _Proc. ICASSP_ , 2017. 

- [50] A. Baevski, W. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli, “data2vec: A general framework for self-supervised learning in speech, vision and language,” in _Proc. ICML_ , 2022. 

- [51] S. Chen _et al._ , “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” _IEEE J. Sel. Top. Signal Process._ , 2022. 

- [52] S. Yadav and Z. Tan, “Audio mamba: Selective state spaces for selfsupervised audio representations,” in _Proc. Interspeech_ , 2024. 

- [53] K. Wilkinghoff, H. Yang, J. Ebbers, F. G. Germain, G. Wichern, and J. L. Roux, “Keeping the balance: Anomaly score calculation for domain generalization,” in _Proc. ICASSP_ , 2025. 

- [54] M. Matsumoto, T. Fujimura, W. Huang, and T. Toda, “Adjusting bias in anomaly scores via variance minimization for domain-generalized discriminative anomalous sound detection,” in _Proc. DCASE_ , 2025. 

- [55] K. Dohi _et al._ , “Description and discussion on DCASE 2023 Challenge Task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring,” in _Proc. DCASE_ , 2023. 

- [56] Y. Liu, J. Guan, Q. Zhu, and W. Wang, “Anomalous sound detection using spectral-temporal information fusion,” in _Proc. ICASSP_ , 2022. 

- [57] N. Harada, D. Niizumi, Y. Ohishi, D. Takeuchi, and M. Yasuda, “Firstshot anomaly sound detection for machine condition monitoring: A domain generalization baseline,” in _Proc. EUSIPCO_ , 2023. 

- [58] K. Wilkinghoff, “Design choices for learning embeddings from auxiliary tasks for domain generalization in anomalous sound detection,” in _Proc. ICASSP_ , 2023. 

- [59] Q. Hou, A. Jiang, W. Zhang, P. Fan, and J. Liu, “Decoupling detectors for scalable anomaly detection in AIoT systems with multiple machines,” in _Proc. GLOBECOM_ , 2023. 

- [60] K. Wilkinghoff, “Self-supervised learning for anomalous sound detection,” in _Proc. ICASSP_ , 2024. 

- [61] Y. Zhang, J. Liu, Y. Tian, H. Liu, and M. Li, “A dual-path framework with frequency-and-time excited network for anomalous sound detection,” in _Proc. ICASSP_ , 2024. 

- [62] K. Wilkinghoff, “AdaProj: Adaptively scaled angular margin subspace projections for anomalous sound detection with auxiliary classification tasks,” in _Proc. DCASE_ , 2024. 

- [63] J. Yin, W. Zhang, M. Zhang, and Y. Gao, “Self-supervised augmented diffusion model for anomalous sound detection,” in _Proc. APSIPA_ , 2024. 

- [64] J. Yin, Y. Gao, W. Zhang, T. Wang, and M. Zhang, “Diffusion augmentation sub-center modeling for unsupervised anomalous sound detection with partially attribute-unavailable conditions,” in _Proc. ICASSP_ , 2025. 

- [65] R. Giri, S. V. Tenneti, K. Helwani, F. Cheng, U. Isik, and A. Krishnaswamy, “Unsupervised anomalous sound detection using selfsupervised classification and group masked autoencoder for density estimation,” DCASE2020 Challenge, Tech. Rep., 2020.
