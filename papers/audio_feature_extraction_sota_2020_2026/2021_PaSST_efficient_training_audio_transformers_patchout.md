<!-- page: 1 -->

## **Efficient Training of Audio Transformers with Patchout** 

_Khaled Koutini_[1] _[,]_[2] _, Jan Schl¨uter_[1] _, Hamid Eghbal-zadeh_[1] _[,]_[2] _, Gerhard Widmer_[1] _[,]_[2] 

Institute of Computational Perception[1] & LIT AI Lab[2] , Johannes Kepler University Linz, Austria first.last@jku.at 

## **Abstract** 

The great success of transformer-based models in natural language processing (NLP) has led to various attempts at adapting these architectures to other domains such as vision and audio. Recent work has shown that transformers can outperform Convolutional Neural Networks (CNNs) on vision and audio tasks. However, one of the main shortcomings of transformer models, compared to the well-established CNNs, is the computational complexity. In transformers, the compute and memory complexity is known to grow _quadratically_ with the input length. Therefore, there has been extensive work on optimizing transformers, but often at the cost of degrading predictive performance. In this work, we propose a novel method to optimize and regularize transformers on audio spectrograms. Our proposed models achieve a new state-of-the-art performance on Audioset and can be trained on a single consumer-grade GPU. Furthermore, we propose a transformer model that outperforms CNNs in terms of both performance and training speed.[1] **Index Terms** : transformers, audio-tagging, attention models, 

## **1. Introduction** 

The transformer architecture [1] has proven very successful in sequence modeling. It allows learning dependencies between different items in the sequence regardless of their positions or their separation in the sequence. Transformers are the state-of-the-art models in different natural language processing tasks [2–4]. More recently, they have been adapted to computer vision [5–7] by extracting small patches from the input image and adding a learnable positional encoding to each patch. The resulting patches form a sequence that can be fed to the transformer. These vision transformer models achieve state-of-theart performance on image classification tasks, but require large amounts of training data (e.g, in Vision Transformer (ViT) [5]), or heavily depend on extensive data augmentation and knowledge distillation from a CNN model (e.g, in Data-efficient Image Transformers (DeiT) [8]). Gong et al. [9] further adapted vision transformers to audio spectrograms, achieving state-ofthe-art performance on Audioset [10] by using pre-trained models from computer vision and using overlapping patches from audio spectrograms for fine-tuning. 

The transformer architecture consists of a series of selfattention layers [1]. Each layer relies on calculating a distance between each pair of items from the input sequence. Although this allows each input item to attend to any other item in the sequence, this results in a complexity of _O_ ( _n_[2] ) with respect to the input sequence length _n_ , in terms of both memory and computation effort. Reducing the quadratic complexity has been the target of several approaches in natural language processing, the idea being to restrict each input item (token) to attend 

> 1Source code and pretrained models: https://github.com/ kkoutini/PaSST 

**==> picture [227 x 179] intentionally omitted <==**

**----- Start of picture text -----**<br>
AST CNN PaSST<br>0.480<br>0.475<br>PaSST-S<br>0.470<br>PaSST-U PaSST-N-S<br>0.465<br>0.460 AST PaSST-L-S<br>0.455 AST-N<br>0.450<br>0.445<br>0.440 1 GB/s 0.5 GB/s RF-CNN CNN14<br>0.435<br>20 40 60 80 100 120 140 160 180<br>Training speed: sample/second<br>mAP<br>**----- End of picture text -----**<br>


Figure 1: _Performance vs training speed on Audioset. The radius of the circle indicates the required GPU memory per sample for training. On the largest publicly available audio dataset, our approach PaSST-S can reach the state-of-the-art performance in less than 2 days on a single consumer GPU. Details are presented in Table 1._ 

only to a pre-selected subset of input items (tokens). One example is to allow attending only to neighbours inside a sliding window [11, 12]. Additionally, Kitaev et al. [13] use localitysensitive hashing to approximate attention, reducing the attention complexity to _O_ ( _n_ log _n_ ). BigBird [14] combines sliding windows, global attention, and random interaction between the sequence items. Masking portions of the input sequence has been shown to be an extremely effective method for training encoder/decoder transformers in NLP [2]. In computer vision, the idea of removing patches during _inference_ was investigated to assess the vision transformer’s _robustness_ against input perturbations [15]. 

In this paper, we focus on applying transformers to _audio processing_ . We address the shortcomings of current audio transformers from the aspect of computational complexity and memory requirements by introducing a new simple yet effective method for training transformers on spectrograms. In summary, the main contributions of our work are as follows: 

- We propose _Patchout_ , a method that significantly reduces the computation and memory complexity of training transformers for the audio domain. Patchout also functions as a regularizer, improving the generalization of the trained transformers. 

- We disentangle the transformer’s positional encoding [5, 8,9] into time and frequency positional encoding, allowing for straightforward inference on audio snippets of

<!-- page: 2 -->

variable length without the need for fine-tuning or interpolating positional encodings. 

- We investigate additional methods for reducing training complexity and demonstrate how they affect performance on the larger general-purpose Audioset as well as domain-specific downstream tasks. 

Our proposed models can achieve state-of-the-art performance on several audio tagging and classification tasks using a single consumer GPU, in a relatively short time (see Figure 1). When different complexity reduction methods are combined, the models outperform CNNs in terms of training speed and memory requirements, in addition to generalization. 

## **2. The Patchout faSt Spectrogram Transformer (PaSST)** 

The Vision Transformer (ViT) [5] works by extracting small _patches_ from an input image and projecting these patches linearly onto a sequence of embeddings. The sequence is then augmented by adding trainable positional encodings as biases to the input sequence. A special classification embedding (classification token) is then appended to the sequence, which is connected to a classifier after the self-attention layers. In Dataefficient Image Transformers (Deit) [8] another special embedding for distillation (distillation token) is added. Gong et al. [9] showed that _overlapping_ the extracted patches improves the performance when training ViT on spectrograms. On the other hand, patch overlapping increases the total number of patches, i.e., the input sequence length. Therefore, overlapping greatly increase the memory and compute requirements for training the transformers. We propose a new method called _Patchout_ (Section 2.2) to overcome these issues. 

Figure 2 summarizes the proposed transformer architecture: The pipeline starts at the upper left with an audio spectrogram being fed into the model as input. (1) is the patch extraction and linear projection steps as explained in [5]. In (2), frequency and time positional encodings are added, as discussed below. In (3), we apply _Patchout_ as explained in Section 2.2, and add the classification token. In (4), we flatten the sequence and pass through _d_ layers blocks of self-attention ( _d_ is the depth of the transformer). Finally, a classifier operates on the mean of the transformed C and D tokens. 

In Addition to _Patchout_ (step 3 in Figure 2), the second main difference between our work and previous work [5, 8, 9] is that we disentangle the positional encoding for the time and frequency dimensions, and as a result, we have two positional encodings: one representing the frequency, and one for time (step 2 in Figure 2). This makes inference and the tuning of the pre-trained models on downstream tasks with shorter audio length simpler. When fine-tuning or inference on shorter audio clips, we simply crop the time positional encoding parameters, without changing the frequency encoding parameter. 

## **2.1. Complexity Analysis** 

Multi-head attention layers [1] rely on computing a distance between each pair of positions in the input sequence (in the form of an attention matrix), therefore having a complexity of _O_ ( _n_[2] ) where _n_ is the input sequence length. As the sequence length grows – for example, when overlapping input patches or for longer audio clips – the compute and memory requirements quickly become problematic. More specifically, given an input of _b_ samples of the dimension ( _n × e_ ), where _b_ is the batch size, 

Figure 2: _The Patchout transformer (PaSST) architecture as explained in Section 2. The Self-attention layer + FFN (Feedforward network) is explained in detail in [5]. C: classification token. D: distillation token (only for models based on DeiT)._ 

_n_ is the input sequence length, _e_ is the embeddings size, each _multi-head_ attention layer projects each input sample to _h_ query _Q_ , key _K_ , and value _V_ matrices, where _h_ is the number of attention heads [1]. Each of _Q_ , _K_ and _V_ has a shape of ( _n × h[e]_[)][.] The _attention matrix A_ is then computed by the matrix multiplication _Q × K[T]_ , scaling, and applying the soft-max activation function _A_ = _Softmax_ ( _[Q][×]_ ~~_√_~~ _[K] d[T]_ ). _A_ has a shape of ( _n × n_ ) and is multiplied with _V_ giving the attention output: _A × V_ resulting in a new sequence with the same shape as the input ( _n × e_ ). As a result, the computation complexity (and memory requirements) for all the operations on the attention matrix _A_ grow quadratically _O_ ( _n_[2] ) with sequence length _n_ , while the operations in the rest of the network have a linear complexity relationship with _n_ [1,5,8]. In short, reducing the sequence length would have a large impact on the computational complexity of these models. 

## **2.2. Patchout** 

Motivated by (a) the impact of reducing the sequence length on the computation complexity of training transformer models; (b) the fact that audio events are expected to be spread out in time and frequency in an audio clip; (c) the insight that CNNs can benefit from having a small receptive field during training for different audio tasks, as shown in [16], we propose _Patchout_ , a method to efficiently train transformer models on audio spectrograms. The idea is to drop parts of the transformer’s input sequence when training, encouraging the transformer to perform the classification using an incomplete sequence. We first extract small overlapping patches from the input spectrograms and linearly project them to vectors, forming the transformer input sequence. We augment the patches with both frequency and time encoding. When training, we randomly drop parts of the sequence, reducing the sequence length, and effectively regularizing the training process. Similar to DropOut [17], during inference, the whole input sequence is presented to the transformer. We distinguish between different types of Patchout as follows: 

**Unstructured Patchout** is the basic form of Patchout,

<!-- page: 3 -->

where we select the patches randomly regardless of their position. We refer to models trained with this method as _PaSST-U_ . 

**Structured Patchout:** We randomly pick some _frequency bins_ / _time frames_ and remove a whole column/row of extracted patches. This structure is inspired by SpecAugment [18]. We refer to models trained with this method as _PaSST-S_ . 

## **2.3. Further Complexity Reduction Methods** 

## _2.3.1. Reducing the extracted patches’ overlap_ 

Reducing the overlap between patches results in a lower number of extracted patches, and therefore a smaller transformer input sequence length. However, Gong et al. [9] showed that reducing the overlap (or training without overlapping) degrades the performance of the transformer on Audioset. Patchout can also be used even when there is no overlap between patches. We refer to the system without patch overlapping as _PaSST-N_ . 

## _2.3.2. Reducing the depth of the transformer_ 

The depth of the transformer is the number of successive selfattention blocks ( _d_ in Figure 2). The depth has a linear relationship with the overall training and inference complexity and influences the total number of parameters of the model. Since we are starting the training from models pre-trained on Imagenet [19] (as explained in Section 3.2), we remove every other self-attention block. This allows us to benefit from the pretraining, compared to removing consecutive blocks, since the residual activations will have a less sudden change. We will refer to the model with the removed blocks as _PaSST-L_ . It has _d_ = 7 self-attention blocks and 50M parameters compared to 87M in the full model. 

||mAP|Speed<br>(_×_AST)|Mem|Seq|
|---|---|---|---|---|
|Baseline [10]<br>PANNs [20]<br>AST[9]|.314<br>.439<br>.459|-<br>131.0<br>(5_._7_×_)<br>23.1<br>(1_×_)|-<br>.213<br>2.33|-<br>-<br>1212|
|AST [9]_⋆_<br>AST-N [9]_⋆_<br>CNN[23]_⋆_|.459<br>.454<br>.438|23.1<br>(1_×_)<br>80.<br>(3_._4_×_)<br>126.3<br>(5_._5_×_)|2.33<br>.534<br>.213|1190<br>498<br>-|
|PaSST-B_⋆_<br>PaSST-U_⋆_<br>PaSST-S_⋆_<br>PaSST-S-L_⋆_<br>PaSST-S-N_⋆_|.462<br>.466<br>**.471**<br>.459<br>.466|23.1<br>(1_×_)<br>43.2<br>(1_._9_×_)<br>88.7<br>(3_._8_×_)<br>148.6<br>(6_._4_×_)<br>**184.2**<br>(8_×_)|2.33<br>1.14<br>.513<br>.311<br>**.202**|1190<br>790<br>474<br>474<br>254|



Table 1: _Single-model results on Audioset. ⋆ indicates our run. mAP is the mean average precision (also referred to as precision/recall area under curve). Speed: training throughput in spectrograms per second on an Nvidia Titan RTX GPU (showing the speedup compared to AST [9]). Mem: the required GPU memory to train per sample. Seq: the training sequence length._ _**B** : Baseline without Patchout._ _**U** : Unstructured Patchout._ _**S** : Structured Patchout._ _**N** : no-overlap of the extracted patches._ _**L** : lighter model with reduced depth=7._ 

ments.Our base model is DeiT B↑384 [8]. We also achieve a comparable performance using computationally more complex ViT models such as stripped-down ViT-hug224 [5]; by removing half of the self-attention blocks, its depth was reduced to only 16 blocks (with the methods explained in Section 2.3.2); this will not be further explored in this paper. 

## **3.3. Data Augmentation** 

## **3. Experiment Setup** 

We train our models on Audioset [10], the largest publicly available audio dataset, consisting of around 2 million audio clips from Youtube. The task is to tag the audio clips with labels from 527 possible classes. Furthermore, we fine-tune the models trained on Audioset on various audio classification and tagging tasks, namely, instrument recognition, environmental audio classification, and acoustic scene classification. 

## **3.1. Preprocessing and Training Setup** 

We use mono audio with a sampling rate of 32 kHz. We extract Mel features from a window of 25 ms with a hop length of 10 ms, resulting in 128 mel bands, similar to [9]. Kong et al. [20] showed the importance of balancing Audioset; therefore, we balance our training data using importance sampling. We assign a sampling weight to each sample proportional to the inverse frequency of its label _freq_ ( _label_ 1 )+100[.][We][train][on] 1 _,_ 893 _,_ 693 (approx. 2M) training segments, and evaluate on 18 _,_ 951 audio clips. For each epoch, we sample 200k samples from the full 2M Audioset without replacement. We use the AdamW [21] optimizer with weight decay of 10 _[−]_[4] , with a maximum learning rate of 10 _[−]_[5] . We use a linear learning rate decay from epoch 50 to 100, dropping the learning rate to 10 _[−]_[7] and fine-tune the model for a further 20 epochs. 

## **3.2. ImageNet Pretraining** 

Gong et al. [9] showed that using pre-trained models on Imagenet significantly improves their performance on Audioset. Therefore, we will use pre-tranined models in all our experi- 

The transformer models are very prone to overfitting, therefore data augmentation plays an essential role in the training process [8]. In our experiments, the following augmentation strategies are used: 

**Two-level Mix-Up:** We use Mix-up [22] since it has been shown to improve performance [9, 23]. We mix both the raw waveforms randomly from the dataset as well as the final spectrograms. 

**Specaugment:** We use SpecAugment [18] by masking up to 48 frequency bins and 192 time frames similar to [9]. 

**Rolling:** We roll the waveforms randomly over time. **Random Gain:** We multiply the audio waveforms to change the gain by _±_ 7 dB. 

## **4. Results** 

## **4.1. Audio Tagging on Audioset** 

Table 1 shows the mean average precision mAP (also referred to as precision-recall area under-curve) results on Audioset [10]. As can be seen, the proposed model _PaSST_ achieves a new state-of-the-art performance on the largest available audio tagging dataset. The proposed model outperforms AST [9] and significantly outperforms CNNs. Using Patchout not only improves the performance of the transformer architecture, but also increases the training speed approximately _4 times_ , and reduces the required GPU memory to _less than_ 25%. As a result, it is possible to train PaSST on a single Nvidia RTX 2080ti (consumer GPU), achieving state-of-the-art performance in 50 hours. Furthermore, _PaSST-L-S_ (with a scaled down depth of _d_ = 7) and _PaSST-S-N_ (without patch overlap) significantly

<!-- page: 4 -->

outperform CNNs while maintaining a higher training throughput, and with similar GPU memory requirements. _PaSST-S-L_ and _PaSST-S-N_ can be trained on a single GPU to reach _._ 459 and _._ 466 mAP in approximately 25 hours. Applying Patchout on the transformer without overlap ( _PaSST-S-N_ ) outperforms the baseline _PaSST-B_ (without Patchout) and _AST_ [9] while being up to 8 times faster, and requiring less than 10% of the GPU memory for training. The results are also illustrated in Figure 1. The only difference between the baseline _PaSST-B_ and the _AST_ model is the positional encoding. _AST_ [9], like vision transformers [5, 8], employs grid positional encoding. _PaSST-B_ , on the other hand, utilises disentangled time and frequency positional encoding (see Section 2). 

||mAP|
|---|---|
|Baseline[10]|.314|
|PSLA (Ensemble-S) [24]<br>PSLA (Ensemble-M) [24]<br>AST (Ensemble-M) [9]<br>AST(Ensemble-M) [9]|.469<br>.474<br>.475<br>.485|
|PaSST-S S16,14 (2 models)_⋆_<br>PaSST-S S10-16 (4 models)_⋆_<br>PaSST-S S10-16 (5 models)_⋆_<br>PaSST-S S10-16(9 models)_⋆_|.486<br>.493<br>**.495**<br>**.496**|



Table 2: _Ensemble results on Audioset. ⋆ indicates our run. mAP is the mean average precision (also referred to as precision/recall area under curve)._ 

Table 2 shows the result of ensemble models. We ensemble models with different overlap values between input patches (Figure 2). **S** indicates the patches stride, S16 means no overlap between the patches. The first ensemble (2 models) averages the logits of a model with no patches overlap and a model with an overlap of 2 (stride=14). S10-S16 indicates that the models used have strides of 10,12,14 and 16. 

## **4.2. Fine-tuning and Transfer to Downstream Tasks** 

We fine-tune the pre-trained (on Audioset) models on several downstream audio tagging and classification tasks with different dataset sizes, Table 3 summarizes the results. _PaSST(B,U,S)_ models use the pre-trained _PaSST-S_ on Audioset, but for fine-tuning, we use no Patchout, unstructured Patchout, and structured Patchout respectively. It is worth noting that the transformer models can be fine-tuned using a small number of epochs. The results suggest that researchers and practitioners can use pre-trained PaSST and fine-tune them on downstream tasks without the need for large computational resources. 

In summary, fine-tuning the transformer model outperforms state-of-the-art CNNs on all tasks. Patchout results in significant speedups and, in many cases, improved generalization. When combined with Structured Patchout ( _-S_ ), reducing complexity by removing patch overlap ( _-N_ ) performs better than reducing transformer depth ( _-L_ ) and enables faster fine-tuning. 

We only replace the MLP classifier in the pre-trained models for fine-tuning. When we use Patchout, we randomly remove roughly half of the input sequence. Each experiment was repeated three times, and the average results are reported. The speedup in Table 3 is relative to _PaSST-B_ and is rounded up to the nearest integer. Details on the setup of each task can be found in our github repository. 

**Polyphonic Musical Instrument Recognition:** The task here is to detect all the instruments present in an audio clip. The 

_OpenMIC_ dataset [25] consists 20,000 audio clips. Each clip is 10 seconds long and can be assigned multiple tags out of 20 classes. The metric for the task is the mean average precision. The state-of-the-art methods for this task are CNNs with restricted receptive fields [23]. _PaSST-S-N_ reaches the state-ofthe-art performance in less than 30 minutes on a single consumer GPU. 

**Environmental Sound Classification:** The _ESC50_ dataset [26] consists of 2,000 environmental 5-second audio clips. The task is to classify each clip into one out of 50 possible classes. We report the accuracy averaged over the 5 official folds [26]. All PaSST variants (with Patchout) can be fine-tuned on this dataset in less than 5 minutes on a single GPU. The state-of-the-art performance was achieved using the AST transformer model [9]. The difference between AST and PaSST-B is in the positional encoding, as explained in Section 2. 

**Acoustic Scene Classification:** The task is to recognize the acoustic scene of 10-second audio clips. We use the TAU Urban Acoustic Scenes 2020 Mobile dataset [27] as used in the DCASE 2020 challenge ( _DCASE20_ ). The audio clips are recorded with different devices and further simulated devices are introduced. The performance is measured using accuracy on a dataset including unseen devices. The first place in the challenge used CNNs [28]. Patchout accelerates training on this task, reaching state-of-the-art in less than an hour. Patchout also allows for fine-tuning on a single consumer GPU. It does, however, lead to a decrease in accuracy. 

**Sound Event Recognition (Tagging) on FSD50K:** The _FSD50K_ dataset [29] consists of 51K audio clips annotated with 200 sound event classes taken from the Audioset ontology [10]. The dataset contains 100 hours of audio and is the second largest publicly available general purpose sound event recognition dataset after Audioset. Furthermore, the FSD50K evaluation set is of high quality, with each evaluation label being double-checked and assessed by two to five independent annotators [29]. The reported results are on the official evaluation subset of FSD50K using the best model on the validation subset. The state-of-the-art in _PSLA_ [24] is achieved through CNN architecture and a collection of performance-improving methods such as ImageNet pre-training, label enhancement, balancing, data augmentation, and weight averaging. On this dataset, our approach significantly outperforms the current state-of-the-art. Fine-tuning _PaSST-S_ and _PaSST-S-N_ takes less than 2 hours and 1 hour, respectively. 

||OpenMIC|ESC50|DCASE20|FSD50K|
|---|---|---|---|---|
|Baseline<br>SOTA|.795 [25]<br>.831[23]|76.9 [26]<br>95.6[9]|54.1 [27]<br>73.7[28]|.434 [29]<br>.558[24]|
|-B_⋆_<br>-U_⋆_<br>-S_⋆_<br>-S-L_⋆_<br>-S-N_⋆_|.837_|_1_×_<br>**.843**_|_4_×_<br>**.843**_|_4_×_<br>.841_|_6_×_<br>.840_|_8_×_|96.3_|_1_×_<br>96.5_|_2_×_<br>**96.8**_|_2_×_<br>95.5_|_2_×_<br>96.4_|_4_×_|**76.3**_|_1_×_<br>75.6_|_4_×_<br>75.6_|_4_×_<br>73.7_|_6_×_<br>73.9_|_8_×_|.649_|_1_×_<br>.639_|_4_×_<br>**.653**_|_4_×_<br>.584_|_6_×_<br>.637_|_8_×_|



Table 3: _Results in performance and speedup compared to the base model PaSST-B for the downstream tasks: polyphonic instrument tagging using OpenMIC [25] dataset (mean average precision), Environmental Sound Classification ESC50 [26] (accuracy), Cross-device Acoustic Scene Classification DCASE20 [27] (accuracy). Sound Event Recognition (Tagging) in FSD50K [29](mean average precision). The second part of the table compares different PaSST variants ( Table 1)._

<!-- page: 5 -->

## **5. Conclusion** 

We propose a new method for efficiently training transformers on audio spectrograms, achieving state-of-the-art performance on Audioset as well as several downstream tasks. Furthermore, Patchout significantly reduces compute complexity and memory requirements for training transformers. We investigate additional methods for reducing training complexity and propose two models, _PaSST-S-L_ and _PaSST-S-N_ , that outperform CNNs while having a faster training speed and comparable memory requirements. Our pre-trained models can be fine-tuned on several audio downstream tasks with little resources and little additional training time. 

## **6. ACKNOWLEDGMENT** 

This work has been supported by the COMET-K2 Center of the Linz Center of Mechatronics (LCM) funded by the Austrian Federal Government and the Federal State of Upper Austria. The LIT AI Lab is financed by the Federal State of Upper Austria. The computational results presented have been achieved in part using the Vienna Scientific Cluster (VSC). 

## **7. References** 

- [1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in _Advances in Neural Information Processing Systems_ , 2017, pp. 5998–6008. 

- [2] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pretraining of deep bidirectional transformers for language understanding,” in _NAACL-HLT Minneapolis, MN, USA, June 2-7, 2019, Volume 1_ , 2019, pp. 4171–4186. 

- [3] D. R. So, W. Manke, H. Liu, Z. Dai, N. Shazeer, and Q. V. Le, “Primer: Searching for efficient transformers for language modeling,” _Advances in neural information processing systems_ , 2021. 

- [4] I. Yamada, A. Asai, H. Shindo, H. Takeda, and Y. Matsumoto, “LUKE: deep contextualized entity representations with entityaware self-attention,” in _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP_ . Association for Computational Linguistics, 2020, pp. 6442–6454. 

- [5] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in _International Conference on Learning Representations ICLR 2021, Virtual, May 3-7_ , 2021. 

- [6] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in _2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021_ . IEEE, 2021, pp. 9992–10 002. 

- [7] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable DETR: deformable transformers for end-to-end object detection,” in _International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7_ , 2021. 

- [8] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. J´egou, “Training data-efficient image transformers & distillation through attention,” in _ICML 2021, 18-24 July 2021, Virtual_ , vol. 139, 2021, pp. 10 347–10 357. 

- [9] Y. Gong, Y.-A. Chung, and J. Glass, “AST: Audio Spectrogram Transformer,” in _Interspeech 2021, Brno, Czechia, 30 August - 3 September_ , 2021, pp. 571–575. 

- [10] J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in _ICASSP 2017_ , New Orleans, LA, 2017. 

- [11] Z. Wang, P. Ng, X. Ma, R. Nallapati, and B. Xiang, “Multipassage BERT: A globally normalized BERT model for opendomain question answering,” in _EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7,_ , 2019, pp. 5877–5881. 

- [12] S. Sukhbaatar, E. Grave, P. Bojanowski, and A. Joulin, “Adaptive attention span in transformers,” in _Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2_ , 2019, pp. 331–335. 

- [13] N. Kitaev, L. Kaiser, and A. Levskaya, “Reformer: The efficient transformer,” in _International Conference on Learning Representations ICLR 2020, Addis Ababa, Ethiopia, April 26-30_ , 2020. 

- [14] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Onta˜n´on, P. Pham, A. Ravula, Q. Wang, L. Yang, and A. Ahmed, “Big bird: Transformers for longer sequences,” in _Advances in neural information processing systems_ , 2020. 

- [15] M. Naseer, K. Ranasinghe, S. H. Khan, M. Hayat, F. S. Khan, and M. Yang, “Intriguing properties of vision transformers,” in _Advances in neural information processing systems_ , 2021. 

- [16] K. Koutini, H. Eghbal-zadeh, M. Dorfer, and G. Widmer, “The Receptive Field as a Regularizer in Deep Convolutional Neural Networks for Acoustic Scene Classification,” in _EUSIPCO 2019_ , A Coru˜na, Spain, 2019. 

- [17] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A simple way to prevent neural networks from overfitting,” _Journal of Machine Learning Research_ , vol. 15, no. 56, pp. 1929–1958, 2014. 

- [18] D. S. Park, W. Chan, Y. Zhang, C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, “Specaugment: A simple data augmentation method for automatic speech recognition,” in _Interspeech 2019, Graz, Austria, 15-19 September_ , 2019, pp. 2613–2617. 

- [19] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in _IEEE Conference On Computer Vision and Pattern Recognition._ , 2009, pp. 248–255. 

- [20] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition,” _IEEE ACM Trans. Audio Speech Lang. Process._ , vol. 28, pp. 2880–2894, 2020. 

- [21] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in _International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019_ , 2019. 

- [22] H. Zhang, M. Ciss´e, Y. N. Dauphin, and D. Lopez-Paz, “Mixup: Beyond empirical risk minimization,” in _International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings_ , 2018. 

- [23] K. Koutini, H. Eghbal-zadeh, and G. Widmer, “Receptive field regularization techniques for audio classification and tagging with deep convolutional neural networks,” _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , vol. 29, pp. 1987–2000, 2021. 

- [24] Y. Gong, Y.-A. Chung, and J. Glass, “PSLA: Improving audio tagging with pretraining, sampling, labeling, and aggregation,” _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 2021. 

- [25] E. Humphrey, S. Durand, and B. McFee, “OpenMIC-2018: An open data-set for multiple instrument recognition,” in _ISMIR 2018, Paris, France, September 23-27_ , 2018, pp. 438–444. 

- [26] K. J. Piczak, “ESC: Dataset for Environmental Sound Classification,” in _Proceedings of the 23rd Annual ACM Conference on Multimedia_ . ACM Press, 2015, pp. 1015–1018. 

- [27] T. Heittola, A. Mesaros, and T. Virtanen, “Acoustic scene classification in DCASE 2020 Challenge: generalization across devices and low complexity solutions,” in _DCASE2020 Workshop_ , 2020. 

- [28] S. Suh, S. Park, Y. Jeong, and T. Lee, “Designing Acoustic Scene Classification Models with CNN Variants,” DCASE2020 Challenge, Tech. Rep., 2020.

<!-- page: 6 -->

- [29] E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra, “FSD50K: an open dataset of human-labeled sound events,” _IEEE ACM Trans. Audio Speech Lang. Process._ , vol. 30, pp. 829–852, 2022.
