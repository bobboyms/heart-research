<!-- page: 1 -->

# **CONTRASTIVE LEARNING OF GENERAL-PURPOSE AUDIO REPRESENTATIONS** 

_Aaqib Saeed_[1] _[∗] , David Grangier_[2] _, Neil Zeghidour_[2] 

1Eindhoven University of Technology, Eindhoven, The Netherlands 2Google Research, Paris, France 

## **ABSTRACT** 

We introduce COLA, a self-supervised pre-training approach for learning a general-purpose representation of audio. Our approach is based on contrastive learning: it learns a representation which assigns high similarity to audio segments extracted from the same recording while assigning lower similarity to segments from different recordings. We build on top of recent advances in contrastive learning for computer vision and reinforcement learning to design a lightweight, easy-toimplement self-supervised model of audio. We pre-train embeddings on the large-scale Audioset database and transfer these representations to 9 diverse classification tasks, including speech, music, animal sounds, and acoustic scenes. We show that despite its simplicity, our method significantly outperforms previous self-supervised systems. We furthermore conduct ablation studies to identify key design choices and release a library[1] to pre-train and fine-tune COLA models. 

_**Index Terms**_ **—** self-supervised learning, audio, sound 

## **1. INTRODUCTION** 

Self-supervised pre-training has recently emerged as a successful technique to leverage unlabeled data to learn representations beneficial to supervised problems. This success spans a wide range of tasks and modalities [1, 2, 3, 4]. Among these methods, Discriminative Pre-Training (DPT) is particularly effective. This approach learns a representation from pairs of _similar_ inputs from unlabeled data, exploiting e.g. temporal consistency [5, 4, 6] or data augmentation [7] and trains a model to recognize _similar_ elements among negative distractors. In contrast with generative encoder-decoder approaches [8, 9, 10, 11, 12], DPT is computationally efficient as it avoids input reconstruction entirely. 

Amidst DPT models for audio, [6] used a metric learning approach with a triplet loss to minimize the distance between embeddings of anchor and positive pairs and maximize it among the negatives. The instance generation is achieved through noise injection, shifting along time-frequency di- 

> _∗_ This work was conducted while interning at Google. 

> 1https://github.com/google-research/google-research/ tree/master/cola 

mensions, and extracting samples in temporally close neighborhoods. Along similar lines, [13] proposed a benchmark for comparing speech representations on non-semantic tasks. Through utilizing a triplet loss as an unsupervised objective with a subset of AudioSet [14] for model training, they showed improved performance on several downstream speech classification tasks. Inspired from seminal work in NLP [15], the work in [16] adopted a similar approach to learn audio representations (i.e. AUDIO2VEC) along with another “pretext” task of estimating temporal distance between audio segments. The pre-trained models are tested on several downstream tasks, from speaker identification to music recognition. Despite recent progress, most work on learning representations of audio focuses on speech tasks [17, 18, 19] (with the exception of [6, 16]) and ignores other audio tasks such as acoustic scene detection or animal vocalizations. Moreover, triplet-based objectives heavily rely on the mining of negative samples, and the quality of learned features can vary significantly with the sample generation scheme. 

In this work, we propose COLA (COntrastive Learning for Audio), a simple contrastive learning framework to learn general-purpose representations of sounds beyond speech. We build upon recent advances in contrastive learning [2] for computer vision (SIMCLR [7], MOCO [20]) and reinforcement learning (CURL [21]). We generate similar pairs by simply sampling segments from the same audio clip, which avoids exploring augmentation strategies entirely unlike SIMCLR, MOCO, CURL and others [22]. Our dissimilar pairs simply associate segments from different clips in the same batch, which does not require maintaining a memory bank of distractors as in MOCO. Our approach allows us to consider a large number of negatives for each positive pair in the loss function and bypass the need for a careful choice of negative examples, unlike triplet-based approaches [6, 13]. COLA is also different from CPC [2] as it does not predict future latent representations from past ones. 

We demonstrate the effectiveness of COLA over challenging and diverse downstream tasks, including speech, music, acoustic scenes, and animal sounds. After pre-training on the large-scale AudioSet database [14], we show that a linear classifier trained over a COLA embedding gets close to the performance of a fully-supervised in-domain convolutional

<!-- page: 2 -->

**==> picture [496 x 91] intentionally omitted <==**

**----- Start of picture text -----**<br>
O 1 Sample Pair Generation O 2 Instance Discrimination Contrastive Learning (COLA)  O 3 Downstream Training<br>log-compressed  h, Zp<br>mel-filterbanks fixed (or fine-tuned)<br>Encoder ProjectionHead<br>Pre-trained<br>Encoder C1 task 1<br>~ Je s t 2h Maximize  »( [ -o-<br>Agreement<br>Raw Waveform<br>Audio DatasetLarge-scale Unlabeled  g o ne Encoder (oe ProjectionHead Pre-trained Encoder CN task N<br>y+ 9 RE K O Je  - L<br>...<br>**----- End of picture text -----**<br>


**Fig. 1** . Overview of the contrastive self-supervised learning for audio. 

network and exceeds it when using fine-tuning. Moreover, our system outperforms previous unsupervised approaches on most downstream tasks. These experiments demonstrate that COLA offers a simple, easy-to-implement method to learn general-purpose audio representations without supervision. 

## **2. METHOD** 

We learn general-purpose audio representations from unlabeled data by pre-training a neural network with a contrastive loss function. Our objective function maximizes an agreement between the latent embedding of segments extracted from the same audio clip while using different audio clips as negative classes, as shown in Figure 1. This objective pre-trains a convolutional feature extractor on unlabeled audio data. After pre-training, we combine our feature extractor with an additional classification layer for solving various audio understanding tasks across several datasets. 

Contrastive learning extracts a latent space in which the similarity between an anchor example and a related example should be greater than the similarity between the same anchor and unrelated examples. In our case, an anchor and its corresponding positive are audio segments from the same clip. This contrasts with approaches that generate positives as perturbations of the anchor [23, 22]. For negative examples, we take segments from different audio clips in the current training batch. This strategy allows to consider a large number of negatives and is efficient since batch examples are used both as positives and negatives without additional computation. 

COLA computes the similarity between audio segments in two steps. First, an encoder _f_ maps a log-compressed melfilterbanks **x** _∈_ R _[N][×][T]_ , with _N_ and _T_ the number of frequency bins and time frames respectively, into a latent representation _h_ = _f_ ( _x_ ) _∈_ R _[d]_ . This is the representation that we will transfer to downstream classification, after pre-training. Then, a shallow neural network _g_ maps _h_ onto a space _z_ = _g_ ( _h_ ), where bilinear comparisons are performed. If we denote with _W_ the bilinear parameters, the similarity between two segments ( _x_ , _x[′]_ ) is, therefore: 

**==> picture [191 x 13] intentionally omitted <==**

Bilinear similarity has been used in the past [2] but is less 

common than cosine similarity, e.g. SIMCLR and MOCO. In Section 3, we perform an ablation study on the choice of similarity measure. Table 3 shows that a bilinear similarity outperforms a simple cosine similarity ( _∥[g] g_[(] ( _[f] f_[(] ( _[x] x_[))] )) _[⊤] ∥∥[·] g[g]_ ([(] _f[f]_ ([(] _x[x][′][′]_ ))[))] _∥_[)][on] all downstream tasks. In the rest of this paper, we use this method when not stated otherwise. 

As an objective function, we rely on multi-class cross entropy applied to similarities, i.e. 

**==> picture [181 x 35] intentionally omitted <==**

where _x_[+] is the positive associated to anchor _x_ , while _X[−]_ ( _x_ ) refers to the set of negative distractors. This loss, unlike the triplet loss [24], leverages multiple distractors at a time. 

As mentioned earlier, we train our model with positive segment pairs sampled from the same audio clip. For each pair, we use one segment as the anchor and the other element as the positive. Positive segments are used as negatives for all other anchors in the batch. This strategy is more efficient than keeping a memory bank of negatives [23, 20] since the representation of an example is paired with every anchor in the batch either as a positive or as a negative segment. In particular, we experiment with batch sizes varying from 256 to 2048, as shown in Table 4. A large batch size allows the model to see many negative samples per anchor and helps accuracy on end tasks. It is important to note that we sample segment pairs on-the-fly and reshuffle the data at each training epoch to maximize the diversity of positive and negative pairs seen during training. The sample generation procedure is illustrated in Figure 1. 

## **3. EXPERIMENTS** 

We evaluate our method by pre-training COLA embeddings on a large-scale audio dataset and then transferring it to downstream tasks in the following ways: 1) training a linear classifier on top of a frozen embedding, used as a feature extractor and 2) fine-tuning the entire network on the end-task. Importantly, we assess the performance on several diverse datasets to determine the transferability of learned representations across audio domains and recording conditions.

<!-- page: 3 -->

**Table 1** . Test accuracy (%) on downstream tasks. 

||**Random **|**Supervised**|**COLA**|**COLA**|
|---|---|---|---|---|
|**Task**|**Init.**||**Frozen **|**Fine-tuned**|
|**Speaker Id. (LBS)**|0.4|**100.0**|**100.0**|**100.0**|
|**Speech commands (V1)**|62.9|97.2|71.7|**98.1**|
|**Speech commands (V2)**|4.0|94.3|62.4|**95.5**|
|**Acoustic scenes**|8.6|98.2|94.1|**99.2**|
|**Speaker Id. (Voxceleb)**|0.0|31.7|29.9|**37.7**|
|**Birdsong detection**|49.6|79.4|77.0|**80.2**|
|**Music, Speech and Noise**|56.8|99.3|99.1|**99.4**|
|**Language Id.**|59.1|**85.0**|71.3|82.9|
|**Music instrument**|20.8|70.7|63.4|**73.0**|
|**Average**|29.1|83.9|74.3|**85.1**|



## **3.1. Datasets and Tasks** 

We pre-train COLA embeddings on the diverse, large-scale Audioset [14]. It contains 2 millions excerpts of 10 seconds audio from YouTube videos that are annotated in a multi-label fashion with over 500 classes. This dataset has been used by [16, 25, 13] for self-supervised pre-training. Since our method is self-supervised, we never use Audioset labels. As described earlier, we randomly sample audio clips to generate examples. Likewise, for the extraction of anchors and positives, segments of audio are selected uniformly at random inside a sequence. 

We perform downstream evaluation on a variety of tasks, including both speech and non-speech. To allow for comparison with previous methods, we rely on datasets that have been previously used by [16, 25, 13]. For speaker identification, we use a 100-hours subset of LibriSpeech (LBS) [26] that contains audio of books read by 251 speakers, as well as the Voxceleb [27] subset used in [13], with 1 _,_ 251 speakers. For keyword spotting, we use Speech Commands (SPC) [28] V1 and V2 to recognize 11 and 35 spoken commands (classes) from one second of audio, respectively. For acoustic scene classification, we use TUT Urban Acoustic Scenes 2018 (TUT) [29], consisting of labeled audio segments from 10 different acoustic scenes. For animal vocalizations, we use the Bird Song Detection (BSD) dataset [30] from DCASE 2018 Challenge to solve a binary classification problem. For music recognition, we use MUSAN [31] that differentiates audio samples across 3 classes (speech, music and noise), as well as the NSynth dataset [32] of musical notes, labeled with the family of the instrument (11 classes). For language identification, we use the Voxforge dataset [33] to categorize audio clips into six classes based on the spoken language. 

## **3.2. Model Architecture and Implementation Details** 

Given an audio input sequence, we extract log-compressed mel-filterbanks with a window size of 25 ms, a hop size of 

10 ms, and _N_ = 64 mel-spaced frequency bins in the range 60–7800 Hz for _T_ = 96 frames, corresponding to 960 ms. These features are passed through an encoder _f_ based on EfficientNet-B0 [34], a lightweight and highly scalable convolutional neural network. Even though EfficientNet-B0 has been originally proposed for computer vision, the 2D structure of mel-filterbanks allows using this architecture without any adjustment. We apply a global max-pooling to the last layer of the encoder to get an embedding _h_ of size 1280. During pre-training, we pass _h_ through the projection head _g_ , which contains a fully-connected layer with 512 units followed by a Layer Normalization [35] and a tanh activation. We discard the projection head for the downstream tasks and train a linear classifier on top of the encoder directly. We pre-train all our models with ADAM [36] and a learning rate of 10 _[−]_[4] , for 500 epochs. We explore the impact of the batch size and report the results in Table 4. We train the downstream classifiers with a batch size of 64 and a learning rate of 10 _[−]_[3] , on randomly selected 960ms segments, as for pre-training. However, we evaluate downstream classifiers on entire sequences using the following procedure: we split the sequence into non-overlapping 960ms segments, pass them through the encoder and linear classifier, and average the predictions. 

## **3.3. Results** 

Table 1 reports the accuracy on the 9 downstream datasets. We compare our approach against multiple baselines: a linear classifier trained on a randomly initialized fixed encoder and a fully-supervised model trained directly on downstream datasets which indicates the performance achievable with EfficientNet-B0 on these datasets. First, we evaluate pretrained COLA embeddings with a linear classifier on top of frozen representations, following the same procedure as [16, 2, 7, 20]. This outperforms drastically the performance of a linear classifier trained on a random embedding (74 _._ 3% against 29 _._ 1% on average), showing that the encoder has learned useful representations. This is remarkable as we pre-train a single COLA embedding, which performs well across many tasks. Next, we also use a pre-trained COLA as initialization and fine-tune one model per downstream task. Table 1 shows that on all tasks but language identification, initializing a supervised model with COLA improves the performance over training from scratch (85 _._ 1% against 83 _._ 9% on average), which demonstrates the benefits of transferring COLA representations even in a fully supervised setting. 

We then compare COLA to prior self-supervised methods proposed in [16, 25], including a standard triplet loss, AUDIO2VEC (CBoW and SG) and temporal gap prediction models. Here, the CBoW and SG are generative models inspired from WORD2VEC, trained to reconstruct a randomly selected temporal slice of log-mel spectrograms given the rest or vice versa. Likewise, TemporalGap trains a model to predict the temporal distance between two pairs of audio segments. Ta-

<!-- page: 4 -->

**Table 2** . Test accuracy (%) of a linear classifier trained on top of COLA embeddings or baseline pre-trained representations. 

||**CBoW [16, 25]**|**SG [16, 25]**|**TemporalGap [16, 25]**|**Triplet Loss [16, 25]**|**TRILL [13]**|**COLA**|
|---|---|---|---|---|---|---|
|**Speaker Id. (LBS)**|99.0|**100.0**|97.0|**100.0**|-|**100.0**|
|**Speech commands (V2)**|30.0|28.0|23.0|18.0|-|**62.4**|
|**Acoustic scenes**|66.0|67.0|63.0|73.0|-|**94.0**|
|**Birdsong detection**|71.0|69.0|71.0|73.0|-|**77.0**|
|**Music, Speech and Noise**|98.0|98.0|97.0|97.0|-|**99.1**|
|**Music instrument**|33.5|34.4|35.1|25.7|-|**63.4**|
|**Speech commands (V1)**|-|-|-|-|**74.0**|71.7|
|**Speaker Id. (Voxceleb)**|-|-|-|-|17.7|**29.9**|
|**Language Id.**|-|-|-|-|**88.1**|71.3|
|**Average (TRILL tasks)**|-|-|-|-|**59.9**|57.6|
|**Average (non-TRILL)**|66.25|66.0|64.3|64.4|-|**82.5**|



**Table 3** . Test accuracy (%) with different similarity functions. 

||**Cosine Similarity**|**Bilinear Similarity**|
|---|---|---|
|**Speaker Id. (LBS)**|99.9|**100.0**|
|**Speech commands (V1)**|64.5|**71.7**|
|**Speech commands (V2)**|42.4|**62.4**|
|**Acoustic scenes**|87.5|**94.1**|
|**Speaker Id. (Voxceleb)**|15.2|**29.9**|
|**Birdsong detection**|76.5|**77.0**|
|**Music, Speech and Noise**|99.0|**99.1**|
|**Language Id.**|62.3|**71.3**|
|**Music instrument**|58.3|**63.4**|
|**Average**|67.2|**74.3**|



**Table 4** . Impact of pre-training batch size on downstream test accuracy (%), using a bilinear similarity. 

||**256**|**512**|**1024**|**2048**|
|---|---|---|---|---|
|**Speaker Id. (LBS)**|99.9|99.9|**100.0**|99.9|
|**Speech commands (V1)**|66.9|69.4|71.7|**72.9**|
|**Speech commands (V2)**|44.4|54.2|62.4|**64.2**|
|**Acoustic scenes**|86.4|90.7|**94.1**|90.2|
|**Speaker Id. (Voxceleb)**|17.6|21.6|**29.9**|22.8|
|**Birdsong detection**|75.9|76.9|**77.0**|76.4|
|**Music, Speech and Noise**|98.8|**99.1**|**99.1**|98.8|
|**Language Id.**|65.6|64.0|**71.3**|68.4|
|**Music instrument**|62.3|57.3|**63.4**|56.6|
|**Average**|68.6|70.3|**74.3**|72.2|



ble 2 shows that COLA embeddings consistently outperform all these methods. In particular, on acoustic scene classification, we obtain a competitive accuracy of 94% compared to 73% achieved with a triplet loss in [16]. We also considerably improve the performance on speech commands and musical instrument classification by an absolute 30% margin on both tasks. We also compare with the recent self-supervised learning framework TRILL [13] on three speech-related tasks, benchmarking against TRILL-19 (the best self-supervised system of [13]). Our general-purpose COLA embeddings are competitive with TRILL, despite the fact that TRILL is pre-trained specifically on the part of Audioset that contains speech, and is evaluated only across speech tasks, while we train and evaluate COLA across speech, music, acoustic scenes, and animal sounds. 

To investigate the role of the similarity measure in the quality of learned representations, we perform an ablation study to compare model pre-training with cosine and bilinear similarity. With the cosine similarity, we use a temperature _τ_ = 0 _._ 2 to normalize the scores before computing the loss. Table 3 reports the results obtained on downstream classifiers using encoders pre-trained with each of the similarity estima- 

tion techniques. We observe that the best results are obtained using bilinear similarity in all cases. We also conduct an experiment to measure the impact of pre-training batch size, as larger batch sizes result in more negative samples and facilitate convergence [7]. Table 4 shows that, on average, a batch size as large as 1024 provides better representations compared to smaller ones. However, increasing the batch size up to 2048 worsens the performance in most cases. 

## **4. CONCLUSIONS** 

We introduce COLA , a simple, easy-to-implement, selfsupervised contrastive algorithm for general-purpose audio representation learning. Our approach achieves remarkable performance improvements over earlier unsupervised methods on a wide variety of challenging downstream tasks in a linear evaluation protocol as well as significantly improves results over supervised baselines through fine-tuning. We believe that the simplicity of our system, combined with its strong transferability across audio tasks, will pose it as a go-to baseline for future work in self-supervised learning for audio.

<!-- page: 5 -->

## **5. REFERENCES** 

- [1] J.B Grill, F. Strub, F. Altch´e, C. Tallec, P.H. Richemond, E. Buchatskaya, C. Doersch, B.A. Pires, Z.D. Guo, M. G. Azar, et al., “Bootstrap your own latent: A new approach to selfsupervised learning,” _arXiv preprint arXiv:2006.07733_ , 2020. 

- [2] A. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” _arXiv preprint arXiv:1807.03748_ , 2018. 

- [3] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “Bert: Pretraining of deep bidirectional transformers for language understanding,” _arXiv preprint arXiv:1810.04805_ , 2018. 

- [4] A. Owens and A.A. Efros, “Audio-visual scene analysis with self-supervised multisensory features,” in _ECCV_ , 2018. 

- [5] K. Kawakami, L. Wang, C. Dyer, P. Blunsom, et al., “Learning robust and multilingual speech representations,” _arXiv preprint arXiv:2001.11128_ , 2020. 

- [6] A. Jansen, M. Plakal, R. Pandya, D.P.W. Ellis, S. Hershey, J. Liu, R.C. Moore, and R.A. Saurous, “Unsupervised learning of semantic audio representations,” in _2018 Proceedings of the ICASSP_ . IEEE, 2018. 

- [7] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” _arXiv preprint arXiv:2002.05709_ , 2020. 

- [8] Y.A. Chung, C.C. Wu, C.H. Shen, H.Y Lee, and L.S. Lee, “Audio word2vec: Unsupervised learning of audio segment representations using sequence-to-sequence autoencoder,” _arXiv preprint arXiv:1603.00982_ , 2016. 

- [9] O. Plchot, L. Burget, H. Aronowitz, and P. Matejka, “Audio enhancing with dnn autoencoder for speaker recognition,” in _2016 Proceedings of the ICASSP_ . IEEE, 2016. 

- [10] M. Meyer, J. Beutel, and L. Thiele, “Unsupervised feature learning for audio analysis,” _arXiv preprint arXiv:1712.03835_ , 2017. 

- [11] V. Wan, Y. Agiomyrgiannakis, H. Silen, and J. Vit, “Google’s next-generation real-time unit-selection synthesizer using sequence-to-sequence lstm-based autoencoders.,” in _INTERSPEECH_ , 2017, pp. 1143–1147. 

- [12] S. Pascual, M. Ravanelli, J. Serr`a, A. Bonafonte, and Y. Bengio, “Learning problem-agnostic speech representations from multiple self-supervised tasks,” _arXiv preprint arXiv:1904.03416_ , 2019. 

- [13] J. Shor, A. Jansen, R. Maor, O. Lang, O. Tuval, F. de Chaumont Quitry Quitry, M. Tagliasacchi, I. Shavitt, D. Emanuel, and Y. Haviv, “Towards learning a universal non-semantic representation of speech,” _arXiv preprint arXiv:2002.12764_ , 2020. 

- [14] J. F. Gemmeke, D.P.W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R.C. Moore, M. Plakal, and M. Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in _ICASSP_ . IEEE, 2017. 

- [15] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” _arXiv preprint arXiv:1301.3781_ , 2013. 

- [16] M. Tagliasacchi, B. Gfeller, F. de Chaumont Quitry, and D. Roblek, “Self-supervised audio representation learning for mobile devices,” _arXiv preprint arXiv:1905.11796_ , 2019. 

- [17] A. Baevski, S. Schneider, and M. Auli, “vq-wav2vec: Selfsupervised learning of discrete speech representations,” _arXiv preprint arXiv:1910.05453_ , 2019. 

- [18] M. Rivi`ere, A. Joulin, P.E. Mazar´e, and E. Dupoux, “Unsupervised pretraining transfers well across languages,” in _ICASSP_ . IEEE, 2020. 

- [19] A. Baevski, H. Zhou, A.R. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” _arXiv preprint arXiv:2006.11477_ , 2020. 

- [20] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in _Proceedings of the IEEE/CVF CVPR_ , 2020. 

- [21] A. Srinivas, M. Laskin, and P. Abbeel, “Curl: Contrastive unsupervised representations for reinforcement learning,” _arXiv preprint arXiv:2004.04136_ , 2020. 

- [22] E. Kharitonov, M. Rivi`ere, G. Synnaeve, Lior Wolf, P.E. Mazar´e, M. Douze, and E. Dupoux, “Data augmenting contrastive learning of speech representations in the time domain,” _arXiv preprint arXiv:2007.00991_ , 2020. 

- [23] Z. Wu, Y. Xiong, S. Yu, and D. Lin, “Unsupervised feature learning via non-parametric instance-level discrimination,” _arXiv preprint arXiv:1805.01978_ , 2018. 

- [24] X. Wang and A. Gupta, “Unsupervised learning of visual representations using videos,” in _ICCV_ , 2015. 

- [25] M. Tagliasacchi, B. Gfeller, F. de Chaumont Quitry, and D. Roblek, “Pre-training audio representations with selfsupervision,” _IEEE Signal Processing Letters_ , 2020. 

- [26] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an asr corpus based on public domain audio books,” in _ICASSP_ , 2015. 

- [27] A. Nagrani, J.S. Chung, and A. Zisserman, “Voxceleb: a large-scale speaker identification dataset,” _arXiv preprint arXiv:1706.08612_ , 2017. 

- [28] P. Warden, “Speech commands: A dataset for limited-vocabulary speech recognition,” _arXiv preprint arXiv:1804.03209_ , 2018. 

- [29] T. Heittola, A. Mesaros, and T. Virtanen, “Tut urban acoustic scenes 2018, development dataset,” April 2018. 

- [30] D. Stowell, M.D. Wood, H. Pamuła, Y. Stylianou, and H. Glotin, “Automatic acoustic detection of birds through deep learning: the first bird audio detection challenge,” _Methods in Ecology and Evolution_ , 2019. 

- [31] D. Snyder, G. Chen, and D. Povey, “Musan: A music, speech, and noise corpus,” _arXiv preprint arXiv:1510.08484_ , 2015. 

- [32] J. Engel, C. Resnick, et al., “Neural audio synthesis of musical notes with wavenet autoencoders,” in _ICML_ . PMLR, 2017, pp. 1068–1077. 

- [33] K. MacLean, “Voxforge,” _Ken MacLean.[Online]. Available: http://www. voxforge. org/home.[Acedido em 2012]_ , 2018. 

- [34] M. Tan and Q.V Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” _arXiv preprint arXiv:1905.11946_ , 2019. 

- [35] J.L Ba, J.R. Kiros, and G.E. Hinton, “Layer normalization,” _arXiv preprint arXiv:1607.06450_ , 2016. 

- [36] D. P Kingma and J. Ba, “Adam: A method for stochastic optimization,” _arXiv preprint arXiv:1412.6980_ , 2014.
