---
layout: default
title: ""          # ← 비우면 page.title이 사라져 site.title을 사용
---

## 1. Introduction
With the rapid advancement of deep learning technologies, high-performance deep neural networks (DNNs) have been extensively employed across diverse domains, including image classification, object detection, and natural language processing [1-2]. Prominent architectures such as VGGNet [3], ResNet [4], and Vision Transformer (ViT) [5] have demonstrated remarkable performance on various tasks. **However, as the depth and width of these models increase, there is a corresponding substantial growth in parameter count and computational complexity.** For example, ResNet-50 requires over 3.8 GFLOPs for processing a single image, whereas VGG-19 demands approximately 19.7 GFLOPs and occupies about 548 MB of memory for storing parameters [6].
<p align="center">
    <img src='./Fig1.VGG-19 and ResNet-34.png' width="500">
    <br>
    <i>Figure 1.  VGG-19 [3] and ResNet-34 [4]</i>
</p>
<p align="center">
    <img src='./Fig2. ViT.png' width="650">
    <br>
    <i>Figure 2. Vision Trasnsformer (ViT) [5]</i>
</p>

Contemporary vision models and large language models (LLMs) now comprise tens to hundreds of billions of parameters, necessitating hundreds of gigabytes of memory and over 10 TFLOPs of computational throughput per second at FP16 precision. These considerable computational requirements result in prohibitively high training costs, even within data centers equipped with advanced GPUs. Furthermore, **edge devices such as mobile phones, IoT sensors, and wearable devices have limited DRAM capacity, constrained computational power, and insufficient battery life, complicating the efficient deployment of large-scale models.** Consequently, these devices often experience increased inference latency and heightened heat and power consumption, limiting their usability in real-time applications. Therefore, model compression strategies that effectively reduce model size and computational complexity, while preserving performance, are crucial. Model compression techniques have emerged as essential methods for realizing lightweight deep learning models and facilitating practical deployment [3].

**Model compression methods aim to optimize deep learning model efficiency by reducing memory consumption, storage requirements, and computational demands.** Additionally, decreasing model complexity through compression can mitigate overfitting, reduce training time and energy consumption [7], and shorten inference latency. Among the various compression strategies, pruning has become the most extensively studied and widely applied approach. **Pruning involves introducing sparsity by identifying and removing less critical parameters within neural networks.** Initially, magnitude-based pruning, which selects parameters for removal based solely on their magnitude, was predominantly employed [8]. Recently, more sophisticated mathematical techniques, including gradient-based methods and sensitivity analyses utilizing Hessian matrices, have emerged [10–12].
<p align="center">
    <img src='./Fig3. Unstructured pruning.png' width="500">
    <br>
    <i>Figure 3. Unstructured Pruning</i>
</p>
<p align="center">
    <img src='./Fig4. Structured pruning.png' width="850">
    <br>
    <i>Figure 4. Structured Pruning [26] </i>
</p>

**Pruning techniques can generally be classified into unstructured pruning (Fig. 3) and structured pruning (Fig. 4)**, based on the granularity of the units removed. **Unstructured pruning** introduces sparsity at the individual weight level, significantly reducing parameter count. However, from a hardware implementation perspective, it encounters challenges in translating this sparsity into actual reductions in multiply-accumulate (MAC) operations or inference speed improvements due to irregular memory access patterns and computational structures [8, 13]. In contrast, **structured pruning** removes coarse-grained components—such as filters, channels, or even entire layers—thereby facilitating efficient execution on hardware accelerators and parallel computing platforms. Thus, structured pruning is better suited for enhancing inference speed and energy efficiency in practical deployment environments [14, 15]. Nonetheless, structured pruning is typically more susceptible to performance degradation because of the relatively larger granularity of the removed units [14, 16].

**Despite these challenges, recent advancements in structured pruning have significantly progressed.** Techniques such as network slimming, which utilize the scaling parameters in Batch Normalization layers, have been introduced [17]. Several studies have also demonstrated methods to minimize accuracy loss from structured pruning. Additionally, combining intelligent fine-tuning with knowledge distillation after pruning has occasionally resulted in performance surpassing that of the original model [18, 19]. These developments indicate that structured pruning is gradually overcoming the traditional trade-off between accuracy and computational efficiency.

The efficacy of pruning largely hinges on the choice of importance metrics, which assess the significance of parameters or structural components targeted for removal [8, 10]. **Importance metrics are critical for accurately identifying less essential parameters or structures, thereby maximizing compression efficiency while minimizing performance degradation [20].** Various importance metrics proposed in the literature include magnitude-based metrics that rely on parameter absolute values [21], gradient-based metrics that leverage gradient information from the loss function [22], Hessian-based metrics that utilize second-order derivatives from the Hessian matrix [12], and activation sensitivity metrics reflecting the sensitivity of activation values [9]. The effectiveness of each metric can vary depending on specific model architectures or application scenarios, and a consensus regarding a universally optimal metric has not yet been reached.

**Considerable research has been conducted on evaluating pruning importance in convolutional neural networks (CNNs)**. For instance, Molchanov et al. [10] comparatively analyzed pruning performance using L2 norm-based and gradient-based metrics across various CNN models (Tab. 1). However, there remains a noticeable research gap regarding the generalization and performance validation of these importance evaluation metrics in the recently emerging Transformer architecture.

<p align="center">
    <img src='./Table 1.png' width="850">
    <br>
    <i>Table 1. Evaluation of Pruning Performance Using L2 Norm Importance Metric Across Multiple CNN Architectures [10]</i>
</p>

**This gap arises primarily due to structural differences between Transformers and CNNs.** Transformers possess unique architectural characteristics distinct from traditional convolutional and recurrent networks, notably their attention mechanisms, as depicted in Figure 5. Consequently, importance evaluation methods developed for CNN or RNN-based pruning may exhibit limited effectiveness when directly applied to Transformers [23]. In particular, there is a significant need for further research on developing new metrics specifically for evaluating component importance within multi-head attention and feed-forward networks and conducting systematic performance analyses [24].

<p align="center">
    <img src='./Figure 5. The architectures - transformer and multi-head attention.png' width="650">
    <br>
    <i>Figure 5. The Architectures - Transformer and Multi-Head Attention</i>
</p>

To address these limitations, **this study systematically investigates the behavior of identical pruning metrics across different model architectures.** Specifically, four representative importance metrics—L1 norm, L2 norm, Random, and Taylor expansion—were selected and applied to two structurally distinct model families: Vision Transformer (ViT) and ResNet. The primary goal of this study is to quantitatively assess how the selected importance metrics impact pruning performance within each architecture and to elucidate how the evaluation outcomes and pruning effectiveness vary when the same metrics are employed across different models. The results from this comparative analysis will provide a fundamental framework for establishing optimized pruning strategies tailored to specific model architectures.

Additionally, **sparsity ratios were set at levels comparable to widely adopted precision-based quantization formats in industry (FP16, INT8, INT4)**. Specifically, sparsity levels were adjusted to 25%, 50%, and 75% to reflect the lightweight effects achievable by each quantization format. While direct comparisons between pruning and quantization are beyond the scope of this research, the experimental design serves as groundwork for future integrated analyses of these techniques. Detailed descriptions of the sparsity conditions and experimental setup are presented in Section 3.1, *Experiment Setup*.


## 2. Related Work
### 2.1. CNN Model - ResNet

In this study, we employed **ResNet-56** as a convolution-based model. ResNet, initially proposed by He et al. [4], is widely recognized **for effectively mitigating the gradient vanishing problem inherent in deep neural networks and significantly enhancing training stability through residual connections.** These connections simplify the function space the network must learn by introducing a shortcut path that directly transmits input x to the subsequent block, effectively representing the learned function as  $$F(x) + x$$ (Fig. 6).

<p align="center">
    <img src='./Figure 6. Basic Residual Connection in ResNet .png' width="400">
    <br>
    <i>Figure 6. Basic Residual Connection in ResNet</i>
</p>

Figure 6 visually represents the hierarchical structure of ResNet-34, illustrating **the arrangement of residual blocks and shortcut connections.** ResNet-56, utilized in this research, follows a similar architecture optimized for the CIFAR dataset series. Due to its clear structural definition and uniform block operations, ResNet-56 is particularly suitable for in-depth architectural-level analyses of the role and influence of pruning importance metrics. Thus, ResNet-56 serves as a primary subject to quantitatively evaluate the impact of various importance metrics on pruning performance.

### 2.2. Vision Transformer (ViT) 
The **Vision Transformer (ViT)** model, introduced by Dosovitskiy et al. [5], is employed as another primary experimental subject. **ViT segments images into two-dimensional patches (typically 16×16 pixels), which are transformed into one-dimensional token sequences** processed by a Transformer encoder originally designed for natural language processing. This method leverages **multi-head self-attention (MHSA)** to learn global contextual relationships within images while minimizing inductive biases related to local receptive fields and spatial invariances found in CNNs.

As illustrated in Figure 2, ViT incorporates a class token ([CLS]) analogous to sentence classification tokens, positioned at the sequence's beginning to represent the entire image, alongside weighted positional embeddings to encode spatial information. For this study, we adopted a ViT model pretrained on the CIFAR dataset. The repetitive and uniform module structure—**comprising tokenization, self-attention, and residual connections**—of the Transformer architecture facilitates hierarchical analysis of importance metrics across subcomponents (heads, channels, blocks) and their effects on structural pruning. Consequently, both ResNet-56 and ViT models were selected to quantitatively compare the effects of diverse importance metrics on pruning performance.

### 2.3. Channel-Wise Pruning in CNN
**Channel-wise structured pruning involves the removal of entire output channels from convolutional layers, including associated filters and feature maps.** This approach effectively reduces computational complexity and parameters while preserving the hierarchical network structure. Figure 8 visually illustrates channel-wise pruning, highlighting the method of selectively removing less critical channels through importance evaluation, thus enabling model lightweighting.

<p align="center">
    <img src='./Figure 7. Channel-Wise Structured Pruning.png' width="850">
    <br>
    <i>Figure 7. Channel-Wise Structured Pruning</i>
</p>

**Channel-wise pruning contrasts notably with unstructured pruning, which achieves high compression through weight-level sparsity but faces limitations due to inefficient hardware implementation resulting from sparse matrices and irregular memory access.** Conversely, channel-wise pruning maintains model structural integrity, with clearly defined pruning units, rendering it highly suitable for parallel hardware execution and offering simultaneous storage and computational efficiency [27].
Additionally, studies [14,15] empirically demonstrated **the effectiveness of channel-based pruning in enhancing inference speed and energy efficiency within real-world device contexts.** Experiments across multiple CNN models revealed that channel-wise structured pruning exhibits lower overhead, improved hardware compatibility, and practical performance gains compared to unstructured sparsity pruning methods.

### 2.4. Modular Structured Pruning in ViT: Attention Heads and FFN Neurons 
Concurrently, structural pruning methodologies, analogous to the channel-wise pruning described in Section 2.3, are actively investigated within Transformer-based models. Vision Transformer (ViT), in particular, facilitates **the application of distinct pruning strategies tailored to each module** due to its modular architecture, comprising self-attention and feed-forward network (FFN) modules arranged in parallel.
**Attention head pruning** refers to the removal of entire individual heads within the multi-head self-attention mechanism. An attention head consists of query (Q), key (K), value (V), and output (O) weight matrices, along with the corresponding -dimensional output space. Heads exhibiting relatively lower importance are identified through importance evaluation metrics and sequentially pruned. Removal of a head involves simultaneous deletion of corresponding columns from the Q, K, and V matrices, as well as rows from the O matrix, thus reducing the concatenated projection dimension by . Consequently, this results in a linear reduction in FLOPs and parameters, and alleviates memory bandwidth constraints due to fewer parallel head executions.

<p align="center">
    <img src='./Figure 8. FFN and head pruning.png' width="650">
    <br>
    <i>Figure 8. FFN and Head Pruning [29]</i>
</p>

**FFN neuron pruning** is a channel-wise structured pruning technique aimed at removing individual neurons constituting the FFN’s internal expansion dimension, . As depicted in Figure 8, pruning a neuron concurrently removes the corresponding row from the input projection matrix  and the corresponding column from the output projection matrix . This pruning reduces the width of the FFN while preserving the tensor dimensions of adjacent layers. Importance is evaluated based on various criteria, including the L1 or L2 norm averages of activations per token or the Fisher information. Uniform neuron pruning across all blocks maintains consistent tensor dimensions throughout the model, simplifying implementation and enhancing computational efficiency.

Unlike unstructured pruning, **structured pruning provides significant hardware advantages.** Unstructured pruning achieves high compression by setting individual weights to zero but often lacks performance gains without specialized GEMM kernels due to overhead from sparse matrix storage and indexing during inference. Conversely, structured pruning of attention heads and neurons directly reduces computation and memory usage by decreasing tensor dimensions, thereby leveraging standard dense GEMM operations. This consistently improves batch processing latency, reduces power consumption and heat generation, and allows easier performance prediction and optimization, particularly in mobile GPUs and AI accelerator environments. Additionally, preserving structural integrity enhances training stability during subsequent fine-tuning or knowledge distillation phases. Thus, leveraging the parallel modular characteristics of ViT through structured pruning optimizes hardware efficiency and offers practical advantages over unstructured pruning methods, which primarily generate sparse weight matrices.

### 2.5. Importance Metrics
**An importance metric assesses the relative significance of each parameter or structural unit during pruning, critically balancing model compression and performance retention by identifying and removing redundant components**. Selecting an appropriate importance metric directly influences the minimization of accuracy degradation and maximization of computational efficiency post-pruning.

#### 2.5.1. L1 Norm-based Metric 
L1 norm-based methods **quantify unit importance by computing the absolute sum of weight vectors associated with each channel, neuron, or attention head.** The L1 norm metric is mathematically defined as follows:

$$ 
l_1 = |x_1| + |x_2| + \cdots + |x_n|. 
$$

Generally, **smaller L1 norm values imply a lower contribution to the model output, thus identifying them as targets for pruning**. Li et al. [30] proposed a pruning approach for CNNs that effectively reduces computational complexity by removing filters with low L1 norm values while minimizing accuracy degradation. Subsequently, this method has been successfully extended to various model structures, including attention heads and hidden units within Transformer models' MLP blocks. The advantage of this method is that it can evaluate importance using only pretrained weight information, eliminating the need for additional training or gradient computations [31]. Specifically, the L1 norm naturally aligns with structured pruning methods such as channel-wise and head-wise pruning, providing structural consistency. This alignment avoids complex sparse mask management on hardware and achieves practical reductions in computational cost and enhanced inference speed. Consequently, the L1 norm has become a widely adopted importance evaluation metric, combining practicality and efficiency [23].

However, **the L1 norm may yield relatively conservative pruning results** because it does not consider input data or dynamic gradient information [20]. To address this limitation, more sophisticated importance evaluation techniques, such as gradient-based and Hessian-based methods, are actively being explored.

#### 2.5.2. L2 Norm-based Metric 
The L2 norm-based method quantifies the importance of structural units such as convolutional channels, MLP units, or attention heads by **computing the sum of squares of their corresponding weight vectors (i.e., the L2 norm).** This method is formally defined as follows:

$$ 
l_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}. 
$$

In general, weight vectors characterized **by small L2 norm values are targeted for pruning due to their minimal contribution to the model output**. Compared to the L1 norm, the L2 norm imposes a relatively higher penalty on larger weights, resulting in stronger suppression of smaller weight values. This property becomes particularly prominent when weight distributions are broad or contain outliers, typically leading to more conservative pruning outcomes than those obtained using the L1 norm [32]. He et al. [33] demonstrated pruning using an L2 norm-based metric to evaluate filter importance in residual networks. Their findings suggest that this approach can be generalized effectively across diverse architectures, including multi-head attention and MLP blocks within attention-based models as well as residual networks.

**The L2 norm-based evaluation approach is practically advantageous due to its low computational complexity and the possibility of immediate pruning using solely pretrained weights.** Moreover, it integrates easily with structural pruning techniques, thereby being extensively utilized in practical model compression tasks. Nevertheless, similar to the L1 norm approach, the L2 norm-based method has inherent limitations in terms of precise pruning due to the exclusion of input data and gradient information, thereby neglecting learning dynamics [20].

#### 2.5.3. Taylor Expansion-based Metric
Pruning based on Taylor expansion offers a theoretically grounded approach **to estimating sensitivity concerning model performance and assessing pruning importance, rendering it a more sophisticated and interpretable pruning strategy.** This method evaluates pruning importance by approximating the change in the loss function upon removal of specific parameters or structural units via first- or second-order Taylor expansions. Molchanov et al. [1] introduced a representative first-order Taylor-based pruning technique, which estimates the change in the loss function  attributable to removing a specific weight  using the first derivative. The corresponding importance metric is thus formally defined:

$$ 
I(w) = \left| \frac{\partial \mathcal{L}}{\partial w} \cdot w \right|. 
$$

The term $$\frac{\partial \mathcal{L}}{\partial w}$$ represents the gradient of the loss function with respect to the weight $$w_{i}$$. By multiplying this gradient by the corresponding weight, the quantitative impact of pruning that weight on the model performance can be estimated. In contrast to L1 or L2 norm-based methods, **Taylor-based approaches beneficially incorporate gradient information, thereby reflecting input data characteristics and learning dynamics.** LeCun et al.'s *Optimal Brain Damage* [11] and Hassibi & Stork's *Optimal Brain Surgeon* [33] utilized second-order Taylor expansions involving Hessian matrix calculations for more precise importance assessment. Despite their accuracy, these methods are computationally intensive and challenging to implement, leading to wider practical adoption of simpler first-order gradient-weight multiplication approximations.

Taylor-based pruning methods are effective not only for finely-tuned high-performance models but also for evaluating the importance of Transformer attention heads [31]. These techniques are particularly praised for their accurate prediction of pruning outcomes at relatively low computational cost compared to fully learning-based importance metrics. However, **the mandatory gradient computations imply that pruning cannot be executed solely with pretrained weights, posing a significant practicality trade-off compared to L1 or L2 norm-based methods.**

#### 2.5.4. Random Metric
Random pruning is the simplest pruning strategy, **selecting and removing target units entirely at random, independent of any importance metrics.** This approach disregards weight magnitude, gradient information, or input characteristics, pruning weights, channels, or attention heads based solely on predefined sparsity ratios. Intuitively considered inefficient, random pruning nonetheless serves widely as a baseline for comparative analysis across numerous studies. 

$$ 
\text{Random(Unit)}. 
$$

Blalock et al. [20], in a systematic review of pruning methods, noted that **random pruning could yield surprisingly competitive performance under certain scenarios**. Particularly, even uninformed pruning approaches might attain accuracy levels comparable to those using explicit importance metrics after adequate fine-tuning [34]. Moreover, random pruning frequently functions as a baseline to evaluate whether specific importance metrics meaningfully contribute to pruning performance enhancement. For instance, Molchanov et al. [10] validated gradient-based pruning effectiveness through comparative analysis against randomly pruned models. Nonetheless, **random pruning may induce irregular structural changes, especially problematic in structured pruning environments, leading to computational unit imbalances and diminished hardware efficiency.**

## 3. Method
### 3.1. Experiment Setup
**Dataset.** The experiments were conducted using the CIFAR-100 dataset, which comprises a total of 60,000 32×32 RGB images distributed across 100 classes. Of these, 50,000 images were used for training, while the remaining 10,000 images were reserved for testing. As a standard benchmark frequently used to evaluate image classification performance across diverse categories, CIFAR-100 is well suited for the quantitative assessment of pruning performance in the ResNet and Vision Transformer (ViT) models adopted in this study. The importance scores used for pruning were calculated using a calibration dataset, which consisted of 3,000 images randomly sampled from the CIFAR-100 training set. This calibration set was constructed to ensure balanced representation across all classes, thereby minimizing bias during the pruning process and enabling the model to comprehensively learn a wide range of features.

**Importance Metric.** For pruning experiments, we adopted **ResNet-56 and ViT models pretrained on CIFAR-100 as the baseline architectures**. In the case of **ResNet-56**, channel-wise pruning was performed on the output channels of each convolutional layer. The importance of each channel was measured using four different metrics: L1-norm, L2-norm, gradient-based, and random. L1-norm and L2-norm were computed based on the sum of the absolute values and the sum of the squared weights, respectively, for each channel. For the gradient-based metric, the average gradient magnitude with respect to each channel was used as the importance score. The random metric served as a baseline, where channels were randomly selected for pruning. For the **Vision Transformer model**, structural characteristics necessitated separate pruning of the feed-forward network (FFN) and attention heads. L2-norm was consistently applied to the FFN pruning, as prior studies have shown robust performance using this metric [35]. The importance score for each FFN neuron was determined by the sum of the squares of its weights. For pruning attention heads, all four metrics (L1-norm, L2-norm, gradient-based, and random) were employed to facilitate comparative analysis. Specifically, the L1-norm and L2-norm for each attention head were calculated as the sum of the absolute values and the sum of the squared weights, respectively, of the corresponding weight matrices. The gradient-based metric computed the importance score by running forward and backward passes on the calibration dataset and measuring the gradient magnitude of the loss with respect to the weights of each head. The random metric assigned importance scores randomly across heads. Based on these importance scores, the pruning order for attention heads was determined. Both attention head and FFN pruning were performed by applying the above metrics in a single pass to compute importance scores simultaneously.

**Fine-Tuning.** After pruning, a fine-tuning phase was performed to minimize performance degradation and recover any losses incurred during pruning. To balance computational cost and convergence stability, the number of training epochs during fine-tuning was limited to 50. Preliminary experiments of Figure 9, indicated that extending training beyond 50 epochs yielded no significant improvement in loss, supporting this decision as a measure to prevent overfitting and maintain efficient training.

<p align="center">
    <img src='./Figure 9. Loss curves during 100 epochs of fine-tuning for two architectures.PNG' width="800">
    <br>
    <i>Figure 9. Loss Curves During 100 Epochs of Fine-Tuning for Two architectures</i>
</p>

During fine-tuning, the Adam optimizer was used with a learning rate of 1e-4 to ensure stable initial learning. Weight decay was set to 1e-3 to further prevent overfitting and effectively restore any loss in performance due to pruning.

**Evaluation.** Model performance was evaluated in terms of classification accuracy and the number of parameters. Accuracy was measured on the CIFAR-100 test set, while the number of parameters was determined by comparing the total parameter count before and after pruning. These clear evaluation criteria enabled a fair and comprehensive comparison of the pruning methods based on the four importance metrics examined in this study.

## 4. Results
### 4.1.  Channel-wise Structured Pruning in CNN
The ResNet-56 model pretrained on the CIFAR-100 dataset has approximately 0.826M parameters and 32.725 GFLOPs of computational cost (Table 2). In this study, three pruning ratios of 25%, 50%, and 75% were applied, reducing the parameters and FLOPs to 0.620M/26.673G, 0.443M/19.933G, and 0.327M/14.486G, respectively (Table 3). As the pruning ratio increased, significant reductions in both parameters and computational cost were observed, and performance was effectively recovered through fine-tuning after pruning in all cases.

<p align="center">
    <img src='./Table 2. Original ResNet-56 Model.png' width="800">
    <br>
    <i>Table 2. Original ResNet-56 Model </i>
</p>
<p align="center">
    <img src='./Table 3. Pruning Methods on the ResNet-56 Model at Various Pruning Ratio.png' width="800">
    <br>
    <i>Table 3. Pruning Methods on the ResNet-56 Model at Various Pruning Ratio</i>
</p>

**At a 25% pruning ratio**, the random metric exhibited the best performance prior to fine-tuning, with top-1 accuracy of 5.71% and top-5 accuracy of 16.12%. However, after fine-tuning, the Taylor expansion-based pruning method achieved the highest performance, with a top-1 accuracy of 71.50% and top-5 accuracy of 91.92%. L1 and L2 norm-based pruning methods showed similar performance levels. **At a 50% pruning ratio,** the L2 norm-based pruning slightly outperformed before fine-tuning with a top-1 accuracy of 1.22% and top-5 accuracy of 6.49%. Post fine-tuning, the Taylor expansion-based method showed the best performance with top-1 accuracy of 69.69% and top-5 accuracy of 91.17%, closely followed by random pruning with a top-5 accuracy of 91.45%. **At a 75% pruning ratio**, random pruning achieved the highest top-1 accuracy of 65.96% and top-5 accuracy of 89.63% after fine-tuning, although the performance difference compared to the Taylor expansion-based method was negligible (less than 0.1%). L1 and L2 norm-based models exhibited relatively lower accuracies.

Overall, **at low to moderate pruning rates, the Taylor expansion–based criterion—which leverages gradient information—demonstrated the greatest efficacy.** In residual networks such as ResNet, each block’s output is formed by the sum of a residual function F(x) and an identity shortcut, allowing critical information to bypass individual convolutional layers. Consequently, pruning based solely on simple magnitude metrics such as L1 or L2 norm risks eliminating parameters with small absolute values that nevertheless contribute substantially to loss reduction, thereby degrading overall performance. By contrast, the Taylor expansion–based method evaluates the first‐order approximation of the change in the loss function, preserving parameters with high loss sensitivity regardless of their magnitude and removing those with large magnitude but low sensitivity, thus maintaining the representational capacity of the residual architecture more finely. **At high pruning rates, however, the performance gap between different importance metrics narrows and even random selection yields competitive results.** This suggests that, under extreme pruning, structural damage to the network becomes the primary driver of performance degradation, thereby diminishing the relative influence of any individual importance criterion. After retraining, the performance gaps between metrics were virtually eliminated, underscoring that the refinement phase’s capacity to adjust the surviving parameters and locate superior minima is more pivotal to recovery than the initially selected importance criterion.

### 4.2. Modular Structured Pruning in ViT: Attention Heads and FFN Neurons
The Vision Transformer (ViT) model pretrained on the CIFAR-100 dataset comprises approximately 85.723 million parameters and requires 33.726 GFLOPs for inference, as shown in Table 4. In this study, we conducted experiments with pruning ratios set to 25%, 50%, and 75%. After pruning, the model sizes and computational costs were reduced to 64.474M/25.360G, 43.224M/16.994G, and 21.974M/8.628G, respectively (Table 5). As the pruning ratio increased, the model complexity was significantly reduced, while fine-tuning effectively restored model performance in most cases.

<p align="center">
    <img src='./Table 4. Original ViT Model.png' width="800">
    <br>
    <i>Table 4. Original ViT Model </i>
</p>

<p align="center">
    <img src='./Table 5. Pruning Methods on The ViT Model at Various Pruning Ratio.png' width="800">
    <br>
    <i>Table 5. Pruning Methods on The ViT Model at Various Pruning Ratio</i>
</p>

**For the 25% pruning scenario**, prior to fine-tuning, the Taylor expansion-based pruning method achieved the highest accuracy, recording a top-1 accuracy of 59.92% and a top-5 accuracy of 82.80%. After fine-tuning, the Taylor expansion method still yielded the best top-1 accuracy at 87.70%, while the random pruning baseline achieved the highest top-5 accuracy at 97.83%. Pruning methods based on L1 and L2 norms showed performance levels comparable to Taylor expansion and random pruning. **Under the 50% pruning condition**, the Taylor expansion-based method again exhibited strong performance before fine-tuning, achieving 21.72% top-1 and 44.17% top-5 accuracy. However, after fine-tuning, L1-norm-based pruning achieved the highest top-1 accuracy of 85.12%, while L2-norm-based pruning recorded the highest top-5 accuracy of 97.20%. **For the 75% pruning case**, Taylor expansion-based pruning continued to outperform other methods before fine-tuning (top-1: 4.01%, top-5: 13.07%). However, after fine-tuning, L2-norm-based pruning achieved the best performance, with a top-1 accuracy of 79.10% and a top-5 accuracy of 94.84%. Other importance-based pruning criteria showed comparatively lower accuracy in this setting.

**The exceptional performance of Taylor expansion-based pruning in Vision Transformer (ViT) models before fine-tuning is closely related to the unique architecture and operational principles of ViT.** Unlike convolutional neural networks (CNNs), which utilize local convolutional filters, ViT processes input data through self-attention mechanisms and feed-forward networks, distributing information globally across the entire input sequence. Due to this architectural property, magnitude-based pruning criteria such as L1 or L2 norm may indiscriminately remove parameters with small absolute values, even if those parameters are highly sensitive to the loss function. In ViT models, relationships captured by attention heads and feed-forward neurons can play a crucial role in maintaining model performance, even when the corresponding weights are small. As a result, simple magnitude-based pruning can inadvertently remove subtle yet essential components, leading to performance degradation. In contrast, Taylor expansion-based pruning leverages the first-order approximation of each parameter’s impact on the loss (i.e., gradient information), preserving parameters that, despite their small magnitude, significantly influence the loss function, while selectively removing those with minimal impact. By directly considering sensitivity to the loss, this approach is especially effective in ViT models, where global interactions and complex dependencies are fundamental to model performance, thus helping maintain the functional integrity of attention mechanisms and the model's representational power. **After fine-tuning, however, the differences in performance among various importance metrics became marginal**, suggesting that the ability of the fine-tuning process to effectively readjust the remaining weights and find new optima played a more critical role in performance recovery than the choice of metric itself.
### 4.3. Fine-tuning Time for Both Architectures
<p>&nbsp;</p>
<p align="center">
    <img src='./Figure 10. Training time on fine-tuning.jpg' width="500">
    <br>
    <i>Figure 10. Training Time on Fine-Tuning</i>
</p>

**Figure 10 presents a comparative analysis of the fine-tuning times for ResNet and ViT models under varying pruning ratios**. The results indicate a marked reduction in training time for the ViT model as the pruning ratio increases. In particular, the fine-tuning time for the ViT model decreased dramatically from approximately 350 seconds to below 150 seconds as the pruning ratio was raised from 25% to 75%. This demonstrates that, for the ViT model, increased pruning not only simplifies the model structure but also substantially reduces computational load and training time. In contrast, the ResNet model exhibited little to no change in fine-tuning time across different pruning ratios, consistently maintaining a training time of around 50 seconds. This discrepancy can be attributed to the inherently higher structural complexity of ViT compared to ResNet, which allows ViT to benefit more significantly from pruning in terms of computational efficiency. These findings confirm that, in addition to maintaining model performance, pruning can also effectively accelerate the training process for the ViT model. This highlights the potential of pruning as a practical approach for improving the training efficiency of large and complex models like ViT.

## 5. Discussion and Conclusion
### 5.1. Discussion
When pruning was applied to ResNet-56, the *Random* metric achieved the highest Top-1 accuracy (5.71%) at 25% sparsity, whereas the *Taylor* metric outperformed others at both 50% (1.22%) and 75% (4.01%) sparsity. This indicates that, in CNNs, the effectiveness of gradient-based importance (*Taylor*) becomes more pronounced as sparsity increases beyond a certain threshold. In the case of ViT, the *Taylor* metric consistently achieved the highest immediate post-pruning Top-1 accuracy across all sparsity levels (25%: 59.92%, 50%: 21.72%, 75%: 4.01%). This suggests that **the** **first-order Taylor approximation accurately captures the relative importance of self-attention heads**. **After fine-tuning**, *Taylor* continued to yield the highest Top-1 accuracy for ResNet at 25% and 50% sparsity (71.50%, 69.69%, respectively), but *Random* marginally outperformed *Taylor* at 75% sparsity (65.96%). This result implies that, when the convolutional layers are pruned to extreme levels, the gradient-based score can no longer sufficiently compensate for the structural damage. For ViT, the optimal metric differed by sparsity level: *L1* norm achieved the highest Top-1 accuracy at 25% and 50% sparsity (87.40%, 85.12%, respectively), while *L2* norm was the best at 75% (79.10%). This indicates that, as pruning becomes more aggressive in Transformers, simple norm-based metrics can match or even slightly surpass gradient-based metrics in performance.

According to Figure 10, **ViT experienced a substantial reduction in training time at 75% sparsity, dropping from approximately 350 seconds to 150 seconds (a decrease of ~57%**). The reductions in parameter count and FLOPs translated directly into faster training. In contrast, ResNet’s training time remained almost unchanged at around 50 seconds across the same range, likely because (i) the model is already lightweight and (ii) I/O bottlenecks and data augmentation overheads dominate, limiting the impact of reduced computation on training speed.

**Despite using the same metric, differences in model structure result in distinct patterns of accuracy degradation and recovery.** In CNNs, channel-wise redundancy is distributed relatively evenly, so even *Random* pruning can maintain reasonable performance. In contrast, ViT exhibits varying degrees of information concentration across heads and MLP blocks, making gradient-based metrics more effective. As the pruning ratio increases, structural damage becomes the primary factor in performance degradation, narrowing the gap between metrics. Notably, for ViT at 75% sparsity, the Top-1 accuracy difference between metrics converges to within 2 percentage points, indicating that the fine-tuning process effectively re-optimizes the remaining weights to compensate for the damage.

### 5.2. Conclusion
This study empirically investigated whether pruning metrics—originally developed and extensively studied for CNN-based models—exhibit similar effectiveness in Transformer architectures. Furthermore, by comparing the effects of different pruning metrics within Transformers, we aimed to highlight the need for differentiated pruning strategies tailored to model architectures. To this end, we systematically evaluated and compared four pruning metrics (L1, L2, Random, and Taylor) applied to both CNNs (ResNet-56) and Vision Transformers (ViT) at sparsity levels of 25%, 50%, and 75%, assessing model performance, training efficiency, and the effectiveness of each metric. The experimental results yield the following insights:

**The suitability of a metric is architecture-dependent:** In CNNs, the Taylor metric demonstrated competitive performance at moderate sparsity levels (≤50%), while the Random metric was most effective at extreme sparsity (75%). In contrast, for Transformers, the Taylor metric consistently outperformed others before fine-tuning, whereas norm-based metrics performed equally well or even better after fine-tuning. **fine-tuning diminishes the influence of the chosen metric**: as sparsity increases, the performance gap between metrics narrows, suggesting that fine-tuning effectively re-optimizes the remaining parameters to restore local minima. **The extent of computational and time efficiency gains also varies by architecture**: while ViT experienced a substantial reduction in training time due to decreased FLOPs following pruning, ResNet saw only limited gains, likely due to I/O and data augmentation bottlenecks.

These findings underscore the necessity of designing pruning metrics that are tailored to the structural characteristics of the model. In particular, for Transformer-based vision models such as ViT, there is a need for the development of importance evaluation metrics and pruning methods specialized for their unique architectures (e.g., head-wise or MLP-wise pruning). Future work should aim to develop unified and modular pruning frameworks that not only encompass both CNNs and Transformers but also account for the distinctive properties of each architecture.

## 6. Limitations
While this study compared and analyzed the effectiveness of pruning metrics across different model architectures (CNN and Transformer), thereby highlighting the need for architecture-specific pruning strategies, several limitations remain.

First, although ResNet and ViT were selected as representative models for their respective architectures, **the limited number of models** examined makes it difficult to directly generalize the findings to other CNN or Transformer architectures. In particular, the impact of pruning metrics on models with varying depths or widths requires further investigation. Second, the experiments were conducted using **fixed pruning ratios of 25%, 50%, and 75%**. As a result, this study did not analyze performance trends or threshold effects at more granular levels of sparsity. Additional experiments with a wider range of pruning ratios are necessary to more precisely assess the sensitivity of models to pruning. Third, all experiments were performed using **a single dataset (CIFAR-100)**. Thus, the study does not address how the effectiveness of pruning metrics may vary with dataset complexity or domain. Future research should extend the evaluation to a broader set of datasets in both visual and natural language domains to more thoroughly assess the generalizability and robustness of pruning metrics.

Based on these limitations, future studies should be designed to encompass diverse architectures, a wider range of pruning ratios, and multiple datasets, ultimately providing more generalizable guidelines for pruning strategies.

## References
[1] YUAN, T., LIU, W., HAN, J., and LOMBARDI, F. High performance CNN accelerators based on hardware and algorithm co-optimization. *IEEE Transactions on Circuits and Systems I: Regular Papers*, 2021, 68(1), pp. 250–263.

[2] BARINOV, R., GAI, V., KUZNETSOV, G., and GOLUBENKO, V. Automatic evaluation of neural network training results. *Computers*, 2023, 12(1), p. 26.

[3] WANG, Limin, et al. Places205-VGGNet models for scene recognition. *arXiv preprint arXiv:1508.01667*, 2015.

[4] HE, Kaiming, et al. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016. pp. 770–778.

[5] DOSOVITSKIY, Alexey, et al. An image is worth 16×16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

[6] LI, Zhuo, LI, Hengyi, and MENG, Lin. Model compression for deep neural networks: A survey. *Computers*, 2023, 12(3), p. 60.

[7] CHENG, Y., et al. A survey of model compression and acceleration for deep neural networks. *IEEE Signal Processing Magazine*, 2018, 35(1), pp. 126–136.

[8] HAN, S., et al. Learning both weights and connections for efficient neural networks. In: *Advances in Neural Information Processing Systems*. 2015. pp. 1135–1143.

[9] YU, R., et al. NISP: Pruning networks using neuron importance score propagation. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2018. pp. 9194–9203.

[10] MOLCHANOV, P., et al. Pruning convolutional neural networks for resource efficient inference. *arXiv preprint arXiv:1611.06440*, 2017.

[11] LECUN, Y., et al. Optimal brain damage. In: *Advances in Neural Information Processing Systems*. 1989. pp. 598–605.

[12] DONG, X., et al. Hessian-aware pruning and optimal neural implant. In: *Advances in Neural Information Processing Systems*. 2017.

[13] GALE, T., et al. The state of sparsity in deep neural networks. *arXiv preprint arXiv:1902.09574*, 2019.

[14] LI, H., et al. Pruning filters for efficient ConvNets. *arXiv preprint arXiv:1608.08710*, 2017.

[15] HE, Y., et al. AMC: AutoML for model compression and acceleration on mobile devices. In: *Proceedings of the European Conference on Computer Vision (ECCV)*. 2018. pp. 815–832.

[16] LUO, J.-H., et al. ThiNet: A filter level pruning method for deep neural network compression. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2017. pp. 5058–5066.

[17] LIU, Z., et al. Learning efficient convolutional networks through network slimming. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2017. pp. 2736–2744.

[18] FRANKLE, J., and CARBIN, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. *arXiv preprint arXiv:1803.03635*, 2019.

[19] HINTON, G., et al. Distilling the knowledge in a neural network. *NIPS Deep Learning Workshop*, 2014.

[20] BLALOCK, D., et al. What is the state of neural network pruning? *Proceedings of Machine Learning and Systems*, 2020, 2, pp. 129–146.

[21] ZHU, M., and GUPTA, S. To prune, or not to prune: Exploring the efficacy of pruning for model compression. *ICLR Workshop*, 2018.

[22] LEE, N., et al. SNIP: Single-shot network pruning based on connection sensitivity. In: *International Conference on Learning Representations (ICLR)*. 2019.

[23] WANG, B., et al. Structured pruning of large language models. In: *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. 2020.

[24] VASWANI, A., et al. Attention is all you need. In: *Advances in Neural Information Processing Systems*. 2017, 30.

[25] QIAN, H., et al. A fire monitoring and alarm system based on channel-wise pruned YOLOv3. *Multimedia Tools and Applications*, 2022, pp. 1–19.

[26] HE, Y., and XIAO, L. Structured pruning for deep convolutional neural networks: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2023, 46(5), pp. 2900–2919.

[28] WEN, W., et al. Learning structured sparsity in deep neural networks. In: *Advances in Neural Information Processing Systems*. 2016. pp. 2074–2082.

[29] LIU, Y., LIN, Z., and YUAN, F. Rosita: Refined BERT compression with integrated techniques. In: *Proceedings of the AAAI Conference on Artificial Intelligence*. 2021. pp. 8715–8722.

[30] LI, H., et al. Pruning filters for efficient ConvNets. In: *International Conference on Learning Representations (ICLR)*. 2017.

[31] MICHEL, P., LEVY, O., and NEUBIG, G. Are sixteen heads really better than one? In: *Advances in Neural Information Processing Systems*. 2019.

[32] HU, H., et al. Network trimming: A data-driven neuron pruning approach towards efficient deep architectures. In: *European Conference on Computer Vision (ECCV)*. 2016. pp. 408–424.

[33] HE, Y., et al. Channel pruning for accelerating very deep neural networks. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2017. pp. 1389–1397.

[33] HASSIBI, B., and STORK, D. G. Second order derivatives for network pruning: Optimal brain surgeon. In: *Advances in Neural Information Processing Systems*. 1993. pp. 164–171.

[34] LIU, Z., et al. Rethinking the value of network pruning. *arXiv preprint arXiv:1810.05270*, 2019.

[35] SANDRI, F., CUNEGATTI, E., and IACCA, G. 2SSP: A two-stage framework for structured pruning of LLMs. *arXiv preprint arXiv:2501.17771*, 2025.
