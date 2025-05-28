---
bookToc: true
weight: 1
---

# XC-CACHE: Cross-Attending to Cached Context for Efficient LLM Inference
Submitted on 23 Apr 2024 by João Monteiro1, Étienne Marcotte1,*, Pierre-André Noël1,*, Valentina Zantedeschi1,*, David Vázquez1, Nicolas Chapados1, 2, Christopher Pal1, 2, Perouz Taslakian11, ServiceNow Research.
Posted by HyunDong Kim, Sangil Han

## Introduction
With the rapid advancement of deep learning technologies, high-performance deep neural networks (DNNs) have been extensively employed across diverse domains, including image classification, object detection, and natural language processing [1-2]. Prominent architectures such as VGGNet [3], ResNet [4], and Vision Transformer (ViT) [5] have demonstrated remarkable performance on various tasks. However, as the depth and width of these models increase, there is a corresponding substantial growth in parameter count and computational complexity. For example, ResNet-50 requires over 3.8 GFLOPs for processing a single image, whereas VGG-19 demands approximately 19.7 GFLOPs and occupies about 548 MB of memory for storing parameters [6].
<p align="center">
    <img src='./VGG-19 and ResNet34.png' width="650">
    <br>
    <i>Figure 1.  VGG-19 [3] and ResNet-34 [4]</i>
</p>
Contemporary vision models and large language models (LLMs) now comprise tens to hundreds of billions of parameters, necessitating hundreds of gigabytes of memory and over 10 TFLOPs of computational throughput per second at FP16 precision. These considerable computational requirements result in prohibitively high training costs, even within data centers equipped with advanced GPUs. Furthermore, edge devices such as mobile phones, IoT sensors, and wearable devices have limited DRAM capacity, constrained computational power, and insufficient battery life, complicating the efficient deployment of large-scale models. Consequently, these devices often experience increased inference latency and heightened heat and power consumption, limiting their usability in real-time applications. Therefore, model compression strategies that effectively reduce model size and computational complexity, while preserving performance, are crucial. Model compression techniques have emerged as essential methods for realizing lightweight deep learning models and facilitating practical deployment [3].

Model compression methods aim to optimize deep learning model efficiency by reducing memory consumption, storage requirements, and computational demands. Additionally, decreasing model complexity through compression can mitigate overfitting, reduce training time and energy consumption [7], and shorten inference latency. Among the various compression strategies, pruning has become the most extensively studied and widely applied approach. Pruning involves introducing sparsity by identifying and removing less critical parameters within neural networks. Initially, magnitude-based pruning, which selects parameters for removal based solely on their magnitude, was predominantly employed [8]. Recently, more sophisticated mathematical techniques, including gradient-based methods and sensitivity analyses utilizing Hessian matrices, have emerged [10–12].


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