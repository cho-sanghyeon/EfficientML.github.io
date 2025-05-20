---
layout: post
title: "Model Compression for Efficient AI"
date: 2025-04-28
tags: [deep learning, model compression, edge computing]
categories: [blogposts]
author: "Your Name"
---

## 1. Introduction

With the rapid development of deep learning technology, high-performance deep neural networks (DNNs) have been actively utilized in various fields such as image classification, object detection, and natural language processing [1, 2]. In particular, models such as VGGNet [4], ResNet [5], and Vision Transformer (ViT) [6] have demonstrated excellent performance in applications such as image classification, object detection, and natural language understanding, but the number of parameters and computational effort increases rapidly as the depth and width of the model increases. For example, ResNet-50 requires more than 3.8 GFLOPs of computation to process a single image, and VGG-19 requires about 19.7 GFLOPs of computation and 548 MB of parameters [3].

Recent vision models and large language models (LLMs) contain tens to hundreds of billions of parameters, requiring hundreds of gigabytes of memory and more than 10 TFLOPs of computation per second at FP16. While these large-scale models work well in data centers and high-performance GPU environments, they are difficult to run directly on edge devices such as mobile devices, IoT sensors, and wearables due to limited DRAM capacity, low computational power, and insufficient battery capacity. The result is increased inference latency and spikes in heat and power consumption, which are limiting for applications that require real-time response. Therefore, a lightweighting strategy that reduces model size and computational complexity while minimizing performance degradation is essential, and model compression has emerged as a key technology for lightweighting and making deep learning practical [3].


