---
layout:     post
title:      "The Price of Sparsity"
subtitle:   "Generalization and Memorization in Sparse Neural Networks"
date:       2022-08-21
author:     "Ziyu Ye"
header-img: "img/post-bg-zhihu1.jpg"
catalog:    true
mathjax:    true
tags:
    - Deep Learning
    - Information
    - Sparsity
---
<!-- # The Price of Sparsity -->

Two things that i believe: learning comes from compression, and models should be sparse. In this post, we will talk about the latter, focusing on the training techniques for sparse models.

## Motivation
Training large models is costly. Making models sparse (*e.g.*, by setting certain weights to be constantly zero, which is effectively removing them) can accelerate both training and inference, saving a lot computation. We refer to this process as sparse training. Our objective is to **train sparse neural networks with matching generalization performance as their dense counterparts**, even at a very high sparsity level.

## Backgrounds
The above is a challenging task due to the fundamental tradeoff between efficiency and performance (*c.f.* the rate distortion tradeoff, [[Dai at al., 2018]](http://proceedings.mlr.press/v80/dai18d/dai18d.pdf) and [[Gao et al., 2019]](https://icml.cc/media/Slides/icml/2019/102(11-14-00)-11-15-15-4454-rate_distortion.pdf)). That being said, while higher sparsity level implies higher computational efficiency, it is often brings performance degradation.



<p align="center">
  <img src="/img/in-post/sc-v-ft.png" alt="Description" width="600">
</p>


<!-- Few things in the world are accomplished in one step.  -->
The most efficient approach is to directly train a sparse neural network from scratch (*a.k.a.*, `sparse scratch`, or pruning at initialization, as depicted by `Init-S -> Sol-S` above). However, in practice, it often results in relatively low generalization performance. People find that it is often better to first pre-train on a dense network, then to prune the trained solution, and finally to finetune on the resulting weights (*a.k.a.*, `sparse finetuning`, or pruning after training, as depicted by `Init-D -> Sol-D -> Init-F -> Sol-F`).


In [[Evci et al., 2020]](https://arxiv.org/pdf/1906.10732.pdf), the authors compare `sparse scratch` and `sparse finetuning`, and find that even with the same pruning mask (which means the only difference lies in their initializations, `Init-S` and `Init-F`), the former still perform worse than the latter. We verify this phenomenon as below on CIFAR-100 with ResNet32 (for now, you may ignore the two regimes; we will explain it later).

<p align="center">
  <img src="/img/in-post/perf-gap.png" alt="Description" width="600">
</p>

This is not good. Sparse finetuning is a tortuous path and is still very computationally expensive. Is there a way for us to get rid of the dense pre-training phase while still achieving comparable generalization performance?

Things are not desperate.

<p align="center">
  <img src="/img/in-post/linear-path.png" alt="Description" width="300">
</p>

In the above figure, we show that there exist a path, from `Init-S` to `Sol-F`, where the loss is almost monotonically decreasing (first discovered in [[Evci et al., 2020]](https://arxiv.org/pdf/1906.10732.pdf); the loss on the y-axis is computed by linearly interpolating between `Init-S` and `Sol-F`). This means that it is possible for `sparse scratch` to have the same results as `sparse finetuning`.

## The Quest

Before we can reach the same performance as dense training, let's first make `sparse scratch` being on par with `sparse finetuning`, which would already provide us with remarkable efficiency in practice.

We thus ask the following research questions:
- **What is the *root cause* for the performance gap between `sparse scratch` and `sparse finetuning`?**
- **How can we close the performance gap?**

For the first question, the secrets lies in the training dynamics. One of the most fundamental and general way to depict the training dynamics, from my own perspective, is by **information measures**, which is easy for humans to understand and help design principled training approach.  We will go in depth later, but as a teaser here, Fisher information is a local discrepancy measure for the prediction distribution of neural networks, and we found the **Fisher information for `sparse scratch` is much higher than that of `sparse finetuning`**. This simple metric unveils crucial dynamics impacting both generalization and optimization – see our analysis forthcoming.

For the second question, the strategy is direct: to control (or stabilize) Fisher information. How can we do that? Recall that Fisher information is derived from the prediction distribution, thus so simplest and the most direct way is to adjust the training data distribution fed into the network (*e.g.*, by using data subsets with lower Fisher information in the earlier phase of training). This is just like building a data curriculum as discussed in [Bengio et al. (2009)](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf), which works like a charm in sparse training.


## A Tale of Two Regimes
YZY is working on this: just discuss the mechanism in the two regimes. Fisher information for `sparse scratch` is higher than `sparse finetuning` in both regimes.

## The Proposed Solutions
YZY is working on this.

## Technical Details for Fisher Information
YZY is working on this.
<!-- ## The Problem
Training a sparse neural network from scratch cannot match the performance of training a sparse neural network from a pre-trained dense network.

Existing work:
- [Frankle et al., 2021](https://openreview.net/forum?id=Ig-VyQc-MLK):
- [Stosic and Stosic, 2021](https://arxiv.org/abs/2105.12920): -->

## Next steps
- Extend the pruning criteria (currently we use magnitude-based approach; we can consider gradient-based approach etc.)
- Extend the pruning procedure (*e.g.*, iterative pruning or dynamic pruning and growing)
- Extend the work from classification tasks to generation tasks as well (*e.g.*, on both stable diffusion and LLAMA)
- Discuss the further implication of this work

## Notes
1. Traditional GPUs are optimized for dense computations. For sparse training to be really useful, it is necessary to improve such hardware support as well, e.g., they should be able to handle sparse matrix operations and optimize for the memory access pattern.
2. The title is a tribute to `The Price of Salt`. I wish I were at the age of 19 waiting for the Christmas to come.
