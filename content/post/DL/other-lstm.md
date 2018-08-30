---
title: "几个LSTM的变种"
date: 2018-06-16T19:20:14+08:00
draft: true
tags: ["Tensorflow", "RNN", "LSTM", "Deep learing", "Note"]
categories: ["DL"]

comment: false
toc: false
# reward: false
mathjax: true
expirydate: 2018-04-06
---

前段时间用RNN做了一些实验，这些天总结下在`TensorFlow`里面看到的一些新奇的RNN模型，具体参考[`tf.contrib.rnn`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/rnn)

# 主要介绍下面几种：
- **Grid LSTM** [
Grid Long Short-Term Memory](https://arxiv.org/abs/1507.01526)
- **Phased LSTM** [Phased LSTM: Accelerating Recurrent Network
Training for Long or Event-based Sequences](https://arxiv.org/pdf/1610.09513v1.pdf)
- **G-LSTM** [Factorization tricks for LSTM networks](https://arxiv.org/abs/1703.10722)


## 0x01: Grid LSTM
第一眼看到它的模型结构图的时候还以为是个类似`Stacked LSTM`，但是认真看完他的模型结构后，发现这可是截然不同的新东西。如果说`LSTM`的结构是横躺着的话，那么它的结构就是可以竖直起来也可以横躺着的。我们知道在标准的`LSTM`在时间序上会把上一层的memory传递到下一层去，但是在垂直方向上并没有把memory传下去在`Stacked LSTM`这种结构上。

它每个cell的2-D的结构图如下：
<div style="text-align:center"><img src ="/images/dl/grid_lstm_f1.png" /></div>

它的整体结构跟`Stacked LSTM`对比如下：
<div style="text-align:center"><img src ="/images/dl/grid_lstm_f2.png" /></div>

从上图对比可以看出，不同与`Stacked LSTM`，2-D的`Grid LSTM`横竖结构都是一样的，memory经过计算后不止在横向上传递，也在垂直方向传递。

那么作者为什么要这样设计呢：

- 首先，从作者的原文可以看到`Grid LSTM`在字符预测，翻译和MNIST分类这些任务中表现都有所提升
- 从`Highway Network`借鉴过来的结构，同样计算的时候可以在垂直方向通过`Gate`来控制那些layer可以跳过，当网络deep到一定程度的时候，提高稳定性同时还能加快收敛速度。


## 0x02: Phased LSTM
这里有它的演讲视频[Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences](https://www.youtube.com/watch?v=ZMyVR3nwgAQ&t=18s)
