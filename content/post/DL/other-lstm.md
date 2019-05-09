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
# expirydate: 2018-04-06
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
这里有它的演讲视频~~[Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences](https://www.youtube.com/watch?v=ZMyVR3nwgAQ&t=18s)~~ (视频好像打不开了= =: 2019-02-09) 

不过大家可以去看它的[论文](http://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf)

从视频介绍上来看`Phased LSTM`解决的是输入的不同维度的特征周期不一致的问题，视频中介绍到在IOT上的应用。不同的传感器收集的数据周期不一样，比如距离传感器时刻收集着信号，但是像湿度传感器这类只有触发了才会收集到数据。

<div style="text-align:center"><img src ="/images/dl/phased_lstm_f1.png" /></div>

在这之前我们一般做法都是把不同周期的特征直接放在一块处理，希望网络能自己学习出来并区分。`Phased LSTM`的不同之处就是加入了一个时间轴的概念把这些不同周期的Features区分开，通过这样的定义：

$$ \phi_t = \frac{(t-s)mod\tau}{\tau} $$ 
(t表示当前的时刻，s是相位偏移，$\tau$ 是周期)

然后`Phased LSTM` 是这样定义了Time Gate function的:

<div style="text-align:center"><img src ="/images/dl/phased_lstm_f2.png" /></div>

更多细节可以参考论文。

虽然博主没有在工程中用到过`Phased LSTM`，但是对于处理不同周期的数据任务有了新的思路。


## 0x03: G-LSTM
**TODO**

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>