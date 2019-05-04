---
title: "从另一个角度看深度神经网络"
date: 2019-02-03T19:23:14+08:00
draft: true
tags: ["Deep Learning", "Discuss"]
categories: ["DL"]

comment: false
toc: false
# reward: false
mathjax: true
# expirydate: 2019-09-06
---

# 0x00 你的网络怎么处理问题的
最开始尝试用DL去做NLP相关任务时候，时不时会想要怎么解析训练出来的网络是怎么去执行这些任务呢？看过NG老师课程的应该都知道`CNN`在图片的边缘识别上是怎么处理的，这样就比较容易理解在像`VGG`这样的深度网络中必定有那么几层是表示着要物体的边缘。但是对于NLP任务，要怎么解析`CNN`的处理过程，怎么可以确定深度网络中就会训练出类似于词性，命名实体等特征？

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/head1.png" style="width:100%"/>
</div>

后来去年刚开始用DL做量化的时候，有个搞金融的朋友问我“你用DL训练出来的模型是怎么做策略的？” Emmm，说实话当时不知道怎么解析好，就只是从概率统计的角度上说了下。然后他问我都用了些什么指标，我说没有，只用了K线和买卖量的数据。后来我在想，训练过的网络中应该会有像EMA这样指标。为了验证这样的想法，下面就开始讨论下。


# 0x01 从最简单的线性问题
假设有这么个问题，现在有一批用户数据，如果用户是IOS用户，或者已经充值过的，都被划分为优质用户。定义$x_1$表示是否IOS，$x_2$表示是否充值过，取值为`[1, -1]`. 

一般情况对于这样简单的划分，可以直接写一段code来实现这样的功能了。
```
if x1 == 1 or x2 == 1:
    return True
else:
    return False
```

如果把他作为一个线性划分问题来解决, 令决策边界`y = WX + b`, 输出`H(x) = hardlims(y) = hardlims(WX + b)`. 这里直接取 W = $\begin{bmatrix} 1 & 1 \end{bmatrix}$, b = $-1$
$$H(x) = hardlims(
    \begin{bmatrix} 1 & 1 \end{bmatrix}
    \begin{bmatrix} x_1 \\\\ x_2 \end{bmatrix}
    - 1)$$

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/example1.jpeg" style="width:60%"/>
    <div><a>图1</a></div>
</div>

这里可以试下反过来想，对于式子$hardlims(
    \begin{bmatrix} 1 & 1 \end{bmatrix}
    \begin{bmatrix} x_1 \\\\ x_2 \end{bmatrix}
    - 1)$ 的理解，就是如果$x_1$等于1或者$x_2$等于1，就是A类，否则就是B类。

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/example1_2.png"/>
    <div><a>图1-2</a></div>
</div>

从实现来看,上面的例子用用线性划分来解决似乎有点多此一举, 直接一个`if...else...`就可以解决. 但是实际当中需要处理的问题都比较复杂,比如实际上要真正评定用户的消费能力，除了跟用户渠道，购买力有关系，还会与活跃度，用户的交际圈，甚至是性别年龄等有关系。

因此实际问题中，输入$X\begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ ... \\ x_n \end{pmatrix}$的维度一般都会比较高，用代码来实现这些规则的话就变成了这样:

```python
if rule1():
    if rule2() and rule3():
        do_something1()
    elif rule4() or rule5():
        if rule6() and rule7():
            do_something2()
    else:
        do_something3()
elif rule8():
    if rule9():
        do_something4()
        if rule10():
            do_something5()
            if rule11() or rule12():
                do_something6()
                if rule20():
                    do_something8()
        elif rule13():
            do_something7()
```

我们经常说的一些恶心业务代码就像是上面这种代码, 要写好多好多恶心的code不说, 维护起来更是困难，重点是很难写出综合最优的结果。但是用机器学习来把这个问题做成一个模型, 那就简洁清晰很多了.


# 0x02 神经网络与特征空间

我们把上面的例子再复杂化一些, 变成一个非线性问题:　对于平面上的点, 满足 $x_1 + x_2 > 0$ 并且 $x_1 - x_2 < 0$ 是A类, 满足 $x_1 + x_2 < 0$ 并且 $x_1 - x_2 > 0$ 是B类, 其他都是C类. 如图:

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/example2_4.png" style="width:60%"/>
    <div><a>图2-1</a></div>
</div>

## 手工设计网络

现在我们想通过神经网络来拟合出这个分类问题来, 假设存在这样的一个网络能够实现这样的分类, 并像下面这样:

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/example2_5.png" style="width:80%"/>
</div>

网络中的`Hidden layer 1`通过适当的参数设置，拟合出$x_1 + x_2$ 和 $x_1 - x_2$ 这两个特征。 `Hidden layer 2`通过适当的参数设置，拟合出$
\begin{cases}
x_1 + x_2 > 0 \\\\\\
x_1 - x_2 < 0
\end{cases}
$
$
\begin{cases}
x_1 + x_2 < 0 \\\\\\
x_1 - x_2 > 0
\end{cases}
$
这样的筛选条件

从这个运算来看这个网络的确是可以解决上面这个非线性问题, 但是实际单中是否可以训练的到这样的网络呢? 答案是可以的, 下面来看下.

## 实验验证

首先我们先随机出一批数据并打上标签作为样本:
```python
x1, x2 = make_samples()
labels = target_sample(x1, x2)
```
然后同样生成我们的预测数据
```python
p_x1, p_x2 = make_samples()
predict_labels = target_sample(p_x1, p_x2)
```
然后定义我们的神经网络并进行训练
```
hidden_layer1 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)
hidden_layer2 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)
out_put_layer = tf.keras.layers.Dense(3, activation=tf.nn.softmax)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2, )),
    hidden_layer1,
    hidden_layer2,
    out_put_layer
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, labels, epochs=2000)
```
训练结束后我们的`accuracy`值基本接近$0.99$, 然后看一下我们的预测结果, 基本上也是没有分错类

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/example2_7.png" style="width:60%"/>
</div>

接着我们来看一下两个隐藏层和输出层的参数:
```python
for layer in [hidden_layer1, hidden_layer2, out_put_layer]:
    [w, b] = layer.trainable_variables
    print(f"w: {w.numpy()} \t b: {b.numpy()}")

>>> w: [[-2.4261363 -1.5472057]
>>>  [ 2.1145093 -1.357698 ]] 	 b: [ 0.27367106 -0.01343301]
>>> w: [[ 1.5651494 -1.3211046]
>>>  [-1.723171   1.4522014]] 	 b: [0.9739141  0.74031043]
>>> w: [[-3.013817    0.06987078  0.7952662 ]
>>>  [ 2.1728668  -2.602164    1.4540561 ]] 	 b: [-0.53234714 -0.22513795  0.43315166]
```

我们可以发现第一层的参数 $
\begin{bmatrix} -2.4261363 &  -1.5472057 \\\\ 2.1145093 & -1.357698 \end{bmatrix}
\begin{bmatrix} 0.27367106 & -0.01343301 \end{bmatrix}$
进行 
$\frac{1}{2} w + \frac{1}{2} b$ 线性变换后, 就非常接近上面设计网络的第一层隐藏层的参数, 同样第二层隐藏层的参数跟我们设计的也是很接近.

**结论**, 通过这个例子, 神经网络中的确能拟合出与$x_1-x_2$和$x_1+x_2$很接近的特征. 但是这也不能说明网络一定能拟合出我们想要的特征.

## 逆向问题

现在我们反过来思考这个问题, 现在我们有一批数据, 分了A, B, C三类, 也就是 上面 `图2-1` 那样的分布, 但是现在并不知道他的分类规则, 也就是之前图中的边界. 现在我们想通过训练模型来发现这批数据的分类规则，我们通过画图观察发现, 的确存在 $x_1 + x_2 = 0$ 和 $x_1 - x_2 = 0$这两条边界能把数据区分开, 但是我们并不清楚网络能否拟合出这样的特征出来, 也不清楚具体怎么样结构的网络才会有效. 

### 实验1

不过我们一般都是一边实验一边摸索, 先用一层隐藏层试试看, 隐藏层包含两个单元
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2, )),
    tf.keras.layers.Dense(2, activation=tf.nn.tanh)
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
```

这样的网络, 最后训练出来的结果 `accuracy` 只有$0.7333$

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/result1.png" style="width:60%"/>
</div>

从分类结果来看， 似乎只是通过$x_2$维度来进行了分类，然后我们再看下隐藏层的参数
```
w: [[-0.03244739 -0.09983221]
 [ 1.0642205   2.0832403 ]] 	 b: [-0.71257067  0.2465051 ]
```
$x_1$相对于$x_2$的权重少了两个量级， $x_1$的权重几乎为0. 这么看来2个单元的隐藏层并不能拟合出这个分类规律来。

### 实验2
下面换成4个单元的隐藏层测试一下：
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2, )),
    tf.keras.layers.Dense(4, activation=tf.nn.tanh)
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
```
最后训练出来的`accuracy`非常接近 $0.99$， 从图形来看基本把边界拟合出来了

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/result2.png" style="width:60%"/>
</div>

从隐藏层中时候拟合出了$x_1-x_2$和$x_1+x_2$这样的特征？从输出的结果来看，的确也存在一对类似的特征

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/result3.png" style="width:60%"/>
</div>

但是现在我们最后的问题是
$
\begin{cases}
x_1 + x_2 > 0 \\\\\\
x_1 - x_2 < 0
\end{cases}
$
$
\begin{cases}
x_1 + x_2 < 0 \\\\\\
x_1 - x_2 > 0
\end{cases}
$
这样的筛选条件

从剩下的两个单元的参数来看， 一层隐藏层很难直观地表现出上面的筛选条件，他是怎么做到把数据分类开的，我们也很难直观地理解. 但是我们可以肯定的是，网络他已经把边界学习出来. 实际上网络中的每个隐藏层通过激活函数，把输入特征映射到更加高的维度，类似于SVM的核函数，在当前维度数据线性不可分，但如果把数据映射到更高维度的空间，数据就会变得线性可分。

### 实验3

为了方便可视化，我们把隐藏层单元改成3个进行实验：
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2, )),
    tf.keras.layers.Dense(3, activation=tf.nn.tanh)
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
```

最后训练出来的`accuracy`非常接近 $0.99$，隐藏层各参数如下：
```
w: [[-2.4996152 -3.0480654 -0.552722 ]
 [ 2.4173682 -3.100619   0.9217639]] 	 b: [-0.04109955 -0.06980375 -0.61598945]
```
我们已隐藏层的输出作为基，把数据投影到三维空间上：

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/result4.gif"/>
    <div><a>图2-1</a></div>
</div>

发现在映射过后的数据是线性可分的，在这里就是存在一个平面， 能将A类数据（红色）和其他数据区分开，也存在另一个平面，能将B类数据（绿色）和其他数据区分开。

结论，在这个问题上，人为观察数据可以发现出
$
\begin{cases}
x_1 + x_2 > 0 \\\\\\
x_1 - x_2 < 0
\end{cases}
$
$
\begin{cases}
x_1 + x_2 < 0 \\\\\\
x_1 - x_2 > 0
\end{cases}
$
这样的规则，但是对于神经网络而言，他的操作跟人类是不一样的（#为什么？#），就像人为了飞行，造出来的飞机飞行原理跟鸟类是不同的，神经网络要学会分类，靠的是把数据投影一个高维空间，一个存在超平面把数据划分开的空间来达到分类的效果。

### 输出层与超平面

上面一直在讨论的是隐藏层，输出层作为网络的最后一层，输出便是分类的结果。按照上面所说隐藏层把数据投影到高维空间中，并存在一个超平面可以划分数据。超平面作为分类的界限，输出层也是网络最后分类的依据，那么输出层与超平面是否存在关系？为了验证，下面尝试把输出层的参数做为平面函数的参数，在空间中画出来。

<div style="text-align:center">
    <img src ="/images/dl/other_side_deep_netword/result5.png"/>
</div>

最后我们惊奇地发现，用输出层的参数作为超平面函数画出来的平面，刚好是能把数据集区分开的平面。


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
