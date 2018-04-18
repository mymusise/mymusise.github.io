---
title: "线性回归模型"
date: 2018-04-11T20:19:14+08:00
draft: true
tags: ["Sklearn", "Linear", "Note"]
categories: ["ML"]

comment: false
toc: false
# reward: false
mathjax: true
# expirydate: 2018-04-06
---

# 0x00

这篇短文主要讲**Scikit-learn**下面的三种线性模型使用，也是比较熟悉和常用的几个：
- 普通线性回归
- 逻辑回归
- 线性判断分析

# 0x01逻辑回归
这个模型是最简单也是最基础的一个模型，通过求际观测数据和预测数据之间的最小差方和来拟合方程，一般可以用来。单地做些数据预测。公逻辑也是回归式表示如下：

<div style="text-align:center"><img src ="http://sklearn.lzjqsdd.com/_images/math/e8e92a5482d9327d939e7a17946a8a1b98006018.png" /></div>

下面讲个简单的例子，先生成一些随即数据，数据根据一个线性方程±一些随即数得到
```python
import numpy as np

x = np.arange(40)
delta = np.random.uniform(-50, 50, size=(40,))
w = np.random.randint(-10, 10)
y = w * x + delta
```
我们用**Scikit-learn**的`LinearRegression`模型来对数据进行拟合
```python
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(np.array(x).reshape(-1, 1), y)
```
`LinearRegression`把数据`fit()`进去后就会进行拟合，通过调用`reg.coef_`可以获取到各个特征值系数(除了`θ0`)，常数系数(也就是一般公式里面的θ0)在Sklearn叫截距系数`reg.intercept_`

下面我们把上面的数据和拟合好的函数用`matplotlib`画出来看看

```python
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y, 'o', label='Samples') #描绘数据集

_XX = np.array([min(x), max(x)])
XX = np.ones((2, 2))
XX[:, :-1] = _XX.reshape((2, 1))
YY = np.array([*reg.coef_, reg.intercept_]).dot(np.array(XX).T)
ax.plot(_XX, YY) #描绘拟合好的函数

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

结果如下：

<div style="text-align:center"><img src ="/images/ml/linear_model_f1.png" /></div>

对于一般的线性回归问题，可以用这种方法简单地进行拟合

# 0x02 逻辑回归（Logistic regression）

逻辑回归有些书上也叫“对数几率回归”，一般用它来解决一些二分类问题，所以它还有一个名字叫最大熵分类器(MaxEnt)。最近很火的深度学习，逻辑回归也是里面最基本的组成单元。它的Cost Function为：

<div style="text-align:center"><img src ="http://sklearn.lzjqsdd.com/_images/math/760c999ccbc78b72d2a91186ba55ce37f0d2cf37.png" /></div>

下面举个例子

- 假设有个叫颜值系数F(0~100)，F越大说明这个人长得越好看，那么它脱单的几率越高，反之就越低了。现在我们在某学校计院里面随即抽取了13个人，然后给他们取了编号分别为1～13，测量他们的颜值系数以及是否单身，情况如下：

| 编号    | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  | 11  | 12  | 13  |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 颜值系数F | 40  | 30  | 81  | 66  | 43  | 94  | 37  | 77  | 67  | 12  | 53  | 12  | 86  |
| 是否有对象 | 0   | 0   | 1   | 0   | 1   | 1   | 0   | 1   | 1   | 0   | 0   | 0   | 1   |

我们定义下这些数据，画坐标轴上是一个这样的分布

```python
from matplotlib import pyplot as plt

x = [40, 30, 81, 66, 43, 94, 37, 77, 67, 12, 53, 12, 86]
y = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1]

fig, ax = plt.subplots()
ax.plot(x, y, 'o')
plt.show()
```

<div style="text-align:center"><img src ="/images/ml/linear_model_f2.png" /></div>

我们使用Sklearn提供的`LogisticRegression`Model进行训练：
```python
from sklearn import linear_model
import numpy as np

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
X = np.array(x).reshape(-1, 1)
clf.fit(X, y)
print(clf.coef_, clf.intercept_)
```
这里得到`θ1`的系数是`0.036`，`θ0`是`-1.76`，画出来大概是这样

<div style="text-align:center"><img src ="/images/ml/linear_model_f3.png" /></div>

上面数据和例子只是为了配合实验造的，事实并非这样。而且事实上，找对象这个东西还有很多因素干扰着，并不是完全靠脸。

而且生活中很多问题也并非是只有两类，种类会很多，然而逻辑回归只能解决的是二分类问题。

