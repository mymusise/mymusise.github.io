---
title: "决策树分类器"
date: 2018-04-13T20:19:14+08:00
draft: true
tags: ["Sklearn", "Decision Tree", "Machine learning", "Note"]
categories: ["ML"]

comment: false
toc: false
# reward: false
mathjax: true
# expirydate: 2018-04-06
---


# 0x00

这篇主要介绍一下决策树(Decision tree)，它一般被用来做分类器，用它做的分类器有几种优势：

- 计算简单，易于理解，可解释性强
- 比较适合处理有缺失属性的样本
- 能够处理不相关的特征
- 在相对短的时间内能够对大型数据源做出可行且效果良好的结果

但是呢它也有几个缺点：

- 容易发生过拟合（随机森林可以很大程度上减少过拟合）
- 忽略了数据之间的相关性

下面举个简单的例子，假如我们有一堆矿石需要给他们进行分类，在我们数据库里有这几类矿石的硬度和密度数据，我们可以用Sklearn快速实现：

利用sklearn`make_blobs()`函数先定义三类数据，并描绘出来
```python
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np

X, Y = datasets.make_blobs(n_features=2, centers=3, center_box=[1.5, 8.5]) # 定义了三类矿石数据

cluster1, cluster2, cluster3 = [], [], []
for x, y in zip(X, Y):
    if y == 0:
        cluster1.append(x)
    elif y == 1:
        cluster2.append(x)
    elif y == 2:
        cluster3.append(x)
cluster1 = np.array(cluster1)
cluster2 = np.array(cluster2)
cluster3 = np.array(cluster3)

fig, ax = plt.subplots()
plt.xlim(0, 10)
plt.ylim(0, 10)
ax.scatter(cluster1[:, 0], cluster1[:, 1], c='r', marker='*')
ax.scatter(cluster2[:, 0], cluster2[:, 1], c='g', marker='+')
ax.scatter(cluster3[:, 0], cluster3[:, 1], c='b', marker='o')
plt.show()
```

<div style="text-align:center"><img src ="/images/ml/decision_tree_f1.png" /></div>

然后我们把这些训练数据用Sklearn的`DecisionTreeClassifier`进行训练，由于决策树很难把边界刻画出来，所以我们先把训练好的决策树给描绘出来：

```python
from sklearn import tree
import graphviz

clf = tree.DecisionTreeClassifier().fit(X, Y)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['硬度', '密度'],
                                class_names=['A矿石', 'B矿石', 'C矿石'],
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('result')
```

<div style="text-align:center"><img src ="/images/ml/decision_tree_f2.svg" /></div>

这样我们就快速用决策树实现了一个分类器，下面用训练好的模型对预测数据进行分类：

```python
#          硬度, 密度
test_data = [[3, 3],
            [4, 6],
            [7, 2],
            [1, 9],
            [2, 8]]
results = clf.predict(test_data)
for _x, result in zip(test_data, results):
    print(_x, result)
```
结果如下：
```text
[3, 3] B矿石
[4, 6] A矿石
[7, 2] B矿石
[1, 9] C矿石
[2, 8] C矿石
```
