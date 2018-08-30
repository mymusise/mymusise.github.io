---
title: "几种词向量的实现"
date: 2018-08-26T19:20:14+08:00
draft: true
tags: ["NLP", "words", "index", "Note"]
categories: ["DL"]

comment: false
toc: false
# reward: false
mathjax: true
# expirydate: 2018-04-06
---

# 相关理论

要进行文本分析，首先得要把文本特征化，转成程序可以处理的数据格式。特征化一般要把文本切分成词的形式，所以处理文本时都有一部分分词的工作。
对于作为文本里最小单位的词，词的特征化一般有两种：
- 独热编码（One-Hot Encoding）：One-Hot型的编码计算起来比较方便，但是维度很高，会导致参数`W`巨大，难以训练。
- 稠密编码/特征嵌入（Embedding）：目前比较普遍的做法，可以大大降低维度，通过嵌入表去特征映射，可以说这个也是基于One-Hot做的优化


# 数据准备

下面都用`20-newsgroups`数据集来进行说明，使用方法：

```python
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
dataset = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)
```

数据大概是这样子：

```Python
In [36]: print(dataset.data[:2])
Out[36]: 
['From: sd345@city.ac.uk (Michael Collier)\nSubject: Converting images to HP LaserJet III?\nNntp-Posting-Host: hampton\nOrganization: The City University\nLines: 14\n\nDoes anyone know of a good way (standard PC application/PD utility) to\nconvert tif/img/tga files into LaserJet III format.  We would also like to\ndo the same, converting to HPGL (HP plotter) files.\n\nPlease email any response.\n\nIs this the correct group?\n\nThanks in advance.  Michael.\n-- \nMichael Collier (Programmer)                 The Computer Unit,\nEmail: M.P.Collier@uk.ac.city                The City University,\nTel: 071 477-8000 x3769                      London,\nFax: 071 477-8565                            EC1V 0HB.\n',
 "From: ani@ms.uky.edu (Aniruddha B. Deglurkar)\nSubject: help: Splitting a trimming region along a mesh \nOrganization: University Of Kentucky, Dept. of Math Sciences\nLines: 28\n\n\n\n\tHi,\n\n\tI have a problem, I hope some of the 'gurus' can help me solve.\n\n\tBackground of the problem:\n\tI have a rectangular mesh in the uv domain, i.e  the mesh is a \n\tmapping of a 3d Bezier patch into 2d. The area in this domain\n\twhich is inside a trimming loop had to be rendered. The trimming\n\tloop is a set of 2d Bezier curve segments.\n\tFor the sake of notation: the mesh is made up of cells.\n\n\tMy problem is this :\n\tThe trimming area has to be split up into individual smaller\n\tcells bounded by the trimming curve segments. If a cell\n\tis wholly inside the area...then it is output as a whole ,\n\telse it is trivially rejected. \n\n\tDoes any body know how thiss can be done, or is there any algo. \n\tsomewhere for doing this.\n\n\tAny help would be appreciated.\n\n\tThanks, \n\tAni.\n-- \nTo get irritated is human, to stay cool, divine.\n"]
```

# One-Hot Encoding

下面说两种实现方式：

## 自己写一个

这功能比较简单，自己可以写一个。虽然不是很提倡，但是下面做个简单的示范，拿英语为例子***（因为这样子分词会比较简单，要自己写一个中文分词会比较麻烦）***：

```python
from sklearn.datasets import fetch_20newsgroups
from scipy.sparse import csr_matrix
import re


class Counter():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.col_length = 0
        self.row_length = 0
        self.clean_re = re.compile(r"[^a-zA-Z?.!,¿]+") # 去掉一些不是内容的字符集
        self.token_re = re.compile(r"[\.|\?|\ |\,|\']") # 分词

        self.create_index() # 构建mapping

    def tokenizer(self, text):
        text = self.clean_re.sub(" ", text)
        return self.token_re.split(text)

    def create_index(self):
        for text in self.lang:
            tokens = self.tokenizer(text)
            if self.col_length < len(tokens):
                self.col_length = len(tokens)
            self.vocab.update(tokens)

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

    def transform2index(self, texts):
        for text in texts:
            tokens = self.tokenizer(text)
            indices = [self.word2idx.get(token, 0) for token in tokens]
            indices += [0 for i in range(self.col_length - len(indices))]
            yield indices

    def transform2matrix(self, texts):
        indices = self.transform2index(texts)
        self.row_length = len(self.vocab)
        for index in indices:
            _row = list(range(len(index)))[:self.row_length]
            _col = index
            data = [1 for i in _row]
            yield csr_matrix((data, (_row, _col)), shape=(self.col_length, self.row_length))
```

我们拿上面的数据试一下：

```python
In [72]: counter = Counter(dataset.data)
    ...: print(list(counter.transform2index(["I'm Groot"])))
    ...: print(list(counter.transform2matrix(["I'm Groot"])))
[[6136, 28031, 0]]
[<11000x39831 sparse matrix of type '<class 'numpy.int64'>'
	with 3 stored elements in Compressed Sparse Row format>]
```


## Scikit-Learn
第一种，用的是`scikit-learn`提供的接口`CountVectorizer`， 方法如下：

```python
from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer()
counter.fit(dataset.data)
```

通过上面的`fit`操作，`CountVectorizer`就会帮你构建一个词表，查看词表如下：

```Python
In [39]: counter.vocabulary_
Out[39]: 
{'from': 14887,
 'sd345': 29022,
 'city': 8696,
 'ac': 4017,
 'uk': 33256,
 'michael': 21661,
 'collier': 9031,
...
}
```

有人说，这个值不值持中文或者其它语言。答案是当然了，实际上`CountVectorizer`在构建这个mapping的时候，会把输入text进行`tokenization`，也就是我们平时说的分词。负责这个是`tokenizer`函数，[具体细节大家可以去看下源码](https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/feature_extraction/text.py#L239)。比如中文，你可以调用`jieba`分词来定义它：

```python
import jieba

def tokenizer(text):
    return list(jieba.cut(text, cut_all=False))
```

然后我们在创建`CountVectorizer`的对象时候，加上它：

```python
counter = CountVectorizer(tokenizer=tokenizer)
```

利用上面的`counter`把字符串转成向量，返回的是一个稀疏矩阵

```python
In [42]: counter.transform(["hello i'm goot"])
Out[42]: 
<1x35788 sparse matrix of type '<class 'numpy.int64'>'
	with 1 stored elements in Compressed Sparse Row format>
```

如果你需要在一开始拿去fit的数据都转成vector，你可以用`fit_transform`来代替掉`fit`，这样它会在训练完之后就返回那个稀疏矩阵。


# Embedding （词嵌入）

上面讲的是One-Hot编码的，因为要进行Embedding之前，都需要把Word转成词表。同样下面说两种方法：

## Gensim

Gensim可以说是NLP界处理文本的神器,最常用就是拿它来做文本特征，Gensim用的就是`WORD2VEC`算法。
当你拿去构建词表的文本不多时候，可以直接这么做：

```python
import re
from gensim.models import Word2Vec

texts = [re.sub("[^a-zA-Z?.!,¿]+", " ", text) for text in dataset.data]
texts = [re.split("[\.|\?|\ |\,|\']", text) for text in texts]
model = Word2Vec(texts)
```

当然啦，很多时候我们要处理的文本量都比较大，我们没法一次都加载到内存中，这时候我们可以这么做：

```python

def text_generator():
    for text in dataset.data"
        text = re.sub("[^a-zA-Z?.!,¿]+", " ", text)
        text = re.split("[\.|\?|\ |\,|\']", text)
        yield text

model = Word2Vec(workers=4)
model.build_vocab(text_generator())
model.train(text_generator())
```

[^_^]:
    # 说明下这个 build_vocab 和 train 的过程

#### 保存模型

如果你的文本量很大的话，训练一次这个模型也是很费时间的，所以我们训练完之后可以把它保存起来，后面使用的时候只需要load进来就可以了。
```python
model.save('path_to_save') # 保训练好的模型

model = Word2Vec.load('path_to_save') # 加载训练过的模型
```

# Tensorflow

准确的说，实际上用的是`Keras`的模块，不错博主平时用的是`Tensorflow`，这里就偷懒不去用`Keras`来说明了。
使用之前，也是需要把文本转成id:

```python
counter = Counter(dataset.data)
indices = counter.transform2index(dataset.data)
```

其实这里应该可以把`Embedding`看成是网络中的一层，降输入层映射到高维。由于输入的文本太多，我们用`Dataset`去加载。

[^_^]:
    # 说明下这个 Embedding 的算法

```python
import tensorflow as tf

tf.enable_eager_execution()

data = tf.data.Dataset().from_generator(lambda :indices, output_types=(tf.float32))
data = data.batch(100).make_one_shot_iterator()
X = data.get_next()
embedding = tf.keras.layers.Embedding(len(counter.vocab), 200)
embedding(X)
```
输出：
```python
Out[12]: 
<tf.Tensor: id=77, shape=(100, 11000, 200), dtype=float32, numpy=
array([[[0.7414243 , 0.07747948, 0.60920155, ..., 0.46979737,
         0.9815726 , 0.17995226],
        [0.03834593, 0.39614272, 0.3659439 , ..., 0.24742019,
         0.9771553 , 0.30359387],
        [0.5240742 , 0.58283293, 0.95961046, ..., 0.5749551 ,
         0.846769  , 0.9823524 ],
        ...,
        [0.3502736 , 0.988209  , 0.85835266, ..., 0.626845  ,
         0.56388843, 0.13658428],
        [0.3502736 , 0.988209  , 0.85835266, ..., 0.626845  ,
         0.56388843, 0.13658428],
        [0.3502736 , 0.988209  , 0.85835266, ..., 0.626845  ,
         0.56388843, 0.13658428]]], dtype=float32)>
```