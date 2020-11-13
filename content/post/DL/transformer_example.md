---
title: "用Transformer构建自己的GPT2模型"
date: 2020-11-03T19:23:14+08:00
draft: true
tags: ["GPT2", "Tensorflow", "Transformer"]
categories: ["DL"]

comment: false
toc: false

# reward: false

mathjax: true

# expirydate: 2019-09-06

---

# 前言

`OpenAI` 发表 `GPT2` 已经过去一年多了，在网络上也看到有很多个实现的版本。近期想找一个别人训练好的中文模型进行Finetune，网上找了一圈发现大部分都是用Pytorch实现的，虽然Github上已经有几个用TF训练好的模型，但感觉代码写的太复杂，不适合上手，要么就是还是`TF1.X`版本的。作为TF2.0的少年，之前了解过 Huggingface 团队出了个 Transformer 库，里面也包含了GPT2模型，看了下文档整体调用也很简洁，所以决定用 Transformer 搞一个。

最终实现代码： [mymusise/gpt2-quickly](https://github.com/mymusise/gpt2-quickly)

# 踩坑之旅

## - TF的支持

🤗 `Transformer` 默认用的是 `Pytorch` 的API，而且从文档上可以体现出团队更倾向 `Pytorch` ，部分API暂时还不支持 `TF` 版本的，比如 `TextDataset` 。不过官方给出可以通过改写 `Dataset` 的[ `set_format` ](https://github.com/huggingface/transformers/issues/8190)方法，来实现 `TextDataset` 或者 `LineByLineTextDataset` 的功能。

## - Train/Finetune的文档

如果用keras的API去训练 `TFGPT2LMHeadModel` ，loss是个坑。看官网其他model的例子，以为直接compile就可以了。

``` python
    loss = model.compute_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=loss)
```

结果这样直接报错，实际上model的output维度比label要高，包含了每个layer的输出。

最后通过看源码和翻他们的issue才找到关于loss的定义。

``` python
    loss = model.compute_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=loss=[loss, *[None] * model.config.n_layer])
```

如果用 `TFTrainer` 就不会涉及上面loss的定义问题。但如果你的版本是(3.4.0)（当前只测试了这个版本，其他版本也有可能），可能会直接报找不到Pytorch的bug，这个Bug官方会在下一个版本（>3.4.0）修复。3.5.0目前已经发布。

<!-- ## Tokenizer

不知道为什么，GPT2Tokenizer好像不支持中文的，（补充） -->

# 正文

## 数据集

作为测试，可以先从 [ `chinese-poetry` ](https://github.com/chinese-poetry/chinese-poetry) download 几篇诗词过来。当前项目采用rawtext的形式，对于json格式的数据可能需要转换下格式。转化后的数据例子： [test/raw.txt](https://github.com/mymusise/gpt2-quickly/blob/main/dataset/test/raw.txt)

``` 

$ head -n 3 dataset/test/raw.txt 
忆秦娥 唐诗：【风淅淅。夜雨连云黑。滴滴。窗外芭蕉灯下客。除非魂梦到乡国。免被关山隔。忆忆。一句枕前争忘得。】
送兄 唐诗：【别路云初起，离亭叶正飞。所嗟人异雁，不作一行归。】
再赠 唐诗：【弄玉有夫皆得道，刘纲兼室尽登仙。君能仔细窥朝露，须逐云车拜洞天。】
```

## Vocabulary

GPT2官方给出的字典大小为50257，如果只是进行小样本测试，可以通过[ `huggingface/Tokenizers` ](https://github.com/huggingface/tokenizers) 构建自己的字典，一般小样本的字典集合大小都在1000左右的范围内，这样可以打打缩小模型维度，方便我们测试。以 `BertWordPieceTokenizer` 为例：

``` python
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=['your raw text file'],
                vocab_size=52_000, min_frequency=5)
tokenizer.save_model('path/to/save/')
```

实际上，现在大部分中文语言模型，相对于Google的21128大小的字典，我发现大家一般会选[ `CLUE` ](https://github.com/CLUEbenchmark/CLUEPretrainedModels)提供的8021大小的字典。

## Tokenizer

Tokenization之前，我们需要对数据进行切片预处理，方法参考了[gpt2-ml](https://github.com/imcaspar/gpt2-ml)的预处理过程。我们知道GPT2最大支持的输入文本是1024长度，假设先设定每个sample的大小是64（1024同样道理），以 `。？！` 标点符号为分界，对文本进行分句。并每个sample加入上一个sample的最后一句。按照这种处理方式，上面三行样例就变成：

``` 

1. 忆秦娥 唐诗：【风淅淅。夜雨连云黑。滴滴。窗外芭蕉灯下客。除非魂梦到乡国。免被关山隔。忆忆。一句枕前争忘得。】[PAD][PAD]...[PAD]
2. 一句枕前争忘得。】\n送兄 唐诗：【别路云初起，离亭叶正飞。所嗟人异雁，不作一行归。】[PAD][PAD]...[PAD]
3. ....

```

接下来把切片好的raw text丢给Tokenizer进行编码, 下面拿刚刚的样例举个例子：

``` python
In [5]: tokenizer = BertTokenizer.from_pretrained('path/you/save/')

In [6]: tokenizer("忆秦娥 唐诗：【风淅淅。夜雨连云黑。滴滴。窗外芭蕉灯下客。除非魂梦到乡国。免被关山隔。忆忆。一句枕前争忘得。】[PAD][PAD]", return_attention_mask=False, return_token_type_ids=False)
Out[6]: {'input_ids': [2, 405, 713, 1, 230, 843, 1003, 8, 973, 1, 1, 7, 267, 952, 885, 53, 1, 7, 628, 628, 7, 724, 265, 1, 1, 636, 15, 305, 7, 942, 962, 990, 559, 155, 43, 242, 7, 1, 827, 123, 336, 947, 7, 405, 405, 7, 10, 196, 541, 157, 49, 407, 399, 7, 9, 0, 0, 3]}

```

实际一般需要预处理的文本量都很大，都是几个G以上甚至几十个G，如果单进程处理会很长时间，这里提供一种多进程Tokenizer的方法供大家参考：[predata.py](https://github.com/mymusise/gpt2-quickly/blob/main/predata.py) 

这里把数据按照进程数进行均分，并分给每个进程encode，encode好的token转成numpy的数组。博主比较懒，看到 `TFRecord` 和 `TFExample` “臃肿”的API就不想用（如果大家知道有什么场景用 `TFRecord` 更好，麻烦在评论里纠正下博主），所以最后用pickle分别导出到对应的二进制文件文件了，像这样：

``` 

$ ls dataset/train 
data_0.pickle   data_1.pickle  data_2.pickle
```

## Model initialization

这个没什么好说的， `Transformer` 都给包装好了，先定义下模型的参数:

``` python
from transformers import GPT2Config

config = GPT2Config(
    architectures=["TFGPT2LMHeadModel"],   # pretrain的时候用来预加载模型
    model_type="TFGPT2LMHeadModel",        # 定义模型类型，导出给`AutoConfig`用，如果要上传到hub请必填
    tokenizer_class="BertTokenizer",       # 定义tokenizer类型，导出给`AutoTokenizer`用，如果要上传到hub请必填
    vocab_size=8021,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=6,
    n_head=6,
    pad_token_id=tokenizer.pad_token_id,   # 前面构建的tokenizer的 PAD ID
    task_specific_params={
        "text-generation": {
            "do_sample": True,
            "max_length": 120
        }
    }
)
```

然后构建模型, 直接把上面定义好的 `configs` 丢给 `TFGPT2LMHeadModel` 就创建好了。如果要通过 `Keras` 的API进行训练的话，需要对模型进行compile一下，前面也提到loss这里会有坑。

``` python
from transformers import TFGPT2LMHeadModel

model = TFGPT2LMHeadModel(config)
loss = model.compute_loss
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)

model.compile(
    optimizer=optimizer,
    loss=[loss, *[None] * model.config.n_layer],
)
```

## Train

训练前可以自定义个callback，每个epochs结束后保存下模型

``` python
class AutoSaveCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained("path/to/save")

callbacks = [AutoSaveCallback()]

model.fit(
    train_dataset,
    epochs=50,
    steps_per_epoch=2000,
    callbacks=callbacks,
)
```

## 一些例子

- 你可以在colab上尝试整个训练过程: [gpt2_quickly.ipynb](https://colab.research.google.com/github/mymusise/gpt2-quickly/blob/main/examples/gpt2_quickly.ipynb)
- 一个还在测试中的mediun量级的GPT2中文模型: [gpt2_medium_chinese.ipynb](https://colab.research.google.com/github/mymusise/gpt2-quickly/blob/main/examples/gpt2_medium_chinese.ipynb)
- 基于上面的模型，Finetune的小说生成模型: [ai_noval_demo.ipynb](https://colab.research.google.com/github/mymusise/gpt2-quickly/blob/main/examples/ai_noval_demo.ipynb)


<div style="text-align:center">
    <img src ="/images/dl/transformer_example/gpt2-medium-chinese-homepage.jpeg" style="width:80%"/>
    <div><a>gpt2_medium_chinese</a></div>
</div>

