
# 简介

最近几年，由于像OpenAI著名的[GPT2](https://openai.com/blog/better-language-models/)这种基于百万级web数据训练出来的大型Transformer模型兴起，开放领域的语言模型越来越多了。尤其像[GPT2](https://openai.com/blog/better-language-models/#samples)、[XLNet](https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e)、[CTRL](https://blog.einstein.ai/introducing-a-conditional-transformer-language-model-for-controllable-generation/)在开放领域的条件写作结果出乎意料的好。除了改进Transformer的结构和喂更多的数据意外，更好的解码方法也很大程度影响输出的结果。

这篇文章主要介绍各种不同的解码策略，而且还会分享如何用`Transformer`来实现它们!

`自回归语言模型`的生成方法都可以用以下的公式直接概括（[点这可以复习下](http://jalammar.github.io/illustrated-gpt2/)）。总的来说，`自回归语言模型`都是基于这样的设定：整个字符串的概率，可以用每个字的条件概率的乘积来表示：

$$ P(w_{1:T} | W_0 ) = \prod_{t=1}^T P(w_{t} | w_{1: t-1}, W_0) \text{ ,with }  w_{1: 0} = \emptyset, $$

其中 $W_0$ 是第一个字，字符串的长度T就是你要生成的长度，并且包括t=T时刻对应的`EOS` token 也是通过 $P(w_{t} | w_{1: t-1}, W_{0})$ 生成的。 (翻译得可能不对)

现在`GPT2`,`XLNet`, `OpenAi-GPT`, `CTRL`, `TransfoXL`, `XLM`, `Bart`, `T5` 这些自回归语言模型都支持PyTorch和TF2了。

本篇教程主要给大家介绍 *Greedy search*, *Beam search*, *Top-K sampling* and *Top-p sampling* 这几种目前效果最好的编码方法。

现在我们可以快速安装transformer以及加载模型，下面用Tensorflow 2.1 作为示例，PyTorch的调用方法也是一摸一样的。

```python
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q tensorflow==2.1
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
```


# Greedy Search （贪心搜索）

Greedy Search 其实很简单，它每一步都取当前最有可能的word作为下一步的结果: $w_t = argmax_{w}P(w | w_{1:t-1})$ 。 下面是Greedy Search的流程图。

<img src="/images/nlp/greedy_search.png" alt="greedy search" style="margin: auto; display: block;">

最一开始的单词是"The", 然后选择了概率最高的单词"nice"，然后同样再选择了"woman"。这样最终生成的句子("The", "nice", "woman")的整个概率为： 0.5×0.4=0.2.

下面我们试一下用GPT2生成一个句子，输入为：("I","enjoy","walking","with","my","cute","dog")，我们看下怎么用`transformers`实现Greeddy Search。
```python
# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
```

