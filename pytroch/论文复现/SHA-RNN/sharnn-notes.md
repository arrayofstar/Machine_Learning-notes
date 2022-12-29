# 《Single Headed Attention RNN: Stop Thinking With Your Head》——单头注意力的RNN模型

[论文pdf](https://arxiv.org/abs/1911.11423)|[原文翻译1](https://zhuanlan.zhihu.com/p/94387032)|[原文翻译2](https://blog.csdn.net/weixin_37993251/article/details/103364977)

整体评价：这篇文章整体上文风不是很严谨，更像是一篇博客的写作风格，源代码中也有很多的注释，部分地方的代码和原文中的流程图也有一定的出入，想要使用源代码中的模型架构的话，还需要花比较长的时间进行逻辑调整，但不可否认的是，这篇文章是我看的第一篇单独使用注意力机制和LSTM这样循环网络结合的文章。

模型本质：实际上算是LSTM和自注意力机制的一个串联，并且在对`enwik8数据集`处理时，使用了4个block，其中只有第三个加上的注意力机制。

该论文中提出了单头注意力机制在语言序列预测中的应用，在Transformer流行的期间，依靠简单朴实的LSTM，结合**单头注意力机制**，依旧可以做出比较好的效果。

- 单头注意力机制与LSTM架构的结合。
- 使用了Boom的模块，利用linear层先升维再降维。
- 
- 

# 论文参数表-未修改

| 参数名称     | 参数含义                                                     |
| ------------ | ------------------------------------------------------------ |
| data         | 数据语料库的位置，默认值为 data/enwik8/                      |
| model        | 使用循环神经网络模型的类型，可选LSTM, QRNN, GRU；默认值LSTM  |
| emsize       | word embeddings的大小，默认值为1024                          |
| nhid         | 每个层中隐藏单元的数量，默认值为4096                         |
| nlayers      | 层的数量，默认值为4                                          |
| lr           | 初始化学习率，默认值为 2e-3 (0.002)                          |
| clip         | 梯度裁剪参数，默认值为0.25                                   |
| epochs       | 训练的轮次，默认值为8000                                     |
| batch_size   | 单次batch训练的大小，默认值为16                              |
| bptt         | 序列长度 {可变的序列长度}，默认值为1024                      |
| warmup       | 新增，学习率预热，默认值为800                                |
| cooldown     | 新增，学习率下降，默认值为None                               |
| accumulate   | 新增，梯度更新前需要累积的batch数量，默认值为1               |
| dropout      | 默认值为0.1                                                  |
| dropouth     | 应用于隐藏层中的dropout，默认值为0.1                         |
| dropouti     | 应用于embedding层的dropout，默认值为0.1                      |
| dropoute     | 在embedding层中去除的词的比例，默认值为0.1                   |
| wdrop        | 应用于RNN隐藏到隐藏矩阵的权重丢失量，默认值为0.0，相当于代码中未使用 |
| alpha        | RNN激活的alpha L2正则化（alpha=0表示无正则化），默认值为2，代码中未使用 |
| beta         | β-缓慢正则化应用于RNN激活（β=0表示无正则化），默认值为1，代码中未使用 |
| wdecay       | 应用于所有权重的权重衰减，默认值为 1.2e-6，代码中未使用      |
| optimizer    | 选择的优化器，可选sgd和adam，默认为sgd                       |
| seed         | 随机种子，默认值为1111                                       |
| nonmono      | 随机种子，默认值为5                                          |
| cuda         | 使用CUDA，无默认值                                           |
| log-interval | 报告结果的间隔，默认值为10                                   |
| save         | 最终模型的储层路径，默认值为 randomhash+'.pt' (进行了修改，不用随机的哈希值) |
| resume       | 要恢复的模型路径，默认值为 None                              |
| optimizer    | 新增，优化器的选择，默认值为lamb                             |
| when         | 何时（哪个时期）将学习率除以10-接受倍数，默认值为[-1]        |
| tied         | 权重共享，默认值为True                                       |
|              |                                                              |




# 论文复现笔记

## 数据集

论文数据集使用`getdata.sh`文件命令进行加载，window用户直接下载对应链接然后改名字即可：

# 一些源码中的问题

- [UserWarning: RNN module weights are not part of single contiguous chunk of memory. and how to generate probability of a setence      #7](https://github.com/salesforce/awd-lstm-lm/issues/7)

> I am getting the following warning:
>
> UserWarning: RNN module weights are not part of a single  contiguous chunk of memory. This means they need to be compacted at  every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().
>
> I am using pytorch 0.20 and python 3.5
>
> Also, how do you generate a probability of a given sentence?

> Hi [@dixiematt8](https://github.com/dixiematt8),
>
> This is an expected UserWarning and isn't a problem. As  the weights are changed before each call of the LSTM, they'd need to be  compacted anyway.
>
> Re: probability of a given sentence, that is not defined.  As language is infinite in the sentences it can generate, there's no  well defined probability distribution. You can see which sentences may  be more of less likely but that's about the best I could suggest.
>
> For that you'd multiply the probabilities (or sum the log  probabilities for a numerically more stable approach) of each of the  words according the model's prediction and select the sentence with the  larger probability.
>
> Hope that helps!