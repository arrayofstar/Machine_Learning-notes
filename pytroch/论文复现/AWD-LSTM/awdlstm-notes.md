# 《Regularizing and Optimizing LSTM Language Models》——LSTM语言模型的正则化和优化器策略

[论文pdf](https://arxiv.org/abs/1708.02182v1)|

该论文中提出了一系列基于词的语言模型的正则化和优化策略。

- 基于DropConnect(2013)的在对隐藏权重的正则方法，作用于隐状态的权重矩阵。
- 使用非单调条件触发的平均随机梯度下降(NT-ASGD)。
- 其他正则化方法：
  - 可变长度反向传播序列(Variable length backpropagation sequences)
  - 变分丢弃(Variational dropout)：使用mask进行dropout
  - 词嵌入层丢弃(Embedding dropout)：
  - 权重共享(Weight tying)：共享嵌入层和输出层的权重矩阵，减少参数量。

# 论文参数表

| 参数名称    | 参数含义                                                    |
| ----------- | ----------------------------------------------------------- |
| data        | 数据语料库的位置，默认值为 data/penn/                       |
| model       | 使用循环神经网络模型的类型，可选LSTM, QRNN, GRU；默认值LSTM |
| emsize      | word embeddings的大小，默认值为400                          |
| nhid        | 每个层中隐藏单元的数量，默认值为1150                        |
| nlayers     | 层的数量，默认值为3                                         |
| lr          | 初始化学习率，默认值为30 {????}                             |
| clip        | 梯度裁剪参数，默认值为0.25                                  |
| epochs      | 训练的轮次，默认值为8000                                    |
| batch_size  | 单次batch训练的大小，默认值为80                             |
| bptt        | 序列长度 {可变的序列长度？}，默认值为70                     |
| dropout     | 默认值为0.4                                                 |
| dropouth    | 应用于rnn层的dropout，默认值为0.3                           |
| dropouti    | 应用于embedding层的dropout，默认值为0.65                    |
| dropoute    | 在embedding层中去除的词的比例，默认值为0.1                  |
| wdrop       | 应用于RNN隐藏到隐藏矩阵的权重丢失量，默认值为0.5            |
| alpha       | RNN激活的alpha L2正则化（alpha=0表示无正则化），默认值为2   |
| beta        | β-缓慢正则化应用于RNN激活（β=0表示无正则化），默认值为1     |
| wdecay      | 应用于所有权重的权重衰减，默认值为 1.2e-6                   |
| optimizer   | 选择的优化器，可选sgd和adam，默认为sgd                      |
| seed        | 随机种子，默认值为1111                                      |
| nonmono     | 随机种子，默认值为5                                         |
| log-interva | 报告结果的间隔，默认值为200                                 |
| save        | 最终模型的储层路径，默认值为 randomhash+'.pt'               |
| resume      | 要恢复的模型路径，默认值为 None                             |
| when        | 何时（哪个时期）将学习率除以10-接受倍数，默认值为[-1]       |
|             |                                                             |
|             |                                                             |




# 论文复现笔记

## 数据集

论文数据集使用`getdata.sh`文件命令进行加载，window用户直接下载对应链接然后改名字即可：

## 模型组网





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