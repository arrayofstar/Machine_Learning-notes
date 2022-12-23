# 《Regularizing and Optimizing LSTM Language Models》——LSTM语言模型的正则化和优化器策略

[论文pdf]()|[B站视频]()|[知乎文章



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
| bptt        | 序列长度 {？？？}，默认值为70                               |
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
|             |                                                             |
|             |                                                             |




# 论文复现笔记

## 数据集

论文数据集使用`getdata.sh`文件命令进行加载，window用户直接下载对应链接然后改名字即可：



## 模型组网

