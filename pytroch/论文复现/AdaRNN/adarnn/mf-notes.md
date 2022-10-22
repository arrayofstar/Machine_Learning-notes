# 《AdaRNN:Adaptive Learning and Forecasting of Time Series》——时间序列预测新范式：基于迁移学习的AdaRNN方法

[论文pdf](https://arxiv.org/abs/2108.04443)|[B站视频](https://www.bilibili.com/video/BV1Gh411B7rj/?vd_source=3aa88642179030efe4ce362bda4fea11)|[知乎文章](https://zhuanlan.zhihu.com/p/398036372)

![image-20221015151553888](notes.assets/image-20221015151553888.png)

## 背景

- 时间序列是一种重要的具有广泛应用的数据类型
  - 股票预测
  - 健康状态控制
  - 天气预报
  - 价格管理
  - 能源消耗

准确的时间序列预测依旧是一个具有挑战且没有被完全解决的问题。

![image-20221015152153684](notes.assets/image-20221015152153684.png)

### 已经存在的时间序列方法

1. 马尔科夫假设
   - 每一个观测值只和前一个时刻的状态有关，而与其他无关
   - HMM-隐马尔科夫模型
   - Dynamic Bayesian network - 动态贝叶斯网络
   - Kalman filter - 卡尔曼滤波
   - RIMA. ARIMA(自回归移动平均模型，Autoregressive integrated Moving Average Model)... - 统计学家常用 
2. RNN(循环神经网络。Recurrent Neural Networks)
   - 能够发现长时间序列数据中的非线性和复杂的依赖关系 - can find highly non-linear and complex relationships in long-time periods
   - RNN/LSTM/GRU
   - Conv-LSTM...
3. Transformers
   - 一种seq2seq的模型，可以并行建模（Parallel modeling）， 可以捕获更长时的数据依赖。
   - Transformers

![image-20221015152254704](notes.assets/image-20221015152254704.png)

### 动机-Motivation

- 非平稳时间序列
  - 时间序列的统计特征是跟随时间不断变化的

数据虽然在整体上是动态的波动，但是在局部看来是一个固定的分布。即由若干个未知的单一分布复合合成的。 时间序列中存在一种内部的分布不一致性，在某一段小的区间里面这个分布是趋向一致的。这个问题之前是没有人研究的，我们将这个问题定义时序协方差漂移或时序变量漂移(Temporal Covariate Shift)。一个时间序列可以分成很多段，段内的分布是一致的，段之间分布不一样。

![image-20221015180546691](notes.assets/image-20221015180546691.png)

### 问题定义- Preblem formulation

![image-20221015180714652](notes.assets/image-20221015180714652.png)

### **时序分布漂移** - Temporal Covariate Shift(TCS)

在原始变量漂移(Covariate Shift)定义的基础上定义了**时序分布漂移**(Temporal Covariate Shift)，其中变量偏移(Covariate Shift)属于数据集偏移(Dataset shift)中的一个分类。

[数据集偏移](https://zhuanlan.zhihu.com/p/205183444)(Dataset shift)的概念是指，在一般的机器学习项目中，我们将采集到的真实场景的数据分成训练集和测试集（或验证集），并假设训练集和测试集符合独立同分布，这样保证了训练集上表现良好的模型同样适用于测试集（即真实场景）。**但是**，当某些原因导致训练集和测试集分布不同时，便会发生dataset shift (or drifing)。

数据集偏移类型一般可分为三种：(1)协变量偏移（Covariate shift）(2)先验概率偏移（Prior probability shift）(3)概念偏移。具体解释这里先不赘述。

![image-20221015181241053](notes.assets/image-20221015181241053.png)

## 如何解决TCS的问题 - How to solve TCS?

- 找出最不一样的分布区间方案
- 匹配最大分布时的谷
- 得到好的模型

![image-20221015181330846](notes.assets/image-20221015181330846.png)

### 我的方法 - Our approach

- AdaRNN：Adaptive RNNs
  - 时序相似性量化：时间分布特征刻画
  - 时序分布匹配：时间分布特征匹配

![image-20221015181950711](notes.assets/image-20221015181950711.png)

### ***时序相似性量化\*** - Temporal Distribution Characterization

- TDC
  - 找到K最不相关的部分
  - 如何来定义相似性呢？—— 分布距离$D$
  - 为什么是最不相似的部分？—— 多样性的分布可以帮助模型的泛化
  - 对分布做一个类似与聚类的操作。

这个问题是可以转化为一个动态规划的问题， 但是我们使用了贪心算法来更高效的求解这个问题。

![image-20221015182559878](notes.assets/image-20221015182559878.png)

### 时序分布匹配 - Temporal Distribution Matching

-   常规方法

  - 领域泛化(domain generalization)问题 - K 领域
  - 该方法忽略了RNN中的隐藏表示分布

- TDM

  - 我们的方法：一种可适应的权重矩阵来表示每一个隐藏状态。

  ![image-20221015183238989](notes.assets/image-20221015183238989.png)

- 如何去学习权重矩阵呢？
  - 常规的方法是加一个注意力机制(attention)，但是会因以下原因而失效：
    - 在初始阶段，通过固定$\theta$和$\alpha$中的一个来表示隐藏层的状态是没有意义的，都会导致对权重不充分的学习。
    - 网络会因为非常复杂和耗时，而很容易卡住
  - 我们的解决方式
    - 提出了一个方法(boosting-based importance evaluation)

![image-20221015184247729](notes.assets/image-20221015184247729.png)

![image-20221015202320936](notes.assets/image-20221015202320936.png)

## 实验

- 数据集：行为数据集-分类(UCI activity)、空气质量预测(Air quality)、电力消耗(Electric power)、股票价格-私有(Stock price)
- 基线
  - 传统方法：ARIMA、GRU、LightGBM
  - 已存在的DA/DG方法：MMD、DANN（没有对TS特别有效的方法，所以我们扩展了他们）
  -  Transformer
  - Latest TS methods: LSTNet(SIGIR-18), STRIPE(NeurIPS-20)
  - 基础网络：2层GRU

![image-20221015204115578](notes.assets/image-20221015204115578.png)

## 结果

![image-20221015204142111](notes.assets/image-20221015204142111.png)

![image-20221015204216936](notes.assets/image-20221015204216936.png)

## 分析

- 时间分布特征刻画
  - 分区的不同反映了不同的分布信息

- 我们的TDC算法给出了最好的分区结果
  - 比随机和翻转（翻转这里还不是很理解）更好。

![image-20221015204501424](notes.assets/image-20221015204501424.png)

- 时间分布特征匹配
  - 学习权重$\alpha$是有效的
  - 我们的boosting方法获得了最佳的结果
  - 在不同分布特征匹配距离时，TDM是不可知的(agnostic)

![image-20221015204936976](notes.assets/image-20221015204936976.png)

### 更细致的分析 - More detailed analysis

- 分布距离
  - 我们的方法给出了最小的距离
- 多步预测
  - 我们的方法给出了最好的多步预测结果

![image-20221015205348881](notes.assets/image-20221015205348881.png)

### 收敛性和训练时间

- 收敛性 - Convergence
  - 我们的方法可以在很少的迭代次数下收敛
- 训练时间
  - 我们的方法不会产生更大的计算负担，而且甚至比SOTA模型更高效 

![image-20221015205945985](notes.assets/image-20221015205945985.png)

### 个案研究 - Case study

![image-20221015210122955](notes.assets/image-20221015210122955.png)

### 更进一步：扩展到Transformer

![image-20221015210147205](notes.assets/image-20221015210147205.png)

## 总结

三个创新点

- 全新的问题
  - 我们首次发现并提出了时间序列数据中时间变量偏移的问题
- 新颖的方法
  - 为了解决时间变量偏移的问题，我们提出了AdaRNN的方法来学习时间不变性(temporally-invariant)的模型来确保好的泛化能力
- 良好的表现
  - 在多个数据集上的实验证明有效。

![image-20221015222951449](notes.assets/image-20221015222951449.png)

# 论文复现-mf

参数表

| 参数名称     | 参数含义                                           |
| ------------ | -------------------------------------------------- |
| batch_size   | 默认值36                                           |
| class_num    |                                                    |
| d_feat       |                                                    |
| data_mode    |                                                    |
| data_path    |                                                    |
| dropout      |                                                    |
| **dw**       |                                                    |
| early_stop   |                                                    |
| gpu_id       |                                                    |
| hidden_size  |                                                    |
| len_seq      |                                                    |
| len_win      |                                                    |
| log_file     | 运行日志                                           |
| loss_type    |                                                    |
| lr           | 学习率，默认值为0.0005                             |
| model_name   | 模型名称，可选“AdaRNN”和“Boosting”，默认值为AdaRNN |
| n_epochs     |                                                    |
| num_domain   |                                                    |
| num_layers   | 层的数量                                           |
| outdir       |                                                    |
| overwrite    |                                                    |
| pre_epoch    |                                                    |
| seed         |                                                    |
| smooth_steps |                                                    |
| station      |                                                    |

其他参数

| 函数名          | 参数名           | 参数含义                                                     |
| --------------- | ---------------- | ------------------------------------------------------------ |
|                 | dis_type         |                                                              |
| AdaRNN.__init__ | use_bottleneck   | 默认值为True                                                 |
|                 | bottleneck_width | 默认值为64                                                   |
| AdaRNN.__init__ | n_input          | 默认值为6                                                    |
| AdaRNN.__init__ | num_layers       | 默认值为2                                                    |
| AdaRNN.__init__ | hiddens          | GRU中的隐藏层长度，因为有两层GRU，默认值为[64, 64]           |
| AdaRNN.__init__ | n_output         | 默认值为1                                                    |
| AdaRNN.__init__ | model_type       | 默认值为'AdaRNN'                                             |
| AdaRNN.__init__ | trans_loss       | 默认值为'adv'                                                |
| AdaRNN.__init__ | len_seq          | 默认值为24                                                   |
|                 | in_size          | GRU层中输入长度，默认值为6                                   |
| train_AdaRNN    | dist_mat         | 默认值为None                                                 |
|                 | weight_mat       | 默认值为None                                                 |
| TDC             | start_time       | 2013-03-01                                                   |
|                 | end_time         | 2016-06-30                                                   |
|                 | split_N          | 10                                                           |
|                 | selected         | [0, 10]                                                      |
|                 | candidate        | [1, 2, 3, 4, 5, 6, 7, 8, 9]                                  |
|                 | dis_type         | 默认为coral，支持的有mmd(mmd_lin), mmd_rbf, coral, cosine, kl, js, mine, adv |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |
|                 |                  |                                                              |

# 论文复现笔记

## 数据集定义与加载

论文中所使用的数据集为[Beijing Multi-Site Air-Quality Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data) 。数据集中包含了从2013年3月到2017年2月北京12个站点收集到的每小时空气质量信息。作者随机选择了四个站点（Dongsi, Tiantan, Nongzhanguan, and Dingling）和六个特征（PM2.5, PM10, SO2, NO2, CO, and O3）。因为原始数据中包含一些缺失值，因此使用了平均值进行替换。随后，还对数据集进行好标准化处理，将所有的特征扩展到相同的范围。作者还提供的处理之后数据的对应链接： [dataset link](https://box.nju.edu.cn/f/2239259e06dd4f4cbf64/?dl=1) or [百度云](https://pan.baidu.com/s/1xkLyd9YPgK7h8B1-acCImg) (密码：1007) 

备注：数据集文件为.pkl格式，包含了三个部分，'features', 'label', 'label_reg'. 'label'为空气质量的分类标签(如. excellence, good, middle)，在文中没有使用。 'lable_reg' 为预测的值。

在实际数据读取中，训练集使用TDC算法按用户指定的`number_domain`进行划分。时间段上，

- 训练集：2013-3-1 00:00 —— 2016-6-30 23:00
- 验证集：2016-7-2 00:00 —— 2016-10-30 23:00
- 测试集：2016-11-2 00:00 —— 2017-2-18 23:00

## 模型组网

