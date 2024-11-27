# -*- coding: utf-8 -*-
# @Time    : 2024/11/26 17:08
# @Author  : Dreamstar
# @File    : dataloader.py
# @Link    :
# @Desc    : 根据《动手学深度学习》中的代码，来复现一下BERT的预训练、微调和预测过程

import os
import random
import torch


# from d2l import torch as d2l

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _read_wiki(data_dir, mode='train'):
    file_name = os.path.join(data_dir, f'wiki.{mode}.tokens')
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    """有半分之50的情况，下一个句子不是真正的下一个，并且返回一个is_next"""
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """将句子增加<cls>和<sep>,如果句子过长了，就不在增加下一句的预测了"""
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 下面的3，考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
                max_len - len(token_ids)), dtype=torch.long))  # 输入序列的词元索引 - 给序列后添加<pad> - 对应索引1 shape:(max_len, ) todo 这里的索引1 也许会有问题
        all_segments.append(torch.tensor(segments + [0] * (
                max_len - len(segments)), dtype=torch.long))  # 上下句判断 - 给序列后添加 0 shape:(max_len, )
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))  # 输入序列有效长度 shape:(1, )
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
                max_num_mlm_preds - len(pred_positions)), dtype=torch.long))  # 预测位置 - 给序列后添加 0 shape: (max_num_mlm_preds, )
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] *
                                            (max_num_mlm_preds - len(pred_positions)),
                                            dtype=torch.float32))  # 这里可以理解为预测的掩码矩阵，实际上暂时是向量 - Y_mask
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
                max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))  #预测词元索引 - 给序列后添加 0 shape: (max_num_mlm_preds, )
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))  # 判断是否是下一句
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


def count_corpus(tokens):
    """统计词元的频率
    Defined in :numref:`sec_text_preprocessing`"""
    import collections
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将所有的词元信息展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 计算词元的出现频率，随后进行排序。 - 利用 collections库
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 添加未知词元<unk>，并将其索引定义为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元
    Defined in :numref:`sec_text_preprocessing`"""
    if token == 'word':
        text = [line.split() for line in lines]
        return text
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # step1：获取下一句子预测任务的数据 - 构造预测下一个句子的文本数据集
        # 此时，examples为一个列表，包含了 tokens - 储存词元的列表, segments - 用0和1表示是上一句还是下一句, is_next - 下一个句子是否为真
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # step2：获取遮蔽语言模型任务的数据
        # 此时，examples为一个列表，包含了 token_ids - 储存词元索引的列表, pred_positions - 掩码需要预测的位置,
        # mlm_pred_label_ids - 掩码需要预测位置的词元索引, segments - 用0和1表示是上一句还是下一句, is_next - 下一个句子是否为真
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len, mode='train'):
    """加载WikiText-2数据集"""
    # data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    data_dir = r'.\wikitext-2-v1\wikitext-2//'
    paragraphs = _read_wiki(data_dir, mode=mode)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab


if __name__ == '__main__':
    # 1. 设计一个函数可以将输入的token_a和token_b加上特殊类别词元<cls>和特殊分隔词元<sep>

    # 2. 读取WikiText-2数据集，并形成_WikiTextDataset类用于数据迭代
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len, mode='train')

    # 显示迭代器其中一次的输出结果
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
         mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
              pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
              nsp_y.shape)
        break

    pass
