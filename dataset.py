# -*- coding: utf-8 -*-
import csv
from torch.utils.data import Dataset
from hanziconv import HanziConv
import pandas as pd
import torch
from config import *


class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, LCQMC_file):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        # self.max_seq_len = max_char_len
        print(f'最大长度为{max_seq_len}')
        # self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(LCQMC_file)
        # self.seqs, self.seq_masks, self.labels = self.get_input(LCQMC_file)
        self.data = self.get_input(LCQMC_file)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]
        return {k: v[idx] for k, v in self.data.items()}

    # 获取文本与标签
    def get_input(self, file):
        """
        通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        入参:
            dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """
        if n_nums:
            df = pd.read_csv(file, engine='python', error_bad_lines=False, nrows=n_nums)
        else:
            df = pd.read_csv(file, engine='python', error_bad_lines=False)
        self.length = len(df)
        print(f"数据集个数为{len(df)}")
        if len(csv_rows) == 2:
            sentences = df[csv_rows[0]].tolist()
            labels = df[csv_rows[1]].tolist()
        elif len(csv_rows) == 3:
            if input_mode == 'add':
                sentences_1 = df[csv_rows[0]].tolist()
                sentences_2 = df[csv_rows[1]].tolist()
                labels = df[csv_rows[2]].tolist()
                sentences = [i + j for i, j in zip(sentences_1, sentences_2)]
            else:
                pass

        else:
            raise AssertionError("csv输入数据不对")
        self.bert_tokenizer.model_max_length = max_seq_len
        # seqs = [self.bert_tokenizer(i, padding=True, truncation=True)['input_ids'] for i in sentences]
        data = self.bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # 返回结果为类字典 {'input_ids':[[1,1,1,], [1,2,1]], 'token_type_ids':矩阵, 'attention_mask':矩阵,...}
        labels = torch.Tensor(labels).type(torch.long)
        data['labels'] = labels
        return data


if __name__ == '__main__':
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    bert_tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
    dataset = DataPrecessForSentence(bert_tokenizer, train_file)
    # for i in data:
    #     print(i)
    #     break
    print(len(dataset))
    d = DataLoader(dataset, batch_size=20)
    print(len(d))
    for ii, i in enumerate(d):
        # print(i)
        print(ii)
        break

    # tokenizer(["Hello, my dog is cute", 'plase call me shen'], return_tensors="pt", padding=True, truncation=True)