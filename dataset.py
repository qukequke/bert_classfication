# -*- coding: utf-8 -*-
# import csv
from collections import Counter

import re
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
import pandas as pd
import torch
from config import *


class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, args, type='train'):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.type_ = type
        # self.args = args

        self.data = self.get_input(args)
        # print(self.args)
        # print('----')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]
        return {k: v[idx] for k, v in self.data.items()}

    def plot_data_info(self, labels, args):
        import matplotlib.pyplot as plt
        # 支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        print('各个分类标签个数')
        # label_count = {id2label[k]: v for k, v in Counter(labels).items()}
        label_count = Counter(labels)
        x = list(label_count.keys())
        y = list(label_count.values())
        plt.barh(x,y,height=0.7,left=0,color='c',edgecolor='r')
        plt.savefig(args.data_info_file)

        # labels = torch.Tensor([eval(i) for i in labels]).type(torch.long)
    # 获取文本与标签
    def get_input(self, args):
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
        # mode = 'train'
        if self.type_ == 'train':
            file = args.train_file
        elif self.type_ == 'dev':
            file = args.dev_file
        else:
            file = args.test_file

        # if isinstance(file, list):
        #     mode = 'test'
        #     print('测试模式')
        #     df = pd.DataFrame([{'text': i} for i in file])
        # else:
        if self.type_ == 'train' and args.read_n_num:
            df = pd.read_csv(file, engine='python', encoding=args.csv_encoding, error_bad_lines=False, nrows=args.read_n_num, sep=args.csv_sep)
        else:
            df = pd.read_csv(file, engine='python', encoding=args.csv_encoding, error_bad_lines=False, sep=args.csv_sep)

        self.length = len(df)
        if self.type_ == 'train':
            self.bert_tokenizer.model_max_length = args.max_seq_len
        print(f"数据集个数为{len(df)}")

        sentences = df[args.text_col_name].tolist()
        if args.class_col_name:
            labels = df[args.class_col_name].tolist()
        else:
            print('没有标签，全部设置为1')
            labels = [1 for _ in range(self.length)]
        data = self.bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # 返回结果为类字典 {'input_ids':[[1,1,1,], [1,2,1]], 'token_type_ids':矩阵, 'attention_mask':矩阵,...}
        self.labels = labels.copy()
        if args.problem_type == 'multi_label_classification':
            # labels = torch.Tensor([i.type(torch.long) for i in labels])
            # labels = torch.Tensor(labels).type(torch.long)
            labels = torch.Tensor([eval(i) for i in labels]).type(torch.float)
        else:
            if self.type_ == 'train':
                self.plot_data_info(labels, args)
        labels = torch.Tensor(labels).type(torch.long)
        data['labels'] = labels
        print('输入例子')
        print(sentences[0] if isinstance(sentences[0], str) else sentences[0][0])
        for k, v in data.items():
            print(k)
            print(v[0])
        print(f"实际序列转换后的长度为{len(data['input_ids'][0])}, 设置最长为{args.max_seq_len}")
        return data


class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, args, type='train'):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.type_ = type
        # self.args = args

        self.data = self.get_input(args)
        # print(self.args)
        # print('----')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]
        return {k: v[idx] for k, v in self.data.items()}

    def plot_data_info(self, labels, args):
        import matplotlib.pyplot as plt
        # 支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        print('各个分类标签个数')
        # label_count = {id2label[k]: v for k, v in Counter(labels).items()}
        label_count = Counter(labels)
        x = list(label_count.keys())
        y = list(label_count.values())
        plt.barh(x,y,height=0.7,left=0,color='c',edgecolor='r')
        plt.savefig(args.data_info_file)

        # labels = torch.Tensor([eval(i) for i in labels]).type(torch.long)
    # 获取文本与标签
    def get_input(self, args):
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
        # mode = 'train'
        if self.type_ == 'train':
            file = args.train_file
        elif self.type_ == 'dev':
            file = args.dev_file
        else:
            file = args.test_file

        # if isinstance(file, list):
        #     mode = 'test'
        #     print('测试模式')
        #     df = pd.DataFrame([{'text': i} for i in file])
        # else:
        if self.type_ == 'train' and args.read_n_num:
            df = pd.read_csv(file, engine='python', encoding=args.csv_encoding, error_bad_lines=False, nrows=args.read_n_num, sep=args.csv_sep)
        else:
            df = pd.read_csv(file, engine='python', encoding=args.csv_encoding, error_bad_lines=False, sep=args.csv_sep)

        self.length = len(df)
        # if self.type_ == 'train':
        self.bert_tokenizer.model_max_length = args.max_seq_len
        print(f"数据集个数为{len(df)}")

        sentences = df[args.text_col_name].tolist()
        if args.class_col_name:
            labels = df[args.class_col_name].tolist()
        else:
            print('没有标签，全部设置为1')
            labels = [1 for _ in range(self.length)]
        data = self.bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # 返回结果为类字典 {'input_ids':[[1,1,1,], [1,2,1]], 'token_type_ids':矩阵, 'attention_mask':矩阵,...}
        self.labels = labels.copy()
        if args.problem_type == 'multi_label_classification':
            # labels = torch.Tensor([i.type(torch.long) for i in labels])
            # labels = torch.Tensor(labels).type(torch.long)
            labels = torch.Tensor([eval(i) for i in labels]).type(torch.float)
        else:
            if self.type_ == 'train':
                self.plot_data_info(labels, args)
            labels = torch.Tensor(labels).type(torch.long)
        data['labels'] = labels
        print('输入例子')
        print(sentences[0] if isinstance(sentences[0], str) else sentences[0][0])
        for k, v in data.items():
            print(k)
            print(v[0])
        print(f"实际序列转换后的长度为{len(data['input_ids'][0])}, 设置最长为{args.max_seq_len}")
        return data

if __name__ == '__main__':
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    bert_tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
    dataset = DataPrecessForSentence(bert_tokenizer, train_file)
    for i in dataset:
        print(i)
        break
    # print(len(dataset))
    d = DataLoader(dataset, batch_size=20)
    # print(len(d))
    for ii, i in enumerate(d):
        print(i)
        print(ii)
        break

    # tokenizer(["Hello, my dog is cute", 'plase call me shen'], return_tensors="pt", padding=True, truncation=True)
