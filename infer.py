# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/12/29 17:05
@Author  : quke
@File    : test_online.py
@Description:
---------------------------------------
'''
# -*- coding: utf-8 -*-
import time

from sys import platform
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import re
from torch.utils.data import Dataset
import pandas as pd
import torch

# dir_name = 'yewuliucheng'
dir_name = 'gangwei'
target_file = f'models/{dir_name}/best.pth.tar'  # 模型存储路径
bert_path_or_name = 'bert-base-chinese'  # 使用模型
batch_size = 32
csv_rows = ['text', 'label']
max_seq_len = 103
# num_labels = 10
num_labels = 7
problem_type = 'multi_label_classification'
# problem_type = 'single_label_classification'  # 单分类


class BertModelTest(nn.Module):
    def __init__(self):
        super(BertModelTest, self).__init__()
        config = BertConfig.from_pretrained(bert_path_or_name, num_labels=num_labels, problem_type=problem_type)
        self.bert = BertForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device("cuda")

    def forward(self, **input_):
        if 'labels' in input_:
            input_.pop('labels')  # 预测不要标签
        data = self.bert(**input_)
        loss = data.loss
        logits = data.logits
        probabilities = nn.functional.softmax(logits, dim=-1)

        return loss, logits, probabilities


def correct_predictions(output_probabilities, targets):
    if problem_type == 'multi_label_classification':
        preds = nn.functional.sigmoid(output_probabilities)
        preds2 = torch.where(preds > 0.50001, torch.ones(preds.shape).to('cuda'), torch.zeros(preds.shape).to('cuda'))
        correct = sum((i == j).all() for i, j in zip(preds2, targets))
    else:
        _, out_classes = output_probabilities.max(dim=1)
        correct = (out_classes == targets).sum()
    return correct.item()


def a_test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    all_pred = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for tokened_data_dict in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
            labels = tokened_data_dict['labels']
            # seqs, masks, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_labels.to(device)
            _, _, probabilities = model(**tokened_data_dict)  # [batch_size, n_label]
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            # all_prob.extend(probabilities[:, 1].cpu().numpy())
            if problem_type == 'multi_label_classification':
                preds = nn.functional.sigmoid(probabilities)
                out_classes = torch.where(preds > 0.50001, torch.ones(preds.shape).to('cuda'),
                                     torch.zeros(preds.shape).to('cuda'))
                # out_classes = out_classes.type(torch.long)
            else:
                _, out_classes = probabilities.max(dim=1)
            # _, out_classes = probabilities.max(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_pred.extend(out_classes.cpu().numpy())
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, all_labels, all_pred


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
        mode = 'train'
        if isinstance(file, list):
            mode = 'test'
            print('测试模式')
            df = pd.DataFrame([{'text': i} for i in file])
        df[csv_rows[0]] = df[csv_rows[0]].apply(lambda x: re.sub('^\d+(\.\d+)+', '', x))  # 去掉1.1.11之类

        self.length = len(df)
        self.bert_tokenizer.model_max_length = max_seq_len
        print(f"数据集个数为{len(df)}")
        # if MODE == 'test':  # 测试模式
        if mode == 'test':  # 测试模式
            labels = [1 for _ in range(self.length)]
            # if len(csv_rows) == 1:
            sentences = df[csv_rows[0]].tolist()
            # else:
            #     sentences = [df[csv_rows[0]].tolist(), df[csv_rows[1]].tolist()]
        data = self.bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # 返回结果为类字典 {'input_ids':[[1,1,1,], [1,2,1]], 'token_type_ids':矩阵, 'attention_mask':矩阵,...}
        self.labels = labels.copy()
        labels = torch.Tensor(labels).type(torch.long)
        data['labels'] = labels
        print('输入例子')
        print(sentences[0] if isinstance(sentences[0], str) else sentences[0][0])
        for k, v in data.items():
            print(k)
            print(v[0])
        print(f"实际序列转换后的长度为{len(data['input_ids'][0])}, 设置最长为{max_seq_len}")
        return data


class Inferenve:
    def __init__(self):
        self.device = torch.device("cuda")
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
        print(20 * "=", " Preparing for testing ", 20 * "=")
        print(target_file)
        if platform == "linux" or platform == "linux2":
            checkpoint = torch.load(target_file)
        else:
            checkpoint = torch.load(target_file, map_location=self.device)
        # Retrieving model parameters from checkpoint.
        self.model = BertModelTest().to(self.device)
        self.model.load_state_dict(checkpoint["model"])

    def get_ret(self, sentences):
        print("\t* Loading test data...")
        if isinstance(sentences, str):
            sentences = list(sentences)
        test_data = DataPrecessForSentence(self.bert_tokenizer, sentences)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        print("\t* Building model...")
        print(20 * "=", " Testing roberta model on device: {} ".format(self.device), 20 * "=")
        batch_time, total_time, accuracy, all_labels, all_pred = a_test(self.model, test_loader)
        return all_pred

if __name__ == "__main__":
    with open(f'data/{dir_name}/label.txt', 'r', encoding='utf-8') as f:
        labels = f.readlines()
    id2label_dict = {i: label.strip() for i, label in enumerate(labels)}
    infer = Inferenve()
    while True:
        data = input("请输入要预测的句子\n")
        a = infer.get_ret([data, ])[0]
        if problem_type == 'multi_label_classification':
            print([id2label_dict[i] for i in a if i])
        else:
            print(id2label_dict[a])
