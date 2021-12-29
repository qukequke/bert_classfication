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
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, BertTokenizer
from model import BertModelTest
from utils import test
from dataset import DataPrecessForSentence
from config import *


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

    def get_ret(self, sentence):
        print("\t* Loading test data...")
        test_data = DataPrecessForSentence(self.bert_tokenizer, sentence)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        print("\t* Building model...")
        print(20 * "=", " Testing roberta model on device: {} ".format(self.device), 20 * "=")
        batch_time, total_time, accuracy, all_labels, all_pred = test(self.model, test_loader)
        return all_pred[0]
        # print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time, total_time, (accuracy*100)))
        # df = pd.read_csv(test_file, engine='python', encoding=csv_encoding, error_bad_lines=False)
        # df['pred'] = all_pred
        # df['should_label'] = all_labels
        # df.to_csv(test_pred_out, index=False, encoding='utf-8')


if __name__ == "__main__":
    with open('data/yewuliucheng/label.txt', 'r', encoding='utf-8') as f:
        labels = f.readlines()
    id2label_dict = {i: label for i, label in enumerate(labels)}
    infer = Inferenve()
    while True:
        data = input("请输入要预测的句子\n")
        print(data)
        a = infer.get_ret(data)
        print(a)
        print(id2label_dict[a])
