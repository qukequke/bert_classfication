# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig, RobertaForSequenceClassification, RobertaConfig, AutoModelForSequenceClassification, AutoConfig
from config import *
from utils import eval_object


ClassifyClass = eval_object(model_dict[MODEL][1])
ClassifyConfig = eval_object(model_dict[MODEL][2])

class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        # self.bert = RobertaForSequenceClassification.from_pretrained(bert_path_or_name, num_labels=10)
        self.bert = ClassifyClass.from_pretrained(bert_path_or_name, num_labels=num_labels)
        # self.bert = AutoModelForSequenceClassification.from_pretrained(bert_path_or_name, num_labels=10)
        self.device = torch.device("cuda")

    def forward(self, **input_):
        # data = self.bert(input_ids=batch_seqs, attention_mask=attention, labels=labels)
        data = self.bert(**input_)
        loss = data.loss
        logits = data.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BertModelTest(nn.Module):
    def __init__(self):
        super(BertModelTest, self).__init__()
        # config = RobertaConfig.from_pretrained(bert_path_or_name, num_labels=num_labels)
        config = ClassifyConfig.from_pretrained(bert_path_or_name, num_labels=num_labels)
        # config = AutoConfig.from_pretrained(bert_path_or_name, num_labels=num_labels)
        # self.bert = RobertaForSequenceClassification(config)  # /bert_pretrain/
        self.bert = ClassifyClass(config)  # /bert_pretrain/
        # self.bert = AutoModelForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device("cuda")

    def forward(self, **input_):
        if 'labels' in input_:
            input_.pop('labels')  # 预测不要标签
        # data = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)
        data = self.bert(**input_)
        loss = data.loss
        logits = data.logits
        probabilities = nn.functional.softmax(logits, dim=-1)

        return loss, logits, probabilities
