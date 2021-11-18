# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig, RobertaForSequenceClassification, RobertaConfig
from config import *


class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained(bert_path_or_name, num_labels=10)
        self.device = torch.device("cuda")

    def forward(self, **input):
        # data = self.bert(input_ids=batch_seqs, attention_mask=attention, labels=labels)
        data = self.bert(**input)


        loss = data.loss
        logits = data.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BertModelTest(nn.Module):
    def __init__(self):
        super(BertModelTest, self).__init__()
        config = RobertaConfig.from_pretrained(bert_path_or_name, num_labels=num_labels)
        self.bert = RobertaForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device("cuda")

    def forward(self, batch_seqs, attention, labels):
        # data = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)
        data = self.bert(input_ids=batch_seqs, attention_mask=attention, labels=labels)
        loss = data.loss
        logits = data.logits
        probabilities = nn.functional.softmax(logits, dim=-1)

        return loss, logits, probabilities
