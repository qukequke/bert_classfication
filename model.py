# -*- coding: utf-8 -*-
import torch
from torch import nn
# from transformers import BertForSequenceClassification, BertConfig, RobertaForSequenceClassification, RobertaConfig, AutoModelForSequenceClassification, AutoConfig
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel
# from transformers.s import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from config import *
from utils import eval_object


def get_model(args):
    ClassifyClass = eval_object(model_dict[args.model][1])
    ClassifyConfig = eval_object(model_dict[args.model][2])
    bert_path_or_name = model_dict[args.model][-1]

    # config = AutoConfig.from_pretrained(bert_path_or_name, num_labels=num_labels)
    # self.bert = RobertaForSequenceClassification(config)  # /bert_pretrain/

    # class BertModel(nn.Module):

    config = ClassifyConfig.from_pretrained(bert_path_or_name, num_labels=args.num_labels,
                                            problem_type=args.problem_type, label2id=args.label2id,
                                            id2label=args.id2label)
    model = ClassifyClass.from_pretrained(bert_path_or_name, config=config)  # /bert_pretrain/
    # model = BertModel()  # /bert_pretrain/
    model = model.to(args.device)
    return model

#
# class BertModelTest(nn.Module):
#     def __init__(self):
#         super(BertModelTest, self).__init__()
#         # config = RobertaConfig.from_pretrained(bert_path_or_name, num_labels=num_labels)
#         config = ClassifyConfig.from_pretrained(bert_path_or_name, num_labels=num_labels, problem_type=problem_type)
#         # config = AutoConfig.from_pretrained(bert_path_or_name, num_labels=num_labels)
#         # self.bert = RobertaForSequenceClassification(config)  # /bert_pretrain/
#         self.bert = ClassifyClass(config)  # /bert_pretrain/
#         # self.bert = AutoModelForSequenceClassification(config)  # /bert_pretrain/
#         self.device = torch.device("cuda")
#
#     def forward(self, **input_):
#         if 'labels' in input_:
#             input_.pop('labels')  # 预测不要标签
#         # data = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)
#         data = self.bert(**input_)
#         loss = data.loss
#         logits = data.logits
#         probabilities = nn.functional.softmax(logits, dim=-1)
#
#         return loss, logits, probabilities


# import torch.nn.functional as F
#
# def compute_kl_loss(p, q, pad_mask=None):
#     p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
#     q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
#
#     # pad_mask is for seq-level tasks
#     if pad_mask is not None:
#         p_loss.masked_fill_(pad_mask, 0.)
#         q_loss.masked_fill_(pad_mask, 0.)
#
#     # You can choose whether to use function "sum" and "mean" depending on your task
#     p_loss = p_loss.sum()
#     q_loss = q_loss.sum()
#
#     loss = (p_loss + q_loss) / 2
#     return loss
#
# class BertModel(nn.Module):
#     def __init__(self):
#         super(BertModel, self).__init__()
#         # self.bert = RobertaForSequenceClassification.from_pretrained(bert_path_or_name, num_labels=10)
#         # self.bert = ClassifyClass.from_pretrained(bert_path_or_name, num_labels=num_labels, problem_type=problem_type)
#         config = ClassifyConfig.from_pretrained(bert_path_or_name, num_labels=num_labels, problem_type=problem_type)
#         self.bert = ClassifyClass(config)  # /bert_pretrain/
#         self.config = config
#
#         # self.bert = AutoModelForSequenceClassification.from_pretrained(bert_path_or_name, num_labels=10)
#         self.device = torch.device("cuda")
#         # config = ClassifyConfig.from_pretrained(bert_path_or_name, num_labels=num_labels, problem_type=problem_type)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#
#     def forward(self, **input_):
#         # labels = input_['labels']
#         # if 'labels' in input_:
#         labels = input_.pop('labels')  # 预测不要标签
#         outputs = self.bert(**input_)
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         pooled_output2 = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         logits2 = self.classifier(pooled_output2)
#
#         loss = None
#         if labels is not None:
#             if self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 # loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#                 loss = 0.5 * loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1)) \
#                        + 0.5 * loss_fct(logits2.view(-1, self.config.num_labels), labels.view(-1)) \
#                        +  compute_kl_loss(logits, logits2)
#
#
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         # if not return_dict:
#         # output = (logits,) + outputs[2:]
#         # return ((loss,) + output) if loss is not None else output
#
#         data = SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#         loss = data.loss
#         logits = data.logits
#         probabilities = torch.softmax(logits, dim=-1)
#         return loss, logits, probabilities





