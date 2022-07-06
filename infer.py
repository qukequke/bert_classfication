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
from pprint import pprint
import argparse
from importlib import import_module
from transformers import pipeline
from config import model_dict


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert', type=str, required=False, help='使用什么模型')
    parser.add_argument('--pretrain_dir', default='./models/xinwen/bert/', type=str, required=False, help='预训练结果文件夹')
    args = parser.parse_args()
    return args


def eval_object(object_):
    if '.' in object_:
        module_, class_ = object_.rsplit('.', 1)
        module_ = import_module(module_)
        return getattr(module_, class_)
    else:
        module_ = import_module(object_)
        return module_


class Inferenve:
    def get_model_tokenizer(self):
        args = set_args()
        ClassifyClass = eval_object(model_dict[args.model][1])
        TokenizerClass = eval_object(model_dict[args.model][0])
        ConfigClass = eval_object(model_dict[args.model][2])
        model = ClassifyClass.from_pretrained(args.pretrain_dir)
        tokenizer = TokenizerClass.from_pretrained(args.pretrain_dir)
        config = ConfigClass.from_pretrained(args.pretrain_dir)
        return model, tokenizer, config

    def __init__(self):
        model, tokenizer, config = self.get_model_tokenizer()
        self.classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, config=config,
                                   return_all_scores=True)

    def get_ret(self, sentences):
        r = self.classifier(sentences)
        return r


if __name__ == "__main__":
    infer = Inferenve()
    while True:
        data = input("请输入要预测的句子\n")
        a = infer.get_ret(data)
        pprint(sorted(a[0], key=lambda x: x['score'], reverse=False))
