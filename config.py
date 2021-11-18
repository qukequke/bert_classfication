# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/18 15:24
@Author  : quke
@File    : config.py
@Description:
---------------------------------------
'''
epochs = 30
batch_size = 128
lr = 1e-4  # 学习率
patience = 40  # early stop 不变好 就停止
max_grad_norm = 20.0  # 梯度修剪
train_file = "data/xinwen.csv"
dev_file = "data/xinwen.csv"
target_file = 'models/best.pth.tar'  # 模型存储路径
# retrain = False
bert_path_or_name = 'hfl/chinese-roberta-wwm-ext'  # 使用模型
num_labels = 10  # 文本分类个数
max_seq_len = 103
n_nums = None   # 读取csv行数，因为有时候测试需要先少读点

checkpoint = None  # 设置模型路径  会继续训练

# csv_rows = ['premise', 'hypothesis', 'label']
csv_rows = ['text', 'class']  # csv的行标题，文本 和 类（目前类必须是数字）
input_mode = 'add'  # 两句话加一起
# input_mode = 'split'  # 两句话分开输入

test_file = "data/xinwen.csv"