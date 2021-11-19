# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/18 15:24
@Author  : quke
@File    : config.py
@Description:
---------------------------------------
'''
model_dict = {
    'bert': ('transformers.BertTokenizer', 'transformers.BertForSequenceClassification', 'transformers.BertConfig'),
    'roberta': (
        'transformers.BertTokenizer', 'transformers.RobertaForSequenceClassification', 'transformers.RobertaConfig'),
}
epochs = 70
# batch_size = 128
batch_size = 64
lr = 1e-5  # 学习率
patience = 40  # early stop 不变好 就停止
max_grad_norm = 10.0  # 梯度修剪
target_file = 'models/best.pth.tar'  # 模型存储路径
# checkpoint = 'models/best2.pth.tar'   # 设置模型路径  会继续训练
checkpoint = None  # 设置模型路径  会继续训练
max_seq_len = 103
n_nums = None  # 读取csv行数，因为有时候测试需要先少读点 None表示读取所有
freeze_bert_head = False  # freeze bert提取特征部分的权重

test_pred_out = "data/test_data_predict_2.csv"

# 切换任务时 数据配置
# num_labels = 10  # 文本分类个数
num_labels = 2  # 文本分类个数
# csv_rows = ['premise', 'hypothesis', 'label']
# csv_rows = ['text', 'class']  # csv的行标题，文本 和 类（目前类必须是数字）
csv_rows = ['sentence1', 'sentence2', 'label']  # csv的行标题，文本 和 类（目前类必须是数字）
# input_mode = 'add'  # 两句话加一起
input_mode = 'split'  # 两句话分开输入

# train_file = "data/xinwen_train.csv"
# dev_file = "data/xinwen_dev.csv"
# test_file = "data/xinwen_test.csv"
train_file = "data/blood_train.csv"
dev_file = "data/blood_train.csv"
test_file = "data/blood_train.csv"
csv_encoding = 'utf-8'
# csv_encoding = 'gbk'

# 训练还是测试任务
MODE = 'train'
MODEL = 'roberta'
# MODEL = 'bert'
# bert_path_or_name = 'bert-base-chinese'  # 使用模型
bert_path_or_name = 'hfl/chinese-roberta-wwm-ext'  # 使用模型
