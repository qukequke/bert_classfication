# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/18 15:24
@Author  : quke
@File    : config.py
@Description:
---------------------------------------
'''

model_dict = {  # 这里可以添加更多模型
    'bert': (
        'transformers.BertTokenizer',
        'transformers.BertForSequenceClassification',
        'transformers.BertConfig',
        'bert-base-chinese',  # 使用模型参数
    ),
    'bert_wwm': (
        'transformers.BertTokenizer',
        'transformers.BertForSequenceClassification',
        'transformers.BertConfig',
        'hfl/chinese-bert-wwm',  # 使用模型参数
    ),
    'raw_bert': (  # 原始bert，为了可以调整损失函数
        'transformers.BertTokenizer',
        'transformers.BertModel',
        'transformers.BertConfig',
        'bert-base-chinese',  # 使用模型参数
    ),

    'roberta': (
        'transformers.BertTokenizer',
        'transformers.RobertaForSequenceClassification',
        'transformers.RobertaConfig',
        'hfl/chinese-roberta-wwm-ext'
    ),
    'ernie': (
        'transformers.AutoTokenizer',
        'transformers.BertForSequenceClassification',
        'transformers.AutoConfig',
        "nghuyong/ernie-1.0",  # 使用模型参数
    ),
    'ernie_healthy': (
        'transformers.AutoTokenizer',
        'transformers.BertForSequenceClassification',
        'transformers.AutoConfig',
        # "nghuyong/ernie-1.0",  # 使用模型参数
        "nghuyong/ernie-health-zh",  # 使用模型参数
    ),
    'albert': (
        'transformers.AutoTokenizer',
        'transformers.AlbertForSequenceClassification',
        'transformers.AutoConfig',
        "voidful/albert_chinese_tiny",  # 使用模型参数
    ),
    'reformer': (
        'transformers.RoFormerTokenizer',
        'transformers.RoFormerForSequenceClassification',
        'transformers.RoFormerConfig',
        "junnyu/roformer_chinese_base",  # 使用模型参数
    )


    # 'bert_token_classify': ('transformers.BertTokenizer', 'transformers.BertForTokenClassification', 'transformers.BertConfig'),
}
# MODEL = 'roberta'
# MODEL = 'ernie'
# MODEL = 'ernie_healthy'
# MODEL = 'albert'
# MODEL = 'raw_bert'
# MODEL = 'bert'
# MODEL = 'bert_wwm'
# MODEL = 'reformer'

# epochs = 3
# batch_size = 16
# # batch_size = 64
# lr = 1e-5  # 学习率
# patience = 40  # early stop 不变好 就停止
# max_grad_norm = 10.0  # 梯度修剪
# # target_file = 'models/yewuliucheng/best.pth.tar'  # 模型存储路径
# dir_name = 'haodaifu'
# # dir_name = 'haodaifu2'
# # dir_name = 'senti_classify'
# # dir_name = 'jigouduixiang'
# # dir_name = 'xinwen'
# # dir_name = 'yewuliucheng'
# # dir_name = 'gangwei'
# # dir_name = 'jigouduixiang'
# problem_type = 'single_label_classification'  # 单分类
# # problem_type = 'multi_label_classification'  # 多分类 损失函数不同 训练数据label需要是列表
# target_file = f'models/{dir_name}/{MODEL}_best.pth.tar'  # 模型存储路径
# pretrain_dir = f'./models/{dir_name}/{MODEL}/'  # 模型存储路径,存储了两种类型，一种是torch.save 一种是model.from_pretrained
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
# # checkpoint = f'models/{dir_name}/{MODEL}_best.pth.tar'   # 设置模型路径  会继续训练
# checkpoint = None  # 设置模型路径设置成target_file可以继续训练, None则重新训练
# max_seq_len = 256  # 序列最长长度
# n_nums = None  # 读取csv行数，因为有时候测试需要先少读点 None表示读取所有
# # n_nums = 1000 # 读取csv行数，因为有时候测试需要先少读点 None表示读取所有
# freeze_bert_head = False  # freeze bert提取特征部分的权重
#
# test_pred_out = f"data/{dir_name}/test_data_predict.csv"
#
# # 切换任务时 数据配置
# # num_labels = 10  # 文本分类个数，应该等于class行数
# csv_rows = ['text', 'class']  # csv的行标题，文本 和 类（目前类必须是数字）
# csv_sep = ','
# # csv_sep = '\t'
# input_mode = 'add'  # 两句话加一起, 对应两句话做输入时的参数,单据输入不用管
#
# # input_mode = 'split'  # 两句话分开输入
#
# train_file = f"data/{dir_name}/train.csv"
# dev_file = f"data/{dir_name}/dev.csv"
# test_file = f"data/{dir_name}/test.csv"
# json_dict = f"data/{dir_name}/class.txt"
# data_info_file = f"data/{dir_name}/label_count.png"
# csv_encoding = 'utf-8'
# # csv_encoding = 'gbk'
#
# # 训练还是测试任务
# MODE = 'train'
# # bert_path_or_name = 'bert-base-chinese'  # 使用模型
# # bert_path_or_name = 'hfl/chinese-roberta-wwm-ext'  # 使用模型
#
# # 下面为文本不均衡过采样参数，待优化
# use_sample = False
# validate_iter = 20  # 过采样情况下 手动设置迭代次数
# PRINT_TRAIN_COUNT = False  # 是否打印训练集各类个数，样本不均衡时调试用
#
#
#
# with open(json_dict, 'r', encoding='utf-8') as f:
#     classes = f.readlines()
# label2id = {label.strip():i for i, label in enumerate(classes)}
# id2label = {v:k for k, v in label2id.items()}
# num_labels = len(classes)
# print(f"num_labels 是{num_labels}")
#
#
#
# # 自动创建文件夹
# if not os.path.exists(pretrain_dir):
#     os.makedirs(pretrain_dir)
#
# target_dir = os.path.dirname(target_file)
# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)
