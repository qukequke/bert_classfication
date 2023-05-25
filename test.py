# -*- coding: utf-8 -*-
import argparse

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sys import platform
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, BertTokenizer
# from model import BertModel
from utils import test, eval_object
from dataset import DataPrecessForSentence
from config import *



def get_model_tokenizer(args):
    ClassifyClass = eval_object(model_dict[args.model][1])
    TokenizerClass = eval_object(model_dict[args.model][0])
    model = ClassifyClass.from_pretrained(args.pretrain_dir)
    model = model.to(args.device)
    tokenizer = TokenizerClass.from_pretrained(args.pretrain_dir)
    return model, tokenizer



def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert', type=str, required=False, help='使用什么模型')
    parser.add_argument('--problem_type', default='single_label_classification', type=str, required=False, help='单标签分类还是多标签分类')
    parser.add_argument('--dir_name', default='xinwen', type=str, required=False, help='训练集存放目录,里面包含train.csv test.csv dev.csv')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='训练的batch size')
    parser.add_argument('--max_seq_len', default=150, type=int, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--text_col_name', default='text', type=str, required=False, help='train.csv文本列名字')
    parser.add_argument('--class_col_name', default=None, type=str, required=False, help='train.csv标签列名字')
    parser.add_argument('--csv_sep', default=',', type=str, required=False, help='csv列间隔')
    parser.add_argument('--csv_encoding', default='utf-8', type=str, required=False, help='csv编码格式')
    args = parser.parse_args()
    return args

def init(args):
    """
    增加其他参数，以及创建文件夹
    """
    # target_file = f'models/{args.dir_name}/{args.model}_best.pth.tar'  # 模型存储路径
    pretrain_dir = f'./models/{args.dir_name}/{args.model}/'  # 模型存储路径,存储了两种类型，一种是torch.save 一种是model.from_pretrained
    test_pred_out = f"data/{args.dir_name}/test_data_predict.csv"
    # train_file = f"data/{args.dir_name}/train.csv"
    # dev_file = f"data/{args.dir_name}/dev.csv"
    test_file = f"data/{args.dir_name}/test.csv"
    json_dict = f"data/{args.dir_name}/class.txt"
    # data_info_file = f"data/{args.dir_name}/label_count.png"


    with open(json_dict, 'r', encoding='utf-8') as f:
        classes = f.readlines()
    label2id = {label.strip():i for i, label in enumerate(classes)}
    id2label = {v:k for k, v in label2id.items()}
    num_labels = len(classes)
    print(f"num_labels 是{num_labels}")

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.pretrain_dir = pretrain_dir
    args.test_pred_out = test_pred_out
    args.test_file = test_file
    args.id2label = id2label
    args.label2id = label2id
    args.num_labels = num_labels


def main():
    args = set_args()
    init(args)
    model, tokenizer = get_model_tokenizer(args)
    # print(20 * "=", " Preparing for testing ", 20 * "=")
    # print(target_file)
    # print("\t* Loading test data...")
    test_data = DataPrecessForSentence(tokenizer, args, 'test')
    # test_data = tokenizer(pd.read_csv(args.test_file)[args.text_col_name].tolist(),
    #                       padding=True, truncation=True,return_tensors='pt')
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
    for dict_ in test_loader:
        print(dict_)
    print("\t* Building model...")
    # model = BertModelTest().to(device)
    # model = BertModel().to(device)
    # model.load_state_dict(checkpoint["model"])

    print(20 * "=", " Testing model on device: {} ".format(args.device), 20 * "=")
    batch_time, total_time, accuracy, all_labels, all_pred = test(model, test_loader, args)
    print(
        "\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time,
                                                                                                            total_time,
                                                                                                            (
                                                                                                                    accuracy * 100)))
    df = pd.read_csv(args.test_file, engine='python', encoding=args.csv_encoding, error_bad_lines=False)
    df['pred'] = [i.cpu().numpy() for i in all_pred]
    # df['pred'] = all_pred.cpu()
    if args.problem_type == 'multi_label_classification':
        # df['ret'] = df['pred'] == (df[csv_rows[-1]].apply(lambda x: eval(x)))
        df['all_pred'] = [[args.id2label[jj] for jj, j in enumerate(i) if j] for i in all_pred]
    else:
        df['pred'] = df['pred'].apply(int)
        # df['ret'] = df['pred'] == df[csv_rows[-1]]
    # print(confusion_matrix(df[csv_rows[-1]].tolist(), df['pred'].tolist()))
    # print(classification_report(all_pred, all_labels, target_names=id2label_dict.values()))
    df.to_csv(args.test_pred_out, index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
