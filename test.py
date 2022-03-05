# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, BertTokenizer
from model import BertModelTest
from utils import test
from dataset import DataPrecessForSentence
from config import *
bert_path_or_name = model_dict[MODEL][-1]

def main():

    device = torch.device("cuda")
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path_or_name)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    print(target_file)
    if platform == "linux" or platform == "linux2":

        checkpoint = torch.load(target_file)
    else:
        checkpoint = torch.load(target_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    print("\t* Loading test data...")    
    test_data = DataPrecessForSentence(bert_tokenizer, test_file)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("\t* Building model...")
    model = BertModelTest().to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, all_labels, all_pred = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time, total_time, (accuracy*100)))
    df = pd.read_csv(test_file, engine='python', encoding=csv_encoding, error_bad_lines=False)
    df['pred'] = [i.cpu().numpy() for i in all_pred]
    if problem_type=='multi_label_classification':
        df['ret'] = df['pred'] == (df[csv_rows[-1]].apply(lambda x:eval(x)))
    else:
        df['ret'] = df['pred'] == df[csv_rows[-1]]
    print(df['ret'].value_counts())
    df.to_csv(test_pred_out, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()