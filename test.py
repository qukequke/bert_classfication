# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from model import BertModelTest
from utils import test
from dataset import DataPrecessForSentence
from config import *

def main():

    device = torch.device("cuda")
    bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    print(target_file)
    if platform == "linux" or platform == "linux2":

        checkpoint = torch.load(target_file)
    else:
        checkpoint = torch.load(target_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    print("\t* Loading test data...")    
    test_data = DataPrecessForSentence(bert_tokenizer, test_file)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    print("\t* Building model...")
    model = BertModelTest().to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing roberta model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy= test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time, total_time, (accuracy*100)))


if __name__ == "__main__":
    main()