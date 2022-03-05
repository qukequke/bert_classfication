# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import DataPrecessForSentence
from utils import train, validate, eval_object, my_plot
# from transformers import BertTokenizer, AutoTokenizer
from model import BertModel
from transformers.optimization import AdamW
from config import *

Tokenizer = eval_object(model_dict[MODEL][0])
bert_path_or_name = model_dict[MODEL][-1]


# print(Tokenizer)

def main():
    bert_tokenizer = Tokenizer.from_pretrained(bert_path_or_name)

    # bert_tokenizer = AutoTokenizer.from_pretrained(bert_path_or_name)
    device = torch.device("cuda")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    target_dir = os.path.dirname(target_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(bert_tokenizer, train_file)
    labels = train_data.labels
    if use_sample:
        from collections import Counter
        count_dict = Counter(labels)
        count_dict = {k: 1 - (v / len(train_data)) for k, v in count_dict.items()}
        print(count_dict)
        sampler = WeightedRandomSampler([count_dict[int(i)] for i in labels], batch_size, replacement=True)

        train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(bert_tokenizer, dev_file)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = BertModel().to(device)
    # -------------------- Preparation for training  ------------------- #
    # 待优化的参数
    for name, para in model.named_parameters():
        if len(para.size()) < 2:
            continue
        if 'classifier' in name:
            nn.init.xavier_normal_(para)
        # para.requires_grad = True
        if freeze_bert_head:
            if 'classifier' in name:
                para.requires_grad = True
            else:
                para.requires_grad = False
        # print(name)
    param_optimizer = list(model.named_parameters())
    param_optimizer = [(i, k) for i, k in param_optimizer if k.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters(), lr=lr))
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    train_acc_list, dev_acc_list = [], []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint_save = torch.load(checkpoint)
        print(checkpoint_save.keys())
        start_epoch = checkpoint_save["epoch"] + 1
        best_score = checkpoint_save["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint_save["model"])
        # optimizer.load_state_dict(checkpoint_save["optimizer"])
        epochs_count = checkpoint_save["epochs_count"]
        train_losses = checkpoint_save["train_losses"]
        valid_losses = checkpoint_save["valid_losses"]
    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate(model, dev_loader)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss, (valid_accuracy * 100), ))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        # print(model)
        epoch_time, epoch_loss, train_epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (train_epoch_accuracy * 100)))
        print("* Validation for epoch {}:".format(epoch))
        # epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader)
        epoch_time, epoch_loss, epoch_accuracy = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}% \n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            print('save data')
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        # "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        train_acc_list.append(train_epoch_accuracy)
        dev_acc_list.append(epoch_accuracy)
        my_plot(train_acc_list, dev_acc_list,train_losses)  # 画图

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    # train_file = "../data/TextClassification/mnli/train.csv"
    # df = pd.read_csv(train_file, engine='python', error_bad_lines=False)

    main(
        # "../data/TextClassification/mnli/train.csv",
        # "../data/TextClassification/mnli/dev.csv",
        # "data/TextClassification/qqp/train.csv",
        # "../data/TextClassification/qqp/dev.csv",
        # "../data/TextClassification/imdb/train.csv",
        # "../data/TextClassification/imdb/dev.csv",
        # "models",
        # checkpoint='models/best.pth.tar'
    )
