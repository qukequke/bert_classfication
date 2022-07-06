# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 02:08:46 2020

@author: zhaog
"""
import os
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from config import *


def eval_object(object_):
    if '.' in object_:
        module_, class_ = object_.rsplit('.', 1)
        module_ = import_module(module_)
        return getattr(module_, class_)
    else:
        module_ = import_module(object_)
        return module_


def generate_sent_masks(enc_hiddens, source_lengths):
    """ Generate sentence masks for encoder hidden states.
    @param enc_hiddens (Tensor): encodings of shape (b, src_len, h), where b = batch size,
                                 src_len = max source length, h = hidden size. 
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.len = batch size
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, b = batch size.
    """
    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


def correct_predictions(pred, targets, args):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    if args.problem_type == 'multi_label_classification':
        # preds = torch.sigmoid(output_probabilities)
        # preds = preds.cpu().numpy()
        # preds2 = output_probabilities
        # preds = output_probabilities
        # preds2 = torch.where(preds >= 0.5, 1, 0)
        # preds2 = torch.where(preds > 0.50001, torch.ones(preds.shape), torch.zeros(preds.shape))
        # preds2 = np.where(preds >= 0.5, 1, 0)
        # correct = sum((i == j).all() for i, j in zip(preds2, targets.cpu().numpy()))
        correct = sum((i == j).all() for i, j in zip(pred, targets))
    else:
        # print(pred.shape)
        # print(targets.shape)
        correct = (pred == targets).sum()
    return correct.item()
        # _, out_classes = output_probabilities.max(dim=1)
        # if pro:
        #     out_classes = pred.argmax(axis=1)
        # else:
            # _, out_classes = pred.max(dim=1)
            # out_classes = output_probabilities
        # print(out_classes.shape)
        # print(targets.shape)


def validate(model, dataloader, args):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    print('正在对验证集进行测试')
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0

    if args.problem_type == 'multi_label_classification':
        all_pred = torch.zeros(0, args.num_labels)
        all_labels = torch.zeros(0, args.num_labels)
    else:
        all_pred = torch.zeros(0, )
        all_labels = torch.zeros(0, )
    all_pred = all_pred.to(device)
    all_labels = all_labels.to(device)

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for tokened_data_dict in dataloader:
            tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}

            loss, probabilities, pred = get_batch_ret(model, args, **tokened_data_dict)
            labels = tokened_data_dict['labels']
            # output_data = model(**tokened_data_dict)
            # loss = output_data.loss
            # logits = output_data.logits
            # if args.problem_type == 'multi_label_classification':
            #     probabilities = torch.softmax(logits, dim=-1)
            # else:
            #     probabilities = torch.sigmoid(logits)
            running_loss += loss.item()
            # running_accuracy += correct_predictions(probabilities.cpu().numpy(), labels.cpu().numpy())
            # all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_pred = torch.concat([all_pred, pred])
            all_labels = torch.concat([all_labels, labels])
            # all_labels.extend(labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    # if args.problem_type == 'multi_label_classification':
    #     all_pred = torch.where(all_pred >= 0.5, 1, 0)
    # else:
    #     all_pred = all_pred.argmax(axis=1)
            # _, all_pred = probabilities.max(dim=1)
    # print(all_labels.shape)
    # print(all_pred.shape)
    print(classification_report(all_labels.cpu(), all_pred.cpu(), labels=list(args.label2id.values()), target_names=args.label2id.keys()))
    acc_num = correct_predictions(all_pred, all_labels, args)
    epoch_accuracy = acc_num / (len(dataloader.dataset))
    # correct = sum((i == j).all() for i, j in zip(out_classes, labels))
    # np.set_printoptions(threshold=np.inf)
    # print(confusion_matrix(all_labels, out_classes))
    # return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob)
    return epoch_time, epoch_loss, epoch_accuracy


def test(model, dataloader, args):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    if args.problem_type == 'multi_label_classification':
        all_pred = torch.zeros(0, args.num_labels)
        all_labels = torch.zeros(0, args.num_labels)
    else:
        all_pred = torch.zeros(0, )
        all_labels = torch.zeros(0, )
    all_pred = all_pred.to(device)
    all_labels = all_labels.to(device)
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for tokened_data_dict in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
            labels = tokened_data_dict['labels']
            # seqs, masks, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_labels.to(device)
            # data = model(**tokened_data_dict)  # [batch_size, n_label]

            loss, probabilities, pred = get_batch_ret(model, args, **tokened_data_dict)
            # loss = data.loss
            # logits = data.logits
            batch_time += time.time() - batch_start
            # all_prob.extend(probabilities[:, 1].cpu().numpy())
            # if args.problem_type == 'multi_label_classification':
            #     probabilities = torch.sigmoid(logits)
            #     # accuracy += correct_predictions(probabilities, labels)
            #     out_classes = torch.where(probabilities >= 0.5, 1, 0)
            #     # out_classes = out_classes.type(torch.long)
            # else:
            #     probabilities = torch.softmax(logits, dim=-1)
            #     _, out_classes = probabilities.max(dim=1)
            accuracy += correct_predictions(pred, labels, args)
            all_pred = torch.concat([all_pred, pred])
            all_labels = torch.concat([all_labels, labels])
            # all_labels.extend(labels.cpu().numpy())
            # all_pred.extend(out_classes)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, all_labels, all_pred


def get_batch_ret(model, args, **tokened_data_dict):
    """
    input:
    return :
      loss:损失函数结果
      probabilityies: bert后 取softmax 或sigmoid的结果
      all_pred:probabilities 转化为预测结果 ，也就是>0.5变为1 ，或者argmax得到最大值
    """
    data = model(**tokened_data_dict)
    loss = data.loss
    logits = data.logits
    if args.problem_type == 'multi_label_classification':
        probabilities = torch.sigmoid(logits)
        all_pred = torch.where(probabilities>= 0.5, 1, 0)
    else:
        probabilities = torch.softmax(logits, -1)
        _, all_pred = probabilities.max(dim=1)
    return loss, probabilities, all_pred


def train(model, dataloader, optimizer, args):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    max_gradient_norm = args.max_grad_norm
    model.train()
    device = model.device
    model.to(device)
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    tqdm_batch_iterator = tqdm(dataloader)
    # for batch_index, (batch_seqs, batch_mask, batch_seq_segments, batch_labels) in enumerate(tqdm_batch_iterator):
    # if not use_sample:
    #     global validate_iter
    #     validate_iter = 1
    # for i in range(validate_iter):
    #     print(f'i:{i}')
    if args.problem_type == 'multi_label_classification':
        all_pred = torch.zeros(0, args.num_labels)
        all_labels = torch.zeros(0, args.num_labels)
    else:
        all_pred = torch.zeros(0, )
        all_labels = torch.zeros(0, )
    all_pred = all_pred.to(device)
    all_labels = all_labels.to(device)

    for batch_index, (tokened_data_dict) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        tokened_data_dict = {k: v.to(device) for k, v in tokened_data_dict.items()}
        labels = tokened_data_dict['labels']
        # if PRINT_TRAIN_COUNT:
        #     print(Counter(list(labels.cpu().numpy())))
        optimizer.zero_grad()

        loss, probabilities, pred = get_batch_ret(model, args, **tokened_data_dict)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        # correct_preds += correct_predictions(probabilities,labels, pro=False)
        # correct_preds += correct_predictions(probabilities.cpu().detach().numpy(), labels.cpu().numpy(), pro=True)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}".format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
        all_pred = torch.concat([all_pred, pred])
        all_labels = torch.concat([all_labels, labels])
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    correct_preds = correct_predictions(all_pred, all_labels, args)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def get_max(x, y):
    max_x_index = np.argmax(y)
    max_x = x[max_x_index]
    max_y = y[max_x_index]
    return max_x, max_y


def my_plot(train_acc_list, dev_acc_list, losses, args):
    plt.figure()
    plt.plot(train_acc_list, color='r', label='train_acc')
    plt.plot(dev_acc_list, color='b', label='dev_acc')
    x = [i for i in range(len(train_acc_list))]
    for add, list_ in enumerate([train_acc_list, dev_acc_list]):
        max_x, max_y = get_max(x, list_)
        plt.text(max_x, max_y, f'{(max_x, max_y)}')
        plt.vlines(max_x, min(min(train_acc_list), min(dev_acc_list)), max_y, colors='r' if add == 0 else 'b',
                   linestyles='dashed')
        plt.hlines(max_y, 0, max_x, colors='r' if add == 0 else 'b', linestyles='dashed')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(args.train_file), f'{args.model}_acc.png'))
    plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(os.path.dirname(args.train_file), f'{args.model}_loss.png'))
