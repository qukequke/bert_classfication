#!/bin/sh
python train.py --model bert --dir_name xinwen_multi_label --epochs 20 --batch_size 32 --problem_type multi_label_classification > data/xinwen_multi_label/bert_train.log
python train.py --model ernie --dir_name xinwen_multi_label --epochs 20 --batch_size 32 --problem_type multi_label_classification  > data/xinwen_multi_label/ernie_train.log
python train.py --model ernie_healthy --dir_name xinwen_multi_label --epochs 20 --batch_size 32 --problem_type multi_label_classification  > data/xinwen_multi_label/ernie_healthy_train.log
python train.py --model albert --dir_name xinwen_multi_label --epochs 20 --batch_size 32 --problem_type multi_label_classification  > data/xinwen_multi_label/albert_train.log
python train.py --model roberta --dir_name xinwen_multi_label --epochs 20 --batch_size 32 --problem_type multi_label_classification  > data/xinwen_multi_label/roberta_train.log
python train.py --model bert_wwm --dir_name xinwen_multi_label --epochs 20 --batch_size 32 --problem_type multi_label_classification  > data/xinwen_multi_label/bert_wwm_train.log
python train.py --model reformer --dir_name xinwen_multi_label --epochs 20 --batch_size 32 --problem_type multi_label_classification  > data/xinwen_multi_label/reformer_train.log
