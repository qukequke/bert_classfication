# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/11/17 17:05
@Author  : quke
@File    : aa.py
@Description:
---------------------------------------
'''

import pandas as pd

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')


def change_data(file_name_in, file_name_out):
    with open(file_name_in, 'r', encoding='utf-8') as f:
        a = [i.rsplit(maxsplit=1) for i in f.readlines()]
        a = [{'text': i, 'class': j} for i, j in a]
    pd.DataFrame(a).to_csv(file_name_out, index=False, encoding='utf-8')


if __name__ == '__main__':
    change_data('train.txt', 'xinwen/xinwen_train.csv')
    change_data('test.txt', 'xinwen/xinwen_test.csv')
    change_data('dev.txt', 'xinwen/xinwen_dev.csv')
