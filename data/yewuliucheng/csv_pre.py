# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2021/12/29 16:46
@Author  : quke
@File    : csv_pre.py
@Description:
---------------------------------------
'''
import pandas as pd

data_col = '业务流程'
file_name = 'label.txt'
df = pd.read_csv("血站技术操作规程1_out.csv", encoding='gbk')
df = df[['text', data_col]]
df[data_col] = df[data_col].fillna('其他')
print(df[data_col].value_counts())
labels = df[data_col].unique()
label_dict = {d: i for i, d in enumerate(labels)}
df[data_col] = df[data_col].apply(lambda x: label_dict[x])
with open(file_name, 'w', encoding='utf-8') as f:
    f.writelines([i + '\n' for i in labels])
df = df.rename(columns={data_col: 'class'})
df.to_csv('train.csv', index=False, encoding='utf-8')
df.to_csv('dev.csv', index=False, encoding='utf-8')
df.to_csv('test.csv', index=False, encoding='utf-8')
