# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2022/7/5 15:07
@Author  : quke
@File    : change.py
@Description:
---------------------------------------
'''
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('raw_train.csv')
df['label'] = df['label'].apply(lambda x: x.split('|'))
labels = list(set(sum(df['label'].tolist(), [])))
labels.sort()
# df['class'] =
print(labels)
df['class'] = df['label'].apply(lambda x: [1 if i in x else 0 for i in labels])
print(df)
with open('class.txt', 'w', encoding='utf-8') as f:
    f.writelines([i + '\n' for i in labels])
df = df.rename(columns={'content': 'text'})
train_df, dev_df = train_test_split(df, test_size=0.2)
train_df.to_csv('train.csv', index=False)
dev_df.to_csv('dev.csv', index=False)
