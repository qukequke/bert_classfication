# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2022/1/17 8:58
@Author  : quke
@File    : main.py
@Description:
---------------------------------------
'''

import glob
import pandas as pd


def get_label_dict(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        line_list = f.readlines()
        label_dict = {i.strip():ii for ii, i in enumerate(line_list)}
        return label_dict


files = glob.glob('*_ann.txt')
ret = []
for file_name in files:
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            _, middle, text = line.strip().split('\t')
            class_ = middle.split()[0]
            ret.append({'class': class_, 'text': text})
# print(ret)
df = pd.DataFrame(ret)
# print(df['class'].value_counts())
# df.to_csv('train.csv', encoding='utf-8', index=False)
# print(df)

yewuliucheng_dict = get_label_dict('../yewuliucheng/label.txt')
zhishileixing_dict = get_label_dict('../zhishileixing/label.txt')

df_yewuiliucheng = df.loc[df['class'].isin(yewuliucheng_dict.keys()), :]
df_zhishileixing = df.loc[df['class'].isin(zhishileixing_dict.keys()), :]
print(df_yewuiliucheng)
print(df_zhishileixing)
df_yewuiliucheng['class'] = df_yewuiliucheng['class'].apply(lambda x:yewuliucheng_dict[x])
df_zhishileixing['class'] = df_zhishileixing['class'].apply(lambda x:zhishileixing_dict[x])


df_yewuiliucheng_lk = pd.read_csv('../yewuliucheng/train_lk.csv')
df_zhishileixing_lk = pd.read_csv('../zhishileixing/train_lk.csv')

yewu_raw_len = len(df_yewuiliucheng)
zhishi_raw_len = len(df_zhishileixing)
df_yewuiliucheng = df_yewuiliucheng.append(df_yewuiliucheng_lk)
df_zhishileixing = df_zhishileixing.append(df_zhishileixing_lk)

yewu_raw_len_new = len(df_yewuiliucheng)
zhishi_raw_len_new = len(df_zhishileixing)

print(f"业务流程,合并前长度为{yewu_raw_len},合并后{yewu_raw_len_new}")
print(f"知识类型,合并前长度为{zhishi_raw_len},合并后{zhishi_raw_len_new}")

print(df_yewuiliucheng['class'].value_counts())
print()
print()
print(df_zhishileixing['class'].value_counts())

df_yewuiliucheng.to_csv('../yewuliucheng/train.csv', index=False)
df_zhishileixing.to_csv('../zhishileixing/train.csv', index=False)
# df_yewuiliucheng_lk = pd.read_csv('../yewuliucheng/train_lk.csv')
