# -*- coding: utf-8 -*-
'''
---------------------------------------
@Time    : 2022/1/20 13:43
@Author  : quke
@File    : check_data.py
@Description:
---------------------------------------
'''
import glob
import os

import pandas as pd


def parse_ann_2_df():
    files = glob.glob('*_ann.txt')
    ret = []
    for file_name in files:
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                _, middle, text = line.strip().split('\t')
                class_ = middle.split()[0]
                ret.append({'class': class_, 'text': text, 'src': os.path.basename(file_name)})
    df = pd.DataFrame(ret)
    return df


def parse_csv():
    data_col = '业务流程'
    # file_name = 'label.txt'
    df = pd.read_csv("血站技术操作规程1_out.csv", encoding='gbk')
    df1 = df[['text', data_col]].rename(columns={'业务流程': 'class'})
    df2 = df[['text', '知识类型']].rename(columns={'知识类型': 'class'})
    df = df1.append(df2)
    df = df.fillna('其他')
    df['src'] = '血站技术操作规程_liukun'

    # df.dropna(inplace=True)

    return df


df = parse_ann_2_df()
df2 = parse_csv()
df = df.append(df2)
label_dict = {
    "献血者健康征询": "业务流程",
    "献血者一般检查": "业务流程",
    "血液采集": "业务流程",
    "采血准备": "业务流程",
    "献血者采后管理": "业务流程",
    "献血核查登记": "业务流程",
    "献血前血液检测": "业务流程",
    "血液制备": "业务流程",
    "血液检测": "业务流程",
    "参考方案": "知识类型",
    "操作流程": "知识类型",
    "风险知识": "知识类型",
    "指导意见": "知识类型",
    "行业标准": "去掉",
    "国家标准": "去掉",
    "体检医师岗": "岗位",
    "信息管理岗": "岗位",
    "数据": "岗位",
    "检验岗": "岗位",
    "血源管理岗": "岗位",
    "采血护士岗": "岗位",
}
df['knowledge_label'] = df['class'].apply(lambda x: label_dict.get(x, '其他'))
df.to_csv('all_train.csv', index=False)


# print(df.groupby(['knowledge_label', 'class']).count())
# print(df.groupby('knowledge_label').count())


def gen_one(df, data_col, ouput_dir):
    df = df.loc[df['knowledge_label'].isin([data_col, '其他']), :]
    labels = df['class'].unique()
    labels.sort()
    label_dict = {d: i for i, d in enumerate(labels)}
    df['label'] = df['class'].apply(lambda x: label_dict[x])
    with open(os.path.join(ouput_dir, 'label.txt'), 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in labels])
    df.to_csv(os.path.join(ouput_dir, 'train.csv'), index=False, encoding='utf-8')
    df.to_csv(os.path.join(ouput_dir, 'dev.csv'), index=False, encoding='utf-8')
    df.to_csv(os.path.join(ouput_dir, 'test.csv'), index=False, encoding='utf-8')
    print('done')


def gangwei(df, data_col, output_dir):
    df = df.loc[df['knowledge_label'].isin([data_col, '其他']), :]
    labels = df['class'].unique()
    labels.sort()
    # print(labels)
    label_dict = {d: i for i, d in enumerate(labels)}
    with open(os.path.join(output_dir, 'label.txt'), 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in labels])
    # print(df.groupby(['text']).apply(list))
    df_new = df.groupby(['text'])['class'].apply(lambda x: list(set(x))).to_frame()
    # print(df_new.index)
    print(df_new.columns)
    # df_new['text'] = df_new.index
    df = df_new.rename_axis('text').reset_index()
    # print(df_new.columns)
    # df_new.reset_index()
    # print(df_new)
    # print(df_new.columns)
    # df = df_new.join(df, on=)
    df['label'] = df['class'].apply(lambda x: [1 if k in x else 0 for k, v in label_dict.items()])
    # # print(data_new['label'].value_counts())
    # # print(df.head())
    # print(label_dict)
    print(df['class'].value_counts())
    print(df['label'].value_counts())
    df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, encoding='utf-8')
    df.to_csv(os.path.join(output_dir, 'dev.csv'), index=False, encoding='utf-8')
    df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, encoding='utf-8')


# gen_one(df, '业务流程', '../yewuliucheng/')
# gen_one(df, '知识类型', '../zhishileixing/')

gangwei(df, '岗位', '../gangwei/')
