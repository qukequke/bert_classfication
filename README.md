# RoBerta-Chinese-Text-Classification-Pytorch

中文文本分类，RoBerta，基于pytorch，开箱即用。

## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。数据以字为单位输入模型。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


### 更换自己的数据集
 - 按照我数据集的格式来格式化你的中文数据集。  



会在线下载预训练模型
## 具体参数可看config.py
```
# 训练
python train.py 
# 测试
python test.py
```
