# 导入必要的库，用于数据处理、可视化和机器学习
import pandas as pd
import numpy as np


from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict


# 定义一个情感字典，用于将情感标签转化为人类可读的文本
SENTIMENT_DICT = {0: 'positive', 1: 'negative', 2: 'neutral'}


class TextSentiment:
    '''文本情感分析的清理和可视化模块'''
    def __init__(self):
        # 初始化时将数据框设置为 NaN
        # self.df = np.nan
        self.df = pd.DataFrame()

    def read_data(self, file_path):
        # 从 CSV 文件读取数据，假设该文件没有标题行，使用 ISO-8859-1 编码
        self.df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
        # 将列重命名为 'sentiment' 和 'text'，分别代表情感标签和文本内容
        self.df.rename(columns={0: 'sentiment', 1: 'text'}, inplace=True)

    def drop_duplicates(self):
        # 删除文本列中的重复值，只保留第一次出现的记录
        self.df.drop_duplicates(subset=['text'], keep='first', inplace=True)

    def make_labels(self):
        # 为情感创建标签，初始时所有情感标签设置为2（neutral）
        self.df['label'] = 2
        # 根据情感列值设置具体标签，正面情感为 0，负面情感为 1
        self.df['label'] = self.df.apply(lambda x: 0 if x['sentiment'] == 'positive' else x['label'], axis=1)
        self.df['label'] = self.df.apply(lambda x: 1 if x['sentiment'] == 'negative' else x['label'], axis=1)

    def clean_text(self):
        # 清理文本数据，去掉换行符，将其替换为空格
        self.df['text'] = self.df['text'].replace(r'\n', ' ', regex=True)

    def get_data(self):
        # 返回处理好的 DataFrame
        return self.df


text_data = TextSentiment()
text_data.read_data('all-data.csv')
text_data.drop_duplicates()
text_data.make_labels()
text_data.clean_text()

# print(text_data.df)
# 原来的代码用于处理 IMDB 数据集
# dataset = load_dataset("imdb")
# print(dataset)

# 将处理后的数据传入 dataset
# dataset = text_data.get_data()

# 假设原始数据是一个 pandas DataFrame，转换为 Hugging Face Dataset
# dataset = Dataset.from_pandas(dataset)



# text_data.data 是处理过后的 DataFrame，并包含两列 'text' 和 'label'
data_dict = {
    'train': text_data.df[['text', 'label']].iloc[:int(len(text_data.df)*0.8)],  # 80% 作为训练集
    'test': text_data.df[['text', 'label']].iloc[int(len(text_data.df)*0.8):]    # 20% 作为测试集
}

# 将数据转换为 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(data_dict['train'])
test_dataset = Dataset.from_pandas(data_dict['test'])

# 创建 DatasetDict 对象，类似于 load_dataset("imdb") 的返回结果
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})


"""
1-标记化数据(就是编码,特征提取)，使用Transformers中的BertTokenizer对数据进行标记
"""

# 文本长度是默认的，可以自行设置，max_length=xxx
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 将文本转换为模型可以处理的格式，包括截断和填充
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)
    # Token Embedding、 Segment Embedding和Position Embedding


# 使用map函数把数据处理应用到每个元素上，这里的batched=True参数表示tokenize_function将在数据集的批次上执行，
# 而不是单独对每个样本执行。
tokenized_datasets = dataset.map(tokenize_function, batched=True)


"""
2-划分训练集和验证集
"""
# 按照8:2,把训练集进一步划分为训练集和测试集
train_testvalid = tokenized_datasets['train'].train_test_split(test_size=0.2)
train_dataset = train_testvalid['train']
valid_dataset = train_testvalid['test']

# 删除多余字段： 在将数据传入模型前删除多余字段，确保只传入 input_ids、token_type_ids、attention_mask 和 label
train_dataset = train_dataset.remove_columns(['text', '__index_level_0__'])
valid_dataset = valid_dataset.remove_columns(['text', '__index_level_0__'])


"""
3-为训练集和验证集创建数据加载器
"""
# 训练数据随机打乱,增加模型的泛化能力
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=4)


"""
4-加载模型，bert-base-uncased是BERT的基本版本,将文本的所有字母都转换为小写有助于减少模型的词汇大小
12个Transformer-Encoder,隐藏层维度768,总共有110M个参数
"""
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)


"""
5-定义模型微调过程的参数
"""
training_args = TrainingArguments(
    # 微调后的结果存放位置
    output_dir='./results',
    # 验证集在每个训练周期(epoch)结束后用于评估模型的性能,模型通过与验证集进行交互来调整自己的参数
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()


"""
6-分类模型评价指标: 准确度、精确度、召回率和 F1 分数等
"""
metrics = trainer.evaluate()
print(metrics)


"""
7-使用微调后的模型进行预测
"""
predictions = trainer.predict(valid_dataset)
print(predictions)

