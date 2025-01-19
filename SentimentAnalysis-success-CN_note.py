# 导入必要的库，用于数据处理、可视化和机器学习
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# 定义一个情感字典，用于将情感标签转化为人类可读的文本
SENTIMENT_DICT = {0: 'positive', 1: 'negative', 2: 'neutral'}


class TextSentiment:
	'''文本情感分析的清理和可视化模块'''

	def __init__(self):
		# 初始化时将数据框设置为 NaN
		self.df = np.nan

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

	def plot_word_cloud(self, sentiment=np.nan):
		# 根据输入的情感类型生成词云，如果未指定情感，则生成整个数据集的词云
		if sentiment != sentiment:  # 处理 NaN 情况
			text = " ".join([x for x in self.df.text])
		else:
			# 只针对指定情感类型的数据生成词云
			text = " ".join([x for x in self.df.text[self.df.sentiment == sentiment]])

		# 生成词云图像，设置背景颜色为白色
		wordcloud = WordCloud(background_color='white').generate(text)
		plt.figure(figsize=(8, 6))  # 设置图像大小
		plt.imshow(wordcloud, interpolation='bilinear')  # 显示词云
		plt.axis('off')  # 隐藏坐标轴
		plt.show()

	def plot_counts(self):
		# 统计不同情感类型的分布，并用条形图展示
		sns.countplot(self.df.sentiment)

	def get_data(self):
		# 返回处理好的 DataFrame
		return self.df


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import sklearn.metrics
import torch


SENTIMENT_DICT = {0: 'negative', 1: 'neutral', 2: 'positive'}  # 预测标签的映射


class Model:
	'''情感分类模型的训练和评估模块'''

	def __init__(self):
		# 初始化时模型设置为 NaN，表示未定义
		self.model = None
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	def train_test_split(self, df, train_pct=0.8):
		'''将数据集划分为训练集和测试集，训练集占比默认为 80%'''
		train_set, test_set = train_test_split(df, test_size=1 - train_pct)
		train_df = train_set[['text', 'label']]
		test_df = test_set[['text', 'label']]
		return train_df, test_df

	def tokenize(self, data_dict):
		'''对数据进行标记化处理'''

		def tokenize_function(examples):
			return self.tokenizer(examples['text'], padding="max_length", truncation=True)

		# 将数据转换为 Hugging Face Dataset 格式
		# train_dataset = Dataset.from_pandas(data_dict['train'])
		train_dataset = Dataset.from_pandas(data_dict)
		# test_dataset = Dataset.from_pandas(data_dict['test'])

		# 创建 DatasetDict 对象
		dataset = DatasetDict({'train': train_dataset})

		# 对文本进行标记化
		tokenized_datasets = dataset.map(tokenize_function, batched=True)
		return tokenized_datasets

	def train(self, train_df):
		'''训练模型'''
		# 划分训练集和验证集
		tokenized_datasets = self.tokenize(train_df)
		train_testvalid = tokenized_datasets['train'].train_test_split(test_size=0.2)
		train_dataset = train_testvalid['train']
		valid_dataset = train_testvalid['test']

		# 删除多余字段
		train_dataset = train_dataset.remove_columns(['text', '__index_level_0__'])
		valid_dataset = valid_dataset.remove_columns(['text', '__index_level_0__'])

		# 创建数据加载器
		# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
		# valid_dataloader = DataLoader(valid_dataset, batch_size=4)

		# 加载模型
		self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

		# 定义训练参数
		training_args = TrainingArguments(
			output_dir='./results',
			evaluation_strategy="epoch",
			learning_rate=2e-5,
			per_device_train_batch_size=1,
			per_device_eval_batch_size=1,
			num_train_epochs=30,
			weight_decay=0.01,
		)

		# 创建 Trainer 对象
		trainer = Trainer(
			model=self.model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=valid_dataset,
		)

		# 训练模型
		trainer.train()

	def predict(self, test_df):
		'''使用训练好的模型进行预测'''
		tokenized_datasets = self.tokenize(test_df)
		# test_dataset = tokenized_datasets['test']
		test_dataset = tokenized_datasets['train']

		# 删除多余字段
		test_dataset = test_dataset.remove_columns(['text', '__index_level_0__'])

		# 创建数据加载器
		# test_dataloader = DataLoader(test_dataset, batch_size=4)

		# 获取预测结果
		trainer = Trainer(model=self.model)
		predictions = trainer.predict(test_dataset)

		return predictions

	def plot_confusion_matrix(self, y_pred, y_true):
		'''绘制混淆矩阵，显示预测与真实标签之间的差异'''
		mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
		df_cm = pd.DataFrame(mat, range(3), range(3))  # 使用 DataFrame 存储矩阵
		sns.heatmap(df_cm, annot=True)  # 使用热力图显示矩阵
		plt.ylabel('True')  # 标记 y 轴
		plt.xlabel('Predicted')  # 标记 x 轴
		plt.show()

	def report_eval_stats(self, y_pred, y_true):
		'''生成分类报告，显示模型的准确率、召回率、F1 分数等评估指标'''
		return sklearn.metrics.classification_report(y_true, y_pred, target_names=['positive', 'neutral', 'negative'])

	def classify(self, text):
		'''对单条文本进行分类，返回预测结果的标签（正面、负面或中性）'''
		if not self.model:
			raise ValueError("Model is not trained yet")

		# Move model to the same device as inputs (e.g., CUDA or CPU)
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model.to(device)

		inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

		# Move inputs to the same device as the model
		inputs = {key: value.to(device) for key, value in inputs.items()}

		outputs = self.model(**inputs)
		logits = outputs.logits
		predicted_class = np.argmax(logits.detach().cpu().numpy(), axis=1)

		return SENTIMENT_DICT[predicted_class[0]]  # 返回相应的情感标签

