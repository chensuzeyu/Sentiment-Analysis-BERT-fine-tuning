# Module for sentiment analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

SENTIMENT_DICT = {0:'positive',1:'negative',2:'neutral'}


class TextSentiment:
	'''Clean and visualize text and sentiment data.'''

	def __init__(self):
		self.df = np.nan

	def read_data(self, file_path):
		self.df = pd.read_csv(file_path,encoding='ISO-8859-1',header=None)
		self.df.rename(columns={0:'sentiment', 1:'text'},inplace=True)

	def drop_duplicates(self):		
		self.df.drop_duplicates(subset=['text'],keep='first',inplace=True)

	def make_labels(self):
		self.df['label'] = 2
		self.df['label'] = self.df.apply(lambda x: 0 if x['sentiment']=='positive' else x['label'], axis=1)
		self.df['label'] = self.df.apply(lambda x: 1 if x['sentiment']=='negative' else x['label'], axis=1)

	def clean_text(self):
		self.df['text'] = self.df['text'].replace(r'\n', ' ', regex=True)

	def plot_word_cloud(self, sentiment=np.nan):
		if sentiment != sentiment:
			text = " ".join([x for x in self.df.text])
		else:
			text = " ".join([x for x in self.df.text[self.df.sentiment==sentiment]])

		wordcloud = WordCloud(background_color='white').generate(text)
		plt.figure(figsize=(8,6))
		plt.imshow(wordcloud,interpolation='bilinear')
		plt.axis('off')
		plt.show()

	def plot_counts(self):
		sns.countplot(self.df.sentiment)


class Model:
	'''Model for sentiment classification.'''

	def __init__(self):
		self.model = np.nan

	def train_test_split(self, df, train_pct=0.8):
		train_set, test_set = train_test_split(df, test_size=1-train_pct)
		train_df = train_set[['text','label']]
		test_df = test_set[['text','label']]
		return train_df, test_df

	def train(self, train_df):
                '''The commented lines: to be completed by students.'''
                
	def predict(self, test_df):
                '''The commented lines: to be completed by students.'''

	def plot_confusion_matrix(self, y_pred, y_true):
		mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
		df_cm = pd.DataFrame(mat, range(3), range(3))
		sns.heatmap(df_cm, annot=True) 
		plt.ylabel('True')
		plt.xlabel('Predicted')
		plt.show()

	def report_eval_stats(self, y_pred, y_true):
		return sklearn.metrics.classification_report(y_true, y_pred,target_names=['positive','neutral','negative'])

	def classify(self, text):
		result = self.model.predict([text])
		pos = np.where(result[1][0] == np.amax(result[1][0]))
		pos = int(pos[0])
		return SENTIMENT_DICT[pos]
