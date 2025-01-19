# Import necessary libraries for data processing, visualization and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Define a sentiment dictionary to convert sentiment labels into human-readable text
SENTIMENT_DICT = {0: 'positive', 1: 'negative', 2: 'neutral'}


class TextSentiment:
    '''Module for cleaning and visualizing text sentiment analysis'''

    def __init__(self):
        # Set the dataframe to NaN during initialization
        self.df = np.nan

    def read_data(self, file_path):
        # Read data from a CSV file, assuming the file has no header row and using ISO-8859-1 encoding
        self.df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
        # Rename the columns to 'sentiment' and 'text', representing sentiment label and text content respectively
        self.df.rename(columns={0: 'sentiment', 1: 'text'}, inplace=True)

    def drop_duplicates(self):
        # Delete duplicate values in the text column, only keeping the first occurrence of the record
        self.df.drop_duplicates(subset=['text'], keep='first', inplace=True)

    def make_labels(self):
        # Create labels for sentiment, initially setting all sentiment labels to 2 (neutral)
        self.df['label'] = 2
        # Set specific labels according to the sentiment column value, with positive sentiment as 0 and negative sentiment as 1
        self.df['label'] = self.df.apply(lambda x: 0 if x['sentiment'] == 'positive' else x['label'], axis=1)
        self.df['label'] = self.df.apply(lambda x: 1 if x['sentiment'] == 'negative' else x['label'], axis=1)

    def clean_text(self):
        # Clean the text data, removing newline characters and replacing them with spaces
        self.df['text'] = self.df['text'].replace(r'\n', ' ', regex=True)

    def plot_word_cloud(self, sentiment=np.nan):
        # Generate a word cloud according to the input sentiment type. If the sentiment is not specified, generate a word cloud for the entire dataset.
        if sentiment!= sentiment:  # Handle the NaN case
            text = " ".join([x for x in self.df.text])
        else:
            # Generate a word cloud only for the data of the specified sentiment type
            text = " ".join([x for x in self.df.text[self.df.sentiment == sentiment]])

        # Generate the word cloud image, setting the background color to white
        wordcloud = WordCloud(background_color='white').generate(text)
        plt.figure(figsize=(8, 6))  # Set the image size
        plt.imshow(wordcloud, interpolation='bilinear')  # Display the word cloud
        plt.axis('off')  # Hide the axes
        plt.show()

    def plot_counts(self):
        # Count the distribution of different sentiment types and display it with a bar chart
        sns.countplot(self.df.sentiment)

    def get_data(self):
        # Return the processed DataFrame
        return self.df


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import sklearn.metrics
import torch


SENTIMENT_DICT = {0: 'negative', 1: 'neutral', 2: 'positive'}  # Mapping of prediction labels



class Model:
    '''Module for training and evaluating the sentiment classification model'''

    def __init__(self):
        # Initialize the model to NaN, indicating it's not defined
        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def train_test_split(self, df, train_pct=0.8):
        '''Split the dataset into training set and test set, with the training set proportion defaulting to 80%'''
        train_set, test_set = train_test_split(df, test_size=1 - train_pct)
        train_df = train_set[['text', 'label']]
        test_df = test_set[['text', 'label']]
        return train_df, test_df

    def tokenize(self, data_dict):
        '''Tokenize the data'''

        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding="max_length", truncation=True)

        # Convert the data to Hugging Face Dataset format
        # train_dataset = Dataset.from_pandas(data_dict['train'])
        train_dataset = Dataset.from_pandas(data_dict)
        # test_dataset = Dataset.from_pandas(data_dict['test'])

        # Create a DatasetDict object
        dataset = DatasetDict({'train': train_dataset})

        # Tokenize the text
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        return tokenized_datasets

    def train(self, train_df):
        '''Train the model'''
        # Split the training set and validation set
        tokenized_datasets = self.tokenize(train_df)
        train_testvalid = tokenized_datasets['train'].train_test_split(test_size=0.2)
        train_dataset = train_testvalid['train']
        valid_dataset = train_testvalid['test']

        # Remove unnecessary fields
        train_dataset = train_dataset.remove_columns(['text', '__index_level_0__'])
        valid_dataset = valid_dataset.remove_columns(['text', '__index_level_0__'])

        # Create data loaders
        # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
        # valid_dataloader = DataLoader(valid_dataset, batch_size=4)

        # Load the model
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

        # Define training parameters
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=30,
            weight_decay=0.01,
        )

        # Create a Trainer object
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )

        # Train the model
        trainer.train()

    def predict(self, test_df):
        '''Use the trained model to make predictions'''
        tokenized_datasets = self.tokenize(test_df)
        # test_dataset = tokenized_datasets['test']
        test_dataset = tokenized_datasets['train']

        # Remove unnecessary fields
        test_dataset = test_dataset.remove_columns(['text', '__index_level_0__'])

        # Create a data loader
        # test_dataloader = DataLoader(test_dataset, batch_size=4)

        # Get the prediction results
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_dataset)

        return predictions

    def plot_confusion_matrix(self, y_pred, y_true):
        '''Draw a confusion matrix to show the differences between predicted and true labels'''
        mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(mat, range(3), range(3))  # Store the matrix using a DataFrame
        sns.heatmap(df_cm, annot=True)  # Display the matrix using a heatmap
        plt.ylabel('True')  # Label the y-axis
        plt.xlabel('Predicted')  # Label the x-axis
        plt.show()

    def report_eval_stats(self, y_pred, y_true):
        '''Generate a classification report to show evaluation metrics such as accuracy, recall, F1-score of the model'''
        return sklearn.metrics.classification_report(y_true, y_pred, target_names=['positive', 'neutral', 'negative'])

    def classify(self, text):
        '''Classify a single text and return the label of the prediction result (positive, negative or neutral)'''
        if not self.model:
            raise ValueError("Model is not trained yet")

        # Move the model to the same device as the inputs (e.g., CUDA or CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Move the inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = np.argmax(logits.detach().cpu().numpy(), axis=1)

        return SENTIMENT_DICT[predicted_class[0]]  # Return the corresponding sentiment label

