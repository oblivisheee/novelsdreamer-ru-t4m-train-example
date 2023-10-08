from datasets import load_dataset
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

class TextPreprocessor:
    def __init__(self, dataset_names=None, text_files_paths=None):
        self.dataset_names = dataset_names if isinstance(dataset_names, list) else [dataset_names]
        self.text_files_paths = text_files_paths if isinstance(text_files_paths, list) else [text_files_paths]
        self.datasets = []
        self.tokenizer = Tokenizer()

    def load_datasets(self):
        for dataset_name in self.dataset_names:
            if dataset_name:
                self.datasets.append(load_dataset(dataset_name))

    def load_text_files(self):
        for text_files_path in self.text_files_paths:
            if text_files_path:
                self.datasets.append(tf.data.TextLineDataset(
                    tf.data.Dataset.list_files(os.path.join(text_files_path, '*.txt'))
                ))

    def clean_text(self, text):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\r', ' ', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text

    def tokenize_and_pad(self, text):
        self.tokenizer.fit_on_texts(text)
        sequences = self.tokenizer.texts_to_sequences(text)
        padded_sequences = pad_sequences(sequences, padding='post')
        return padded_sequences

    def preprocess_datasets(self):
        for i, dataset in enumerate(self.datasets):
            cleaned_text = self.clean_text(dataset)
            tokenized_and_padded_text = self.tokenize_and_pad(cleaned_text)
            self.datasets[i] = tokenized_and_padded_text

    def get_datasets(self):
        return self.datasets

