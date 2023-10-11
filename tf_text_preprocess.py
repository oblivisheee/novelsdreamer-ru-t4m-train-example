from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import os

class TextPreprocess:
    def __init__(self, dataset):
        self.datasets = dataset
        self.tokenizer = Tokenizer()

    def tokenize_text(self, text):
        self.tokenizer.fit_on_texts(text)
        sequences = self.tokenizer.texts_to_sequences(text)
        return sequences

    def pad_sequences(self, sequences):
        padded_sequences = pad_sequences(sequences, padding='post')
        return padded_sequences

    def preprocess_datasets(self):
        for i, dataset in enumerate(self.datasets):
            tokenized_text = self.tokenize_text(dataset)
            padded_text = self.pad_sequences(tokenized_text)
            self.datasets[i] = padded_text
        return self.datasets
    def load_text_files(self, text_files_path):
        if text_files_path:
            self.datasets.append(self._load_text_files(text_files_path))

    def _load_text_files(self, text_files_path):
        return tf.data.Dataset.list_files(os.path.join(text_files_path, '*.txt')).interleave(
            lambda filename: tf.data.TextLineDataset(filename),
            num_parallel_calls=tf.data.AUTOTUNE
        )



class TextClass(TextPreprocess):
    def __init__(self, class_name, dataset):
        super().__init__(dataset)
        self.class_name = class_name

