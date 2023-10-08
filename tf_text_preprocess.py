from datasets import load_dataset
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

class TextPreprocessor:
    def __init__(self, dataset_names=None, text_files_paths=None):
        self.dataset_names = self._ensure_list(dataset_names)
        self.text_files_paths = self._ensure_list(text_files_paths)
        self.datasets = []
        self.tokenizer = Tokenizer()

    def _ensure_list(self, item):
        return item if isinstance(item, list) else [item]

    def load_datasets(self):
        for dataset_name in self.dataset_names:
            if dataset_name:
                self.datasets.append(load_dataset(dataset_name))

    def load_text_files(self):
        for text_files_path in self.text_files_paths:
            if text_files_path:
                self.datasets.append(self._load_text_file(text_files_path))

    def _load_text_file(self, text_files_path):
        return tf.data.TextLineDataset(
            tf.data.Dataset.list_files(os.path.join(text_files_path, '*.txt'))
        )

    def clean_text(self, text):
        text = self._remove_newlines(text)
        text = self._remove_tabs(text)
        text = self._remove_extra_spaces(text)
        return text

    def _remove_newlines(self, text):
        return re.sub(r'\n', ' ', text)

    def _remove_tabs(self, text):
        return re.sub(r'\t', ' ', text)

    def _remove_extra_spaces(self, text):
        return re.sub(r' +', ' ', text)

    def tokenize_text(self, text):
        self.tokenizer.fit_on_texts(text)
        sequences = self.tokenizer.texts_to_sequences(text)
        return sequences

    def pad_sequences(self, sequences):
        padded_sequences = pad_sequences(sequences, padding='post')
        return padded_sequences

    def preprocess_datasets(self):
        for i, dataset in enumerate(self.datasets):
            cleaned_text = self.clean_text(dataset)
            tokenized_text = self.tokenize_text(cleaned_text)
            padded_text = self.pad_sequences(tokenized_text)
            self.datasets[i] = padded_text

    def get_datasets(self):
        return self.datasets

class TextClass(TextPreprocessor):
    def __init__(self, class_name, dataset_names=None, text_files_paths=None):
        super().__init__(dataset_names, text_files_paths)
        self.class_name = class_name

    def load_text_files(self):
        for text_files_path in self.text_files_paths:
            if text_files_path:
                self.datasets.append(self._load_text_file(text_files_path, self.class_name))

    def _load_text_file(self, text_files_path, class_name):
        return tf.data.TextLineDataset(
            tf.data.Dataset.list_files(os.path.join(text_files_path, class_name, '*.txt'))
        )

class EnglishTexts(TextClass):
    def __init__(self, class_name, dataset_names=None, text_files_paths=None):
        super().__init__(class_name, dataset_names, text_files_paths)

class RussianTexts(TextClass):
    def __init__(self, class_name, dataset_names=None, text_files_paths=None):
        super().__init__(class_name, dataset_names, text_files_paths)
