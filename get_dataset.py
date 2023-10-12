import os
import shutil
import tensorflow as tf
import codecs
from docx import Document

def read_file(file_path):
    if file_path.endswith('.txt'):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            return f.read()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return ' '.join([p.text for p in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def gather_dataset(path):
    english_folder = os.path.join(path, 'english')
    russian_folder = os.path.join(path, 'russian')
    if not os.path.exists(english_folder):
        os.mkdir(english_folder)
        print(ValueError(f"Folder {english_folder} does not exist. Created a new path."))
    if not os.path.exists(russian_folder):
        os.mkdir(russian_folder)
        print(ValueError(f"Folder {russian_folder} does not exist. Created a new path."))
    english_classes = os.listdir(english_folder)
    russian_classes = os.listdir(russian_folder)
    if len(english_classes) != len(russian_classes):
        raise ValueError(f"Number of classes in {english_folder} and {russian_folder} do not match")
    english_texts = []
    russian_texts = []
    for english_class, russian_class in zip(english_classes, russian_classes):
        english_class_folder = os.path.join(english_folder, english_class)
        russian_class_folder = os.path.join(russian_folder, russian_class)
        if not os.path.exists(english_class_folder):
            raise ValueError(f"Folder {english_class_folder} does not exist")
        if not os.path.exists(russian_class_folder):
            raise ValueError(f"Folder {russian_class_folder} does not exist")
        english_files = os.listdir(english_class_folder)
        russian_files = os.listdir(russian_class_folder)
        if len(english_files) != len(russian_files):
            raise ValueError(f"Number of files in {english_class_folder} and {russian_class_folder} do not match")
        english_texts_class = []
        russian_texts_class = []
        for english_file, russian_file in zip(english_files, russian_files):
            english_file_path = os.path.join(english_class_folder, english_file)
            russian_file_path = os.path.join(russian_class_folder, russian_file)
            if not os.path.exists(english_file_path):
                raise ValueError(f"File {english_file_path} does not exist")
            if not os.path.exists(russian_file_path):
                raise ValueError(f"File {russian_file_path} does not exist")
            english_text = read_file(english_file_path)
            russian_text = read_file(russian_file_path)
            english_texts_class.append(english_text)
            russian_texts_class.append(russian_text)
        english_texts.append(english_texts_class)
        russian_texts.append(russian_texts_class)
    english_texts = tf.data.Dataset.from_tensor_slices(english_texts)
    russian_texts = tf.data.Dataset.from_tensor_slices(russian_texts)
    dataset = tf.data.Dataset.zip((english_texts, russian_texts))
    return dataset
