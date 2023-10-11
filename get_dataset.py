import os
import shutil
import tensorflow as tf

def gather_dataset(path):
    english_folder = os.path.join(path, 'english')
    russian_folder = os.path.join(path, 'russian')
    if not os.path.exists(english_folder):
        raise ValueError(f"Folder {english_folder} does not exist")
    if not os.path.exists(russian_folder):
        raise ValueError(f"Folder {russian_folder} does not exist")
    english_classes = os.listdir(english_folder)
    russian_classes = os.listdir(russian_folder)
    if len(english_classes) != len(russian_classes):
        raise ValueError(f"Number of classes in {english_folder} and {russian_folder} do not match")
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
        for english_file, russian_file in zip(english_files, russian_files):
            english_file_path = os.path.join(english_class_folder, english_file)
            russian_file_path = os.path.join(russian_class_folder, russian_file)
            if not os.path.exists(english_file_path):
                raise ValueError(f"File {english_file_path} does not exist")
            if not os.path.exists(russian_file_path):
                raise ValueError(f"File {russian_file_path} does not exist")
            if os.path.getsize(english_file_path) != os.path.getsize(russian_file_path):
                raise ValueError(f"Size of {english_file_path} and {russian_file_path} do not match")
            if not english_file_path.endswith('.txt'):
                raise ValueError(f"File {english_file_path} is not in txt format")
            if not russian_file_path.endswith('.txt'):
                raise ValueError(f"File {russian_file_path} is not in txt format")
    print("Dataset validated successfully")
    dataset = tf.data.Dataset.list_files(os.path.join(path, '*.txt'))
    dataset = dataset.map(lambda x: (tf.io.read_file(x),))
    return dataset

path = 'path/to/dataset'
dataset = gather_dataset(path)
print("Dataset gathered successfully")
