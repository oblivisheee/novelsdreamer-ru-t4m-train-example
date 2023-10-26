import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset

class DataGenerator:
    """
    Generates tensors by converting input data.

    Usage:
    >>> datagen = DataGenerator(TRAIN_DATASET_DIR, VALID_DATASET_DIR)

    >>> (train_english, train_russian), (valid_english, valid_russian) = datagen.generate()

    
    """
    def __init__(self, main_dir, train_dir, valid_dir, padding_type='post', trunc_type='post', session_name=str):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.padding_type = padding_type
        self.trunc_type = trunc_type
        if not os.path.exists(os.path.join(main_dir, 'logs')):
            os.mkdir(os.path.join(main_dir, 'logs'))
        log_file_name = f'dataset_logs_{session_name}.txt'
        self.log_file_path = os.path.join(main_dir, 'logs', log_file_name)
        log_file = open(self.log_file_path, "w")
    #Load files and classes.
    def load_data(self, dir_name):
        data = {}
        for class_name in os.listdir(dir_name):
            class_dir = os.path.join(dir_name, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            data[class_name] = []
            for filename in os.listdir(class_dir):
                if os.path.isfile(os.path.join(class_dir, filename)):
                    with open(os.path.join(class_dir, filename), 'r', encoding='utf-8') as f:
                        data[class_name].extend(f.read().splitlines())


            
        return data

    def prepare_data(self, data):
        tokenizer = Tokenizer()
        for class_name in data.keys():
            tokenizer.fit_on_texts(data[class_name])
            sequences = tokenizer.texts_to_sequences(data[class_name])
            if sequences:
                padded = pad_sequences(sequences, padding=self.padding_type, truncating=self.trunc_type)
                data[class_name] = [tf.constant(p, dtype=tf.float32) for p in padded]  # Add an extra dimension at the end to avoid ValueError
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(str(data))
                
        return data


    def generate(self):
        """
        Do main process of converting tensors.
        Usage:
        >>> (train_english, train_russian), (valid_english, valid_russian) = datagen.generate()
        Returns:
        Tuple of two dictionaries. Each dictionary contains class names as keys and lists of tensors as values. Tensors are of dtype float32.
        """
        train_data = self.load_data(self.train_dir)
        valid_data = self.load_data(self.valid_dir)
        train_data = self.prepare_data(train_data)
        valid_data = self.prepare_data(valid_data)

        train_data = {k: v for k, v in train_data.items() if len(v) > 0}
        valid_data = {k: v for k, v in valid_data.items() if len(v) > 0}


        print(f"Train data info: {len(train_data.keys())} classes, {sum([len(v) for v in train_data.values()])} samples")
        print(f"Valid data info: {len(valid_data.keys())} classes, {sum([len(v) for v in valid_data.values()])} samples")
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"\nTrain data info: {len(train_data.keys())} classes, {sum([len(v) for v in train_data.values()])} samples\n")
            log_file.write(f"Valid data info: {len(valid_data.keys())} classes, {sum([len(v) for v in valid_data.values()])} samples\n")

        # Convert the data to TensorFlow Datasets
        train_data = {k: Dataset.from_tensor_slices(v) for k, v in train_data.items()}
        valid_data = {k: Dataset.from_tensor_slices(v) for k, v in valid_data.items()}

        return ((train_data['english'], train_data['russian']), (valid_data['english'], valid_data['russian']))
