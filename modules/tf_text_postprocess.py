import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

class TextPostprocessor:
    def __init__(self, tokenizer: Tokenizer):
        if not isinstance(tokenizer, Tokenizer):
            raise ValueError("tokenizer must be an instance of Tokenizer class")
        self.tokenizer = tokenizer

    def sequences_to_texts(self, sequences):
        if not isinstance(sequences, (list, tf.Tensor)):
            raise ValueError("sequences must be a list or a Tensor")
        if isinstance(sequences, tf.Tensor):
            sequences = sequences.numpy().tolist()
        texts = self.tokenizer.sequences_to_texts(sequences)
        return texts
