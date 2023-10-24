import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from modules.logs import log_train
import numpy as np
from math import log
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# Positional Encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Scaled Dot Product Attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Multi-head Attention
class MultiHeadAttention(layers.Layer):
    """
    Class that allows the model to focus on different positions of the input sequence simultaneously, 
    capturing various aspects of the sequence. 
    It does this by splitting the input into multiple "heads" and applying the attention mechanism to each head independently before recombining the results. This allows the model to capture various types of information from different positions of the input.
    """
    def __init__(self, d_model, num_heads, embedding=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        self.embedding = embedding

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, embedding=None):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, embedding)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Decoder Layer
class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, embedding=None):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, embedding)
        self.mha2 = MultiHeadAttention(d_model, num_heads, embedding)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2

# Encoder
class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1, embedding=None):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = layers.Embedding(input_vocab_size, d_model) if embedding is None else embedding
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, embedding) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

# Decoder
class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1, embedding=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = layers.Embedding(target_vocab_size, d_model) if embedding is None else embedding
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, embedding) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights

# Transformer
class Transformer(tf.keras.Model):
    """
    The Transformer class is a model that uses multi-head attention mechanisms.
    It contains an Encoder, a Decoder, and a final Dense layer that are all applied in sequence.
    This class also includes methods for fitting the model to data, saving the model, and creating masks for the input data.
    
    Usage
    ```python
    transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1, embedding=None)
    ```
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1, embedding=None):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate, embedding)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate, embedding)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    def beam_search_decoder(self, data, k):
        sequences = [[list(), 1.0]]
        for row in data:
            all_candidates = list()
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * -log(row[j])]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            sequences = ordered[:k]
        return sequences

    

    
    def fit_model(self, train_english, train_russian, valid_english, 
                  valid_russian, epochs, model_name, 
                    save_model_each_epoch=False, 
                    logs=True, logs_path= '/logs/plots', callback=None, save_path='results/trained_models'):
        """
        Training the model and save it.

        Usage:
        ```python
        model = Transformer() # Settings of transformer


        epochs = 50
        model.fit_model(train_english, 
                                train_russian,
                                valid_english,
                                valid_russian,
                                epochs=epochs,
                                save_model_each_epoch=True,
                                logs=True,
                                model_name='novelsdreamer-ru-t4m')
                
        ```
        """
        # Create a log file
        log_file_name = 'training_logs.txt'
        log_file_path = os.path.join('logs', 'train', log_file_name)
        # Check if the file already exists. If it does, append to it. If not, create a new file.
        if os.path.isfile(log_file_path):
            log_file = open(log_file_path, "a")
        else:
            log_file = open(log_file_path, "w")

        for epoch in tqdm(range(1, epochs+1)):
            # Ensure the data is in the correct format before creating masks
            train_english_tensor = tf.convert_to_tensor([x for x in train_english], dtype=tf.int32)
            train_russian_tensor = tf.convert_to_tensor([x for x in train_russian], dtype=tf.int32)
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(train_english_tensor, train_russian_tensor)
            with tf.GradientTape() as tape:
                predictions, _ = self.call(train_english_tensor, train_russian_tensor, True, enc_padding_mask, combined_mask, dec_padding_mask)
                loss = self.loss(train_russian_tensor, predictions)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            if len(self.metrics) > 0:
                self.metrics[0].update_state(train_russian_tensor, predictions)
            
            # Validation
            valid_english_tensor = tf.convert_to_tensor([x for x in valid_english], dtype=tf.int32)
            valid_russian_tensor = tf.convert_to_tensor([x for x in valid_russian], dtype=tf.int32)
            enc_padding_mask_valid, combined_mask_valid, dec_padding_mask_valid = self.create_masks(valid_english_tensor, valid_russian_tensor)
            predictions_valid, _ = self.call(valid_english_tensor, valid_russian_tensor, False, enc_padding_mask_valid, combined_mask_valid, dec_padding_mask_valid)
            loss_valid = self.loss(valid_russian_tensor, predictions_valid)
            if len(self.metrics) > 1:
                self.metrics[1].update_state(valid_russian_tensor, predictions_valid)
            print('Epoch {} Loss {:.4f} Validation Loss {:.4f}'.format(epoch, loss, loss_valid))
            log_file.write('Epoch {} Loss {:.4f} Validation Loss {:.4f}\n'.format(epoch, loss, loss_valid))  # Save logs to file
            print(f'Epoch {epoch} finished.')
            if callback is not None:
                callback()
            #if logs:
            #   log_train(predictions_valid, logs_path, epoch)
                

            if save_model_each_epoch:
                self.save_weights(save_path + f'{model_name}_epoch_{epoch+1}')


        
        # Save the model after training
        try:
            self.save_weights(save_path + model_name)
            print('Final weights saved.')
        except:
            print('Happened an error during saving final weights.')
            
        log_file.close()  # Close the log file
        return self

    def create_masks(self, inp, tar):
        """
        Create masks for training.
        """
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    @staticmethod
    def create_padding_mask(seq):
        """
        Create padding mask for sequence.
        """
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    @staticmethod
    def create_look_ahead_mask(size):
        """
        Create look ahead mask for sequence.
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask