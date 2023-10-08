import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from transformer_custom import Transformer, EncoderLayer, DecoderLayer

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

input_vocab_size = 3000
target_vocab_size = 3000
maximum_position_encoding = 10000

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=maximum_position_encoding, 
                          pe_target=maximum_position_encoding,
                          rate=dropout_rate)

# Dummy input
temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = transformer(temp_input, temp_target, training=False, 
                        enc_padding_mask=None, 
                        look_ahead_mask=None,
                        dec_padding_mask=None)

# Print and visualize the shape
print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(np.arange(3), fn_out.shape)
plt.xticks(np.arange(3), ['batch_size', 'tar_seq_len', 'target_vocab_size'])
plt.ylabel('Dimension size')
plt.title('Output tensor dimensions')
plt.show()