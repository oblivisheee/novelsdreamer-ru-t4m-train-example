import tensorflow as tf
import matplotlib.pyplot as plt
import os

class Epoch:
    def __init__(self, transformer, optimizer, metrics, loss_object, log: bool):
        self.transformer = transformer
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss_object = loss_object
        self.log_epoch = log

    @tf.function
    def train_step(self, inp, tar):
        tar_inp = tf.convert_to_tensor(tar)[:, :-1]
        tar_real = tf.convert_to_tensor(tar)[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp, True)
            loss = self.loss_object(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        for metric in self.metrics:
            metric.update_state(tar_real, predictions)
        
        if self.log_epoch:
            self.log_weights()

    def log_weights(self, log_dir='logs/weights'):
        os.makedirs(log_dir, exist_ok=True)
        for i, var in enumerate(self.transformer.trainable_variables):
            fig = plt.figure(figsize=(10, 10))
            plt.title(f'Weight {i}')
            plt.imshow(var.numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(os.path.join(log_dir, f'weight_{i}.png'))
            plt.close(fig)