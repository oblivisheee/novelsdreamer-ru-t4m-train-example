import tensorflow as tf
def parallel_compile(num_cores: int):

    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:{}".format(i) for i in range(num_cores)])

    return strategy.scope()

