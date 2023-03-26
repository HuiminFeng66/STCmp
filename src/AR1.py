import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense

class AR():
    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = Dense(window, 1)

    def call(self, x):
    	# x: [batch, window, n_multiv]
        x = tf.transpose(x, 1, 2)  # x: [batch, n_multiv, window]
        x = self.linear(x)  # x: [batch, n_multiv, 1]
        x = tf.transpose(x, 1, 2)  # x: [batch, 1, n_multiv]
        return x
