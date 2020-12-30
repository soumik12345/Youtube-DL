import tensorflow as tf


class NoobModel(tf.keras.Model):

    def __init__(self):
        super(NoobModel, self).__init__()
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        self.conv_layer_1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.conv_layer_2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.conv_layer_3 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.max_pool_1 = tf.keras.layers.MaxPooling2D()
        self.max_pool_2 = tf.keras.layers.MaxPooling2D()
        self.max_pool_3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layer = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.rescale(inputs)
        x = self.conv_layer_1(x)
        x = self.max_pool_1(x)
        x = self.conv_layer_2(x)
        x = self.max_pool_2(x)
        x = self.conv_layer_3(x)
        x = self.max_pool_3(x)
        x = self.flatten(x)
        x = self.hidden_layer(x)
        return self.output_layer(x)
