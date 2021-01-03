import os
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from .model import NoobModel
from plotly import express as px
from matplotlib import pyplot as plt


class ModelPredictor:

    def __init__(self, unique_labels):
        self.model = None
        self.unique_labels = unique_labels

    def build_model(self, image_size: int, weights_location=None):
        self.model = NoobModel()
        self.model.build((1, image_size, image_size, 3))
        if weights_location is not None:
            self.model.load_weights(weights_location)
        else:
            self.model.load_weights(
                os.path.join(
                    sorted(
                        glob('./checkpoints/*'),
                        key=lambda x: int(x.split('_')[-1]))[-1],
                    'classifier_weights.ckpt'
                )
            )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'],
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        )

    def evaluate(self, test_dataset, using_streamlit: bool):
        loss, accuracy = self.model.evaluate(test_dataset)
        if using_streamlit:
            import streamlit as st
            st.markdown('<strong>Loss: {}</strong>'.format(loss), unsafe_allow_html=True)
            st.markdown('<strong>Accuracy: {}</strong>'.format(loss), unsafe_allow_html=True)

    def predict_from_file(self, file_name: str, image_size, using_streamlit: bool):
        image = tf.io.read_file(file_name)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.expand_dims(image, axis=0)
        y_pred = tf.nn.sigmoid(self.model(image))
        y_pred = [0 if _y[0] < 0.5 else 1 for _y in y_pred.numpy()]
        if using_streamlit:
            import streamlit as st
            figure = px.imshow(np.array(Image.open(file_name)))
            figure.update_layout(
                title='Predicted Label: ' + str(self.unique_labels[y_pred[0]]))
            st.plotly_chart(figure)
        return self.unique_labels[y_pred[0]]

    def predict_batch(self, dataset, using_streamlit: bool):
        x, y = next(iter(dataset))
        y_pred = tf.nn.sigmoid(self.model(x))
        y = y.numpy()
        y_pred = [0 if _y[0] < 0.5 else 1 for _y in y_pred.numpy()]
        plt.figure(figsize=(20, 20))
        for i in range(16):
            axis = plt.subplot(4, 4, i + 1)
            plt.imshow(x[i].numpy().astype(np.uint8))
            plt.title(
                'Actual Label: {}\nPredicted Label: {}'.format(
                    self.unique_labels[y[i]], self.unique_labels[y_pred[i]]))
            plt.axis('off')
        if using_streamlit:
            import streamlit as st
            st.pyplot(plt)
        else:
            plt.show()
