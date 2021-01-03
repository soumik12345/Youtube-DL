import os
import numpy as np
from glob import glob
import streamlit as st
import tensorflow as tf
from .model import NoobModel
from plotly import express as px
from matplotlib import pyplot as plt
from plotly import graph_objects as go


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
            checkpoint_path = sorted(
                glob('./checkpoints/epoch_*'),
                key=lambda x: int(x.split('_')[-1]))[-1]
            st.text(
                'Loaded Checkpoints for Epoch: {}'.format(
                    checkpoint_path.split('_')[-1]))
            self.model.load_weights(
                os.path.join(checkpoint_path, 'classifier_weights.ckpt')
            )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'],
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        )

    def evaluate(self, test_dataset, using_streamlit: bool):
        loss, accuracy = self.model.evaluate(test_dataset)
        if using_streamlit:
            st.markdown('Loss: {}'.format(loss), unsafe_allow_html=True)
            st.markdown('Accuracy: {}'.format(accuracy), unsafe_allow_html=True)

    def _visualize_single_image_prediction(
            self, pil_image, y_pred, probability_class_0, probability_class_1):

        figure = px.imshow(pil_image)
        figure.update_layout(
            title={
                'text': 'Predicted Label: ' + str(self.unique_labels[y_pred[0]]),
                'x': 0.5, 'y': 0.05, 'xanchor': 'center', 'yanchor': 'bottom'
            })
        figure.update_layout(coloraxis_showscale=False)
        figure.update_xaxes(showticklabels=False)
        figure.update_yaxes(showticklabels=False)
        st.plotly_chart(figure, use_container_width=True)

        bar_figure = go.Figure([
            go.Bar(
                x=self.unique_labels[::-1],
                y=[
                    probability_class_0,
                    probability_class_1
                ])
        ])
        bar_figure.update_layout(title='Class Probablities')
        st.plotly_chart(bar_figure)

    def predict_from_image(self, pil_image, image_size, using_streamlit: bool):
        image = tf.keras.preprocessing.image.img_to_array(pil_image)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.expand_dims(image, axis=0)
        y_pred = tf.nn.sigmoid(self.model(image))
        probability_class_0 = y_pred.numpy()[0][0]
        probability_class_1 = 1 - probability_class_0
        y_pred = [0 if _y[0] < 0.5 else 1 for _y in y_pred.numpy()]
        if using_streamlit:
            self._visualize_single_image_prediction(
                pil_image, y_pred, probability_class_0, probability_class_1)
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
