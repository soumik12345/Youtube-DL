import numpy as np
import tensorflow as tf
from .model import NoobModel
from matplotlib import pyplot as plt


class ModelPredictor:

    def __init__(self, unique_labels):
        self.model = None
        self.unique_labels = unique_labels

    def build_model(self, weights_location=None):
        self.model = NoobModel()
        self.model.build((1, self.image_size, self.image_size, 3))
        if weights_location is not None:
            self.model.load_weights(weights_location)

    def infer_from_dataset(self, dataset, using_streamlit: bool):
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
