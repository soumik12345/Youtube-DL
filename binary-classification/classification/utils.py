import numpy as np
from matplotlib import pyplot as plt
from plotly import express as px


class Visualizer:

    def __init__(self, unique_labels):
        self.unique_labels = unique_labels

    def visualize_batch(self, sample_images, sample_labels, using_streamlit: bool):
        plt.figure(figsize=(15, 15))
        for i in range(9):
            axis = plt.subplot(3, 3, i + 1)
            plt.imshow(sample_images[i].numpy().astype(np.uint8))
            plt.title(self.unique_labels[sample_labels[i]])
            plt.axis('off')
        if using_streamlit:
            import streamlit as st
            st.pyplot(plt)
        else:
            plt.show()
