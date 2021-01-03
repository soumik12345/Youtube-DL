import pickle
import streamlit as st
from PIL import Image

from classification.inference import ModelPredictor


def inference_module():
    with open('./checkpoints/unique_labels.pkl', 'rb') as infile:
        unique_labels = pickle.load(infile)
    print(unique_labels)
    predictor = ModelPredictor(unique_labels=unique_labels)
    predictor.build_model(image_size=256, weights_location=None)
    uploaded_files = st.sidebar.file_uploader(
        'Upload an Image', accept_multiple_files=True)
    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            pil_image = Image.open(uploaded_file)
            predictor.predict_from_image(
                pil_image=pil_image, image_size=256, using_streamlit=True)
            st.markdown('---')
