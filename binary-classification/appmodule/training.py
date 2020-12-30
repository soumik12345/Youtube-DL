import os
from glob import glob
import streamlit as st
from classification.train import Trainer


def train_module():

    image_file_pattern = st.text_input('Enter Image File Pattern: ')
    show_sanity_checks = st.checkbox('Show Sanity Checks??? ')
    image_size = st.number_input(
        "Enter size of the images: ",
        min_value=64, max_value=512, value=256, step=16
    )

    val_split = st.number_input(
        "Enter Fraction of Validation Split: ",
        min_value=0.1, max_value=0.5, value=0.2, step=0.01
    )
    col1, col2 = st.beta_columns(2)
    with col1:
        train_batch_size = st.number_input(
            "Enter Training Batch Size: ",
            min_value=4, max_value=256, value=16, step=4
        )
    with col2:
        val_batch_size = st.number_input(
            "Enter Validation Batch Size: ",
            min_value=4, max_value=256, value=16, step=4
        )

    files_existence_list = [os.path.exists(_file) for _file in glob(image_file_pattern)]

    if len(image_file_pattern) > 0:
        if [True] * len(glob(image_file_pattern)) == files_existence_list:
            trainer = Trainer(image_size=image_size)
            if show_sanity_checks:
                st.markdown(
                    '<br><h2>Sample Images for Dataset<h2>',
                    unsafe_allow_html=True
                )
            trainer.build_datasets(
                image_file_pattern=image_file_pattern, show_sanity_checks=show_sanity_checks,
                using_streamlit=True, val_split=val_split, buffer_size=1024,
                train_batch_size=train_batch_size, val_batch_size=val_batch_size
            )
            trainer.build_model(model_schematic_location=None)

            epochs = st.number_input(
                'Enter number of epochs: ',
                min_value=2, max_value=10000, step=2, value=10
            )

            start_training = st.button('Start Training', key='button1')

            if start_training:
                trainer.train(epochs=epochs)
