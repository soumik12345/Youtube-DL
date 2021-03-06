import os
from glob import glob
import streamlit as st
from classification.train import Trainer
from classification.inference import ModelPredictor


def train_module():

    dataset_path = st.text_input('Enter Dataset Path: ')
    image_file_pattern = str(os.path.join(dataset_path, '*[1-2]/*.jpg'))
    print(image_file_pattern)
    show_sanity_checks = st.checkbox('Show Sanity Checks??? ')
    image_size = st.number_input(
        "Enter size of the images: ",
        min_value=64, max_value=512, value=256, step=16
    )

    col1, col2 = st.beta_columns(2)

    with col1:
        val_split = st.number_input(
            "Enter Fraction of Validation Split: ",
            min_value=0.1, max_value=0.5, value=0.2, step=0.01
        )
    with col2:
        test_split = st.number_input(
            "Enter Fraction of Test Split (as a fraction of Validation Split): ",
            min_value=0.1, max_value=0.5, value=0.2, step=0.01
        )

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

    files_existence_list = []
    try:
        files_existence_list = [os.path.exists(_file) for _file in glob(image_file_pattern)]
    except Exception as e:
        print(e)

    if len(dataset_path) > 0:
        if [True] * len(glob(image_file_pattern)) == files_existence_list:
            trainer = Trainer(image_size=image_size)
            if show_sanity_checks:
                st.markdown(
                    '<br><h2>Sample Images for Dataset<h2>',
                    unsafe_allow_html=True
                )
            trainer.build_datasets(
                image_file_pattern=image_file_pattern, show_sanity_checks=show_sanity_checks,
                using_streamlit=True, val_split=val_split, test_split=test_split, buffer_size=1024,
                train_batch_size=train_batch_size, val_batch_size=val_batch_size
            )
            trainer.build_model(model_schematic_location=None)

            epochs = st.number_input(
                'Enter number of epochs: ',
                min_value=2, max_value=10000, step=2, value=10
            )

            start_training = st.button('Start Training', key='button1')

            if start_training:
                st.text('Use the following command to start and view tensorboard:')
                st.text('tensorboard --logdir="./logs" --port 6009')
                trainer.train(epochs=epochs, using_streamlit=True)
                predictor = ModelPredictor(unique_labels=trainer.unique_labels)
                predictor.build_model(image_size=image_size, weights_location=None)
                st.markdown('**Train Dataset Evaluation Result: **')
                predictor.evaluate(test_dataset=trainer.train_dataset, using_streamlit=True)
                st.markdown('**Validation Dataset Evaluation Result: **')
                predictor.evaluate(test_dataset=trainer.val_dataset, using_streamlit=True)
                st.markdown('**Test Dataset Evaluation Result: **')
                predictor.evaluate(test_dataset=trainer.test_dataset, using_streamlit=True)
