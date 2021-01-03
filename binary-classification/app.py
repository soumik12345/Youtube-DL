import streamlit as st
from appmodule.training import train_module
from appmodule.inference import inference_module


def main():

    st.markdown('''
    <h1 align="center">Bee/Wasp Classification Application</h1><hr>
    ''', unsafe_allow_html=True)

    st.markdown('''
    Download [Bee or wasp?](https://www.kaggle.com/jerzydziewierz/bee-vs-wasp) from Kaggle,
    a dataset consisting of 19480 Hand curated photos of bees, wasps and other insects.''')

    option = st.selectbox(
        'What would you like to do???',
        ('', 'Train a Model', 'Inference using a pre-trained model')
    )

    if option == 'Train a Model':
        train_module()
    elif option == 'Inference using a pre-trained model':
        inference_module()


if __name__ == '__main__':
    main()
