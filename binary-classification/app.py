import streamlit as st
from appmodule.training import train_module
from appmodule.inference import inference_module


def main():
    st.markdown('''
    <h1 align="center">Bee/Wasp Classification Application</h1><hr>
    ''', unsafe_allow_html=True)
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
