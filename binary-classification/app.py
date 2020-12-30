import streamlit as st
from appmodule.training import train_module


def main():
    st.markdown('''
    <h1 align="center">Image Classification Application</h1><hr>
    ''', unsafe_allow_html=True)
    option = st.selectbox(
        'What would you like to do???',
        ('', 'Train a Model', 'Inference for a pre-trained model')
    )
    if option == 'Train a Model':
        train_module()
    elif option == 'Inference for a pre-trained model':
        st.warning('Not yet implemented')


if __name__ == '__main__':
    main()
