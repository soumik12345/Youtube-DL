import streamlit as st
import tensorflow as tf


class ClassifierCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        st.info("Started epoch {} of training...".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        st.success("Ended epoch {} of training".format(epoch))

    def on_train_end(self, logs=None):
        st.success("Training Ended!!!")
        st.balloons()
