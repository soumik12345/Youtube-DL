import tensorflow as tf
from datetime import datetime

from .model import NoobModel
from .dataloader import DataLoader
from .callbacks import ClassifierCallback


class Trainer:

    def __init__(self, image_size=256):
        self.image_size = image_size
        self.train_dataset = None
        self.val_dataset = None
        self.model = None

    def build_datasets(
            self, image_file_pattern: str, show_sanity_checks=True, using_streamlit=True,
            val_split=0.2, buffer_size=1024, train_batch_size=16, val_batch_size=16):
        dataloader = DataLoader(
            image_file_pattern=image_file_pattern,
            show_sanity_checks=show_sanity_checks,
            image_size=self.image_size
        )
        self.train_dataset, self.val_dataset = dataloader.build_dataset(
            val_split=val_split, buffer_size=buffer_size, using_streamlit=using_streamlit,
            train_batch_size=train_batch_size, val_batch_size=val_batch_size
        )

    def build_model(self, model_schematic_location='model.png'):
        self.model = NoobModel()
        self.model.build((1, self.image_size, self.image_size, 3))
        self.model.summary()
        if model_schematic_location is not None:
            tf.keras.utils.plot_model(
                self.model, to_file='model.png', show_shapes=True,
                show_layer_names=True, rankdir='TB'
            )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'],
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        )

    def train(self, epochs: int):
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = [
            ClassifierCallback(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./checkpoints/model_best.ckpt', monitor='val_loss',
                save_weights_only=True, save_best_only=True, mode='min', save_freq='epoch'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1,
                update_freq=50, write_images=True
            )
        ]
        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,
            epochs=epochs, callbacks=callbacks
        )
        return history
