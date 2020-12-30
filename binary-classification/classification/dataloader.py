from glob import glob
import tensorflow as tf
from .utils import Visualizer


class DataLoader:

    def __init__(self, image_file_pattern: str, show_sanity_checks: bool, image_size: int):
        self.image_files = glob(image_file_pattern)
        self.show_sanity_checks = show_sanity_checks
        self.image_size = image_size
        self.num_images = len(self.image_files)
        if show_sanity_checks:
            print('Number of images:', self.num_images)
        self.encoded_labels, self.unique_labels = self._build_labels()
        self.visualizer = Visualizer(unique_labels=self.unique_labels)

    def _build_labels(self):
        labels = [_file.split('/')[-2][:-1] for _file in self.image_files]
        unique_labels = list(set(labels))
        if self.show_sanity_checks:
            print(unique_labels)
        return [unique_labels.index(label) for label in labels], unique_labels

    def _read_data(self, file_name, label):
        image = tf.io.read_file(file_name)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        return image, label

    def _get_configured_dataset(self, dataset, buffer_size, batch_size):
        dataset = dataset.map(self._read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(1)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def build_dataset(
            self, val_split: float, buffer_size: int, train_batch_size: int,
            val_batch_size: int, using_streamlit: bool):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.image_files, self.encoded_labels)).shuffle(
            self.num_images, reshuffle_each_iteration=False
        )
        if self.show_sanity_checks:
            for data in dataset.take(5):
                filename, label = data
                print(
                    'Filename: {}\nLabel: {}\n\n'.format(
                        filename, self.unique_labels[label]))
        train_dataset = dataset.skip(int(self.num_images * val_split))
        val_dataset = dataset.take(int(self.num_images * val_split))
        train_dataset = self._get_configured_dataset(
            train_dataset, buffer_size=buffer_size, batch_size=train_batch_size)
        val_dataset = self._get_configured_dataset(
            val_dataset, buffer_size=buffer_size, batch_size=val_batch_size)
        if self.show_sanity_checks:
            print(train_dataset)
            print(val_dataset)
            sample_images, sample_labels = next(iter(train_dataset))
            self.visualizer.visualize_batch(
                sample_images=sample_images, sample_labels=sample_labels,
                using_streamlit=using_streamlit
            )
        return train_dataset, val_dataset
