import os
import numpy as np
import tensorflow as tf

def load_dataset(split_path, label_to_index, input_shape=(32, 40), shuffle=True):
    """
    Loads a dataset split (train/val/test) from .npy spectrograms.

    Args:
        split_path (str): Path to 'train', 'val', or 'test' directory.
        label_to_index (dict): Mapping of folder names to integer labels.
        input_shape (tuple): Shape of the spectrograms.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: (spectrogram, label) dataset.
    """
    data = []
    labels = []

    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)
        if not os.path.isdir(label_path):
            continue

        for fname in os.listdir(label_path):
            if not fname.endswith(".npy"):
                continue

            file_path = os.path.join(label_path, fname)
            spectrogram = np.load(file_path).astype(np.float32)

            # Ensure shape matches
            if spectrogram.shape != input_shape:
                continue

            data.append(spectrogram)
            labels.append(label_to_index[label])

    data = np.array(data)[..., np.newaxis]  # Add channel dimension
    labels = np.array(labels)

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    return dataset
