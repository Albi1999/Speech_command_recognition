import os
from collections import Counter
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(dataset_path, exclude_dirs=["_background_noise_"]):
    """
    Computes class weights based on the number of .wav files in each class directory.

    Args:
        dataset_path (str): Path to the root dataset folder.
        exclude_dirs (list): List of folder names to exclude (e.g., background noise).

    Returns:
        class_weights (dict): {class_index: weight}
        label_to_index (dict): {label_name: index}
        index_to_label (dict): {index: label_name}
    """
    # List class names
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and d not in exclude_dirs
    ])

    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    # Count .wav files per class
    class_counts = Counter()
    for label in class_names:
        label_path = os.path.join(dataset_path, label)
        num_wavs = len([f for f in os.listdir(label_path) if f.endswith(".wav")])
        class_counts[label] = num_wavs

    # Prepare array for sklearn
    y_all = np.concatenate([[label] * class_counts[label] for label in class_names])

    # Compute weights
    weights = compute_class_weight(class_weight="balanced", classes=np.array(class_names), y=y_all)
    class_weights = {label_to_index[class_names[i]]: weights[i] for i in range(len(class_names))}

    return class_weights, label_to_index, index_to_label
