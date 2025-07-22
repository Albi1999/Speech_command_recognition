import os
from collections import Counter
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pickle

def get_class_weights(train_path):
    """
    Computes class weights based on the number of .wav files in each class directory.

    Args:
        train_path (str): Path to the train dataset folder.

    Returns:
        class_weights (dict): {class_index: weight}
        label_to_index (dict): {label_name: index}
        index_to_label (dict): {index: label_name}
    """
    # List class names
    class_names = sorted([
        d for d in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, d))
    ])

    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    class_counts = Counter()
    for label in class_names:
        label_dir = os.path.join(train_path, label)
        num_files = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
        class_counts[label] = num_files

    y_all = np.concatenate([[label] * class_counts[label] for label in class_names])

    weights = compute_class_weight(class_weight="balanced", classes=np.array(class_names), y=y_all)
    class_weights = {label_to_index[class_names[i]]: weights[i] for i in range(len(class_names))}

    return class_weights, label_to_index, index_to_label


if __name__ == "__main__":

    train_path = "Data/processed_dataset/train"
    class_weights, label_to_index, index_to_label = get_class_weights(train_path)

    # Save class weights and mappings
    with open("class_weights.pkl", "wb") as f:
        pickle.dump(class_weights, f)
    with open("label_to_index.pkl", "wb") as f:
        pickle.dump(label_to_index, f)
    with open("index_to_label.pkl", "wb") as f:
        pickle.dump(index_to_label, f)
