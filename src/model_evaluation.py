import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, models_path, label_map_path, processed_dataset_path, batch_size=32, output_dir="output"):
        self.models_path = models_path
        self.label_map_path = label_map_path
        self.processed_dataset_path = processed_dataset_path
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models_info = []

    def load_test_data(self):
        # Load label mapping
        with open(self.label_map_path, "rb") as f:
            label_to_index = pickle.load(f)

        # Load .npy test data
        test_ds, X_test, y_test = self._load_dataset(
            split_path=os.path.join(self.processed_dataset_path, "test"),
            label_to_index=label_to_index,
            shuffle=False
        )

        self.test_ds = test_ds
        self.X_test = X_test
        self.y_test = y_test

    def _load_dataset(self, split_path, label_to_index, input_shape=(40, 101), shuffle=False):
        data, labels = [], []

        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)
            if not os.path.isdir(label_path):
                continue

            for fname in os.listdir(label_path):
                if not fname.endswith(".npy"):
                    continue

                file_path = os.path.join(label_path, fname)
                spectrogram = np.load(file_path).astype(np.float32)

                if spectrogram.shape != input_shape:
                    print(f"Skipping {file_path}, wrong shape: {spectrogram.shape}")
                    continue

                data.append(spectrogram)
                labels.append(label_to_index[label])

        data = np.array(data)[..., np.newaxis]
        labels = np.array(labels)

        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset, data, labels

    def evaluate_all(self):
        self.load_test_data()

        model_folders = [
            d for d in os.listdir(self.models_path)
            if os.path.isdir(os.path.join(self.models_path, d)) and not d.startswith("__")
        ]

        for model_name in model_folders:
            print(f"\nEvaluating {model_name}...")
            folder_path = os.path.join(self.models_path, model_name)

            model_path = os.path.join(folder_path, f"{model_name}_model.keras")
            metadata_path = os.path.join(folder_path, f"metadata_{model_name}.json")

            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                print(f"Skipping {model_name}: Missing model or metadata.")
                continue

            model = load_model(model_path)
            y_pred = np.argmax(model.predict(self.test_ds, verbose=0), axis=1)

            acc = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="macro")
            recall = recall_score(self.y_test, y_pred, average="macro")
            f1_macro = f1_score(self.y_test, y_pred, average="macro")
            f1_weighted = f1_score(self.y_test, y_pred, average="weighted")

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Get model size in MB
            model_size_bytes = os.path.getsize(model_path)
            model_size_mb = round(model_size_bytes / (1024 * 1024), 2)

            # Update metadata with evaluation metrics
            metadata.update({
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "model_size_MB": model_size_mb,
            })

            self.models_info.append(metadata)
            self.plot_confusion_matrix(self.y_test, y_pred, model_name)

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        model_output_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        save_path = os.path.join(model_output_dir, f"confusion_matrix_{model_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")

    def export_results_csv(self, filename="evaluation_results.csv"):
        df = pd.DataFrame(self.models_info)
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"\nEvaluation results saved to: {output_path}")
        return df
