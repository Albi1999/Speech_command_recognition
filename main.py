from src import (SpeechPreprocessor, 
                load_dataset)
from utils import get_class_weights
from models import baseline_cnn
import tensorflow as tf

def main():

    # Path to the dataset
    raw_data_dir = "Data/speech_commands_v0.02" # Raw dataset path
    data_dir = "Data/processed_dataset"

    # Create the preprocessor instance
    processor = SpeechPreprocessor(raw_data_dir)

    # Process all audio files
    processor.process_audio_files()

    # Visualize a sample spectrogram
    processor.visualize_random_sample()

    class_weights, label_to_index, index_to_label = get_class_weights(raw_data_dir) # Get class weights for imbalanced dataset

    # Load datasets
    train_ds = load_dataset(f"{data_dir}/train", label_to_index)
    val_ds = load_dataset(f"{data_dir}/val", label_to_index, shuffle=False)
    test_ds = load_dataset(f"{data_dir}/test", label_to_index, shuffle=False)

    # Batch and prefetch
    BATCH_SIZE = 32
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create and train model
    model = baseline_cnn(input_shape=(32, 40, 1), num_classes=len(label_to_index))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()
