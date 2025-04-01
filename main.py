from src import (SpeechPreprocessor, 
                load_dataset)
from utils import get_class_weights
from models import baseline_cnn
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

def main():

    # Path to the dataset
    raw_data_dir = "Data/speech_commands_v0.02" # Raw dataset path
    data_dir = "Data/processed_dataset"

    class_weights, label_to_index, index_to_label = get_class_weights(raw_data_dir) # Get class weights for imbalanced dataset

    # Load datasets
    train_ds = load_dataset(f"{data_dir}/train", label_to_index)
    val_ds = load_dataset(f"{data_dir}/val", label_to_index, shuffle=False)
    test_ds = load_dataset(f"{data_dir}/test", label_to_index, shuffle=False)

    # Batch and prefetch
    BATCH_SIZE = 64
    train_ds = (
        train_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
        .cache()  # Optional, but helps for re-runs
    )
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create and train model
    print("Creating model...")
    model = baseline_cnn(input_shape=(40, 101, 1), num_classes=len(label_to_index))

    print("Training model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights
    )
    print("Training complete.")

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Plot loss and accuracy curves
    history = model.history.history
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Baseline CNN Accuracy')
    plt.show()
    plt.savefig("baseline_cnn_accuracy.png")
    plt.close()

    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Baseline CNN Loss')
    plt.show()
    plt.savefig("baseline_cnn_loss.png")
    plt.close()

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {val_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")


    # Save the model
    model.save("baseline_cnn_model.h5")
    print("Model saved as baseline_cnn_model.h5")


if __name__ == '__main__':
    main()
