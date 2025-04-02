from src import (SpeechPreprocessor, 
                load_dataset)
from utils import get_class_weights
from models import baseline_cnn, ResidualCNN
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

def main():

    # Path to the dataset
    raw_data_dir = "Data/speech_commands_v0.02" # Raw dataset path
    data_dir = "Data/processed_dataset"

    class_weights, label_to_index, index_to_label = get_class_weights(raw_data_dir) # Get class weights for imbalanced dataset

    with open("class_weights.pkl", "wb") as f:
        pickle.dump(class_weights, f)
    with open("label_to_index.pkl", "wb") as f:
        pickle.dump(label_to_index, f)
    with open("index_to_label.pkl", "wb") as f:
        pickle.dump(index_to_label, f)
    
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
        .cache()
    )
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create and train model
    print("Creating model...")
    residual_cnn = ResidualCNN(input_shape=(40, 101, 1), num_classes=len(label_to_index))
    residual_cnn.compile()
    model = residual_cnn.get_model()

    print("Training model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs = 20,
        class_weight=class_weights
    )
    print("Training complete.")

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Plot loss and accuracy curves
    history = model.history.history
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Residual CNN Accuracy')
    plt.show()
    plt.savefig("output/residual_cnn/residual_cnn_accuracy.png")
    plt.close()

    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Residual CNN Loss')
    plt.show()
    plt.savefig("output/residual_cnn/residual_cnn_loss.png")
    plt.close()

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nValidation accuracy: {val_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")


    # Save the model
    model.save("models/residual_ccn/residual_cnn_model.keras")
    model.save("models/residual_ccn/residual_cnn_model.weights.h5")
    print("Model saved as baseline_cnn_model.h5")


if __name__ == '__main__':
    main()
