import tensorflow as tf
from keras import layers, models

def baseline_cnn(input_shape=(32, 40, 1), num_classes=35):
    """
    Baseline CNN

    Args:
        input_shape (tuple): Shape of the input spectrogram (H, W, C).
        num_classes (int): Number of output classes.

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape), # 32 filters, 3x3 kernel
        layers.BatchNormalization(),
        
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), # 64 filters, 3x3 kernel
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax') # Output layer with softmax activation for multi-class classification
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
