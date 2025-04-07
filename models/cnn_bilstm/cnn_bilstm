import tensorflow as tf
from tensorflow.keras import layers, models

class CNNBiLSTMModel:
    def __init__(self, input_shape=(40, 101, 1), num_classes=12,
                 lstm_units=32, dense_units=64, dropout_rate=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # --- Convolutional Layers ---
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # --- Reshape for RNN ---
        shape = x.shape
        time_steps = shape[1]
        features = shape[2] * shape[3]
        x = layers.Reshape((time_steps, features))(x)

        # --- BiLSTM Layer ---
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=False)
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # --- Dense Classifier ---
        x = layers.Dense(self.dense_units, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 1.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        return models.Model(inputs, outputs)

    def compile(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def fit(self, train_data, val_data, epochs=30, callbacks=None):
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )

    def evaluate(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        return self.model.summary()

    def save(self, path):
        self.model.save(path)

    def get_model(self):
        return self.model