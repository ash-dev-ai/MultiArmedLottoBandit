import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Conv2D, GlobalMaxPooling2D, Dense, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class CNNModel:
    def __init__(self, output_dim, max_sequence_length, num_classes):
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes

    def build_model(self, input_shape):
        input_layer = Input(shape=input_shape)
        reshape_layer = Reshape((input_shape[0], input_shape[1], 1))(input_layer)
        conv_layer = Conv2D(128, (5, 1), activation='relu')(reshape_layer)
        global_pooling_layer = GlobalMaxPooling2D()(conv_layer)
        output_layer = Dense(self.num_classes, activation='softmax')(global_pooling_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def try_cnn_scenario(self, X, y_onehot):
        X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

        # Update the input_shape parameter based on the shape of X_train
        input_shape = X_train.shape[1:]

        model = self.build_model(input_shape)  # Pass the input_shape to the build_model method

        # One-hot encode the target labels
        encoder = LabelEncoder()
        all_labels = pd.concat([y_train, y_test])
        y_all_onehot = tf.keras.utils.to_categorical(encoder.fit_transform(all_labels), num_classes=self.num_classes)

        # Separate the one-hot encoded target labels back into train and test sets
        y_train_onehot = y_all_onehot[:len(y_train)]
        y_test_onehot = y_all_onehot[len(y_train):]

        model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

        _, test_accuracy = model.evaluate(X_test, y_test_onehot)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(np.argmax(y_test_onehot, axis=1), np.argmax(y_pred, axis=1))

        return test_accuracy, mse

    def try_all_cnn_scenarios(self, X, y_onehot, scenarios):
        results = {}

        for scenario_name, scenario_data in scenarios.items():
            X = scenario_data["features"]
            y_onehot = scenario_data["labels"]

            test_accuracy, mse = self.try_cnn_scenario(X, y_onehot)
            results[scenario_name] = {"CNN Test Accuracy": test_accuracy, "CNN Mean Squared Error": mse}

        return results
