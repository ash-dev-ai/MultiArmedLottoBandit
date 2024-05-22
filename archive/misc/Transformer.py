import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, TFDistilBertModel  # Import the tokenizer

class TransformerEncoder:
    def __init__(self, input_dim, output_dim, max_sequence_length):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length

    def __call__(self, input_layer):
        encoder_layer = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                  input_length=self.max_sequence_length)(input_layer)
        return encoder_layer

class Transformer:
    def __init__(self, input_dim, output_dim, max_sequence_length, num_classes):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes  # Set the num_classes attribute
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # Initialize the tokenizer

    def preprocess_data(self, input_data):
        input_ids = []
        attention_masks = []
        for sentence_list in input_data:
            encoded_dict = self.tokenizer.encode_plus(
                ' '.join(str(s) for s in sentence_list),  # Join the elements of the list into a string
                add_special_tokens=True,
                max_length=self.max_sequence_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='tf'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = tf.concat(input_ids, axis=0)
        attention_masks = tf.concat(attention_masks, axis=0)

        return input_ids, attention_masks

    def build_model(self):
        input_layer = Input(shape=(self.max_sequence_length,))
        encoder_layer = TransformerEncoder(self.input_dim, 128, self.max_sequence_length)(input_layer)  # Set the embedding dimension
        flatten_layer = tf.keras.layers.Flatten()(encoder_layer)
        output_layer = Dense(self.num_classes, activation='softmax')(flatten_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    def try_all_transformer_scenarios(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Combine train and test sets to find the unique classes
        all_classes = np.unique(np.concatenate((y_train, y_test), axis=0))

        # Use LabelEncoder to map classes to integer labels
        label_encoder = LabelEncoder()
        label_encoder.fit(all_classes)
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        num_classes = len(all_classes)

        # Print debug information
        print("Number of classes:", num_classes)
        print("y_train_encoded:", y_train_encoded)
        print("y_test_encoded:", y_test_encoded)

        # Convert y_train and y_test to one-hot encodings using the combined set of unique classes
        y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
        y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)


        # Create and compile the model
        model = self.build_model()

        # Preprocess the data
        X_train, attention_masks_train = self.preprocess_data(X_train)
        X_test, attention_masks_test = self.preprocess_data(X_test)

        # Train the model using model.fit() method directly
        model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

        # Evaluate the model
        _, test_accuracy = model.evaluate(X_test, y_test_onehot)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, np.argmax(y_pred, axis=1))

        return test_accuracy, mse
