import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from steps import calculate_steps 

def data_preprocessing(column, split_ratio=0.73, real_or_imaginary='r'):
    df_steps, df_steps_r, df_steps_i = calculate_steps()  # Modified to fetch real and imaginary steps
    df_steps = df_steps_r if real_or_imaginary == 'r' else df_steps_i  # Choose based on the argument
    dataset = df_steps[column].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_size = int(len(scaled_data) * split_ratio)
    train, test = scaled_data[:train_size, :], scaled_data[train_size:, :]

    return train, test, scaler

def create_time_series_data(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def study1(columns_to_study=['dw1', 'dw2', 'dw3', 'dw4', 'dw5', 'dwb'], time_step=1, epochs=78, batch_size=3):
    all_results = {}
    start_time = datetime.now()
    
    for column in columns_to_study:
        results = {}
        
        train, test, scaler = data_preprocessing(column)
        X_train, y_train = create_time_series_data(train, time_step)
        X_test, y_test = create_time_series_data(test, time_step)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        y_train = scaler.inverse_transform([y_train])
        test_predict = scaler.inverse_transform(test_predict)
        y_test = scaler.inverse_transform([y_test])

        mse = mean_squared_error(y_test[0], test_predict[:, 0])

        results['Test_MSE'] = mse
        results['Test_values'] = y_test[0]
        results['Predictions'] = test_predict[:, 0]

        all_results[column] = results

    end_time = datetime.now()
    all_results['Total_time'] = end_time - start_time
    return all_results

if __name__ == "__main__":
    study1_results = study1()