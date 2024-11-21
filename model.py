import tensorflow as tf
import pandas as pd
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_imf_sequences(imfs, look_back):
    X,y = [], []
    for i in range(len(imfs) - look_back):
        X.append(imfs[i: i + look_back])
        y.append(imfs[i + look_back, 0])
    return np.array(X), np.array(y)

def preprocess_and_decompose(df_training, df_testing, target_column, look_back, K):
    transformer = VmdTransformer(K=K)
    imfs_training = transformer.fit_transform(df_training[target_column].values)
    imfs_testing = transformer.transform(df_testing[target_column].values)
    
    print("IMFs Training Shape:", imfs_training.shape)
    print("IMFs Testing Shape:", imfs_testing.shape)
    print("NaN in IMFs Training:", np.isnan(imfs_training).any())
    print("Infinity in IMFs Training:", np.isinf(imfs_training).any())

    X_train, y_train = create_imf_sequences(imfs_training, look_back)
    X_test, y_test = create_imf_sequences(imfs_testing, look_back)
    
    # X_train, y_train = create_imf_sequences(df_training, look_back)
    # X_test, y_test = create_imf_sequences(df_testing, look_back)

    return X_train, y_train, X_test, y_test

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def main():
    # Driver function of this model. Calls main functions and sets all the values for the 
    # VMD - BPNN model.
    print('Model testing and Data Preprocessing starting...')

    start_date_training  = '2016-08-22 00:00:00-04:00'
    end_date_training = '2023-06-16 00:00:00-04:00'
    start_date_testing = '2023-06-20 00:00:00-04:00'
    end_date_testing = '2023-09-21 00:00:00-04:00'
    look_back = 5
    K = 5
    target_column = 'Close'


    file = r'./Data/AAPL.csv'
    columns_to_drop = ['Open','High','Low','Volume','Dividends','Stock Splits'] 
    dataFrame = pd.read_csv(file)
    dataFrame.drop(columns_to_drop, axis=1)

    dataFrame_training = dataFrame[(dataFrame['Date'] >= start_date_training) & (dataFrame['Date'] <= end_date_training)].copy()
    dataFrame_testing = dataFrame[(dataFrame['Date']>= start_date_testing)&(dataFrame['Date'] <= end_date_testing)].copy()
    
    scaler = MinMaxScaler()
    dataFrame_training[target_column] = scaler.fit_transform(dataFrame_training[[target_column]])
    dataFrame_testing[target_column] = scaler.transform(dataFrame_testing[[target_column]])
    
    X_train, y_train, X_test, y_test = preprocess_and_decompose(dataFrame_training, dataFrame_testing, target_column, look_back, K)
    print("Training data shapes:", X_train.shape, y_train.shape)
    print("Testing data shapes:", X_test.shape, y_test.shape)
    print("NaN in X_test:", np.isnan(X_test).any())
    print("Infinity in X_test:", np.isinf(X_test).any())
    print("NaN in y_test:", np.isnan(y_test).any())
    print("Infinity in y_test:", np.isinf(y_test).any())
    print("Min y_test:", np.min(y_test), "Max y_test:", np.max(y_test))

    # TensorFlow BackPropagation Model Building
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, K)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2),
        # tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mse', metrics=['mae', root_mean_squared_error])
    model.summary()

    history = model.fit(X_train,y_train, validation_split = 0.2, epochs= 10, batch_size=10)

    print(history.history['loss'])
    print(history.history['val_loss'])

    plt.plot(history.history['loss'], label= 'Training Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    loss, mae, rmse = model.evaluate(X_test, y_test)
    print('Test loss: ', loss)
    print('Test MAE: ', mae)
    print('Test RMSE: ', rmse)

    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 3))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Closing Prices using IMFs")
    plt.show()
    
if __name__ == "__main__":
    main()