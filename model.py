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

    X_train, y_train = create_imf_sequences(imfs_training, look_back)
    X_test, y_test = create_imf_sequences(imfs_testing, look_back)

    return X_train, y_train, X_test, y_test

def main():
    # Driver function of this model. Calls main functions and sets all the values for the 
    # VMD - BPNN model.
    start_date_training  = '2020-08-21 00:00:00-04:00'
    end_date_training = '2023-08-21 00:00:00-04:00'
    start_date_testing = '2023-08-22 00:00:00-04:00'
    end_date_testing = '2023-09-21 00:00:00-04:00'
    look_back = 30
    K = 10
    target_column = 'Close'


    file = r'./Data/AAPL.csv'
    columns_to_drop = ['Open','High','Low','Volume','Dividends','Stock Splits'] 
    dataFrame = pd.read_csv(file)
    dataFrame.drop(columns_to_drop, axis=1)

    dataFrame_training = dataFrame[(dataFrame['Date'] >= start_date_training) & (dataFrame['Date'] <= end_date_training)]
    dataFrame_testing = dataFrame[(dataFrame['Date']>= start_date_testing)&(dataFrame['Date'] <= end_date_testing)]

    scaler = MinMaxScaler()
    dataFrame_training[target_column] = scaler.fit_transform(dataFrame_training[[target_column]])
    dataFrame_testing[target_column] = scaler.transform(dataFrame_testing[[target_column]])
    
    X_train, y_train, X_test, y_test = preprocess_and_decompose(dataFrame_training, dataFrame_testing, target_column, look_back, K)
    print("Training data shapes:", X_train.shape, y_train.shape)
    print("Testing data shapes:", X_test.shape, y_test.shape)

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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.summary()

    history = model.fit(X_train,y_train, validation_split = 0.2, epochs= 20, batch_size=32)

    loss, mae = model.evaluate(X_test, y_test)
    print('Test loss: ', loss)
    print('Test MAE: ', mae)

    y_pred = model.predict(X_test)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Closing Prices using IMFs")
    plt.show()
    
if __name__ == "__main__":
    main()