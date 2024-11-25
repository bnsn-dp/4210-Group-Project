import tensorflow as tf
import pandas as pd
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def preprocess_and_decompose_NOVMD(dataFrame, target_column, look_back, forecast_horizon=30):
    
    print("dataFrame Shape:", dataFrame.shape)
    
    X_train, y_train, X_test, y_test = [], [], [], []
    values = dataFrame[target_column].values

    for i in range(len(dataFrame[target_column].values) - look_back):
        X_train.append(values[i:i + look_back])
        y_train.append(values[i + look_back])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # converting the data into a 3D shape of samples, timesteps(window), and features(1)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    for i in range(len(values) - forecast_horizon - look_back, len(values) - forecast_horizon):
        X_test.append(values[i:i + look_back])
        y_test.append(values[i + look_back])
    
    X_test = np.array(X_test).reshape((-1, look_back, 1))
    y_test = np.array(y_test)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    
    return X_train, y_train, X_test, y_test

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def main():
    # Driver function of this model. Calls main functions and sets all the values for the 
    # VMD - BPNN model.
    print('Model testing and Data Preprocessing starting...')
    '''
    
    start_date_training  = '2016-08-22 00:00:00-04:00'
    end_date_training = '2023-06-16 00:00:00-04:00'
    start_date_testing = '2023-06-20 00:00:00-04:00'
    end_date_testing = '2023-09-21 00:00:00-04:00'

    '''
    
    look_back = 5
    K = 5
    target_column = 'Close'

    file = r'./Data/AAPL.csv'
    columns_to_drop = ['Open','High','Low','Volume','Dividends','Stock Splits'] 
    dataFrame = pd.read_csv(file)
    dataFrame = dataFrame.drop(columns_to_drop, axis=1)


    #dataFrame_training = dataFrame[(dataFrame['Date'] >= start_date_training) & (dataFrame['Date'] <= end_date_training)].copy()
    #dataFrame_testing = dataFrame[(dataFrame['Date']>= start_date_testing)&(dataFrame['Date'] <= end_date_testing)].copy()
    
    # scaler = MinMaxScaler()
    # dataFrame[target_column] = scaler.fit_transform(dataFrame[[target_column]])
    #dataFrame_testing[target_column] = scaler.transform(dataFrame_testing[[target_column]])
    
    print(dataFrame.info(),dataFrame.head())
          #,dataFrame_testing.info())

    X_train, y_train, X_test, y_test = preprocess_and_decompose_NOVMD(dataFrame, target_column, look_back, 30)

    # TensorFlow CNN Model Building
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2),
        # tf.keras.layers.LSTM(64, return_sequences=False),
        # tf.keras.layers.Dropout(0.3),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mse', metrics=['mae', root_mean_squared_error])
    model.summary()

    history = model.fit(X_train, y_train, validation_split = 0.2, epochs= 20, batch_size=50)

    print(history.history['loss'])
    print(history.history['val_loss'])

    plt.plot(history.history['loss'], label= 'Training Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss(%)")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    '''
    loss, mae, rmse = model.evaluate(X_test)
    print('Test loss: ', loss)
    print('Test MAE: ', mae)
    print('Test RMSE: ', rmse)
    '''

    y_pred = model.predict(X_test)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_pred)), y_test[:len(y_pred)], label="Actual")  # True test values
    plt.plot(range(len(y_pred)), y_pred, label="Predicted")
    plt.xlabel('Day in time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.title("Actual vs Predicted Closing Prices")
    plt.show()
    
if __name__ == "__main__":
    main()