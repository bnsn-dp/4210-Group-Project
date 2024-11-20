import tensorflow as tf
import pandas as pd
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def DataPreprocess(file, drop_columns, start_date_training, end_date_training, start_date_testing, end_date_testing):
    df = pd.read_csv(file)
    df = df.drop(drop_columns, axis=1)
    df_training = df[(df['Date'] >= start_date_training) & (df['Date'] <= end_date_training)]
    df_testing = df[(df['Date'] >= start_date_testing) & (df['Date'] <= end_date_testing)]
    return df_training, df_testing

def main():
    # Driver function of this model. Calls main functions and sets all the values for the 
    # VMD - BPNN model.
    start_date_training  = '2020-08-21 00:00:00-04:00'
    end_date_training = '2023-08-21 00:00:00-04:00'
    start_date_testing = '2023-08-22 00:00:00-04:00'
    end_date_testing = '2023-09-21 00:00:00-04:00'
    
    scaler = MinMaxScaler()
    file = r'./Data/AAPL.csv'
    columns_to_drop = ['Open','High','Low','Volume','Dividends','Stock Splits'] 
    dataFrame_training,dataFrame_test = DataPreprocess(file, columns_to_drop,start_date_training, end_date_training,start_date_testing , end_date_testing)

    dataFrame_training['Close'] = scaler.fit_transform(dataFrame_training[['Close']])
    dataFrame_test['Close'] = scaler.fit_transform(dataFrame_test[['Close']])
    print(dataFrame_training.info(), dataFrame_test.info())
    
    # VMD Transformation on the training data to be sent to the CNN 
    transformer = VmdTransformer(K=10)
    imfs = transformer.fit_transform(dataFrame_training['Close'])
    n_timesteps = imfs.shape[0]
    K = imfs.shape[1]

    # plotting the decomposition of the dataset
    imfs.plot(subplots=True, figsize=(10, 8), title="Intrinsic Mode Functions (IMFs)")
    plt.suptitle("VMD Decomposition - IMFs (08-21-2020 to 08-21-2023)", y=0.95)
    plt.show()
    
    # TensorFlow BackPropagation Model Building
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,K)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activatiun='relu'),
        tf.keras.layers.Dense(1)

    ])

    
if __name__ == "__main__":
    main()