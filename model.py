import tensorflow as tf
import pandas as pd
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def DataFrameDropcolumns(file, drop_columns):
    df = pd.read_csv(file)
    df = df.drop(drop_columns, axis=1)
    return df


def main():
    # Driver function of this model. Calls main functions and sets all the values for the 
    # VMD - BPNN model.
    scaler = MinMaxScaler()
    file = r'./Data/AAPL.csv'
    columns_to_drop = ['Open','High','Low','Volume','Dividends','Stock Splits'] 
    dataFrame = DataFrameDropcolumns(file, columns_to_drop)
    dataFrame['Close'] = scaler.fit_transform(dataFrame[['Close']])

    # to test the VMD function, took 1st 50 samples out of the dataset to save on time processing 
    # and debugging
    
    transformer = VmdTransformer(K=10)
    imfs = transformer.fit_transform(dataFrame['Close'])
    
    # plotting the decomposition of the dataset
    imfs.plot(subplots=True, figsize=(10, 8), title="Intrinsic Mode Functions (IMFs)")
    plt.suptitle("VMD Decomposition - IMFs", y=0.95)
    plt.show()

    # TensorFlow BackPropagation Model Building
    hidden_layer_size = 10
    model = tf.keras.Sequential([
        tf.keras.layers.Dense
    ])


if __name__ == "__main__":
    main()