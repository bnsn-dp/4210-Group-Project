import pandas as pd
#pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)
import numpy as np
from sktime.transformations.series.vmd import VmdTransformer
from sktime.datasets import load_solar


class DataPreprocessor():
    def __init__ (self, dataFrame):
        self.dataFrame = dataFrame
    def getDataFrame (self):
        return self.dataFrame
    def setDataFrame (self, dataFrame):
        self.dataFrame = dataFrame
    def removeEmptyValues(self):
        self.dataFrame.dropna(axis=0,inplace=True)
    
    # Function calculates the unweighted mean of the previous N closing prices
    def SimpleMovingAverage(self):
        self.dataFrame['SMA_20'] = self.dataFrame['Close'].rolling(window=20).mean()
    
    # Function calculates the weighted recent prices and responds to quickly changing prices
    def ExponentialMovingAverage(self):
        self.dataFrame['EMA_20'] = self.dataFrame['Close'].ewm(span=20, adjust=False).mean()
    
    # Function calculates the magnitude of recent prices changes to evalute overbought and oversold conditions
    # Rs is the average of N up day's gains divided by the average of N down days' losses
    def RelativeStrengthIndex(self):
        window_length = 14
        delta = self.dataFrame['Close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window = window_length).mean()
        avg_loss = loss.rolling(window = window_length).mean()

        rs = avg_gain / avg_loss
        self.dataFrame['RSI'] = 100 - (100 / (1 + rs))
    
    # Function calculates the difference of 2 Exponential Moving Averages(EMAs)
    # Typical EMAs are 12-day and 26-days
    def MovingAverageConvergenceDivergence (self):
        short_ema = self.dataFrame['Close'].ewm(span=12, adjust=False)
        long_ema = self.dataFrame['Close'].ewm(span=26, adjust=False)

        self.dataFrame['MACD'] = short_ema - long_ema
        self.dataFrame['Signal_line'] = self.dataFrame['MACD'].ewm(span=9, adjust=False).mean()

    # A middle band (SMA) and an upper and lower band that are typically two standard deviations away from the middle band.
    def BollingerBands(self):
        self.dataFrame['STD'] = self.dataFrame['Close'].rolling(window=20).std()

        self.dataFrame['Upper_Band'] = self.dataFrame['SMA_20'] + (2 * self.dataFrame['STD'])
        self.dataFrame['Lower_Band'] = self.dataFrame['SMA_20'] - (2 * self.dataFrame['STD'])
def testingVMD():
    y = load_solar()
    print(y)
    transformer = VmdTransformer()
    modes = transformer.fit_transform(y)
    print (modes)

def main():
    testingVMD()
    """
    file_location = r"./Data/A.csv"
    df = pd.read_csv(file_location)
    preprocessor = DataPreprocessor(df)
    preprocessor.removeEmptyValues()
    print(preprocessor.getDataFrame().info())
    preprocessor.SimpleMovingAverage()
    newDF = preprocessor.getDataFrame()
    print(newDF['SMA_20'])
    preprocessor.ExponentialMovingAverage()
    newDF2 = preprocessor.getDataFrame()
    print(newDF2['EMA_20'])
    print(newDF2.info())
    """
if __name__ == "__main__":
    main()