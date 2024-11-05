import pandas as pd
#pd.set_option("display.max_columns", None)
#pd.set_option("display.max_rows", None)
import numpy as np

class DataPreprocessor():
    def __init__ (self, dataFrame):
        self.dataFrame = dataFrame
    def getDataFrame (self):
        return self.dataFrame
    def setDataFrame (self, dataFrame):
        self.dataFrame = dataFrame
    def removeEmptyValues(self):
        self.dataFrame.dropna(axis=0,inplace=True)
    def SimpleMovingAverage(self):
        self.dataFrame['SMA_20'] = self.dataFrame['Close'].rolling(window=20).mean()
        self.dataFrame['SMA_20'].fillna(0, inplace=True)
    def ExponentialMovingAverage(self):
        self.dataFrame['EMA_20'] = self.dataFrame['Close'].ewm(span=20, adjust= False).mean()
    def RelativeStrengthIndex(self):
        x=1    
        
    
    

def main():
    file_location = r"./Data/A.csv"
    df = pd.read_csv(file_location)
    preprocessor = DataPreprocessor(df)
    preprocessor.removeEmptyValues()
    print(preprocessor.getDataFrame().head())
    preprocessor.SimpleMovingAverage()
    newDF = preprocessor.getDataFrame()
    print(newDF['SMA_20'])
    preprocessor.ExponentialMovingAverage()
    newDF2 = preprocessor.getDataFrame()
    print(newDF2['EMA_20'])

if __name__ == "__main__":
    main()