import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
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
def main():
    file_location = r"./Data/A.csv"
    df = pd.read_csv(file_location)
    preprocessor = DataPreprocessor(df)
    preprocessor.removeEmptyValues()
    print(preprocessor.getDataFrame())
if __name__ == "__main__":
    main()