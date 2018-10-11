
import pandas as pd
def getData():
    trainingData = pd.read_csv("train.csv").values
    trainingDataX = trainingData[:, :25]
    trainingDataY = trainingData[:, 25:]

    testingData = pd.read_csv("test.csv").values
    testingDataX = testingData[:, :25]
    testingDataY = testingData[:, 25:]
    return trainingDataX, trainingDataY, testingDataX, testingDataY