import numpy as np
import matplotlib.pyplot as plot
from sklearn.metrics import r2_score
import pandas as pd


def Training():
    dataset = pd.read_csv('LinearRegression.csv')
    x_Data = dataset.drop(columns='y').values
    y_Data = dataset['y']
    yMean = np.mean(y_Data)
    xMean = np.mean(x_Data)

    # now calculating the sum((y-yMean)*(x-xMean))
    neu = 0
    denominator = 0
    for data in range(len(x_Data)):
        neu += (x_Data[data] - xMean) * (y_Data[data] - yMean)
        denominator += (x_Data[data] - xMean) ** 2
    slope = neu / denominator
    # intercept calculation
    intercept = yMean - slope * xMean
    # plotting the regression line in a graph
    regressionLine = slope * x_Data + intercept
    return yMean, slope, regressionLine


def Model_Working(x, y, Line):
    plot.scatter(x ,y)
    plot.scatter(x , Line)
    plot.plot(x,y)
    plot.plot(x , Line)
    plot.xlabel('x    ------- > ')
    plot.ylabel('y --------> ')
    plot.title('X vs Y')
    plot.show()


def Performace_Analysis(yMean, Y, y_predicted):
    Accuracy = 0
    neu = 0
    deno = 0
    for point in range(len(y_predicted)):
        neu += (yMean - y_predicted[point]) ** 2
        deno += (yMean - Y[point]) ** 2
    Accuracy = neu / deno
    return Accuracy

yMean , slope , line = Training()
data = pd.read_csv('LinearRegression.csv')
xTest =  data.drop(columns= 'y').values
yTest = data['y']
Model_Working(xTest , yTest , line)
print(r2_score(yTest , line))