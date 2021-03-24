
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ..........................................

salaryData = pd.read_csv('Salary_Data.csv')
xFeature = salaryData.drop(columns='Salary').values
yData = salaryData['Salary'].values
# print(xFeature)
# print(yData)
xtrain , xtest , ytrain , ytest = train_test_split(xFeature , yData, test_size=0.5 , random_state=40)
train = linear_model.LinearRegression()
train.fit(xtrain , ytrain)
y_predict = train.predict(xtest)

plt.scatter(ytest , y_predict)
plt.plot(xtest , y_predict)
plt.show()

print(r2_score(ytest , y_predict))