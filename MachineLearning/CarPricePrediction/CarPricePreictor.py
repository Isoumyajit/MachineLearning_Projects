import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ...................

dataset = pd.read_csv('quikr_car.csv')
# Data preprocessing phase

def Preprocessing(dataset):

    dataset = dataset[dataset['year'].str.isnumeric()]
    dataset['year'] = dataset['year'].astype(int)
    dataset = dataset[dataset['Price'] != "Ask For Price"]
    dataset['Price'] = dataset['Price'].str.replace(',','').astype(int)
    dataset['kms_driven'] = dataset['kms_driven'].str.split(' ').str.get(0)
    dataset['kms_driven'] = dataset['kms_driven'].str.replace(',','')
    dataset = dataset[dataset['kms_driven'].str.isnumeric()]
    dataset['kms_driven'] = dataset['kms_driven'].astype(int)
    dataset = dataset[~dataset['fuel_type'].isna()]
    dataset['name'] = dataset['name'].str.split(' ').str.slice(0 , 3).str.join(' ')
    dataset = dataset[~(dataset['Price'] > 6e6)]
    dataset.reset_index(drop=True)
    return dataset

def Training(data):
    xFeatureData = data.drop(columns='Price')
    yData = data['Price']
    xtrain , xtest , ytrain , ytest = train_test_split(xFeatureData , yData , test_size=0.2)
    Ohe = OneHotEncoder().fit((xFeatureData[['name','company','fuel_type']]))
    column_transform = make_column_transformer((OneHotEncoder(categories=Ohe.categories_), ['name','company' ,'fuel_type']),remainder='passthrough')
    train = linear_model.LinearRegression()
    pipe = make_pipeline(column_transform,train)
    score = []
    for i in range(0,1000):
        xtrain, xtest, ytrain, ytest = train_test_split(xFeatureData, yData, test_size=0.2 , random_state=i)
        pipe.fit(xtrain, ytrain)
        yPredict = pipe.predict(xtest)
        score.append(r2_score(ytest,yPredict))
    xtrain, xtest, ytrain, ytest = train_test_split(xFeatureData, yData, test_size=0.2, random_state=np.argmax(score))
    pipe = make_pipeline(column_transform, train)
    pipe.fit(xtrain, ytrain)
    yPredict = pipe.predict(xtest)
    Accuracy = r2_score(ytest,yPredict)
    return Accuracy , yPredict , ytest

backUp = dataset.copy()
preprocessedData = Preprocessing(dataset)
preprocessedData.to_csv('cleanedData.csv')
acc , Price , ydata = Training(preprocessedData)
plt.scatter(ydata , Price)
plt.show()
print(acc)
print(Price)