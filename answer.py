
# Importing libraries
import pandas as pd             # DataFrames to handle csv files
import numpy as np              # python array implemented in C
import matplotlib.pyplot as plt # to show the graphs
import math                     # contains 'sqrt' function for calculating root mean square value

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor # Algorithm used


# split any column
# col -> column to split
# delim -> delimeter to use to split (For example: '22-03-09' -> ['22', '03', '09'] with delimeter as '-')
# cols -> expected column names of parts of split column
# obj -> obj to map the data of the part to ( For example: '22' -> 22 when obj is int)
def split_col(data, col, delim, cols, obj):
    d = data[col]
    d = d.apply(str.split, args=(delim,))
    for i in range(len(cols)):
        data[cols[i]] = d.apply(lambda x: obj(x[i]))

# Plot line graph for two columns
def plot(data, col1, col2, title):
    col1_vs_col2 = data[[col1,col2]]
    col1_vs_col2 = col1_vs_col2.groupby(col1).mean()
    col1_vs_col2.plot()
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(title)
    plt.show()

# return the Room Mean Squared Score
def RMSE(ypred, ytest):
    rmse = math.sqrt(mean_squared_error(ypred,ytest))
#     print("Root Mean Squared Error: ", rmse)
    return rmse

# split the datetime column into day/month/year and hour/minute/second
def clean_data(data):
    split_col(data, 'datetime', ' ', ['date','time'], str)
    split_col(data, 'date', '-', ['year', 'month', 'day'], int)
    split_col(data, 'time', ':', ['hour','minutes','seconds'], int)
    
# make the submission file for Job-A-Thon
def save_data(row_id, ypred):
    data = pd.DataFrame({'row_id':row_id, 'energy':ypred})
    data.to_csv('data/submission.csv')

def submit():
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    ytest = pd.read_csv("data/sample_submission_jn0a7vR.csv")
    test_data['energy'] = ytest['energy']
    print("Data Imported")
    
    print("Cleaning Data")
    # train_data.ffill(inplace=True)
    train_data.dropna(inplace=True)
    clean_data(train_data)
    clean_data(test_data)
    print("Data has been cleaned")

    plot(train_data, 'year', 'energy', "Yearly growth of energy")
    plot(train_data, 'month', 'energy', "Monthly growth of energy")
    plot(train_data, 'day', 'energy', "Daily growth of energy")
    plot(train_data, 'hour', 'energy', "Hourly growth of energy")

    drop_columns = ['row_id', 'datetime', 'date', 'time', 'energy']
    target = 'energy'
    
    # Using Decision Tree Regressor
    print("Building model")
    dtr = DecisionTreeRegressor()
    xtrain = train_data.drop(drop_columns,axis=1)
    ytrain = train_data[target]    
    xtest = test_data.drop(drop_columns, axis=1)
    ytest = ytest['energy']
    
    dtr.fit(xtrain,ytrain)
    print("Data fit into model")
    
    ypred = dtr.predict(xtest)
    
    print("Root Mean Squared Error: ", RMSE(ypred, ytest))
    
    save_data(test_data['row_id'], ypred)

submit()
