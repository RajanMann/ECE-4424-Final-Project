from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import pandas 
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
import time

'''
The function below is for one-hot encoding the input data, setting the x and y data, and finally splitting
the input data between training and testing. 
'''
def setTrainTest(file1):
    features = pandas.read_csv(file1)
    y = features['SalePrice']
    str_features = features.select_dtypes(include=['object']).columns.tolist()
    str_df = pandas.get_dummies(features[str_features])
    features.drop(str_features, axis=1, inplace=True)
    features = features.join(str_df)
    x = features.drop(['Id', 'SalePrice'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    features.to_csv('testies.csv')
    return x_train, x_test, y_train, y_test


'''
The function below is for setting missing values in the data, training a support vector regression model, making
predictions of housing prices using the model, and finally calculating how accurate the model is. 
'''
def datasetSVR(file1):
    start = time.time()
    x_train, x_test, y_train, y_test = setTrainTest(file1)
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    imputer = imputer.fit(x_train)
    x_train = imputer.transform(x_train)
    imputer = imputer.fit(x_test)
    x_test = imputer.transform(x_test)

    classifier = SVR(kernel='poly', degree=4, epsilon=0.1)
    
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    y_test = y_test.tolist()

    error = mean_squared_error(y_test, prediction, squared=False)
    end = time.time()
    return error, end-start


'''
The function below is for setting missing values in the data, training a linear regression model, making
predictions of housing prices using the model, and finally calculating how accurate the model is. 
'''
def datasetlinearRegression(file1):
    start = time.time()
    x_train, x_test, y_train, y_test = setTrainTest(file1)
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    imputer = imputer.fit(x_train)
    x_train = imputer.transform(x_train)
    imputer = imputer.fit(x_test)
    x_test = imputer.transform(x_test)

    classifier = LinearRegression(copy_X=True)

    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    y_test = y_test.tolist()

    error = mean_squared_error(y_test, prediction, squared=False)
    end = time.time()
    return error, end-start


'''
The function below is for setting missing values in the data, training a perceptron model, making
predictions of housing prices using the model, and finally calculating how accurate the model is. 
'''
def datasetPerceptron(file1):
    start = time.time()
    x_train, x_test, y_train, y_test = setTrainTest(file1)
    imputer = SimpleImputer(strategy='constant', fill_value=0)

    imputer = imputer.fit(x_train)
    x_train = imputer.transform(x_train)
    imputer = imputer.fit(x_test)
    x_test = imputer.transform(x_test)

    classifier = Perceptron(penalty='l2', alpha=0.000001)

    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    y_test = y_test.tolist()

    error = mean_squared_error(y_test, prediction, squared=False)
    end = time.time()
    return error, end - start

'''
The main program below calls each of the machine learning models ten times and stores the error and time it takes
for each model. After the ten epochs, the average error and time for each model is calculated and output. 
'''
if __name__ == "__main__":
    avgError = dict(svr=0, lin=0, per=0)
    avgTime = dict(svr=0, lin=0, per=0)
    epochs = 10
    for i in range(epochs):
        svrError, svrTime = datasetSVR("train.csv")
        avgError['svr'] += svrError
        avgTime['svr'] += svrTime
        linearError, linearTime = datasetlinearRegression("train.csv")
        avgError['lin'] += linearError
        avgTime['lin'] += linearTime
        perceptronError, perceptronTime = datasetPerceptron("train.csv")
        avgError['per'] += perceptronError
        avgTime['per'] += perceptronTime
    print("svr Error: ", avgError['svr'] / epochs, "svr Time (seconds): ", avgTime['svr'] / epochs)
    print("Linear Regression Error: ", avgError['lin'] / epochs, "Linear Regression Time (seconds): ", avgTime['lin'] / epochs)
    print("Perceptron Error: ", avgError['per'] / epochs, "Perceptron Time (seconds): ", avgTime['per'] / epochs)