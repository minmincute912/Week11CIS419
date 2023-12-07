'''
    TEST SCRIPT FOR POLYNOMIAL REGRESSION 1
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
import matplotlib.pyplot as plt
from polyreg import PolynomialRegression
from polyreg import learningCurve
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    '''
        Main function to test polynomial regression
    '''

    # load the data
    filePath = "data/polydata.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = allData[:, 0]
    y = allData[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # regression with degree = d
    d = 8
    model = PolynomialRegression(degree = d, regLambda = 0)
    model.fit(X_train, y_train)
    
    # output predictions
    errorTrain,errorTest = learningCurve(X_train,y_train,X_test,y_test, regLambda = 0, degree = d)
    

    # plot curve
    plt.plot(range(2, len(X_train) + 1), errorTrain, label='Training Error', color='red')
    plt.plot(range(2, len(X_test) + 1), errorTest, label='Testing Error', color='blue')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
