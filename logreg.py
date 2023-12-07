'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''
import numpy as np


class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        m = len(y)
        h = self.sigmoid(X.dot(theta))
        J = (-1/m) * (y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h)))
        reg_term = (regLambda / (2*m))* np.sum(theta[1:]**2)
        J = J+reg_term
        return J.item()
    


    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''

        m = len(y)
        h = self.sigmoid(X.dot(theta))
        grad = (1/m) * X.T.dot(h-y)
        regTerm = (regLambda/m) * np.concatenate([[0], theta[1,:]], axis = 0)
        grad += regTerm
        return grad
    


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''

        n, d = X.shape
        X = np.concatenate([np.ones((n,1)),X],axis = 1)
        self.theta = np.zeros((d+1,1))
        for _ in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            self.theta -=self.alpha *gradient

            cost = self.computeCost(self.theta, X, y, self.regLambda)
            if cost < self.epsilon:
                break


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n = X.shape[0]
        X = np.concatenate([np.ones((n,1)),X],axis = 1)
        probabilities = self.sigmoid(X.dot(self.theta))
        predictions = (probabilities >= 0.5).astype(int)

        return predictions


    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
        
