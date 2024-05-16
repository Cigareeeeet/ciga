import numpy as np

class LinearRegression:
    x=None
    theta=None
    y=None

    def fit(self,x,y): #モデルに学習を適用する関数
        temp=np.linalg.inv(np.dot(x.T,x))
        self.theta=np.dot(np.dot(temp,x.T),y)

    def predict(self, X): #予測する関数
        return np.dot(X, self.theta)

    def score(self,x,y):
        error = self.predict(x)-y
        return(error**2).sum()


        