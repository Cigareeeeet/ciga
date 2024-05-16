import regression
import importlib
import datasets


X,Y=datasets.load_linear_example1()
#model=regression.LinearRegression()
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X,Y)
print(model.predict(X))
print(model.score(X,Y))
