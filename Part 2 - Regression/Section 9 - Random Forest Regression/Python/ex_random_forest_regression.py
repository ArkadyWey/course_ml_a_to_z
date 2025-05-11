"""
* Pick K points from the training set.
* Build a decision tree for those K data points.
* Repeat this. 
* For predicting, make each tree make a prediction, then take the average result. 

* Usually use 500 trees at least.
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(X=X,y=y)

plt.scatter(X,y,color="red")
y_pred = model.predict(X=X)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid,model.predict(X_grid), color="blue")
plt.title("Decision tree")
plt.xlabel("Level")
plt.xlabel("Salary")
plt.show()