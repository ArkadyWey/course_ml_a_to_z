import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Import data 
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Decision tree regression is better for datasets with more features

# Training decision tree on whole dataset
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)
model.fit(X=X,y=y)
import sklearn.tree as tree
tree.plot_tree(decision_tree=model)

# Predict
y_pred = model.predict([[6.5]])
print(y_pred)
plt.show()

plt.scatter(X,y,color="red")
y_pred = model.predict(X=X)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid,model.predict(X_grid), color="blue")
plt.title("Decision tree")
plt.xlabel("Level")
plt.xlabel("Salary")
plt.show()