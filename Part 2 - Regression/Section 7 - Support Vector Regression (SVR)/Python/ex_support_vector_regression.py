import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Import data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(f"X before scaling:{X}")
print(f"y before scaling:{y}")

# Feature scaling
# We need to scale the features because there is no explicit equation between 
# the features and the dependent variable (unlike in linear regression)

# We will appply scaling on the whole dataaset because there is no train test split 
# Because we want to use the whole dataset to predit a new value

# Need things to be 2D arrays before scaling
y = y.reshape((len(y),1))
print(y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Can't use the same object because fit calculates the mean and variance 
# But the features and target don't have the same mean and variance
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
print(f"X after scaling:{X}")
print(f"y after scaling:{y}")

from sklearn.svm import SVR
model = SVR(kernel="rbf") # We need to tell it which kernal
model.fit(X=X, y=y)

# This will return the results in the scaling
# So we will have to unscale to understand the results
X_new = sc_X.transform([[6.5]])
y_pred_scaled = model.predict(X=X_new)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1,1))

print(f"y pred: {y_pred}")

# Visualise
# -----
X_unscaled = sc_X.inverse_transform(X)
y_unscaled = sc_y.inverse_transform(y)

plt.scatter(X_unscaled,y_unscaled,color="red")
y_pred = model.predict(X=X)
plt.plot(X_unscaled,sc_y.inverse_transform(y_pred.reshape(-1,1)), color="blue")
plt.title("SVR")
plt.xlabel("Level")
plt.xlabel("Salary")
plt.show()

# High resolution plot
X_grid = np.arange(min(X_unscaled), max(X_unscaled), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_unscaled,y_unscaled,color="red")
y_pred = model.predict(X=sc_X.transform(X_grid))
plt.plot(X_grid,sc_y.inverse_transform(y_pred.reshape(-1,1)), color="blue")
plt.title("SVR")
plt.xlabel("Level")
plt.xlabel("Salary")
plt.show()