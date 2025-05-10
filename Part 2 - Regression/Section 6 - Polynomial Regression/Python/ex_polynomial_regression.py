import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
dataset = pd.read_csv("./Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Train linear model 
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X=X,y=y)
y_pred = linear_regressor.predict(X=X)

# Train polynomial model 
# Create matrix of powers
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree=2)
X_poly_2 = polynomial_regressor.fit_transform(X=X)
# Fit linear model on matrix of powers
linear_regressor_poly_2 = LinearRegression()
linear_regressor_poly_2.fit(X=X_poly_2,y=y)
y_pred_poly_2 = linear_regressor_poly_2.predict(X=X_poly_2)


# Train polynomial model 
# Create matrix of powers
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree=4)
X_poly_4 = polynomial_regressor.fit_transform(X=X)
# Fit linear model on matrix of powers
linear_regressor_poly_4 = LinearRegression()
linear_regressor_poly_4.fit(X=X_poly_4,y=y)
y_pred_poly_4 = linear_regressor_poly_4.predict(X=X_poly_4)

# Visualise linear regression 
plt.scatter(x=X, y=y, color="red")
plt.plot(X, y_pred, color="blue")
plt.title("Linear regression")
plt.xlabel("Job position level")
plt.ylabel("Salary")
plt.show()

# Visualise polynomial regression 
plt.scatter(x=X, y=y, color="red")
plt.plot(X, y_pred_poly_2, color="blue")
plt.title("Polynomial regression")
plt.xlabel("Job position level")
plt.ylabel("Salary")
plt.show()

# Visualise polynomial regression 
plt.scatter(x=X, y=y, color="red")
plt.plot(X, y_pred_poly_4, color="blue")
plt.title("Polynomial regression")
plt.xlabel("Job position level")
plt.ylabel("Salary")
plt.show()

# Predict a new result
new_linear = linear_regressor.predict([[6.5]])
print(new_linear)

new_poly = linear_regressor_poly_4.predict(X=polynomial_regressor.fit_transform(X=[[6.5]]))
# Have to fill all the powers in as well, but use the same method as we formed the matrix before
print(new_poly)