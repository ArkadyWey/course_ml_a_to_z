import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

# Get data 
# ----
df = pd.read_csv("Salary_Data.csv")

# Get features and target 
# -----
X = df.drop("Salary", axis=1).values
y = df["Salary"].values

# Split data
# ----
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=42)

# Train the Simple Linear Regression model 
# ----
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor = linear_regressor.fit(X=X_train, y=y_train)

# Use the model (predict the test set results)
# ----
y_pred = linear_regressor.predict(X=X_test)

# print(y_test)
# print(y_pred)

# Visualise train set
# -----
import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, linear_regressor.predict(X=X_train), color="blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (£)")
# plt.show()

# Visualise test set
# -----
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, linear_regressor.predict(X=X_train), color="blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (£)")
# plt.show()

# Evaluate the model
# -----
score_train = linear_regressor.score(X=X_train,y=y_train)
score_test = linear_regressor.score(X=X_test,y=y_test)
print(score_train)
print(score_test)

# Predict a new value
# -----
value_pred = linear_regressor.predict(X=[[12]]) # Expects a matrix for X
print(value_pred)

# Get coefficients 
# ------
coef = linear_regressor.coef_
intercept = linear_regressor.intercept_
print(coef)
print(intercept)

# So equation is: 
# Salary = intercept + coef * Experience