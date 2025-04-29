# Multiple Linear Regression

# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') # passthrough keeps the cols that aren't encoded
X = np.array(column_transformer.fit_transform(X))
# print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,train_size=0.8)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor = linear_regressor.fit(X=X_train, y=y_train)

# Predicting the Test set results
y_pred = linear_regressor.predict(X=X_test)

# Evaluate 
score_train = linear_regressor.score(X=X_train,y=y_train)
score_test = linear_regressor.score(X=X_test,y=y_test)
print(score_train)
print(score_test)

