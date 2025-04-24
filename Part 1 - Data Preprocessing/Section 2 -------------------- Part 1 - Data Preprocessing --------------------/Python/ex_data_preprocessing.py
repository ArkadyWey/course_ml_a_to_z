import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset 
dataset = pd.read_csv(filepath_or_buffer="./Data.csv")
print(dataset.head())

# Always want a matrix of features and matrix of target 
X = dataset.iloc[:, 0:-1].values # iloc is locate indexes 
y = dataset.iloc[:, -1].values # values gets numpy array

print(X)
print(y)

# Taking care of missing data
# Could delete the row
# Could replace the missing value by the average of the column 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") # An imputer is an estimator of missing values 
imputer.fit(X[:, 1:3])  # Only takes numerical values so get rid of non-numerical cols
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Actually does the replacement 

print(X)

# Encoding categorical data - turning something like string data into number
# We can't just use numbers, because we don't want the model to think these entries have an order that matters
# Instead, we use one hot encoding - creating identity vectors 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # passthrough keeps the cols that aren't encoded
X = np.array(column_transformer.fit_transform(X))

print(X)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(y)