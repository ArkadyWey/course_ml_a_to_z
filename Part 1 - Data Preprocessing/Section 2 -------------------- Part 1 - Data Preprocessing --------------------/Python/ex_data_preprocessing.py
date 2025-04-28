import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset 
# -----
dataset = pd.read_csv(filepath_or_buffer="./Data.csv")
print(dataset.head())

# Always want a matrix of features and matrix of target 
X = dataset.iloc[:, 0:-1].values # iloc is locate indexes 
y = dataset.iloc[:, -1].values # values gets numpy array

print(X)
print(y)

# Taking care of missing data
# -----
# Could delete the row
# Could replace the missing value by the average of the column 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") # An imputer is an estimator of missing values 
imputer.fit(X[:, 1:3])  # Only takes numerical values so get rid of non-numerical cols
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Actually does the replacement 

print(X)

# Encoding categorical data - turning something like string data into number
# -----

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

# Splitting dataset into training and testing 
# -----
# Note we need to do this BEFORE we scale the features
# This is because the test set is cosidered completely independent of the train set - it's distribvution shouldn't be considered for scaling the train set
# So we do scaling after, to prevent what is called INFORMATION LEAKAGE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

# print(X_train)
print(X_test)

# Feature scaling
# -----
# ML models need all features to be on same scale so that some features are not ignored
# Some don't - e.g., in regression, the coefficients we learn will just scale away any scaling in the features do it's not needed
# There are two main scaling techniques:
# 1. Standardisation - works well all the time
# Subtract the mean and divide by the standard deviation - makes the mean be close to zero and the SD be close to 1 - so all values between -3 and +3 (because almost whole distr is within 3 SD of mean)
# 2. Normalisation - works well when distr is roughly normal
# Subtract the min and divide by the range - so that numbers fall between 0 and 1
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler() # Don't need parameters for standarisation

# Do not apply stndardardisation on features tahtt have been encoded because tey're just 1 and 0 and they'll lose their interpretationa and we don't need to becasue they're all between 0 and 1
X_train[:, 3:] = standard_scaler.fit_transform(X=X_train[:, 3:]) # Fit gets teh mean and SD of features and transform applies scaling to X

# Now apply the same scaling to the test set (do DON'T fit again, only transform)
X_test[:, 3:] = standard_scaler.transform(X=X_test[:, 3:])

print(X_test)