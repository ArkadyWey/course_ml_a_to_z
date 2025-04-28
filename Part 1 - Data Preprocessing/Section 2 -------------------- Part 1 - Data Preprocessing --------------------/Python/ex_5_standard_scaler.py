# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Wine Quality Red dataset
df = pd.read_csv("winequality-red.csv", delimiter=";")

# Separate features and target
X = df.iloc[:,:-1].values 
y = df.iloc[:, -1].values

print(X)
print(y)

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=1)

# Create an instance of the StandardScaler class
ss = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
X_train = ss.fit_transform(X=X_train)


# Apply the transform to the test set
X_test = ss.transform(X=X_test)

# Print the scaled training and test datasets
print(X_train)
print(X_test)