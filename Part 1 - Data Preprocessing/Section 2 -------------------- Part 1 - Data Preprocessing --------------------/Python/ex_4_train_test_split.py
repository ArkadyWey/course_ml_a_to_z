# Import necessary libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
df = pd.read_csv("iris.csv")

# Separate features and target
X = df.iloc[:,0:-1].values 
y = df.iloc[:, -1].values

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, train_size=0.8, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(X.shape)

