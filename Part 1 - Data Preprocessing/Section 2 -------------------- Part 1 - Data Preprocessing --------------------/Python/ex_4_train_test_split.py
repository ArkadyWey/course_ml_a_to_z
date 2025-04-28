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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, train_size=0.8, random_state=42)

# print(X_train.shape)
# print(X_test.shape)
# print(X.shape)

# print(X_train)
# print(X_test)
# print(y_train)
print(y_test)

# Apply feature scaling on the training and test sets
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

# print(X_test)

ss = StandardScaler()
y_train = ss.fit_transform(y_train.reshape(-1,1))
y_test = ss.transform(y_test.reshape(-1,1))

print(y_test)