import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Import data
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0, shuffle=True)

# Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X=X_train, y=y_train)

# Predict value
value = [[30,87000]]
y_value_pred = model.predict(sc.transform(value))
# print(y_value_pred)

# Predict 
y_pred = model.predict(X=X_test)
comparison = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), axis=1)
# print(comparison)

# Making the confusion matrix - shows number of correct and incorrect predictions when result is 0 and 1 
# C_ij is the number of trueths in the ith class that were predicted in the jth class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cm)

# Accuracy score 
# This is the number of correct predictions over total number of predictions
from sklearn.metrics import accuracy_score
a = accuracy_score(y_true=y_test, y_pred=y_pred)
print(a)

# Plot train set
from matplotlib.colors import ListedColormap
X_set = sc.inverse_transform(X_train) 
y_set = y_train

# Create the background by making a grid and then predicting each value
X1_range = np.arange(start=X_set[:,0].min()-10, stop=X_set[:,0].max()+10, step=5)
X2_range = np.arange(start=X_set[:,1].min()-1000, stop=X_set[:,1].max()+1000, step=5)
X1, X2 = np.meshgrid(X1_range, X2_range)

X_train = np.array([X1.ravel(), X2.ravel()]).T
y_pred = sc.transform(X_train)
y_pred = y_pred.reshape(X2.shape)

# plt.contourf(X1, X2, y_pred, alpha=0.5, cmap=ListedColormap(colors=("red", "green")))


# Plot test set
# plt.scatter(X_test[:,0], y_test, color="red")
# plt.scatter(X_test[:,0], y_pred, color="blue", marker=".")

# X_grid = np.arange(min(X_test[:,0]), max(X_test[:,0]), 0.1)
# y_grid_pred = model.predict(X=X_grid)
# plt.plot(X_grid, y_grid_pred.reshape(-1,1))
# plt.show()
