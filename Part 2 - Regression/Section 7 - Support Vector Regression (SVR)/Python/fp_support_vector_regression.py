import pandas as pd
import numpy as np

class LinearSVR:
    def __init__(self, epsilon=0.1, C=1.0, learning_rate=0.01, n_iterations=1000):
        self.epsilon = epsilon # Size of allowance tube
        self.C = C # Importance of support vectors
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None # Weight vector
        self.b = None # Constant

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features) # Weight vector
        self.b = 0 # Constant

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                # Calculate a prediction using linear regression using current w and b
                y_pred = np.dot(X[i], self.w) + self.b

                # Calculate the error for current w and b
                error = y_pred - y[i]

                # Update weights and bias based on the epsilon-insensitive loss
                if abs(error) > self.epsilon: # Check if the loss is bigger than epsilon
                    if error > 0: # If y_pred > y_i -> pred is too large so make it smaller
                        self.w = self.w - self.learning_rate * (self.w + self.C * X[i])
                        self.b = self.b - self.learning_rate * self.C
                    else:
                        self.w = self.w - self.learning_rate * (self.w - self.C * X[i])
                        self.b = self.b + self.learning_rate * self.C
                else:
                    self.w = self.w - self.learning_rate * self.w

        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b

# Load the dataset
data = """Position,Level,Salary
Business Analyst,1,45000
Junior Consultant,2,50000
Senior Consultant,3,60000
Manager,4,80000
Country Manager,5,110000
Region Manager,6,150000
Partner,7,200000
Senior Partner,8,300000
C-level,9,500000
CEO,10,1000000"""

from io import StringIO
df = pd.read_csv(StringIO(data))

# Prepare the data
X = df[['Level']].values
y = df['Salary'].values

# Initialize and train the Linear SVR model
model = LinearSVR(epsilon=10000, C=0.1, learning_rate=0.001, n_iterations=2000)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print the results
print("Predictions:")
for i in range(len(y)):
    print(f"Level: {X[i][0]}, Actual Salary: {y[i]}, Predicted Salary: {y_pred[i]:.2f}")

# You can also try to predict for a new level
new_level = np.array([[11]])
predicted_salary_new = model.predict(new_level)
print(f"\nPredicted salary for Level {new_level[0][0]}: {predicted_salary_new[0]:.2f}")