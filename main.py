# Import necessary libraries
import numpy as np  # For mathematical operations on arrays
import pandas as pd  # For data analysis and manipulation
from sklearn.linear_model import LinearRegression  # For linear regression modeling
import matplotlib.pyplot as plt  # For data visualization

# Read data from Excel file
data = pd.read_excel("Pazarlama.xlsx")

# Convert Pandas DataFrame to NumPy array for further processing
data = np.array(data)

# Extract features and target variables
x = data[:, 1].reshape(-1, 1)  # Feature (independent variable)
y = data[:, 2].reshape(-1, 1)  # Target variable (dependent variable)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(x, y)

# Calculate R-squared score (goodness of fit)
r_sq = model.score(x, y)

# Print model parameters
print(f"R2: {r_sq}")  # R-squared score
print(f"b0: {model.intercept_}")  # Intercept (b0)
print(f"b1: {model.coef_}")  # Slope (b1)

# Print equation of the fitted line
print(f"{model.intercept_} + {model.coef_[0]} x")

# Predict target values for the given features
y_pred = model.predict(x)

# Alternatively, you can calculate predicted values using the equation
# y_pred = model.intercept_ + model.coef_ * x
# or
# y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)

# Print predicted values
print(f"predicted values :\n {y_pred}")

# Visualize the results
plt.scatter(x, y, color="r", marker="o", label="actual")
plt.plot(x, y_pred, color="b", label="predicted")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
