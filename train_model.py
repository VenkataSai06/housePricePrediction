# train_model.py (Corrected)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
import numpy as np

# Create a more realistic dummy dataset
# We'll introduce some randomness to break the perfect linear relationship
np.random.seed(42) # for reproducibility

base_area = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500])
base_bedrooms = np.array([2, 3, 3, 4, 4, 5, 5, 6, 6, 7])
base_bathrooms = np.array([1, 2, 2, 2, 3, 3, 4, 4, 5, 5])

# Add noise to each feature to make them less correlated
area = base_area + np.random.randint(-100, 100, size=len(base_area))
bedrooms = base_bedrooms + np.random.randint(-1, 2, size=len(base_bedrooms))
bathrooms = base_bathrooms + np.random.randint(-1, 2, size=len(base_bathrooms))

# Create a price based on a formula with weights for each feature, plus noise
price = (area * 200 + bedrooms * 15000 + bathrooms * 25000) + np.random.randint(-50000, 50000, size=len(base_area))

data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'price': price
}
df = pd.DataFrame(data)

# Define features (X) and target (y)
features = ['area', 'bedrooms', 'bathrooms']
target = 'price'
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained successfully. Mean Squared Error: {mse:.2f}")
print(f"Model coefficients: Area={model.coef_[0]:.2f}, Bedrooms={model.coef_[1]:.2f}, Bathrooms={model.coef_[2]:.2f}")

# Define the directory to save the model
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Save the trained model to a file using joblib
model_filename = 'house_price_model.joblib'
model_path = os.path.join(model_dir, model_filename)

joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Note: In a real project, you would use a more robust dataset and
# perform more advanced data preprocessing, feature engineering, and
# hyperparameter tuning. This script is for demonstration purposes only.
