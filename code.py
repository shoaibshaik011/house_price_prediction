import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load the dataset
data = pd.read_csv('data/house_prices.csv')

# Feature selection: let's assume we are using 'size', 'bedrooms', and 'bathrooms' to predict 'price'
X = data[['size (sq ft)', 'bedrooms', 'bathrooms']]  # Features
y = data['price']  # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Save the trained model to a file using joblib
joblib.dump(model, 'model/house_price_model.pkl')

# Optionally, load the model back (to demonstrate model persistence)
# loaded_model = joblib.load('model/house_price_model.pkl')
# loaded_model.predict(X_test)  # Use the loaded model to predict
