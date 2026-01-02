import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("measurement_data.csv")

# Feature engineering based on measurement physics
data["inverse_time"] = 1.0 / data["measurement_time"]
data["noise_variance"] = data["noise_std"] ** 2
data["relative_noise"] = data["noise_std"] / data["measured_value"]

# Features for ML model
X = data[
    [
        "measured_value",
        "measurement_time",
        "inverse_time",
        "noise_std",
        "noise_variance",
        "relative_noise",
    ]
]

# Ground truth
y = data["true_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest regressor
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
mse_ml = mean_squared_error(y_test, y_pred)
rmse_ml = np.sqrt(mse_ml)

print(f"ML-corrected RMSE (with engineered features): {rmse_ml:.2f}")
