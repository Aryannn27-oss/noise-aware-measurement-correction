import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


data = pd.read_csv("measurement_data.csv")


true_values = data["true_value"]
measured_values = data["measured_value"]


mse = mean_squared_error(true_values, measured_values)
rmse = np.sqrt(mse)

print(f"Baseline RMSE (no correction applied): {rmse:.2f}")
