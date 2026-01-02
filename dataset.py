import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

true_value = np.random.uniform(100, 500, N)

measurement_time = np.random.uniform(0.1, 1.0, N)

noise_std = 20 / measurement_time

measured_value = true_value + np.random.normal(0, noise_std)

df = pd.DataFrame({
    "measured_value": measured_value,
    "measurement_time": measurement_time,
    "noise_std": noise_std,
    "true_value": true_value
})

df.to_csv("measurement_data.csv", index=False)
print("Dataset saved as measurement_data.csv")
