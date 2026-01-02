import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("measurement_data.csv")

data = data[data["measurement_time"] > 0.18].reset_index(drop=True)

data["inverse_time"] = 1.0 / data["measurement_time"]
data["noise_variance"] = data["noise_std"] ** 2
data["snr"] = data["measured_value"] / data["noise_std"]
data["log_noise"] = np.log(data["noise_std"])
data["correction"] = data["true_value"] - data["measured_value"]

X = data[
    [
        "measured_value",
        "measurement_time",
        "inverse_time",
        "noise_std",
        "noise_variance",
        "snr",
        "log_noise",
    ]
]

y = data["correction"]

sample_weights = 1.0 / data["noise_variance"]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.04,
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=w_train)

predicted_correction = model.predict(X_test)


true_pred = X_test["measured_value"] + predicted_correction


rmse = np.sqrt(
    mean_squared_error(
        data.loc[X_test.index, "true_value"],
        true_pred
    )
)

print(f"ML-corrected RMSE (noise-aware, filtered): {rmse:.2f}")
