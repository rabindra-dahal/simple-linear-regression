# model.py
import logging
import os
import urllib.error
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("InsuranceAPI")


class InsuranceRegressionModel:

    def __init__(self, test_size=0.2, random_state=42):
        self.url = "https://githubusercontent.com"
        self.local_path = "auto-insurance.csv"
        self.test_size = test_size
        self.random_state = random_state
        self.model = LinearRegression()
        self.df = None

    def load_and_train(self):
        """Attempts to load data via URL, falling back to a local file if offline."""
        try:
            logger.info("Attempting to load dataset from remote source...")
            self.df = pd.read_csv(self.url, header=None)
        except (urllib.error.URLError, IOError) as e:
            logger.warning(
                f"Network connection failed ({e}). Falling back to local file context."
            )
            if os.path.exists(self.local_path):
                self.df = pd.read_csv(self.local_path, header=None)
            else:
                logger.error("Both remote URL and local backup data file missing.")
                raise FileNotFoundError(
                    "Dataset asset not found remotely or locally."
                )

        self.df.columns = ["Claims", "Payment"]

        X = self.df[["Claims"]]
        y = self.df["Payment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "slope": float(self.model.coef_[0]),
            "intercept": float(self.model.intercept_),
            "raw_data": self.df.to_dict(orient="list"),
        }
        return metrics

    def predict(self, claims_value):
        """Generates target predictions for a given claim input."""
        input_data = pd.DataFrame([[claims_value]], columns=["Claims"])
        # Extract the element at index 0 from the prediction array
        prediction_array = self.model.predict(input_data)
        return float(prediction_array[0])
