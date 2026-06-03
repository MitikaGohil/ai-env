import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

class ModelTrainer:

    def __init__(self):
        self.models = {
            "Linear Regression":  LinearRegression(),
            "Ridge":              Ridge(alpha=1.0),
            "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        self.best_model = None
        self.results = {}

    def train_all(self, X_train, y_train):
        print("\nTraining models...")
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            self.results[name] = {"model": model, "cv_mean": cv.mean()}
            print(f"  {name}: R² = {cv.mean():.3f} ± {cv.std():.3f}")

    def evaluate(self, X_test, y_test):
        print("\n--- Test Results ---")
        best_r2 = -999
        for name, result in self.results.items():
            preds = result["model"].predict(X_test)
            r2   = r2_score(y_test, preds)
            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            print(f"{name}: R²={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = (name, result["model"])
        print(f"\nBest: {self.best_model[0]}")

    def plot_predictions(self, X_test, y_test):
        name, model = self.best_model
        preds = model.predict(X_test)
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, preds, alpha=0.3, s=10)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted — {name}")
        plt.tight_layout()
        plt.savefig("predictions.png")
        plt.show()

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.best_model[1], f)
        print(f"Model saved: {path}")