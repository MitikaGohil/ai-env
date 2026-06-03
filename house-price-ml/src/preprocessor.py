import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class Preprocessor:

    def __init__(self):
        self.pipeline = None

    def split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
        return X_train, X_test, y_train, y_test

    def build_pipeline(self):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
        ])
        return self.pipeline

    def fit_transform(self, X_train):
        return self.pipeline.fit_transform(X_train)

    def transform(self, X_test):
        return self.pipeline.transform(X_test)