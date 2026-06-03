import pandas as pd
import numpy as np

class DataLoader:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def get_info(self):
        print("\n--- Missing values ---")
        print(self.df.isnull().sum())
        print("\n--- Statistics ---")
        print(self.df.describe().round(2))

    def clean(self):
        before = len(self.df)
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        print(f"Removed {before - len(self.df)} rows")
        return self.df

    def get_features_target(self):
        X = self.df.drop("MedHouseVal", axis=1)
        y = self.df["MedHouseVal"]
        return X, y