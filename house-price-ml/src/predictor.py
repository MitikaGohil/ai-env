import numpy as np
import pandas as pd
import pickle

class Predictor:

    def __init__(self, model_path, pipeline):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.pipeline = pipeline

    def predict_single(self, house: dict):
        df  = pd.DataFrame([house])
        arr = self.pipeline.transform(df)
        price = self.model.predict(arr)[0]
        return round(price * 100_000, 2)

    def explain_features(self):
        if hasattr(self.model, "feature_importances_"):
            names = ["MedInc","HouseAge","AveRooms","AveBedrms",
                     "Population","AveOccup","Latitude","Longitude"]
            pairs = sorted(zip(names, self.model.feature_importances_),
                           key=lambda x: x[1], reverse=True)
            print("\n--- Feature Importance ---")
            for feat, imp in pairs:
                print(f"  {feat:<12} {'█' * int(imp*50)} {imp:.3f}")