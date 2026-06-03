from src.data_loader  import DataLoader
from src.eda          import EDA
from src.preprocessor import Preprocessor
from src.model        import ModelTrainer
from src.predictor    import Predictor

def main():
    print("=" * 50)
    print("  House Price Predictor")
    print("=" * 50)

    # 1. Load & clean
    loader = DataLoader("data/housing.csv")
    df = loader.load()
    loader.get_info()
    df = loader.clean()
    X, y = loader.get_features_target()

    # 2. Explore
    eda = EDA(df)
    eda.summary_stats()
    eda.plot_distributions()
    eda.plot_correlation()
    eda.plot_target_vs_feature("MedInc")

    # 3. Preprocess
    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.split(X, y)
    prep.build_pipeline()
    X_train_sc = prep.fit_transform(X_train)
    X_test_sc  = prep.transform(X_test)

    # 4. Train & evaluate
    trainer = ModelTrainer()
    trainer.train_all(X_train_sc, y_train)
    trainer.evaluate(X_test_sc, y_test)
    trainer.plot_predictions(X_test_sc, y_test)
    trainer.save("model.pkl")

    # 5. Predict a new house
    predictor = Predictor("model.pkl", pipeline=prep.pipeline)
    predictor.explain_features()

    house = {
        "MedInc": 4.5, "HouseAge": 15, "AveRooms": 6.0,
        "AveBedrms": 1.0, "Population": 1200, "AveOccup": 2.5,
        "Latitude": 34.05, "Longitude": -118.25
    }
    price = predictor.predict_single(house)
    print(f"\nPredicted price: ${price:,.2f}")

if __name__ == "__main__":
    main()