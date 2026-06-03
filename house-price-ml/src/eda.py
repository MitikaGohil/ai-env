import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EDA:

    def __init__(self, df):
        self.df = df

    def plot_distributions(self):
        self.df.hist(bins=30, figsize=(14, 10), edgecolor="black")
        plt.suptitle("Feature distributions")
        plt.tight_layout()
        plt.savefig("distributions.png")
        plt.show()

    def plot_correlation(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.savefig("correlation.png")
        plt.show()

    def plot_target_vs_feature(self, feature):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.df[feature], self.df["MedHouseVal"], alpha=0.3, s=10)
        plt.xlabel(feature)
        plt.ylabel("House Value")
        plt.title(f"{feature} vs House Price")
        plt.show()

    def summary_stats(self):
        prices = self.df["MedHouseVal"].values
        print(f"Mean:   {np.mean(prices):.2f}")
        print(f"Median: {np.median(prices):.2f}")
        print(f"Std:    {np.std(prices):.2f}")