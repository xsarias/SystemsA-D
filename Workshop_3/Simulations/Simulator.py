import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


class TournamentSimulator:
    def __init__(self):
        self.model = None
        self.matchups = None
        self.perturbed = None

    def run(self):
        self.load_data()
        self.prepare_matchups()
        self.train_model()
        self.predict_baseline()
        self.predict_perturbed()
        self.analyze()

    def load_data(self):
        """Load seeds and tournament results from CSV files"""
        self.seeds = pd.read_csv("Data/MNCAATourneySeeds.csv")
        self.seeds["SeedNum"] = self.seeds["Seed"].str.extract(r"(\d+)").astype(int)
        self.results = pd.read_csv("Data/MNCAATourneyCompactResults.csv")
        print("Data loaded successfully.")
        print(self.seeds.head())
        print(self.results.head())

    def prepare_matchups(self):
        """Merge seed info into tournament results and create symmetric training samples"""
        seeds_A = self.seeds.rename(columns={"TeamID": "WTeamID", "SeedNum": "SeedA"})
        seeds_B = self.seeds.rename(columns={"TeamID": "LTeamID", "SeedNum": "SeedB"})

        merged = pd.merge(self.results, seeds_A[["Season", "WTeamID", "SeedA"]],
                          on=["Season", "WTeamID"], how="left")
        merged = pd.merge(merged, seeds_B[["Season", "LTeamID", "SeedB"]],
                          on=["Season", "LTeamID"], how="left")
        merged["SeedDiff"] = merged["SeedB"] - merged["SeedA"]

        # First version: winner is TeamA
        win = merged.copy()
        win["TeamA"] = win["WTeamID"]
        win["TeamB"] = win["LTeamID"]
        win["Winner"] = 1

        # Second version: loser is TeamA
        lose = merged.copy()
        lose["TeamA"] = lose["LTeamID"]
        lose["TeamB"] = lose["WTeamID"]
        lose["SeedDiff"] = -lose["SeedDiff"]
        lose["Winner"] = 0

        matchups = pd.concat([win, lose], ignore_index=True)
        self.X = matchups[["SeedDiff"]]
        self.y = matchups["Winner"]

        print("Matchup data prepared:")
        print(matchups[["Season", "TeamA", "TeamB", "SeedDiff", "Winner"]].head())

    def train_model(self):
        """Train logistic regression model and report performance"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        self.model = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"Accuracy: {acc:.4f}")
        print(f"AUC Score: {auc:.4f}")

    def predict_baseline(self):
        """Generate baseline predictions for all possible 2025 matchups"""
        teams = self.seeds[self.seeds["Season"] == 2025][["TeamID", "SeedNum"]]
        pairs = []

        for A, B in combinations(teams["TeamID"], 2):
            sA = teams.loc[teams["TeamID"] == A, "SeedNum"].values[0]
            sB = teams.loc[teams["TeamID"] == B, "SeedNum"].values[0]
            pairs.append({"TeamA": A, "TeamB": B, "SeedDiff": sB - sA})
            pairs.append({"TeamA": B, "TeamB": A, "SeedDiff": sA - sB})

        df = pd.DataFrame(pairs)
        df["Pred"] = self.model.predict_proba(df[["SeedDiff"]])[:, 1]
        df.to_csv("Data/baseline_predictions.csv", index=False)
        self.matchups = df

        print("Baseline predictions generated:")
        print(df.head())

    def predict_perturbed(self):
        """Generate predictions under perturbed input conditions"""
        df = self.matchups.copy()

        # Add random noise (±2% to ±5%) to simulate data uncertainty
        np.random.seed(42)
        noise = np.random.uniform(-0.05, 0.05, size=len(df)) * df["SeedDiff"].abs()
        df["PerturbedSeedDiff"] = df["SeedDiff"] + noise

        # Rename column to match training-time feature name
        df["PerturbedPred"] = self.model.predict_proba(
            df[["PerturbedSeedDiff"]].rename(columns={"PerturbedSeedDiff": "SeedDiff"})
        )[:, 1]

        df.to_csv("Data/perturbed_predictions.csv", index=False)
        self.perturbed = df

        print("Perturbed predictions generated:")
        print(df[["TeamA", "TeamB", "SeedDiff", "PerturbedSeedDiff", "Pred", "PerturbedPred"]].head())

    def analyze(self):
        """Compare baseline vs perturbed predictions and generate histogram"""
        df = self.perturbed.copy()
        df["Delta"] = df["PerturbedPred"] - df["Pred"]
        avg_delta = df["Delta"].abs().mean()
        pct_shifted = (df["Delta"].abs() > 0.10).mean() * 100

        print(f"Average change in prediction: {avg_delta:.4f}")
        print(f" Matchups with >±0.10 change: {pct_shifted:.2f}%")

        # Scatter plot: baseline vs perturbed predictions
        plt.figure(figsize=(6, 6))
        plt.scatter(df["Pred"], df["PerturbedPred"], alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("Baseline Prediction")
        plt.ylabel("Perturbed Prediction")
        plt.title("Prediction Shift: Baseline vs Perturbed")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Data/prediction_shift.png")
        plt.close()
