"""Churn models and the economics of a retention offer.

The offer: a one-off credit to customers we think will leave. It only pays off if
    chance of leaving x chance the offer changes their mind x what keeping them is worth  >  cost of the offer
so who to target depends on each customer's bill, not just their risk.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
# TotalCharges is left out: it's mostly tenure x monthly bill (correlation 0.83 with tenure), so it adds almost
# nothing (AUC 0.845 with it, 0.843 without) and muddles the explanations.
NUMERIC = ["tenure", "MonthlyCharges", "SeniorCitizen"]


def load(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)   # 11 new customers
    return df[CATEGORICAL + NUMERIC], (df["Churn"] == "Yes").astype(int), df


def models(seed=0):
    """Three candidate models, each a full pipeline from raw columns to a churn probability."""
    one_hot = OneHotEncoder(handle_unknown="ignore", drop="if_binary")
    linear = ColumnTransformer([("cat", one_hot, CATEGORICAL), ("num", StandardScaler(), NUMERIC)])
    trees = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")
    return {
        "Logistic regression": Pipeline([("prep", linear), ("model", LogisticRegression(max_iter=2000))]),
        "Random forest": Pipeline([("prep", trees), ("model", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=10, random_state=seed, n_jobs=-1))]),
        "Gradient boosting": Pipeline([("prep", trees), ("model", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
            random_state=seed))]),
    }


def out_of_fold(model, X, y, folds=5, seed=0):
    """Churn probability for every customer, each predicted by a model that never saw that customer."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


@dataclass
class Offer:
    cost: float = 60.0          # one-off credit, dollars (about one average monthly bill)
    success_rate: float = 0.30  # share of would-be leavers the offer keeps
    months_kept: int = 12       # how long a saved customer stays on average
    margin: float = 0.60        # profit margin on each month's bill

    def value_if_saved(self, monthly_charges):
        """Profit from keeping one customer who would otherwise have left."""
        return self.months_kept * self.margin * np.asarray(monthly_charges, dtype=float)


def expected_profit(p_churn, monthly_charges, offer):
    """Expected profit of sending the offer to each customer, using only predicted risk (no hindsight)."""
    return np.asarray(p_churn) * offer.success_rate * offer.value_if_saved(monthly_charges) - offer.cost


def realised_profit(targeted, churned, monthly_charges, offer):
    """Profit a targeting decision would have earned, given who actually left.

    Customers who stay anyway still take the credit; for those who would have left, the offer works
    `success_rate` of the time (counted as its expected value).
    """
    targeted, churned = np.asarray(targeted, bool), np.asarray(churned, bool)
    gain = churned * offer.success_rate * offer.value_if_saved(monthly_charges)
    return float(((gain - offer.cost) * targeted).sum())


def policies(p_churn, churned, monthly_charges, offer):
    """Compare targeting rules on the same customers. Returns profit and share of customers targeted."""
    ev = expected_profit(p_churn, monthly_charges, offer)
    worth_saving = offer.success_rate * offer.value_if_saved(monthly_charges) > offer.cost
    rules = {
        "Nobody": np.zeros(len(p_churn), bool),
        "Everybody": np.ones(len(p_churn), bool),
        "Risk above 50% (default classifier)": np.asarray(p_churn) >= 0.5,
        "Expected profit above zero": ev > 0,
        "Perfect hindsight": np.asarray(churned, bool) & worth_saving,   # leavers whose bill can repay the offer
    }
    return pd.DataFrame({name: {"profit": realised_profit(mask, churned, monthly_charges, offer),
                                "share_targeted": mask.mean(), "customers_targeted": int(mask.sum())}
                         for name, mask in rules.items()}).T
