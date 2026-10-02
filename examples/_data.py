"""Synthetic datasets shared by the examples (each has a known, hidden failure region)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def credit_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = rng.integers(18, 75, n)
    student = rng.random(n) < np.where(age < 26, 0.55, 0.05)
    income = np.round(rng.lognormal(10.75, 0.45, n) * np.where(student, 0.55, 1.0), -2)
    X = pd.DataFrame({
        "age": age,
        "income": income,
        "student": student,
        "region": rng.choice(["north", "south", "east", "west"], n, p=[0.3, 0.3, 0.2, 0.2]),
        "transactions": rng.poisson(10, n).astype(float),
        "account_age": np.round(rng.uniform(0, 25, n), 1),
    })
    X.loc[rng.random(n) < 0.04, "transactions"] = np.nan
    return X


def hidden_subgroup(X: pd.DataFrame) -> np.ndarray:
    """The planted failure region: young, low-income customers."""
    return ((X["age"] < 25) & (X["income"] < 30000)).to_numpy()


def credit_labels(X: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Default label. A different rule applies inside the hidden subgroup."""
    tx = X["transactions"].fillna(10).to_numpy()
    acc = X["account_age"].to_numpy()
    inc = X["income"].to_numpy()
    majority_rule = ((tx > 13) & (acc < 10)) | ((inc < 25000) & (acc < 3))
    subgroup_rule = (tx <= 9) | (acc > 15)
    y = np.where(hidden_subgroup(X), subgroup_rule, majority_rule)
    noise = rng.random(len(X)) < 0.02
    return np.where(noise, ~y, y).astype(int)


def credit_dataset(n_train: int = 20000, n_test: int = 10000, seed: int = 7
                   ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Train/test data where the hidden subgroup is almost absent from training data."""
    rng = np.random.default_rng(seed)
    X_train = credit_features(n_train, rng)
    keep = ~hidden_subgroup(X_train) | (rng.random(n_train) < 0.05)  # under-represented
    X_train = X_train[keep].reset_index(drop=True)
    y_train = credit_labels(X_train, rng)
    X_test = credit_features(n_test, rng)
    y_test = credit_labels(X_test, rng)
    return X_train, y_train, X_test, y_test


def housing_dataset(n_train: int = 20000, n_test: int = 10000, seed: int = 11
                    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Price regression. Prices of large homes in the east district are much noisier, and
    homes with an unknown construction year carry a price premium."""
    rng = np.random.default_rng(seed)

    def make(n: int) -> tuple[pd.DataFrame, np.ndarray]:
        X = pd.DataFrame({
            "sqft": np.round(rng.gamma(9, 180, n)),
            "bedrooms": rng.integers(1, 6, n),
            "year_built": rng.integers(1950, 2024, n).astype(float),
            "district": rng.choice(["north", "south", "east", "west", "central"], n),
            "distance_km": np.round(rng.exponential(8, n), 1),
            "has_garage": rng.random(n) < 0.6,
        })
        price = (60 + 0.11 * X["sqft"] + 8 * X["bedrooms"] - 2.5 * X["distance_km"]
                 + 0.6 * (X["year_built"] - 1950) + 15 * X["has_garage"]
                 + X["district"].map({"central": 60, "north": 20, "south": 0, "east": 10,
                                      "west": 5}))
        hard = ((X["district"] == "east") & (X["sqft"] > 2000)).to_numpy()
        noise = rng.normal(0, np.where(hard, 70, 12), n)
        unknown_year = rng.random(n) < 0.05
        price = price.to_numpy() + noise + np.where(unknown_year, 45, 0)
        X.loc[unknown_year, "year_built"] = np.nan
        return X, price

    X_train, y_train = make(n_train)
    X_test, y_test = make(n_test)
    return X_train, y_train, X_test, y_test
