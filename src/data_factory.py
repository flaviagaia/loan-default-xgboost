from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def build_dataset(seed: int = 42) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=1800,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.74, 0.26],
        class_sep=1.05,
        random_state=seed,
    )
    columns = [
        "income_stability",
        "debt_ratio",
        "credit_utilization",
        "recent_delinquency",
        "installment_burden",
        "credit_history_length",
        "application_velocity",
        "hard_inquiries",
        "asset_buffer",
        "behavior_score",
        "employment_consistency",
        "cash_flow_volatility",
    ]
    df = pd.DataFrame(features, columns=columns)
    df["monthly_income"] = 5800 + df["income_stability"] * 900 - df["cash_flow_volatility"] * 300
    df["loan_to_income"] = (df["installment_burden"] + 2.2) / (np.abs(df["monthly_income"]) / 2800)
    df["utilization_x_delinquency"] = df["credit_utilization"] * df["recent_delinquency"]
    df["target_default"] = target
    return df

