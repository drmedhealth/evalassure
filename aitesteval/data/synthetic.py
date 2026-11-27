"""Synthetic, realistic-looking datasets with *controllable* problems so every scenario
runs offline and demonstrates a specific failure mode (bias, imbalance, drift, leakage...)."""
from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def credit(n: int = 6000, bias: float = 0.8, seed: int = 7, **_: Any) -> dict[str, Any]:
    """Loan approval. `bias` injects historical discrimination against one gender plus a proxy feature."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["female", "male"], size=n, p=[0.45, 0.55])
    age = rng.integers(19, 75, n)
    age_group = np.where(age < 25, "18-24", np.where(age < 60, "25-59", "60+"))
    female = (gender == "female").astype(float)
    income = rng.lognormal(10.8, 0.45, n) * (1 - 0.08 * female)
    debt_ratio = np.clip(rng.beta(2, 5, n), 0, 1)
    history = np.clip(age - 18 - rng.integers(0, 6, n), 0, None).astype(float)
    late = rng.poisson(0.8, n).astype(float)
    employment = np.clip(rng.normal(6, 4, n), 0, None)
    loan = rng.lognormal(9.6, 0.5, n)
    # proxy: "part_time" correlates strongly with gender in this synthetic population
    part_time = (rng.random(n) < (0.08 + 0.62 * female)).astype(float)
    z = (1.6 * (np.log(income) - 10.8) / 0.45 - 2.2 * (debt_ratio - 0.28) / 0.16 + 0.03 * history
         - 0.55 * late + 0.06 * employment - 0.5 * (np.log(loan) - 9.6) / 0.5 + 0.4)
    z = z - bias * 0.8 * female - bias * 1.6 * part_time  # historical bias baked into labels
    y = (rng.random(n) < _sigmoid(z)).astype(int)
    X = np.column_stack([income, debt_ratio, history, late, employment, loan, part_time])
    names = ["income", "debt_ratio", "credit_history_years", "late_payments", "employment_years",
             "loan_amount", "part_time"]
    return {"X": X, "y": y, "feature_names": names,
            "sensitive": {"gender": gender, "age_group": age_group},
            "description": "Synthetic loan-approval data (1 = approved) with injected gender bias and a proxy feature."}


def fraud(n: int = 20000, fraud_rate: float = 0.015, seed: int = 11, **_: Any) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    amount = rng.lognormal(3.8, 1.0, n) * np.where(y == 1, 1.5, 1.0)
    hour = np.where(y == 1, rng.normal(2.5, 3, n) % 24, rng.normal(14, 4.5, n) % 24)
    distance = rng.exponential(12, n) * np.where(y == 1, 2.5, 1.0)
    velocity = rng.poisson(np.where(y == 1, 3.0, 1.3)).astype(float)
    new_device = (rng.random(n) < np.where(y == 1, 0.40, 0.08)).astype(float)
    foreign = (rng.random(n) < np.where(y == 1, 0.25, 0.05)).astype(float)
    age_acct = rng.exponential(np.where(y == 1, 200, 900))
    X = np.column_stack([amount, hour, distance, velocity, new_device, foreign, age_acct])
    names = ["amount", "hour", "distance_km", "tx_last_hour", "new_device", "foreign", "account_age_days"]
    region = rng.choice(["north", "south", "east", "west"], n)
    return {"X": X, "y": y, "feature_names": names, "sensitive": {"region": region},
            "description": f"Card transactions with {fraud_rate:.1%} fraud (severe class imbalance)."}


def churn(n: int = 6000, drift: float = 1.0, seed: int = 21, **_: Any) -> dict[str, Any]:
    """Telecom churn. Returns an additional drifted `production` set (price rise + new-customer surge)."""
    def gen(m: int, rng: np.random.Generator, shift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tenure = np.clip(rng.exponential(30 * (1 - 0.5 * shift), m), 0, 72)
        monthly = rng.normal(65 + 22 * shift, 18, m).clip(15, 160)
        tickets = rng.poisson(1.0 + 1.2 * shift, m).astype(float)
        contract = rng.choice([0, 1, 2], m, p=[0.5 + 0.2 * shift, 0.3 - 0.1 * shift, 0.2 - 0.1 * shift]).astype(float)
        usage = rng.gamma(4, 5, m)
        z = (-0.05 * tenure + 0.035 * (monthly - 65) + 0.45 * tickets - 0.9 * contract - 0.02 * usage + 0.3
             + shift * (0.025 * (monthly - 65)))  # concept drift: price sensitivity increases
        y = (rng.random(m) < _sigmoid(z)).astype(int)
        seg = rng.choice(["consumer", "smb"], m, p=[0.8, 0.2])
        return np.column_stack([tenure, monthly, tickets, contract, usage]), y, seg

    rng = np.random.default_rng(seed)
    X, y, seg = gen(n, rng, 0.0)
    Xp, yp, segp = gen(n // 2, rng, drift)
    return {"X": X, "y": y, "feature_names": ["tenure_months", "monthly_charges", "support_tickets", "contract_type", "usage_gb"],
            "sensitive": {"segment": seg},
            "production": {"X": Xp, "y": yp, "sensitive": {"segment": segp}},
            "description": "Customer churn with a drifted production window (data + concept drift)."}


def medical_triage(n: int = 5000, seed: int = 31, **_: Any) -> dict[str, Any]:
    """3-class triage risk (0 low, 1 medium, 2 high). Under-represented group has noisier labels."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["F", "M"], n)
    ethnicity = rng.choice(["group_a", "group_b", "group_c"], n, p=[0.7, 0.2, 0.1])
    age = rng.integers(18, 95, n).astype(float)
    hr = rng.normal(82, 15, n)
    sbp = rng.normal(125, 20, n)
    spo2 = np.clip(rng.normal(96, 3, n), 70, 100)
    temp = rng.normal(37.0, 0.8, n)
    pain = rng.integers(0, 11, n).astype(float)
    score = 0.03 * (age - 50) + 0.04 * (hr - 82) - 0.03 * (sbp - 125) - 0.35 * (spo2 - 96) + 0.6 * (temp - 37) + 0.15 * pain
    score += rng.normal(0, np.where(ethnicity == "group_c", 1.6, 0.6))
    y = np.digitize(score, [0.6, 2.2]).astype(int)
    X = np.column_stack([age, hr, sbp, spo2, temp, pain])
    return {"X": X, "y": y, "feature_names": ["age", "heart_rate", "systolic_bp", "spo2", "temperature", "pain_score"],
            "sensitive": {"sex": sex, "ethnicity": ethnicity},
            "description": "Emergency triage (0=low,1=medium,2=high risk); minority group has noisier labels."}


def housing(n: int = 4000, seed: int = 41, **_: Any) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sqft = rng.normal(1800, 600, n).clip(400, 6000)
    beds = np.clip(np.round(sqft / 600 + rng.normal(0, 0.7, n)), 1, 7)
    age = rng.integers(0, 100, n).astype(float)
    dist = rng.exponential(10, n)
    school = rng.uniform(1, 10, n)
    price = 50000 + 150 * sqft + 9000 * beds - 600 * age - 4000 * dist + 12000 * school
    price = price * np.exp(rng.normal(0, 0.12, n))  # heteroscedastic noise
    neighbourhood = rng.choice(["urban", "suburban", "rural"], n, p=[0.4, 0.4, 0.2])
    X = np.column_stack([sqft, beds, age, dist, school])
    return {"X": X, "y": price, "feature_names": ["sqft", "bedrooms", "house_age", "distance_city_km", "school_rating"],
            "sensitive": {"neighbourhood": neighbourhood},
            "description": "House prices (regression) with multiplicative noise."}


GENERATORS = {"credit": credit, "fraud": fraud, "churn": churn, "medical_triage": medical_triage, "housing": housing}


def inject_issues(data: dict[str, Any], missing_rate: float = 0.0, duplicate_rate: float = 0.0,
                  leakage: bool = False, seed: int = 0) -> dict[str, Any]:
    """Corrupt a dataset on purpose so data-quality checks and fixes can be demonstrated."""
    rng = np.random.default_rng(seed)
    X, y = data["X"].astype(float).copy(), data["y"].copy()
    names = list(data["feature_names"])
    sens = {k: v.copy() for k, v in data.get("sensitive", {}).items()}
    if leakage:
        leak = y.astype(float) * 2.0 + rng.normal(0, 0.05, len(y))  # e.g. "days_until_decision" known only post-outcome
        X = np.column_stack([X, leak])
        names.append("post_decision_flag")
    if missing_rate > 0:
        mask = rng.random(X.shape) < missing_rate
        mask[:, -1] = False if leakage else mask[:, -1]
        X[mask] = np.nan
    if duplicate_rate > 0:
        k = int(len(X) * duplicate_rate)
        idx = rng.integers(0, len(X), k)
        X = np.vstack([X, X[idx]])
        y = np.concatenate([y, y[idx]])
        sens = {kk: np.concatenate([v, v[idx]]) for kk, v in sens.items()}
    out = dict(data)
    out.update(X=X, y=y, feature_names=names, sensitive=sens)
    return out
