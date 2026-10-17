"""Leakage-safe feature engineering and outcome models for event journeys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REQUIRED_COLUMNS = {"journey_id", "event_name", "event_timestamp"}

DEFAULT_STAGE_MAP: dict[str, int] = {
    "campaign_click": 0,
    "application_view": 0,
    "browse_product": 1,
    "view_cart": 2,
    "add_to_cart": 2,
    "begin_checkout": 3,
    "place_order": 4,
    "place_downpayment": 4,
    "order_shipped": 5,
}

DEFAULT_INDICATOR_EVENTS = (
    "browse_product",
    "view_cart",
    "add_to_cart",
    "begin_checkout",
    "place_order",
    "place_downpayment",
)


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate the public schema and return chronologically sorted events."""
    missing = REQUIRED_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = events.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    frame["journey_id"] = frame["journey_id"].astype(str)
    frame["event_name"] = frame["event_name"].astype(str)
    frame["event_timestamp"] = pd.to_datetime(
        frame["event_timestamp"], utc=True, errors="coerce"
    )
    frame = frame.dropna(subset=["event_timestamp"])
    frame = frame.drop_duplicates(
        subset=["journey_id", "event_name", "event_timestamp"]
    )
    return frame.sort_values(
        ["journey_id", "event_timestamp", "event_name"]
    ).reset_index(drop=True)


def summarize_journeys(
    events: pd.DataFrame,
    *,
    inactivity_days: int = 60,
    success_event: str = "order_shipped",
) -> pd.DataFrame:
    """Assign successful, unsuccessful, or ongoing status from full histories."""
    frame = normalize_events(events)
    if frame.empty:
        raise ValueError("events must contain at least one valid timestamp.")
    dataset_end = frame["event_timestamp"].max()
    grouped = frame.groupby("journey_id", sort=False)
    summary = grouped.agg(
        first_event_time=("event_timestamp", "min"),
        last_event_time=("event_timestamp", "max"),
        total_events=("event_name", "size"),
    )
    success_ids = set(
        frame.loc[frame["event_name"].eq(success_event), "journey_id"]
    )
    stale_before = dataset_end - pd.Timedelta(days=inactivity_days)
    summary["journey_status"] = np.where(
        summary.index.isin(success_ids),
        "successful",
        np.where(
            summary["last_event_time"].le(stale_before),
            "unsuccessful",
            "ongoing",
        ),
    )
    summary["outcome"] = summary["journey_status"].eq("successful").astype(int)
    summary["observed_age_hours"] = (
        dataset_end - summary["first_event_time"]
    ).dt.total_seconds() / 3600.0
    return summary.reset_index()


def observation_age_pool(summary: pd.DataFrame) -> np.ndarray:
    """Return positive observation ages from currently open journeys."""
    ages = summary.loc[
        summary["journey_status"].eq("ongoing"), "observed_age_hours"
    ].to_numpy(dtype=float)
    ages = ages[np.isfinite(ages) & (ages > 0)]
    if ages.size == 0:
        raise ValueError("At least one ongoing journey is needed for pseudo-cut ages.")
    return ages


def _sample_cut_age(
    duration_hours: float,
    age_pool: np.ndarray,
    rng: np.random.Generator,
) -> float:
    eligible = age_pool[age_pool < duration_hours]
    if eligible.size:
        return float(rng.choice(eligible))
    if duration_hours <= 0:
        return 0.0
    return float(duration_hours * rng.uniform(0.25, 0.75))


def build_snapshot_features(
    observed_events: pd.DataFrame,
    cut_times: Mapping[str, pd.Timestamp],
    *,
    stage_map: Mapping[str, int] = DEFAULT_STAGE_MAP,
    indicator_events: Sequence[str] = DEFAULT_INDICATOR_EVENTS,
) -> pd.DataFrame:
    """Build journey features using only events at or before each cut time."""
    frame = normalize_events(observed_events)
    rows: list[dict[str, object]] = []
    for journey_id, journey in frame.groupby("journey_id", sort=False):
        if journey_id not in cut_times:
            continue
        cut_time = pd.Timestamp(cut_times[journey_id])
        if cut_time.tzinfo is None:
            cut_time = cut_time.tz_localize("UTC")
        else:
            cut_time = cut_time.tz_convert("UTC")
        visible = journey.loc[journey["event_timestamp"].le(cut_time)]
        if visible.empty:
            continue
        names = visible["event_name"].tolist()
        first_time = visible["event_timestamp"].iloc[0]
        last_time = visible["event_timestamp"].iloc[-1]
        row: dict[str, object] = {
            "journey_id": journey_id,
            "cut_time": cut_time,
            "n_events": len(visible),
            "n_distinct_events": visible["event_name"].nunique(),
            "journey_age_hours": (cut_time - first_time).total_seconds() / 3600.0,
            "hours_since_last_event": (cut_time - last_time).total_seconds() / 3600.0,
            "last_event": names[-1],
            "furthest_stage": max((stage_map.get(name, 0) for name in names), default=0),
        }
        for event_name in indicator_events:
            row[f"seen_{event_name}"] = int(event_name in names)
        rows.append(row)
    return pd.DataFrame(rows)


def build_pseudo_cut_dataset(
    events: pd.DataFrame,
    *,
    inactivity_days: int = 60,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create resolved training examples that resemble currently open journeys."""
    frame = normalize_events(events)
    summary = summarize_journeys(frame, inactivity_days=inactivity_days)
    ages = observation_age_pool(summary)
    resolved = summary.loc[
        summary["journey_status"].isin(["successful", "unsuccessful"])
    ].copy()
    rng = np.random.default_rng(random_state)
    cut_times: dict[str, pd.Timestamp] = {}
    for row in resolved.itertuples(index=False):
        duration_hours = (
            row.last_event_time - row.first_event_time
        ).total_seconds() / 3600.0
        age = _sample_cut_age(duration_hours, ages, rng)
        cut_times[row.journey_id] = row.first_event_time + pd.Timedelta(hours=age)
    features = build_snapshot_features(frame, cut_times)
    labels = resolved.loc[:, ["journey_id", "outcome"]]
    return features.merge(labels, on="journey_id", how="inner", validate="one_to_one")


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["last_event"]
    excluded = {"journey_id", "cut_time", "outcome", *categorical}
    numeric = [column for column in frame.columns if column not in excluded]
    return numeric, categorical


def make_random_forest_pipeline(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Construct the portfolio version of the report's Random Forest model."""
    numeric = list(numeric_features)
    categorical = list(categorical_features)
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def make_xgboost_pipeline(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    *,
    random_state: int = 42,
) -> Pipeline:
    """Construct the optional XGBoost model described in the project report."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "Install requirements-xgboost.txt to use the XGBoost pipeline."
        ) from exc
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", list(numeric_features)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            ),
        ]
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


@dataclass(frozen=True)
class ProbabilityEvaluation:
    mse: float
    prediction_mean: float
    outcome_mean: float


def evaluate_probabilities(y_true: Sequence[int], probabilities: Sequence[float]) -> ProbabilityEvaluation:
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(probabilities, dtype=float)
    return ProbabilityEvaluation(
        mse=float(mean_squared_error(truth, predicted)),
        prediction_mean=float(predicted.mean()),
        outcome_mean=float(truth.mean()),
    )

