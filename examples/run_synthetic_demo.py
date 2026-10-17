"""Train the leakage-safe journey model on generated event histories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.journey_modeling import (  # noqa: E402
    build_pseudo_cut_dataset,
    evaluate_probabilities,
    feature_columns,
    make_random_forest_pipeline,
)


FUNNEL = [
    "campaign_click",
    "browse_product",
    "view_cart",
    "add_to_cart",
    "begin_checkout",
    "place_order",
    "place_downpayment",
]


def synthetic_events(random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    dataset_end = pd.Timestamp("2026-01-01", tz="UTC")
    rows: list[dict[str, object]] = []

    for index in range(720):
        journey_id = f"synthetic-{index:04d}"
        ongoing = index >= 600
        intent = rng.beta(2.0, 2.0)
        if ongoing:
            start = dataset_end - pd.Timedelta(days=int(rng.integers(2, 55)))
        else:
            start = dataset_end - pd.Timedelta(days=int(rng.integers(80, 500)))
        reached = int(np.clip(np.floor(intent * len(FUNNEL)), 1, len(FUNNEL)))
        event_time = start
        for step, event_name in enumerate(FUNNEL[:reached]):
            repeats = 1 + int(rng.integers(0, 3 if step < 4 else 2))
            for _ in range(repeats):
                rows.append(
                    {
                        "journey_id": journey_id,
                        "event_name": event_name,
                        "event_timestamp": event_time,
                    }
                )
                event_time += pd.Timedelta(hours=int(rng.integers(2, 36)))
        if not ongoing and intent > 0.62:
            rows.append(
                {
                    "journey_id": journey_id,
                    "event_name": "order_shipped",
                    "event_timestamp": event_time + pd.Timedelta(hours=24),
                }
            )
        if ongoing:
            rows.append(
                {
                    "journey_id": journey_id,
                    "event_name": "browse_product",
                    "event_timestamp": dataset_end - pd.Timedelta(hours=int(rng.integers(1, 48))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    snapshots = build_pseudo_cut_dataset(synthetic_events())
    train, test = train_test_split(
        snapshots,
        test_size=0.25,
        stratify=snapshots["outcome"],
        random_state=42,
    )
    numeric, categorical = feature_columns(snapshots)
    model = make_random_forest_pipeline(numeric, categorical)
    model.fit(train[numeric + categorical], train["outcome"])
    probabilities = model.predict_proba(test[numeric + categorical])[:, 1]
    evaluation = evaluate_probabilities(test["outcome"], probabilities)
    print(f"Synthetic snapshots: {len(snapshots)}")
    print(f"Test MSE: {evaluation.mse:.4f}")
    print(f"Prediction mean: {evaluation.prediction_mean:.4f}")
    print(f"Outcome mean: {evaluation.outcome_mean:.4f}")


if __name__ == "__main__":
    main()

