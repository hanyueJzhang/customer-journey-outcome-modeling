import unittest

import numpy as np
import pandas as pd

from src.journey_clustering import cluster_embeddings, head_tail_sequence
from src.journey_modeling import (
    build_snapshot_features,
    summarize_journeys,
)


class JourneyModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.DataFrame(
            [
                ("success", "browse_product", "2025-01-01"),
                ("success", "add_to_cart", "2025-01-03"),
                ("success", "order_shipped", "2025-01-10"),
                ("stale", "campaign_click", "2025-01-01"),
                ("stale", "browse_product", "2025-01-02"),
                ("open", "campaign_click", "2025-03-30"),
                ("open", "browse_product", "2025-04-01"),
            ],
            columns=["journey_id", "event_name", "event_timestamp"],
        )

    def test_status_uses_success_and_inactivity(self) -> None:
        summary = summarize_journeys(self.events, inactivity_days=60).set_index(
            "journey_id"
        )
        self.assertEqual(summary.loc["success", "journey_status"], "successful")
        self.assertEqual(summary.loc["stale", "journey_status"], "unsuccessful")
        self.assertEqual(summary.loc["open", "journey_status"], "ongoing")

    def test_features_exclude_post_cut_shipment(self) -> None:
        features = build_snapshot_features(
            self.events,
            {"success": pd.Timestamp("2025-01-05", tz="UTC")},
        ).set_index("journey_id")
        self.assertEqual(features.loc["success", "last_event"], "add_to_cart")
        self.assertEqual(features.loc["success", "n_events"], 2)
        self.assertEqual(features.loc["success", "furthest_stage"], 2)

    def test_head_tail_truncation(self) -> None:
        sequence = [f"event_{index}" for index in range(10)]
        result = head_tail_sequence(sequence, tokens_per_side=2)
        self.assertEqual(result, "event_0 event_1 event_8 event_9")

    def test_cluster_selection_returns_valid_labels(self) -> None:
        rng = np.random.default_rng(7)
        embeddings = np.vstack(
            [
                rng.normal(-3, 0.2, size=(10, 4)),
                rng.normal(0, 0.2, size=(10, 4)),
                rng.normal(3, 0.2, size=(10, 4)),
            ]
        )
        result = cluster_embeddings(
            embeddings,
            candidate_clusters=(2, 3, 4),
            pca_components=3,
        )
        self.assertEqual(len(result.labels), len(embeddings))
        self.assertIn(result.selected_clusters, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()

