from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import torch

from encoders import EncoderConfig, EvidenceEncoder, REPRESENTATIONS, probability_from_evidence
from experiment import ExperimentConfig, make_synthetic_dataset, prepare_dataset, run_experiment
from temporal import causal_schedule, cdf_drift, plot_temporal_shape, plot_temporal_performance


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(
            scenario="temporal_factors", n_samples=600, n_features=4, n_informative=2,
            warmup_fraction=0.2, encoding_block_size=40, forecast_horizon=5,
            temporal_update="causal", history_window=120, epochs=1, hidden_units=4,
            batch_size=256, tree_min_samples_leaf=5,
        )
        self.encoder_config = EncoderConfig(num_bins=5, num_knots=25)
        self.features = torch.randn(600, 4, generator=torch.Generator().manual_seed(42))
        self.labels = torch.arange(600) % 2

    def prepare(self, config=None, features=None, labels=None):
        fixture = (self.features if features is None else features,
                   self.labels if labels is None else labels)
        with patch("experiment.make_synthetic_dataset", return_value=fixture):
            return prepare_dataset(config or self.config, self.encoder_config)

    def test_schedule_purges_labels_and_excludes_warmup(self):
        for update in ("frozen", "causal"):
            for window in (0, 120):
                config = replace(self.config, temporal_update=update, history_window=window)
                data = self.prepare(config)
                self.assertEqual(sum(len(indices) for indices in data.row_ids.values()), 600 - 120 - 10)
                self.assertEqual(int(data.row_ids["train"][0]), 120)
                self.assertLess(int(data.row_ids["train"][-1]) + 5, int(data.row_ids["validation"][0]))
                self.assertLess(int(data.row_ids["validation"][-1]) + 5, int(data.row_ids["test"][0]))
                for block in data.protocol["blocks"]:
                    self.assertLess(block["latest_label_end"], block["start"])
                    if window:
                        self.assertLessEqual(block["fit_stop"] - block["fit_start"], window)
                if update == "frozen":
                    stops = {block["fit_stop"] for block in causal_schedule(config) if block["split"] != "train"}
                    self.assertEqual(len(stops), 1)

    def test_block_encodings_match_manual_past_only_fit(self):
        data = self.prepare()
        for split in ("train", "validation", "test"):
            block = next(block for block in data.protocol["blocks"] if block["split"] == split)
            fitted = EvidenceEncoder(self.encoder_config).fit(
                self.features[block["fit_start"]:block["fit_stop"]],
                self.labels[block["fit_start"]:block["fit_stop"]],
            )
            mask = (data.row_ids[split] >= block["start"]) & (data.row_ids[split] < block["stop"])
            expected = fitted.transform_probabilities(self.features[block["start"]:block["stop"]])
            for representation in REPRESENTATIONS:
                actual = probability_from_evidence(data.inputs[split][representation][mask], representation)
                torch.testing.assert_close(actual, expected)

    def test_future_features_and_labels_cannot_change_earlier_encodings(self):
        baseline = self.prepare()
        features, labels = self.features.clone(), self.labels.clone()
        features[500:] += 10000
        labels[500:] = 1 - labels[500:]
        changed = self.prepare(features=features, labels=labels)
        for split in ("train", "validation", "test"):
            earlier = baseline.row_ids[split] < 500
            for representation in REPRESENTATIONS:
                torch.testing.assert_close(baseline.inputs[split][representation][earlier],
                                           changed.inputs[split][representation][earlier], rtol=0, atol=0)

    def test_frozen_external_labels_are_never_used_for_fitting(self):
        config = replace(self.config, temporal_update="frozen")
        baseline = self.prepare(config)
        labels = self.labels.clone()
        labels[360:] = 1 - labels[360:]
        changed = self.prepare(config, labels=labels)
        for split in ("train", "validation", "test"):
            for representation in REPRESENTATIONS:
                torch.testing.assert_close(baseline.inputs[split][representation],
                                           changed.inputs[split][representation], rtol=0, atol=0)

    def test_causal_updates_wait_until_labels_mature(self):
        baseline = self.prepare()
        labels = self.labels.clone()
        labels[400:440] = 1
        changed = self.prepare(labels=labels)
        earlier = baseline.row_ids["validation"] < 440
        torch.testing.assert_close(baseline.inputs["validation"]["centered"][earlier],
                                   changed.inputs["validation"]["centered"][earlier], rtol=0, atol=0)
        self.assertFalse(torch.equal(baseline.inputs["test"]["centered"], changed.inputs["test"]["centered"]))

    def test_missing_history_class_fails_explicitly(self):
        labels = self.labels.clone()
        labels[:120] = 0
        with self.assertRaisesRegex(ValueError, "both classes"):
            self.prepare(labels=labels)

    def test_missing_downstream_class_fails_explicitly(self):
        labels = self.labels.clone()
        labels[120:355] = 0
        with self.assertRaisesRegex(ValueError, "downstream training"):
            self.prepare(replace(self.config, history_window=0), labels=labels)

    def test_drift_controls_preserve_the_appropriate_pairing(self):
        stable = replace(self.config, contamination_rate=0, label_noise=0)
        features, labels = make_synthetic_dataset(stable)
        shifted, shifted_labels = make_synthetic_dataset(replace(stable, drift="covariate"))
        torch.testing.assert_close(labels, shifted_labels)
        cutoff = int(stable.n_samples * stable.shift_fraction)
        torch.testing.assert_close(features[:cutoff], shifted[:cutoff], rtol=0, atol=0)
        self.assertFalse(torch.equal(features[cutoff:], shifted[cutoff:]))
        concept, concept_labels = make_synthetic_dataset(replace(stable, drift="concept", hidden_strength=0))
        matched, matched_labels = make_synthetic_dataset(replace(stable, hidden_strength=0))
        torch.testing.assert_close(concept, matched, rtol=0, atol=0)
        torch.testing.assert_close(concept_labels[:cutoff - stable.forecast_horizon],
                                   matched_labels[:cutoff - stable.forecast_horizon])
        self.assertTrue(bool((concept_labels != matched_labels).any()))

    def test_cdf_drift_detects_changed_shape(self):
        reference = torch.arange(20).float()[:, None]
        diagnostics = cdf_drift(reference, {"same": reference, "shifted": reference + 100})
        self.assertEqual(diagnostics["same"][0]["ks_distance"], 0)
        self.assertEqual(diagnostics["shifted"][0]["ks_distance"], 1)
        self.assertEqual(diagnostics["shifted"][0]["outside_history_range"], 1)

    def test_temporal_training_metrics_and_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = self.prepare()
        results = run_experiment(data)
        json.dumps(results, allow_nan=False)
        self.assertEqual(len(results["summary"]), 9)
        for run in results["runs"]:
            test_blocks = [block for block in run["time_metrics"] if block["split"] == "test"]
            self.assertEqual(sum(block["count"] for block in test_blocks), len(data.labels["test"]))
        figure = plot_temporal_shape(data)
        figure.canvas.draw()
        self.assertEqual(len(figure.axes), 4)
        plt.close(figure)
        figure = plot_temporal_performance(results)
        figure.canvas.draw()
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_temporal_preparation(self):
        with patch("experiment.make_synthetic_dataset", return_value=(self.features, self.labels)):
            data = prepare_dataset(self.config, self.encoder_config, "cuda:0")
        self.assertEqual(data.inputs["train"]["centered"].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()