import json
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from encoders import EncoderConfig, REPRESENTATIONS, probability_from_evidence
from experiment import (
    ExperimentConfig, build_model, make_synthetic_dataset, prepare_dataset,
    evidence_diagnostics, plot_data_shape, run_experiment, scale_inputs,
)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(
            n_samples=240, n_features=3, n_folds=2, epochs=2,
            batch_size=128, hidden_units=4, patience=1, tree_min_samples_leaf=5,
        )
        self.encoder_config = EncoderConfig(num_bins=4, num_knots=25)

    def test_config_rejects_invalid_settings(self):
        for changes in (
            {"n_folds": 1}, {"label_noise": 0.5}, {"test_fraction": 0.9},
            {"scaling": "unknown"}, {"epochs": True}, {"mlp_lr": float("nan")},
            {"class_prior": 0}, {"rare_fraction": 1}, {"n_informative": 7},
            {"scenario": "variance", "intercept": 1},
            {"contamination_rate": 1}, {"tail_index": 0}, {"observation_noise": -1},
            {"hidden_strength": -1}, {"factor_persistence": 1}, {"drift": "concept"},
            {"temporal_update": "unknown"}, {"forecast_horizon": 0},
            {"include_raw_baseline": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ExperimentConfig(**changes)

    def test_continuous_generation_is_seeded_and_labels_are_paired(self):
        features, labels = make_synthetic_dataset(self.config)
        repeated_features, repeated_labels = make_synthetic_dataset(self.config)
        torch.testing.assert_close(features, repeated_features)
        torch.testing.assert_close(labels, repeated_labels)
        for distribution in ("normal", "skewed", "heavy_tail"):
            for scenario in ("additive", "interaction", "rare_tail"):
                config = ExperimentConfig(n_samples=240, distribution=distribution, scenario=scenario)
                generated, observed = make_synthetic_dataset(config)
                self.assertTrue(bool(torch.isfinite(generated).all()))
                self.assertEqual(generated.shape, (240, 6))
                self.assertEqual(observed.unique().numel(), 2)

    def test_class_conditional_generators_change_signal_not_nuisance(self):
        for scenario in ("class_conditional", "rare_mixture", "variance"):
            weak = replace(self.config, n_samples=4000, scenario=scenario, signal_strength=1.0,
                           label_noise=0, rare_fraction=0.15)
            strong = replace(weak, signal_strength=5.0)
            first, labels = make_synthetic_dataset(weak)
            second, repeated_labels = make_synthetic_dataset(strong)
            torch.testing.assert_close(labels, repeated_labels)
            torch.testing.assert_close(first[:, 2:], second[:, 2:], rtol=0, atol=0)
            if scenario == "variance":
                self.assertGreater(float(second[labels == 1, 0].std()), 4 * float(first[labels == 1, 0].std()))
                torch.testing.assert_close(first[labels == 0], second[labels == 0], rtol=0, atol=0)
            elif scenario == "class_conditional":
                self.assertGreater(float(second[labels == 1, 0].mean()), float(first[labels == 1, 0].mean()))
                self.assertLess(float(second[labels == 0, 0].mean()), float(first[labels == 0, 0].mean()))
            else:
                changed = (first[:, 0] != second[:, 0]).float().mean()
                self.assertAlmostEqual(float(changed), 0.15, delta=0.03)

    def test_weak_factors_and_contamination_are_paired(self):
        config = replace(self.config, scenario="weak_factors", n_samples=2000,
                         n_features=12, n_informative=8, n_factors=3, label_noise=0)
        clean, labels = make_synthetic_dataset(config)
        contaminated, contaminated_labels = make_synthetic_dataset(replace(config, contamination_rate=0.01))
        stronger, stronger_labels = make_synthetic_dataset(replace(config, contamination_rate=0.01, contamination_scale=100))
        torch.testing.assert_close(labels, contaminated_labels)
        torch.testing.assert_close(labels, stronger_labels)
        affected = contaminated != clean
        self.assertAlmostEqual(float(affected.float().mean()), 0.01, delta=0.003)
        self.assertTrue(torch.equal(affected, stronger != clean))
        self.assertGreater(float((stronger - clean).abs().mean()), 4 * float((contaminated - clean).abs().mean()))
        observed, observed_labels = make_synthetic_dataset(replace(config, hidden_strength=0))
        torch.testing.assert_close(clean, observed, rtol=0, atol=0)
        self.assertGreater(float((labels != observed_labels).float().mean()), 0.1)
        noisy, noisy_labels = make_synthetic_dataset(replace(config, observation_noise=10))
        torch.testing.assert_close(labels, noisy_labels)
        torch.testing.assert_close(clean[:, 8:], noisy[:, 8:], rtol=0, atol=0)

    def test_weak_factor_null_and_contaminated_training_smoke(self):
        for signal in (0.0, 1.0):
            config = replace(self.config, scenario="weak_factors", signal_strength=signal,
                             contamination_rate=0.01, epochs=1)
            results = run_experiment(prepare_dataset(config, self.encoder_config))
            json.dumps(results, allow_nan=False)
            self.assertEqual(len(results["summary"]), 9)

    def test_targeted_factor_generator_has_weak_observed_signal(self):
        from sklearn.metrics import roc_auc_score

        for seed in (42, 43, 44):
            config = replace(self.config, scenario="weak_factors", seed=seed, n_samples=12000,
                             n_features=12, n_informative=8, n_factors=2, signal_strength=1.5,
                             observation_noise=1.5, hidden_strength=1.0, label_noise=0)
            features, labels = make_synthetic_dataset(config)
            observed = (features - torch.linspace(-10, 10, config.n_features)) / torch.logspace(-2, 2, config.n_features)
            score = (observed[:, :8] * torch.linspace(0.2, 1.0, 8)).sum(1)
            self.assertGreater(roc_auc_score(labels.numpy(), score.numpy()), 0.58)
            individual = [roc_auc_score(labels.numpy(), observed[:, feature].numpy()) for feature in range(8)]
            self.assertLess(max(individual), 0.7)
            null_features, null_labels = make_synthetic_dataset(replace(config, signal_strength=0))
            torch.testing.assert_close(features, null_features, rtol=0, atol=0)
            self.assertAlmostEqual(roc_auc_score(null_labels.numpy(), score.numpy()), 0.5, delta=0.04)

    def test_sparse_probabilities_are_extreme_and_recover_with_support(self):
        config = replace(self.config, scenario="weak_factors", n_samples=1200, n_features=12,
                         n_informative=8, n_factors=2, signal_strength=1.5, observation_noise=1.5,
                         hidden_strength=1.0, label_noise=0, n_folds=4)
        encoder = EncoderConfig(num_bins=120, alpha=0.01)
        fractions = []
        for sample_count in (1200, 12000):
            data = prepare_dataset(replace(config, n_samples=sample_count), encoder)
            extreme = data.inputs["validation"]["centered"][:, :8].abs() > 0.9
            fractions.append(float(extreme.float().mean()))
            if sample_count == 1200:
                records = evidence_diagnostics(data)["splits"]["validation"][:8]
                count = sum(record["extreme_count"] for record in records)
                agreement = sum(record["extreme_count"] * record["extreme_direction_accuracy"]
                                for record in records if record["extreme_count"])
                self.assertGreater(count, 0)
                self.assertLess(agreement / count, 0.85)
            for partitions in data.encoder.counts_:
                for counts in partitions:
                    self.assertEqual(int(counts.sum()), len(data.labels["train"]))
        self.assertGreater(fractions[0], 0.02)
        self.assertLess(fractions[1], fractions[0])

    def test_raw_baselines_share_rows_and_training_only_scaling(self):
        config = replace(self.config, include_raw_baseline=True, epochs=1)
        data = prepare_dataset(config, self.encoder_config)
        with patch("experiment.scale_inputs", wraps=scale_inputs) as scaling:
            results = run_experiment(data)
        self.assertEqual(len(results["summary"]), 12)
        self.assertEqual(scaling.call_count, 4)
        for index, split in enumerate(("train", "validation", "test")):
            self.assertIs(scaling.call_args.args[index], data.features[split])
        raw_rows = [row for row in results["summary"] if row["representation"] == "raw"]
        self.assertEqual({row["model"] for row in raw_rows}, {"logistic", "mlp", "tree"})
        for run in results["runs"]:
            self.assertEqual(len(run["predictions"]["test"]), len(data.labels["test"]))
        json.dumps(results, allow_nan=False)

    def test_diagnostics_and_training_shape_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = prepare_dataset(self.config, self.encoder_config)
        diagnostics = evidence_diagnostics(data)
        json.dumps(diagnostics, allow_nan=False)
        self.assertEqual(set(diagnostics["splits"]), {"train", "validation", "test"})
        self.assertEqual(len(diagnostics["full_training_support"]), self.config.n_features)
        self.assertEqual({record["feature"] for record in diagnostics["full_training_support"]},
                 set(range(self.config.n_features)))
        for record in diagnostics["splits"]["train"]:
            quantiles = list(record["q_quantiles"].values())
            self.assertEqual(quantiles, sorted(quantiles))
            self.assertGreaterEqual(record["extreme_fraction"], 0)
            self.assertLessEqual(record["extreme_fraction"], 1)
            if record["extreme_count"]:
                self.assertGreaterEqual(record["extreme_direction_accuracy"], 0)
                self.assertLessEqual(record["extreme_direction_accuracy"], 1)
            else:
                self.assertIsNone(record["extreme_direction_accuracy"])
        figure = plot_data_shape(data)
        self.assertEqual(len(figure.axes), 6)
        figure.canvas.draw()
        plt.close(figure)

    def test_separation_expands_estimated_evidence_range(self):
        weak = replace(self.config, n_samples=2000, scenario="class_conditional",
                       signal_strength=0.2, label_noise=0)
        strong = replace(weak, signal_strength=1.8)
        encoder = EncoderConfig(num_bins=20, alpha=0.5)
        weak_data = prepare_dataset(weak, encoder)
        strong_data = prepare_dataset(strong, encoder)
        fractions = []
        for data in (weak_data, strong_data):
            records = evidence_diagnostics(data)["splits"]["train"][:2]
            fractions.append(sum(record["extreme_fraction"] for record in records) / 2)
        self.assertGreater(fractions[1], fractions[0] + 0.1)

    def test_small_training_runs_for_new_generators(self):
        for scenario in ("class_conditional", "rare_mixture", "variance"):
            config = replace(self.config, scenario=scenario, signal_strength=3.0,
                             rare_fraction=0.1, class_prior=0.25, epochs=1)
            results = run_experiment(prepare_dataset(config, self.encoder_config))
            self.assertEqual(len(results["summary"]), 9)
            json.dumps(results, allow_nan=False)

    def test_splits_are_disjoint_and_share_probability_estimates(self):
        data = prepare_dataset(self.config, self.encoder_config)
        combined = torch.cat(list(data.row_ids.values()))
        torch.testing.assert_close(combined.sort().values, torch.arange(self.config.n_samples))
        for split in ("train", "validation", "test"):
            for representation in REPRESENTATIONS:
                probability = probability_from_evidence(data.inputs[split][representation], representation)
                torch.testing.assert_close(
                    probability, probability_from_evidence(data.inputs[split]["centered"]),
                )
        training = data.features["train"]
        fitted = data.encoder.transform_all(training)
        self.assertFalse(torch.equal(fitted["centered"], data.inputs["train"]["centered"]))
        for column in data.encoder.counts_:
            for counts in column:
                self.assertEqual(int(counts.sum()), len(training))

    def test_external_rows_do_not_change_training_encoding(self):
        baseline = prepare_dataset(self.config, self.encoder_config)
        features, labels = make_synthetic_dataset(self.config)
        external = torch.cat([baseline.row_ids["validation"], baseline.row_ids["test"]])
        features[external] += 10000
        with patch("experiment.make_synthetic_dataset", return_value=(features, labels)):
            changed = prepare_dataset(self.config, self.encoder_config)
        for representation in REPRESENTATIONS:
            torch.testing.assert_close(
                baseline.inputs["train"][representation], changed.inputs["train"][representation],
                atol=0, rtol=0,
            )

    def test_scaling_uses_only_training_statistics(self):
        train = torch.tensor([[1.0, 5], [3.0, 5]])
        validation = torch.tensor([[101.0, 7]])
        test = torch.tensor([[-99.0, 3]])
        scaled = scale_inputs(train, validation, test, "standard")
        torch.testing.assert_close(scaled[0], torch.tensor([[-1.0, 0], [1.0, 0]]))
        torch.testing.assert_close(scaled[1], torch.tensor([[99.0, 2]]))
        torch.testing.assert_close(scaled[2], torch.tensor([[-101.0, -2]]))
        self.assertIs(scale_inputs(train, validation, test, "none")[0], train)

    def test_models_start_identically_but_do_not_share_weights(self):
        for family in ("logistic", "mlp"):
            first = build_model(family, 3, self.config)
            second = build_model(family, 3, self.config)
            for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
                torch.testing.assert_close(first_parameter, second_parameter, atol=0, rtol=0)
                self.assertNotEqual(first_parameter.data_ptr(), second_parameter.data_ptr())

    def _check_experiment(self, device):
        data = prepare_dataset(self.config, self.encoder_config, device)
        results = run_experiment(data)
        self.assertEqual(len(results["summary"]), 9)
        self.assertIn("diagnostics", results)
        json.dumps(results, allow_nan=False)
        for row, run in zip(results["summary"], results["runs"]):
            for metric in ("log_loss", "brier", "roc_auc", "accuracy"):
                self.assertTrue(math.isfinite(row[f"test_{metric}"]))
            self.assertEqual(len(run["predictions"]["test"]), len(data.labels["test"]))
            self.assertEqual(sum(part["count"] for part in run["test_slices"].values()),
                             len(data.labels["test"]))
            if row["model"] != "tree":
                best = min(run["history"], key=lambda epoch: epoch["validation_log_loss"])
                self.assertEqual(row["best_epoch"], best["epoch"])

    def test_nine_model_smoke(self):
        self._check_experiment("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_nine_model_cuda_smoke(self):
        self._check_experiment("cuda:0")


if __name__ == "__main__":
    unittest.main()