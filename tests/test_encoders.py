import unittest

import torch

from encoders import (
    EncoderConfig,
    EvidenceEncoder,
    PercentileEncoder,
    REPRESENTATIONS,
    cross_fit_transform,
    evidence_from_probability,
    probability_from_evidence,
)


class RepresentationTests(unittest.TestCase):
    def test_monotonicity_inverse_and_symmetry(self):
        probability = torch.linspace(0.001, 0.999, 1001, dtype=torch.float64)
        for name in REPRESENTATIONS:
            with self.subTest(representation=name):
                evidence = evidence_from_probability(probability, name)
                self.assertTrue(bool((torch.diff(evidence) > 0).all()))
                torch.testing.assert_close(
                    probability_from_evidence(evidence, name), probability, atol=1e-14, rtol=1e-12,
                )
                torch.testing.assert_close(evidence, -evidence.flip(0), atol=1e-12, rtol=1e-12)
                self.assertAlmostEqual(float(evidence_from_probability(0.5, name)), 0.0)

    def test_centered_log_identity(self):
        probability = torch.tensor([0.001, 0.2, 0.5, 0.9, 0.999], dtype=torch.float64)
        torch.testing.assert_close(
            evidence_from_probability(probability),
            torch.tanh(evidence_from_probability(probability, "log") / 2),
        )

    def test_invalid_transform_inputs(self):
        for probability in (0, 1, -0.1, float("nan"), float("inf")):
            for name in REPRESENTATIONS:
                with self.assertRaises(ValueError):
                    evidence_from_probability(probability, name)
        with self.assertRaises(ValueError):
            evidence_from_probability(0.5, "unknown")
        for name in ("centered", "arcsine"):
            with self.assertRaises(ValueError):
                probability_from_evidence(1.1, name)
        self.assertTrue(bool(torch.isfinite(probability_from_evidence([-1000, 1000], "log")).all()))


class EncoderTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.features = torch.randn(120, 3, generator=generator, dtype=torch.float64)
        self.labels = torch.tensor([0, 1]).repeat(60)

    def test_shared_probabilities_in_all_modes(self):
        for mode in ("fixed", "dual_fixed", "dual_adaptive", "raw_fixed"):
            with self.subTest(mode=mode):
                encoder = EvidenceEncoder(EncoderConfig(
                    mode=mode, num_bins=6, min_bin_count=10,
                )).double().fit(self.features, self.labels)
                probability = encoder.transform_probabilities(self.features)
                before = probability.clone()
                for name, evidence in encoder.transform_all(self.features).items():
                    self.assertEqual(evidence.shape, self.features.shape)
                    torch.testing.assert_close(probability_from_evidence(evidence, name), probability)
                torch.testing.assert_close(encoder.transform_probabilities(self.features), before, rtol=0, atol=0)
                self.assertEqual(list(encoder.parameters()), [])
                for partitions, counts in zip(encoder.edges_, encoder.counts_):
                    for edges, table in zip(partitions, counts):
                        self.assertEqual(table.shape, (2, len(edges) + 1))
                        torch.testing.assert_close(table.sum(1), torch.tensor([60, 60]))

    def test_equal_prior_mass_formula(self):
        features = torch.zeros(100, 1)
        labels = torch.tensor([0] * 90 + [1] * 10)
        for mode in ("fixed", "dual_fixed", "dual_adaptive"):
            encoder = EvidenceEncoder(EncoderConfig(mode=mode)).double().fit(features, labels)
            expected = []
            for partition in encoder.partitions[0]:
                num_bins = partition.counts.shape[1]
                mass = (partition.counts.double() + 1) / (torch.tensor([90, 10]).double()[:, None] + num_bins)
                expected.append(mass[1] / mass.sum(0))
                torch.testing.assert_close(partition.probability, expected[-1])
            probability = encoder.transform_probabilities(features)
            self.assertFalse(bool(torch.allclose(probability, torch.full_like(probability, 0.1))))
            if mode == "dual_adaptive":
                torch.testing.assert_close(probability, torch.full_like(probability, 0.5))

    def test_percentiles_and_tail_behavior(self):
        encoder = PercentileEncoder().double().fit([[0.0], [1.0], [2.0], [3.0]])
        torch.testing.assert_close(
            encoder([[-1], [0], [1], [2], [3], [4]]).flatten(),
            torch.tensor([0, 0.125, 0.375, 0.625, 0.875, 1], dtype=torch.float64),
        )
        constant = PercentileEncoder().fit(torch.ones(5, 1))
        torch.testing.assert_close(constant([[0], [1], [2]]).flatten(), torch.tensor([0, 0.5, 1]))

    def test_knots_and_adaptive_minimum_support(self):
        features = torch.arange(120, dtype=torch.float64)[:, None]
        percentile = PercentileEncoder(num_knots=7).double().fit(features)
        self.assertLessEqual(len(percentile.knots_[0]), 7)
        self.assertTrue(bool((torch.diff(percentile(features).flatten()) >= 0).all()))
        encoder = EvidenceEncoder(EncoderConfig(
            mode="dual_adaptive", num_bins=20, min_bin_count=17,
        )).double().fit(features, self.labels)
        self.assertTrue(bool((encoder.counts_[0][0].sum(0) >= 17).all()))

    def test_schema_and_label_errors(self):
        with self.assertRaises(RuntimeError):
            EvidenceEncoder()(self.features)
        for features, labels in (
            (self.features, self.labels[:-1]), (self.features, torch.zeros(120)),
            (self.features, torch.full((120,), 0.5)),
            (torch.full_like(self.features, float("nan")), self.labels),
        ):
            with self.assertRaises(ValueError):
                EvidenceEncoder().fit(features, labels)
        encoder = EvidenceEncoder().fit(self.features, self.labels)
        with self.assertRaises(ValueError):
            encoder(self.features[:, :1])
        for settings in (
            {"alpha": 0}, {"alpha": float("nan")}, {"num_bins": 2.5},
            {"num_knots": 2}, {"min_bin_count": 0}, {"mode": "discrete"},
        ):
            with self.assertRaises(ValueError):
                EncoderConfig(**settings)

    def test_cross_fitting_excludes_held_out_labels(self):
        splits = [(torch.arange(60, 120), torch.arange(60)),
                  (torch.arange(60), torch.arange(60, 120))]
        for mode in ("fixed", "dual_fixed", "dual_adaptive", "raw_fixed"):
            config = EncoderConfig(mode=mode, num_bins=6)
            encoded, deployment = cross_fit_transform(self.features, self.labels, splits, config)
            changed = self.labels.clone()
            changed[:60] = 1 - changed[:60]
            altered, _ = cross_fit_transform(self.features, changed, splits, config)
            for name in REPRESENTATIONS:
                torch.testing.assert_close(encoded[name][:60], altered[name][:60], rtol=0, atol=0)
            expected = EvidenceEncoder(config).double().fit(self.features[60:], self.labels[60:])
            torch.testing.assert_close(encoded["centered"][:60], expected(self.features[:60]), rtol=0, atol=0)
            self.assertEqual(deployment.n_features_in_, 3)

    def test_invalid_folds(self):
        for splits in (
            [], [(torch.arange(60), torch.arange(60))],
            [(torch.arange(60), torch.arange(60, 119))],
            [(torch.arange(60), torch.tensor([120]))],
            [(torch.arange(60), torch.tensor([60.5]))],
            [(torch.arange(60), torch.tensor([60, 60]))],
            [(torch.arange(60), torch.arange(60, 120))] * 2,
        ):
            with self.assertRaises(ValueError):
                cross_fit_transform(self.features, self.labels, splits)

    def test_refit_and_dtype_transfer(self):
        encoder = EvidenceEncoder().double().fit(self.features, self.labels)
        self.assertTrue(all(buffer.device.type == "cpu" for buffer in encoder.buffers()))
        encoder.float()
        self.assertEqual(encoder(self.features).dtype, torch.float32)
        encoder.fit(self.features[:, :1], self.labels)
        self.assertEqual(encoder(self.features[:, :1]).shape, (120, 1))

    def test_raw_width_partitions_respond_to_outlier_magnitude(self):
        features = torch.linspace(-1, 1, 120, dtype=torch.float64)[:, None]
        changed = features.clone()
        changed[-1] = 1000
        config = EncoderConfig(mode="raw_fixed", num_bins=6)
        first = EvidenceEncoder(config).double().fit(features, self.labels)
        second = EvidenceEncoder(config).double().fit(changed, self.labels)
        self.assertLess(float(first.edges_[0][0][-1]), 1)
        self.assertGreater(float(second.edges_[0][0][0]), 100)
        self.assertEqual(int(second.counts_[0][0].sum(0).max()), 119)
        constant = EvidenceEncoder(config).fit(torch.ones(120, 1), self.labels)
        self.assertEqual(constant.edges_[0][0].numel(), 0)
        torch.testing.assert_close(constant.transform_probabilities(torch.ones(2, 1)), torch.full((2, 1), 0.5))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fit_transfer_cross_fitting_and_mlp(self):
        splits = [(torch.arange(60, 120), torch.arange(60)),
                  (torch.arange(60), torch.arange(60, 120))]
        for mode in ("fixed", "dual_fixed", "dual_adaptive", "raw_fixed"):
            config = EncoderConfig(mode=mode, num_bins=6)
            encoder = EvidenceEncoder(config).double().fit(self.features, self.labels)
            expected = encoder.transform_all(self.features)
            encoder.cuda()
            self.assertTrue(all(buffer.device.type == "cuda" for buffer in encoder.buffers()))
            for name, evidence in encoder.transform_all(self.features.cuda()).items():
                torch.testing.assert_close(evidence.cpu(), expected[name])
            gpu_encoder = EvidenceEncoder(config).cuda().fit(self.features.cuda(), self.labels.cuda())
            self.assertTrue(bool(torch.isfinite(gpu_encoder(self.features.cuda())).all()))
            encoded, deployment = cross_fit_transform(
                self.features.float().cuda(), self.labels.cuda(), splits, config,
            )
            self.assertEqual(deployment(self.features.cuda()).device.type, "cuda")
            model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)).cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(encoded["centered"]).flatten(), self.labels.float().cuda(),
            )
            loss.backward()
            optimizer.step()
            self.assertTrue(bool(torch.isfinite(loss)))


if __name__ == "__main__":
    unittest.main()