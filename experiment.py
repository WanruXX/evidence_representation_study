"""Small, paired synthetic experiments for the three evidence representations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from numbers import Integral
import platform

import sklearn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
import torch
from torch import Tensor, nn

from encoders import EncoderConfig, EvidenceEncoder, REPRESENTATIONS, cross_fit_transform


MODEL_FAMILIES = ("logistic", "mlp", "tree")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_samples: int = 3000
    n_features: int = 6
    scenario: str = "additive"
    distribution: str = "normal"
    signal_strength: float = 1.5
    intercept: float = 0.0
    label_noise: float = 0.05
    class_prior: float = 0.5
    n_informative: int = 2
    rare_fraction: float = 0.05
    n_factors: int = 4
    observation_noise: float = 3.0
    hidden_strength: float = 3.0
    contamination_rate: float = 0.0
    contamination_scale: float = 20.0
    tail_index: float = 1.5
    factor_persistence: float = 0.8
    drift: str = "none"
    shift_fraction: float = 0.65
    recovery_fraction: float = 0.85
    drift_location: float = 3.0
    drift_scale: float = 2.0
    forecast_horizon: int = 5
    warmup_fraction: float = 0.15
    encoding_block_size: int = 100
    history_window: int = 0
    temporal_update: str = "frozen"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    n_folds: int = 4
    scaling: str = "standard"
    epochs: int = 60
    batch_size: int = 256
    hidden_units: int = 32
    logistic_lr: float = 0.02
    mlp_lr: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 12
    tree_max_depth: int = 5
    tree_min_samples_leaf: int = 20
    include_raw_baseline: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.include_raw_baseline, bool):
            raise ValueError("include_raw_baseline must be boolean")
        for name, minimum in (
            ("seed", 0), ("n_samples", 100), ("n_features", 2), ("n_folds", 2),
            ("epochs", 1), ("batch_size", 1), ("hidden_units", 1), ("patience", 1),
            ("n_informative", 1),
            ("n_factors", 1), ("forecast_horizon", 1), ("encoding_block_size", 1),
            ("history_window", 0),
            ("tree_max_depth", 1), ("tree_min_samples_leaf", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.seed >= 2**32 - 1:
            raise ValueError("seed must be less than 2**32 - 1")
        for name in ("logistic_lr", "mlp_lr", "contamination_scale", "tail_index", "drift_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("signal_strength", "observation_noise", "hidden_strength"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.signal_strength == 0 and self.scenario not in {"weak_factors", "temporal_factors"}:
            raise ValueError("zero signal is supported only by factor generators")
        if not 0 <= self.contamination_rate < 1:
            raise ValueError("contamination_rate must lie in [0, 1)")
        if not 0 <= self.factor_persistence < 1:
            raise ValueError("factor_persistence must lie in [0, 1)")
        if self.drift not in {"none", "covariate", "concept", "combined"}:
            raise ValueError("unknown drift type")
        if not 0 < self.shift_fraction < self.recovery_fraction < 1:
            raise ValueError("shift fractions must satisfy 0 < shift < recovery < 1")
        if not math.isfinite(self.drift_location):
            raise ValueError("drift_location must be finite")
        if not 0 < self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must lie in (0, 1)")
        if (self.scenario == "temporal_factors"
                and self.warmup_fraction >= 1 - self.validation_fraction - self.test_fraction):
            raise ValueError("warmup_fraction must lie inside the chronological training period")
        if self.temporal_update not in {"frozen", "causal"}:
            raise ValueError("temporal_update must be frozen or causal")
        if self.scenario != "temporal_factors" and self.drift != "none":
            raise ValueError("drift requires the temporal_factors scenario")
        if not math.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.label_noise < 0.5:
            raise ValueError("label_noise must lie in [0, 0.5)")
        if not 0 < self.class_prior < 1:
            raise ValueError("class_prior must lie in (0, 1)")
        if not 0 < self.rare_fraction < 1:
            raise ValueError("rare_fraction must lie in (0, 1)")
        if self.n_informative > self.n_features:
            raise ValueError("n_informative must not exceed n_features")
        if not (0 < self.validation_fraction < 1 and 0 < self.test_fraction < 1
                and self.validation_fraction + self.test_fraction < 1):
            raise ValueError("validation/test fractions must be positive and sum to less than one")
        if self.scenario not in {
            "additive", "interaction", "rare_tail", "class_conditional", "rare_mixture", "variance",
            "weak_factors", "temporal_factors",
        }:
            raise ValueError("unknown synthetic scenario")
        if self.scenario in {"class_conditional", "rare_mixture", "variance"} and self.intercept != 0:
            raise ValueError("class-conditional scenarios use class_prior, not intercept")
        if self.distribution not in {"normal", "skewed", "heavy_tail"}:
            raise ValueError("distribution must be normal, skewed or heavy_tail")
        if self.scaling not in {"none", "standard"}:
            raise ValueError("scaling must be none or standard")


@dataclass
class PreparedData:
    config: ExperimentConfig
    encoder_config: EncoderConfig
    features: dict[str, Tensor]
    labels: dict[str, Tensor]
    row_ids: dict[str, Tensor]
    inputs: dict[str, dict[str, Tensor]]
    encoder: EvidenceEncoder
    protocol: dict = field(default_factory=dict)


def make_synthetic_dataset(config: ExperimentConfig) -> tuple[Tensor, Tensor]:
    """Generate continuous heterogeneous features using a local CPU generator."""
    generator = torch.Generator().manual_seed(config.seed)
    latent = torch.randn(config.n_samples, config.n_features, generator=generator)
    if config.scenario in {"weak_factors", "temporal_factors"}:
        temporal = config.scenario == "temporal_factors"
        length = config.n_samples + (config.forecast_horizon if temporal else 0)
        factors = torch.randn(length, config.n_factors, generator=generator)
        if temporal:
            innovation_scale = math.sqrt(1 - config.factor_persistence**2)
            for timestamp in range(1, length):
                factors[timestamp] = (config.factor_persistence * factors[timestamp - 1]
                                      + innovation_scale * factors[timestamp])
        loadings = torch.linspace(0.2, 1.0, config.n_informative)
        factor_ids = torch.arange(config.n_informative) % config.n_factors
        latent[:, :config.n_informative] = (
            factors[:config.n_samples, factor_ids] * loadings
            + config.observation_noise * latent[:, :config.n_informative]
        )
        hidden = torch.randn(length, generator=generator)
        signal = factors.sum(1) / math.sqrt(config.n_factors)
        timestamps = torch.arange(length)
        stage = ((timestamps >= int(config.n_samples * config.shift_fraction)).long()
                 + (timestamps >= int(config.n_samples * config.recovery_fraction)).long())
        if temporal and config.drift in {"concept", "combined"}:
            signal = signal * torch.where(stage == 1, -1.0, 1.0)
        if temporal:
            future_signal = signal.unfold(0, config.forecast_horizon, 1)[1:config.n_samples + 1].mean(1)
            future_hidden = hidden.unfold(0, config.forecast_horizon, 1)[1:config.n_samples + 1].sum(1)
            future_hidden = future_hidden / math.sqrt(config.forecast_horizon)
        else:
            future_signal, future_hidden = signal, hidden
        probability = torch.sigmoid(config.signal_strength * future_signal
                                    + config.hidden_strength * future_hidden + config.intercept)
        labels = (torch.rand(config.n_samples, generator=generator) < probability).long()
        if temporal and config.drift in {"covariate", "combined"}:
            current_stage = stage[:config.n_samples, None]
            latent = latent * config.drift_scale**current_stage + config.drift_location * current_stage
    elif config.scenario in {"class_conditional", "rare_mixture", "variance"}:
        labels = (torch.rand(config.n_samples, generator=generator) < config.class_prior).long()
        signed_class = 2 * labels.float()[:, None] - 1
        informative = latent[:, :config.n_informative]
        if config.scenario == "class_conditional":
            informative += signed_class * config.signal_strength
        elif config.scenario == "rare_mixture":
            rare = torch.rand(informative.shape, generator=generator) < config.rare_fraction
            informative += rare * signed_class * config.signal_strength
        else:
            informative *= torch.where(labels[:, None] == 1, config.signal_strength, 1.0)
    else:
        if config.scenario == "additive":
            signal = latent[:, 0] - 0.75 * latent[:, 1]
        elif config.scenario == "interaction":
            signal = latent[:, 0] * latent[:, 1]
        else:
            signal = 3.0 * torch.relu(latent[:, 0] - 1.5) - 0.75
        probability = torch.sigmoid(config.signal_strength * signal + config.intercept)
        labels = (torch.rand(config.n_samples, generator=generator) < probability).long()
    flips = torch.rand(config.n_samples, generator=generator) < config.label_noise
    labels = torch.where(flips, 1 - labels, labels)
    if config.contamination_rate:
        event_generator = torch.Generator().manual_seed(config.seed + 1)
        events = torch.rand(latent.shape, generator=event_generator) < config.contamination_rate
        signs = torch.where(torch.rand(latent.shape, generator=event_generator) < 0.5, -1.0, 1.0)
        uniform = torch.rand(latent.shape, generator=event_generator).clamp_min(torch.finfo(latent.dtype).tiny)
        contamination = config.contamination_scale * uniform.pow(-1 / config.tail_index)
        latent = latent + events * signs * contamination
    if config.distribution == "skewed":
        observed = torch.exp(latent)
    elif config.distribution == "heavy_tail":
        observed = torch.sinh(latent)
    else:
        observed = latent
    scales = torch.logspace(-2, 2, config.n_features)
    offsets = torch.linspace(-10, 10, config.n_features)
    features = observed * scales + offsets
    if not bool(torch.isfinite(features).all()):
        raise ValueError("synthetic features overflowed")
    return features, labels


def prepare_dataset(
    config: ExperimentConfig,
    encoder_config: EncoderConfig,
    device: torch.device | str = "cpu",
) -> PreparedData:
    """Split first, then fit all encoder geometry and counts inside training."""
    if config.scenario == "temporal_factors":
        from temporal import prepare_temporal_dataset
        return prepare_temporal_dataset(config, encoder_config, device)
    features, labels = make_synthetic_dataset(config)
    indices = torch.arange(config.n_samples).numpy()
    train_validation, test = train_test_split(
        indices, test_size=config.test_fraction, stratify=labels.numpy(), random_state=config.seed,
    )
    train, validation = train_test_split(
        train_validation, test_size=config.validation_fraction / (1 - config.test_fraction),
        stratify=labels[train_validation].numpy(), random_state=config.seed + 1,
    )
    row_ids = {name: torch.as_tensor(values, dtype=torch.long) for name, values in (
        ("train", train), ("validation", validation), ("test", test),
    )}
    split_features = {name: features[values].to(device) for name, values in row_ids.items()}
    split_labels = {name: labels[values].to(device) for name, values in row_ids.items()}
    train_labels = split_labels["train"].cpu()
    if int(torch.bincount(train_labels, minlength=2).min()) < config.n_folds:
        raise ValueError("each training class needs at least n_folds rows")
    folds = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    splits = folds.split(train_labels.numpy(), train_labels.numpy())
    train_inputs, encoder = cross_fit_transform(
        split_features["train"], split_labels["train"], splits, encoder_config,
    )
    inputs = {"train": train_inputs}
    for name in ("validation", "test"):
        inputs[name] = encoder.transform_all(split_features[name])
    return PreparedData(config, encoder_config, split_features, split_labels, row_ids, inputs, encoder)


def scale_inputs(
    train: Tensor, validation: Tensor, test: Tensor, scaling: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Use only out-of-fold training inputs to estimate scaling statistics."""
    if scaling == "none":
        return train, validation, test
    if scaling != "standard":
        raise ValueError("scaling must be none or standard")
    mean = train.mean(0)
    standard_deviation = train.std(0, unbiased=False)
    scale = torch.where(standard_deviation > 0, standard_deviation, 1.0)
    return tuple((values - mean) / scale for values in (train, validation, test))


def build_model(family: str, n_features: int, config: ExperimentConfig) -> nn.Module:
    """Reset initialization for each representation, without sharing fitted weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        if family == "logistic":
            return nn.Linear(n_features, 1)
        if family == "mlp":
            return nn.Sequential(
                nn.Linear(n_features, config.hidden_units), nn.ReLU(),
                nn.Linear(config.hidden_units, 1),
            )
    raise ValueError("PyTorch family must be logistic or mlp")


def evidence_diagnostics(data: PreparedData) -> dict:
    """Describe evidence geometry and full-training bin support, feature by feature."""
    quantiles = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], dtype=torch.float64)
    splits = {}
    for split, inputs in data.inputs.items():
        probability = (inputs["centered"].detach().cpu().double() + 1) / 2
        centered = inputs["centered"].detach().cpu().double()
        logged = inputs["log"].detach().cpu().double()
        centered = centered - centered.mean(0)
        logged = logged - logged.mean(0)
        denominator = (centered.square().sum(0) * logged.square().sum(0)).sqrt()
        correlation = (centered * logged).sum(0) / denominator.clamp_min(1e-30)
        records = []
        split_labels = data.labels[split].cpu()
        observed_labels = split_labels.numpy()
        for feature in range(probability.shape[1]):
            values = probability[:, feature]
            extreme = (values < 0.05) | (values > 0.95)
            records.append({
                "feature": feature,
                "q_quantiles": dict(zip(("min", "p01", "median", "p99", "max"),
                                        torch.quantile(values, quantiles).tolist())),
                "extreme_fraction": float(extreme.double().mean()),
                "extreme_count": int(extreme.sum()),
                "extreme_direction_accuracy": float(((values[extreme] >= 0.5) == split_labels[extreme]).double().mean())
                if bool(extreme.any()) else None,
                "central_fraction": float(((values - 0.5).abs() < 0.05).double().mean()),
                "centered_log_correlation": float(correlation[feature]) if denominator[feature] > 0 else None,
                "q_roc_auc": float(roc_auc_score(observed_labels, values.numpy()))
                if len(set(observed_labels)) == 2 else None,
            })
        splits[split] = records
    support = []
    for feature, partitions in enumerate(data.encoder.counts_):
        for partition, counts in enumerate(partitions):
            counts = counts.detach().cpu()
            totals = counts.sum(0)
            support.append({
                "feature": feature, "partition": partition, "n_bins": len(totals),
                "min_bin_count": int(totals.min()), "median_bin_count": float(totals.float().median()),
                "min_class_bin_count": int(counts.min()),
                "empty_bins": int((totals == 0).sum()), "max_bin_count": int(totals.max()),
                "bins_with_class_count_below_5": int((counts.min(0).values < 5).sum()),
            })
    return {"splits": splits, "full_training_support": support,
            "support_reference": data.protocol.get("reference_fit", "full training")}


def plot_data_shape(data: PreparedData):
    """Show training-only raw distributions, estimated q, support and transform geometry."""
    import matplotlib.pyplot as plt

    features = data.features["train"].detach().cpu()
    labels = data.labels["train"].detach().cpu()
    probability = (data.inputs["train"]["centered"].detach().cpu() + 1) / 2
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = ("#267a83", "#c35043")
    for feature in range(2):
        lower, upper = float(features[:, feature].min()), float(features[:, feature].max())
        for label, color in enumerate(colors):
            values = features[labels == label, feature].numpy()
            axes[0, feature].hist(values, bins=40, range=(lower, upper), density=True,
                                  alpha=0.55, color=color, label=f"class {label}")
        axes[0, feature].set(title=f"Feature {feature}: class-conditional density", xlabel="Raw value")
        axes[0, feature].legend()
    for label, color in enumerate(colors):
        points = features[labels == label][:1000]
        axes[0, 2].scatter(points[:, 0], points[:, 1], s=6, alpha=0.25, color=color,
                           label=f"class {label}")
    axes[0, 2].set(title="Training feature geometry (up to 1,000/class)", xlabel="Feature 0", ylabel="Feature 1")
    axes[0, 2].legend()
    for feature in range(min(4, probability.shape[1])):
        axes[1, 0].hist(probability[:, feature].numpy(), bins=30, range=(0, 1),
                        histtype="step", density=True, label=f"feature {feature}")
    axes[1, 0].set(title="Out-of-fold equal-prior q", xlabel="q")
    axes[1, 0].legend()
    counts = torch.cat([partitions[0].sum(0).cpu() for partitions in data.encoder.counts_])
    axes[1, 1].hist(counts.numpy(), bins=25, color=colors[0])
    support_title = "Historical reference-bin support" if data.protocol else "Full-training primary-bin support"
    axes[1, 1].set(title=support_title, xlabel="Rows per bin", ylabel="Bins")
    order = probability[:, 0].argsort()
    for representation in REPRESENTATIONS:
        values = scale_inputs(*(data.inputs[split][representation] for split in (
            "train", "validation", "test",
        )), scaling=data.config.scaling)[0][:, 0].detach().cpu()
        axes[1, 2].plot(probability[order, 0].numpy(), values[order].numpy(), label=representation)
    axes[1, 2].set(title=f"Feature 0 geometry ({data.config.scaling})", xlabel="q", ylabel="Model input")
    axes[1, 2].legend()
    figure.suptitle(f"{data.config.scenario}: {len(features)} training rows, "
                   f"positive fraction {float(labels.float().mean()):.3f}")
    return figure


def _fit_torch(
    family: str, inputs: tuple[Tensor, Tensor, Tensor], labels: dict[str, Tensor],
    config: ExperimentConfig,
) -> tuple[dict[str, Tensor], list[dict], int]:
    train, validation, test = inputs
    model = build_model(family, train.shape[1], config).to(train.device)
    learning_rate = config.logistic_lr if family == "logistic" else config.mlp_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(config.seed)
    best_loss = math.inf
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(train.shape[0], generator=generator).to(train.device)
        total_loss = 0.0
        for batch in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train[batch]).flatten(), labels["train"][batch].float())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.numel()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation).flatten(), labels["validation"].float()))
        if not math.isfinite(validation_loss):
            raise RuntimeError("nonfinite validation loss")
        history.append({"epoch": epoch, "train_log_loss": total_loss / train.shape[0],
                        "validation_log_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = deepcopy(model.state_dict())
        if epoch - best_epoch >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = {name: torch.sigmoid(model(values).flatten()).cpu() for name, values in (
            ("validation", validation), ("test", test),
        )}
    return predictions, history, best_epoch


def _metrics(labels: Tensor, probability: Tensor) -> dict[str, float | None]:
    observed = labels.cpu().numpy()
    predicted = probability.cpu().double().numpy()
    return {
        "log_loss": float(log_loss(observed, predicted, labels=[0, 1])),
        "brier": float(brier_score_loss(observed, predicted)),
        "roc_auc": float(roc_auc_score(observed, predicted)) if len(set(observed)) == 2 else None,
        "accuracy": float(accuracy_score(observed, predicted >= 0.5)),
    }


def run_experiment(data: PreparedData) -> dict:
    """Fit evidence models and optional raw controls, with validation-only selection."""
    config = data.config
    rows = []
    runs = []
    prior = data.protocol.get("reference_prior", float(data.labels["train"].float().mean()))
    baselines = {split: _metrics(data.labels[split], torch.full((len(data.labels[split]),), prior,
                                                             dtype=torch.float64))
                 for split in ("validation", "test")}
    signal_features = {"additive": 2, "interaction": 2, "rare_tail": 1}.get(
        config.scenario, config.n_informative,
    )
    extreme = (data.inputs["test"]["centered"][:, :signal_features].abs() > 0.9).any(1).cpu()
    input_sets = {representation: {split: values[representation] for split, values in data.inputs.items()}
                  for representation in REPRESENTATIONS}
    if config.include_raw_baseline:
        input_sets["raw"] = data.features
    for representation, split_inputs in input_sets.items():
        inputs = scale_inputs(*(split_inputs[name] for name in (
            "train", "validation", "test",
        )), scaling=config.scaling)
        for family in MODEL_FAMILIES:
            if family == "tree":
                model = DecisionTreeClassifier(
                    max_depth=config.tree_max_depth,
                    min_samples_leaf=config.tree_min_samples_leaf, random_state=config.seed,
                )
                model.fit(inputs[0].cpu().numpy(), data.labels["train"].cpu().numpy())
                predictions = {name: torch.from_numpy(model.predict_proba(values.cpu().numpy())[:, 1])
                               for name, values in zip(("validation", "test"), inputs[1:])}
                history, best_epoch = [], None
            else:
                predictions, history, best_epoch = _fit_torch(family, inputs, data.labels, config)
            metrics = {name: _metrics(data.labels[name], predictions[name]) for name in predictions}
            slices = {}
            for name, mask in (("extreme_signal", extreme), ("nonextreme_signal", ~extreme)):
                slices[name] = {
                    "count": int(mask.sum()),
                    "metrics": _metrics(data.labels["test"].cpu()[mask], predictions["test"][mask])
                    if bool(mask.any()) else None,
                }
            rows.append({"model": family, "representation": representation, "scaling": config.scaling,
                         "test_excess_log_loss": metrics["test"]["log_loss"] - baselines["test"]["log_loss"],
                         "test_extreme_count": slices["extreme_signal"]["count"],
                         "test_extreme_log_loss": slices["extreme_signal"]["metrics"]["log_loss"]
                         if slices["extreme_signal"]["metrics"] is not None else None,
                         "best_epoch": best_epoch, **{
                             f"{split}_{metric}": value for split, scores in metrics.items()
                             for metric, value in scores.items()
                         }})
            time_metrics = []
            if data.protocol.get("kind") == "forward":
                from temporal import temporal_metrics
                time_metrics = temporal_metrics(data, predictions)
            runs.append({"model": family, "representation": representation, "history": history,
                         "time_metrics": time_metrics,
                         "test_slices": slices,
                         "predictions": {name: values.tolist() for name, values in predictions.items()}})
    return {
        "config": asdict(config), "encoder_config": asdict(data.encoder_config),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "sklearn": sklearn.__version__},
        "device": str(data.inputs["train"]["centered"].device),
        "row_ids": {name: values.tolist() for name, values in data.row_ids.items()},
        "labels": {name: values.cpu().tolist() for name, values in data.labels.items()},
        "protocol": data.protocol, "baselines": baselines,
        "diagnostics": evidence_diagnostics(data), "summary": rows, "runs": runs,
    }