"""Forward-only evidence fitting and diagnostics for delayed temporal outcomes."""

import torch

from encoders import EvidenceEncoder, REPRESENTATIONS, evidence_from_probability


def causal_schedule(config) -> list[dict]:
    """Require label_end < prediction_start; omit warm-up and boundary-crossing targets."""
    train_end = int(config.n_samples * (1 - config.validation_fraction - config.test_fraction))
    validation_end = int(config.n_samples * (1 - config.test_fraction))
    warmup = int(config.n_samples * config.warmup_fraction)
    horizon = config.forecast_horizon
    boundaries = (
        ("train", warmup, train_end - horizon),
        ("validation", train_end, validation_end - horizon),
        ("test", validation_end, config.n_samples),
    )
    blocks = []
    for split, first, stop in boundaries:
        if first >= stop:
            raise ValueError("warm-up and outcome horizon leave an empty split")
        for start in range(first, stop, config.encoding_block_size):
            cutoff = start if split == "train" or config.temporal_update == "causal" else train_end
            fit_stop = cutoff - horizon
            fit_start = max(0, fit_stop - config.history_window) if config.history_window else 0
            if fit_stop <= fit_start:
                raise ValueError("no matured history is available; increase warm-up or history window")
            blocks.append({"split": split, "start": start,
                           "stop": min(start + config.encoding_block_size, stop),
                           "fit_start": fit_start, "fit_stop": fit_stop})
    return blocks


def prepare_temporal_dataset(config, encoder_config, device="cpu"):
    from experiment import PreparedData, make_synthetic_dataset

    features, labels = make_synthetic_dataset(config)
    features, labels = features.to(device), labels.to(device)
    schedule = causal_schedule(config)
    probabilities = {split: [] for split in ("train", "validation", "test")}
    row_ids = {split: [] for split in probabilities}
    cached = {}
    audit = []
    for block in schedule:
        fit_range = (block["fit_start"], block["fit_stop"])
        if fit_range not in cached:
            fit_start, fit_stop = fit_range
            fitted = EvidenceEncoder(encoder_config).to(device).fit(
                features[fit_start:fit_stop], labels[fit_start:fit_stop],
            )
            cached = {fit_range: fitted}
        fitted = cached[fit_range]
        held = features[block["start"]:block["stop"]]
        probabilities[block["split"]].append(fitted.transform_probabilities(held))
        row_ids[block["split"]].append(torch.arange(block["start"], block["stop"]))
        class_counts = labels[block["fit_start"]:block["fit_stop"]].bincount(minlength=2)
        audit.append({**block, "fit_class_counts": class_counts.cpu().tolist(),
                      "latest_label_end": block["fit_stop"] - 1 + config.forecast_horizon,
                      "low_class_support_bins": sum(
                          int((counts.min(0).values < 5).sum())
                          for partitions in fitted.counts_ for counts in partitions),
                      "feature0_primary_edges": fitted.edges_[0][0].cpu().tolist()})
    row_ids = {split: torch.cat(indices) for split, indices in row_ids.items()}
    inputs = {split: {name: evidence_from_probability(torch.cat(values), name)
                      for name in REPRESENTATIONS} for split, values in probabilities.items()}
    train_end = int(config.n_samples * (1 - config.validation_fraction - config.test_fraction))
    fit_stop = train_end - config.forecast_horizon
    fit_start = max(0, fit_stop - config.history_window) if config.history_window else 0
    reference = EvidenceEncoder(encoder_config).to(device).fit(features[fit_start:fit_stop], labels[fit_start:fit_stop])
    protocol = {
        "kind": "forward", "blocks": audit, "label_rule": "row + horizon < prediction_start",
        "reference_fit": [fit_start, fit_stop], "reference_prior": float(labels[fit_start:fit_stop].float().mean()),
        "warmup_rows": int(config.n_samples * config.warmup_fraction),
        "excluded_boundary_rows_per_boundary": config.forecast_horizon,
        "update": config.temporal_update, "history_window": config.history_window,
        "model_update": "none", "scaler_update": "none",
    }
    prepared = PreparedData(
        config, encoder_config,
        {split: features[indices.to(device)] for split, indices in row_ids.items()},
        {split: labels[indices.to(device)] for split, indices in row_ids.items()},
        row_ids, inputs, reference, protocol,
    )
    if prepared.labels["train"].unique().numel() != 2:
        raise ValueError("both classes must be present in downstream training rows")
    protocol["drift_diagnostics"] = cdf_drift(features[fit_start:fit_stop], prepared.features)
    return prepared


def cdf_drift(reference, splits) -> dict:
    records = {}
    for split, features in splits.items():
        columns = []
        for feature in range(features.shape[1]):
            history = reference[:, feature].sort().values.contiguous()
            future = features[:, feature].sort().values.contiguous()
            points = torch.cat((history, future)).sort().values.contiguous()
            history_cdf = torch.searchsorted(history, points, right=True).double() / len(history)
            future_cdf = torch.searchsorted(future, points, right=True).double() / len(future)
            columns.append({
                "feature": feature, "ks_distance": float((history_cdf - future_cdf).abs().max()),
                "outside_history_range": float(((future < history[0]) | (future > history[-1])).float().mean()),
            })
        records[split] = columns
    return records


def temporal_metrics(data, predictions) -> list[dict]:
    from experiment import _metrics

    records = []
    for split, probability in predictions.items():
        for block in data.protocol["blocks"]:
            if block["split"] != split:
                continue
            selected = (data.row_ids[split] >= block["start"]) & (data.row_ids[split] < block["stop"])
            labels = data.labels[split].cpu()[selected]
            prior = torch.full((len(labels),), data.protocol["reference_prior"], dtype=torch.float64)
            records.append({"split": split, "start": block["start"], "stop": block["stop"],
                            "count": len(labels), "metrics": _metrics(labels, probability[selected]),
                            "reference_prior_metrics": _metrics(labels, prior)})
    return records


def plot_temporal_shape(data):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for split in ("train", "validation", "test"):
        times = data.row_ids[split].numpy()
        values = data.features[split][:, 0].cpu()
        axes[0, 0].plot(times, values.numpy(), linewidth=0.6, alpha=0.7, label=split)
        levels = torch.linspace(0.01, 0.99, 99)
        axes[0, 1].plot(torch.quantile(values, levels).numpy(), levels.numpy(), label=split)
        probability = (data.inputs[split]["centered"][:, 0].cpu() + 1) / 2
        axes[1, 0].plot(times, probability.numpy(), linewidth=0.6, alpha=0.6, label=split)
    axes[0, 0].set(title="Feature 0 over time (retained rows)", xlabel="Time", ylabel="Raw value")
    axes[0, 1].set(title="Feature 0 marginal quantile/CDF shape", xlabel="Raw value", ylabel="Quantile")
    axes[1, 0].set(title="Causally estimated q over time", xlabel="Time", ylabel="q")
    blocks = data.protocol["blocks"]
    axes[1, 1].plot([block["start"] for block in blocks],
                     [block["low_class_support_bins"] for block in blocks])
    axes[1, 1].set(title="Low-class-support bins in each historical fit", xlabel="Prediction block start", ylabel="Bins")
    for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
        if data.config.drift != "none":
            for fraction in (data.config.shift_fraction, data.config.recovery_fraction):
                axis.axvline(int(data.config.n_samples * fraction), color="#c35043", linestyle="--", alpha=0.7)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.legend()
    figure.suptitle(f"Temporal {data.config.drift} | {data.config.temporal_update} | "
                   f"history window {data.config.history_window or 'expanding'}")
    return figure


def plot_temporal_performance(result):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for axis, family in zip(axes, ("logistic", "mlp")):
        family_runs = [run for run in result["runs"] if run["model"] == family]
        for run in family_runs:
            blocks = run["time_metrics"]
            axis.plot([block["start"] for block in blocks],
                      [block["metrics"]["log_loss"] for block in blocks],
                      marker=".", label=run["representation"])
        blocks = family_runs[0]["time_metrics"]
        axis.plot([block["start"] for block in blocks],
                  [block["reference_prior_metrics"]["log_loss"] for block in blocks],
                  linestyle=":", color="black", label="historical prior")
        config = result["config"]
        if config["drift"] != "none":
            for fraction in (config["shift_fraction"], config["recovery_fraction"]):
                axis.axvline(int(config["n_samples"] * fraction), color="#c35043", linestyle="--", alpha=0.5)
        axis.set(title=f"{family}: validation then test blocks", xlabel="Prediction block start", ylabel="Log loss")
        axis.legend()
    return figure