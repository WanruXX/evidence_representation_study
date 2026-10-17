"""PyTorch evidence encoders for finite, continuous feature matrices."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn


REPRESENTATIONS = ("centered", "log", "arcsine")


def _floating(values) -> Tensor:
    values = torch.as_tensor(values)
    if values.is_complex():
        raise ValueError("values must be real")
    return values if values.is_floating_point() else values.to(torch.float32)


def _matrix(values, reference: Tensor) -> Tensor:
    values = torch.as_tensor(values, device=reference.device)
    if values.is_complex():
        raise ValueError("X must be real")
    values = values.to(dtype=reference.dtype)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("X must be a nonempty matrix with shape [samples, features]")
    if values.dtype not in (torch.float32, torch.float64):
        raise ValueError("encoders require float32 or float64")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("X must contain only finite continuous values")
    return values


def _labels(values, X: Tensor) -> Tensor:
    values = torch.as_tensor(values, device=X.device)
    if values.shape != (X.shape[0],) or not bool(((values == 0) | (values == 1)).all()):
        raise ValueError("y must be a binary 0/1 vector aligned with X")
    return values.to(torch.long)


def _representation(name: str) -> None:
    if name not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")


def evidence_from_probability(probability, representation: str = "centered") -> Tensor:
    """Transform equal-prior probabilities without representation-specific clipping."""
    _representation(representation)
    probability = _floating(probability)
    if not bool((torch.isfinite(probability) & (probability > 0) & (probability < 1)).all()):
        raise ValueError("probabilities must be finite and strictly between 0 and 1")
    centered = 2.0 * probability - 1.0
    if representation == "centered":
        return centered
    if representation == "log":
        return torch.log(probability) - torch.log1p(-probability)
    return (2.0 / math.pi) * torch.asin(centered)


def probability_from_evidence(evidence, representation: str = "centered") -> Tensor:
    """Invert a representation, subject to floating-point precision."""
    _representation(representation)
    evidence = _floating(evidence)
    if not bool(torch.isfinite(evidence).all()):
        raise ValueError("evidence must be finite")
    if representation == "log":
        return torch.sigmoid(evidence)
    if not bool(((evidence >= -1) & (evidence <= 1)).all()):
        raise ValueError("bounded evidence must lie in [-1, 1]")
    if representation == "arcsine":
        evidence = torch.sin((math.pi / 2.0) * evidence)
    return (evidence + 1.0) / 2.0


@dataclass(frozen=True)
class EncoderConfig:
    """Shared settings; alpha is a pseudocount per bin and class."""

    mode: str = "fixed"
    num_bins: int = 20
    num_knots: int = 257
    min_bin_count: int = 20
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "dual_fixed", "dual_adaptive", "raw_fixed"}:
            raise ValueError("mode must be fixed, dual_fixed, dual_adaptive or raw_fixed")
        for name, minimum in (("num_bins", 2), ("num_knots", 3), ("min_bin_count", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")


class _PercentileColumn(nn.Module):
    def __init__(self, knots: Tensor, levels: Tensor) -> None:
        super().__init__()
        self.register_buffer("knots", knots.detach().clone())
        self.register_buffer("levels", levels.detach().clone())

    def forward(self, values: Tensor) -> Tensor:
        if self.knots.numel() == 1:
            result = self.levels[0].expand_as(values)
        else:
            right = torch.searchsorted(self.knots, values.contiguous()).clamp(1, self.knots.numel() - 1)
            left = right - 1
            fraction = (values - self.knots[left]) / (self.knots[right] - self.knots[left])
            result = self.levels[left] + fraction * (self.levels[right] - self.levels[left])
        result = torch.where(values < self.knots[0], 0.0, result)
        return torch.where(values > self.knots[-1], 1.0, result)


class PercentileEncoder(nn.Module):
    """Column-wise interpolated empirical CDF, fitted on training inputs only."""

    def __init__(self, num_knots: int = 257) -> None:
        super().__init__()
        if isinstance(num_knots, bool) or not isinstance(num_knots, Integral) or num_knots < 3:
            raise ValueError("num_knots must be an integer >= 3")
        self.num_knots = num_knots
        self.register_buffer("_reference", torch.empty(0), persistent=False)
        self.columns = nn.ModuleList()

    @property
    def knots_(self) -> list[Tensor]:
        return [column.knots for column in self.columns]

    @torch.no_grad()
    def fit(self, X) -> PercentileEncoder:
        X = _matrix(X, self._reference)
        columns = []
        for column in X.unbind(1):
            values, counts = torch.unique(column, sorted=True, return_counts=True)
            if values.numel() <= self.num_knots:
                levels = (counts.cumsum(0).to(X.dtype) - 0.5 * counts) / column.numel()
            else:
                probabilities = torch.linspace(0, 1, self.num_knots, device=X.device, dtype=X.dtype)
                values, inverse, counts = torch.unique(
                    torch.quantile(column, probabilities), sorted=True,
                    return_inverse=True, return_counts=True,
                )
                levels = torch.zeros_like(values).scatter_add_(0, inverse, probabilities) / counts
            columns.append(_PercentileColumn(values, levels))
        self.columns = nn.ModuleList(columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> Tensor:
        if not self.columns:
            raise RuntimeError("fit must be called before transform")
        X = _matrix(X, self._reference)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from the fitted schema")
        return torch.stack([column(X[:, index]) for index, column in enumerate(self.columns)], dim=1)

    def forward(self, X) -> Tensor:
        return self.transform(X)


def _adaptive_edges(values: Tensor, config: EncoderConfig) -> Tensor:
    levels = torch.linspace(0, 1, config.num_bins + 1, device=values.device, dtype=values.dtype)[1:-1]
    candidates = torch.unique(torch.quantile(values, levels), sorted=True)
    candidates = candidates[(candidates > values.min()) & (candidates < values.max())]
    bins = torch.searchsorted(candidates, values.contiguous(), right=True)
    counts = torch.bincount(bins, minlength=candidates.numel() + 1)
    edges = []
    running_count = 0
    for index, count in enumerate(counts.tolist()):
        running_count += count
        if running_count >= config.min_bin_count:
            if index < candidates.numel():
                edges.append(candidates[index])
            running_count = 0
    if running_count and edges:
        edges.pop()
    return torch.stack(edges) if edges else values.new_empty(0)


class _Partition(nn.Module):
    def __init__(self, edges: Tensor, values: Tensor, y: Tensor, alpha: float) -> None:
        super().__init__()
        bins = torch.searchsorted(edges, values.contiguous(), right=True)
        num_bins = edges.numel() + 1
        counts = torch.bincount(bins + y * num_bins, minlength=2 * num_bins).reshape(2, num_bins)
        float_counts = counts.to(torch.float64)
        mass = (float_counts + alpha) / (float_counts.sum(1, keepdim=True) + alpha * num_bins)
        probability = (mass[1] / mass.sum(0)).to(values.dtype)
        if not bool((torch.isfinite(probability) & (probability > 0) & (probability < 1)).all()):
            raise ValueError("probabilities saturated; increase alpha or fit in float64")
        self.register_buffer("edges", edges.detach().clone())
        self.register_buffer("probability", probability.detach())
        self.register_buffer("counts", counts.detach())

    def forward(self, values: Tensor) -> Tensor:
        return self.probability[torch.searchsorted(self.edges, values.contiguous(), right=True)]


class EvidenceEncoder(nn.Module):
    """Shared supervised bin estimator; training rows require out-of-fold encoding.

    Fitted tables are buffers, not trainable parameters. Move before fitting to
    select the fitting device, or after fitting for GPU batch transformation.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else EncoderConfig()
        self.register_buffer("_reference", torch.empty(0), persistent=False)
        self.percentile_encoder_ = PercentileEncoder(self.config.num_knots)
        self.partitions = nn.ModuleList()

    @property
    def edges_(self) -> list[list[Tensor]]:
        return [[partition.edges for partition in column] for column in self.partitions]

    @property
    def counts_(self) -> list[list[Tensor]]:
        return [[partition.counts for partition in column] for column in self.partitions]

    @torch.no_grad()
    def fit(self, X, y) -> EvidenceEncoder:
        X = _matrix(X, self._reference)
        y = _labels(y, X)
        if not bool((torch.bincount(y, minlength=2) > 0).all()):
            raise ValueError("both classes must be present in every fit")
        if self.config.mode in {"fixed", "dual_fixed"}:
            self.percentile_encoder_.fit(X)
        coordinates = self._coordinates(X)
        columns = []
        for column in coordinates.unbind(1):
            if self.config.mode == "raw_fixed":
                primary = (torch.linspace(column.min(), column.max(), self.config.num_bins + 1,
                                          device=X.device, dtype=X.dtype)[1:-1]
                           if bool(column.max() > column.min()) else column.new_empty(0))
            elif self.config.mode == "dual_adaptive":
                primary = _adaptive_edges(column, self.config)
                shifted = (0.5 * primary[:-1] + 0.5 * primary[1:]
                           if primary.numel() >= 2 else primary.clone())
            else:
                primary = torch.arange(1, self.config.num_bins, device=X.device, dtype=X.dtype) / self.config.num_bins
                shifted = (torch.arange(self.config.num_bins, device=X.device, dtype=X.dtype) + 0.5) / self.config.num_bins
            edges = [primary] if self.config.mode in {"fixed", "raw_fixed"} else [primary, shifted]
            columns.append(nn.ModuleList([_Partition(edge, column, y, self.config.alpha) for edge in edges]))
        self.partitions = nn.ModuleList(columns)
        self.n_features_in_ = X.shape[1]
        return self

    def _coordinates(self, X: Tensor) -> Tensor:
        return X if self.config.mode in {"dual_adaptive", "raw_fixed"} else self.percentile_encoder_.transform(X)

    def transform_probabilities(self, X) -> Tensor:
        if not self.partitions:
            raise RuntimeError("fit must be called before transform")
        X = _matrix(X, self._reference)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from the fitted schema")
        coordinates = self._coordinates(X)
        return torch.stack([
            torch.stack([partition(coordinates[:, index]) for partition in column]).mean(0)
            for index, column in enumerate(self.partitions)
        ], dim=1)

    def transform(self, X, representation: str = "centered") -> Tensor:
        return evidence_from_probability(self.transform_probabilities(X), representation)

    def transform_all(self, X) -> dict[str, Tensor]:
        probability = self.transform_probabilities(X)
        return {name: evidence_from_probability(probability, name) for name in REPRESENTATIONS}

    def forward(self, X, representation: str = "centered") -> Tensor:
        return self.transform(X, representation)


@torch.no_grad()
def cross_fit_transform(
    X,
    y,
    splits: Iterable[tuple[Tensor, Tensor]],
    config: EncoderConfig | None = None,
) -> tuple[dict[str, Tensor], EvidenceEncoder]:
    """Encode each row using a disjoint fit; return a full-training deployment fit.

    Splits must cover every row exactly once as held-out data. Neither geometry
    nor label counts use that fold's held-out rows. Use only the training split
    here; the returned deployment encoder transforms external validation/test X.
    """
    X = _floating(X)
    X = _matrix(X, X)
    y = _labels(y, X)
    probability = torch.empty_like(X)
    covered = torch.zeros(X.shape[0], device=X.device, dtype=torch.bool)
    for train, held_out in splits:
        indices = []
        for values in (train, held_out):
            values = torch.as_tensor(values, device=X.device)
            if (values.ndim != 1 or values.numel() == 0
                    or values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                    or bool(((values < 0) | (values >= X.shape[0])).any())
                    or values.unique().numel() != values.numel()):
                raise ValueError("fold indices must be nonempty, unique, valid integer vectors")
            indices.append(values.to(torch.long))
        train, held_out = indices
        train_mask = torch.zeros_like(covered)
        train_mask[train] = True
        if bool(train_mask[held_out].any()) or bool(covered[held_out].any()):
            raise ValueError("folds must be disjoint with no repeated held-out rows")
        encoder = EvidenceEncoder(config).to(device=X.device, dtype=X.dtype).fit(X[train], y[train])
        probability[held_out] = encoder.transform_probabilities(X[held_out])
        covered[held_out] = True
    if not bool(covered.all()):
        raise ValueError("every row must be held out exactly once")
    encoded = {name: evidence_from_probability(probability, name) for name in REPRESENTATIONS}
    deployment = EvidenceEncoder(config).to(device=X.device, dtype=X.dtype).fit(X, y)
    return encoded, deployment