"""Module 5: a small transformer over event sequences.

This is a deliberately compact, fully-working transformer encoder that takes a
patient's ordered events and predicts a risk score (probability of an adverse
trajectory). It is the "temporal modeling of patient trajectories" piece.

Honesty note (also in the README): the model is trained on *synthetic* sequences
generated from a known latent rule, so it is a faithful demonstration of the
method, not a clinically validated predictor. The value is that the architecture,
the featurization, the training loop and the inference path are all real and
runnable, and the synthetic rule is recoverable, so the explainability module has
something genuine to explain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import get_settings
from .schema import EventType, PatientTimeline

# Fixed ordering for the event-type one-hot. Keep in sync with EventType.
EVENT_ORDER = [
    EventType.DIAGNOSIS,
    EventType.STAGE,
    EventType.TREATMENT,
    EventType.RESPONSE,
    EventType.PROGRESSION,
    EventType.OBSERVATION,
    EventType.OTHER,
]
# feature layout: [6 one-hot type][gap_norm][response_score][has_stage]
FEATURE_DIM = len(EVENT_ORDER) + 3


def _month_index(date: str) -> int:
    """Convert YYYY-MM to an absolute month index for computing gaps."""
    y, m = date.split("-")
    return int(y) * 12 + int(m)


def featurize_timeline(timeline: PatientTimeline) -> np.ndarray:
    """Turn a real timeline into a (T, FEATURE_DIM) float array."""
    events = timeline.sorted_events()
    rows = []
    prev_idx = None
    for ev in events:
        onehot = [1.0 if ev.event_type == t else 0.0 for t in EVENT_ORDER]
        idx = _month_index(ev.date)
        gap = 0.0 if prev_idx is None else (idx - prev_idx)
        prev_idx = idx
        gap_norm = min(gap, 24.0) / 24.0  # cap and scale to ~[0,1]

        # response score from the RECIST-ish signal we may have on the event
        resp = 0.0
        desc = (ev.canonical or ev.description or "").lower()
        if ev.event_type == EventType.PROGRESSION or "progress" in desc:
            resp = -1.0
        elif "complete response" in desc:
            resp = 1.0
        elif "partial response" in desc or ev.event_type == EventType.RESPONSE:
            resp = 0.5

        has_stage = 1.0 if ev.stage else 0.0
        rows.append(onehot + [gap_norm, resp, has_stage])
    if not rows:
        rows = [[0.0] * FEATURE_DIM]
    return np.asarray(rows, dtype=np.float32)


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------
@dataclass
class Sample:
    features: np.ndarray  # (T, FEATURE_DIM)
    label: float          # 0.0 / 1.0


def _latent_risk(features: np.ndarray) -> float:
    """The 'true' rule the model has to learn from data.

    Risk goes up with: presence of progression, more treatment lines, and short
    gaps between events (a fast-moving course). Goes down with good responses.
    """
    type_block = features[:, : len(EVENT_ORDER)]
    n_progression = type_block[:, EVENT_ORDER.index(EventType.PROGRESSION)].sum()
    n_treatment = type_block[:, EVENT_ORDER.index(EventType.TREATMENT)].sum()
    resp_score = features[:, len(EVENT_ORDER) + 1].sum()
    mean_gap = features[:, len(EVENT_ORDER)].mean()

    logit = (1.6 * n_progression) + (0.4 * n_treatment) - (1.1 * resp_score) - (1.5 * mean_gap) - 0.5
    return 1.0 / (1.0 + np.exp(-logit))


def generate_synthetic_dataset(n: int = 600, seed: int | None = None) -> list[Sample]:
    """Generate n synthetic patient sequences with labels from the latent rule."""
    rng = np.random.default_rng(seed if seed is not None else get_settings().seed)
    samples: list[Sample] = []
    for _ in range(n):
        length = int(rng.integers(3, 10))
        rows = []
        for i in range(length):
            t = EVENT_ORDER[rng.integers(0, len(EVENT_ORDER) - 1)]  # avoid OTHER
            onehot = [1.0 if t == e else 0.0 for e in EVENT_ORDER]
            gap_norm = float(rng.random())
            if t == EventType.PROGRESSION:
                resp = -1.0
            elif t == EventType.RESPONSE:
                resp = float(rng.choice([0.5, 1.0]))
            else:
                resp = 0.0
            has_stage = 1.0 if (t == EventType.STAGE or (t == EventType.DIAGNOSIS and rng.random() > 0.5)) else 0.0
            rows.append(onehot + [gap_norm, resp, has_stage])
        feats = np.asarray(rows, dtype=np.float32)
        p = _latent_risk(feats)
        label = 1.0 if rng.random() < p else 0.0  # noisy label
        samples.append(Sample(feats, label))
    return samples


# ----------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------
class RiskModel(nn.Module):
    """Transformer encoder over events -> single risk logit."""

    def __init__(self, feature_dim: int = FEATURE_DIM, dim: int | None = None,
                 heads: int | None = None, layers: int | None = None):
        super().__init__()
        s = get_settings()
        dim = dim or s.model_dim
        heads = heads or s.model_heads
        layers = layers or s.model_layers

        self.input_proj = nn.Linear(feature_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, s.max_events, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            batch_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, F), mask: (B, T) with True for padding positions
        h = self.input_proj(x)
        h = h + self.pos[:, : h.size(1), :]
        h = self.encoder(h, src_key_padding_mask=mask)
        if mask is not None:
            keep = (~mask).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        else:
            pooled = h.mean(1)
        return self.head(pooled).squeeze(-1)  # (B,) logits


def _collate(batch: list[Sample], max_len: int):
    """Pad a batch of variable-length sequences to a common length."""
    B = len(batch)
    T = min(max(s.features.shape[0] for s in batch), max_len)
    x = torch.zeros(B, T, FEATURE_DIM)
    mask = torch.ones(B, T, dtype=torch.bool)  # True = padding
    y = torch.zeros(B)
    for i, s in enumerate(batch):
        t = min(s.features.shape[0], T)
        x[i, :t] = torch.from_numpy(s.features[:t])
        mask[i, :t] = False
        y[i] = s.label
    return x, mask, y


def train_model(samples: list[Sample] | None = None, epochs: int = 12,
                batch_size: int = 32, lr: float = 1e-3, verbose: bool = True) -> RiskModel:
    """Train the risk model on synthetic data and return it."""
    s = get_settings()
    torch.manual_seed(s.seed)
    np.random.seed(s.seed)
    if samples is None:
        samples = generate_synthetic_dataset()

    model = RiskModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        np.random.shuffle(samples)
        total = 0.0
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            x, mask, y = _collate(batch, s.max_events)
            opt.zero_grad()
            logits = model(x, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if verbose:
            print(f"[temporal] epoch {epoch + 1:02d}/{epochs}  loss={total / len(samples):.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict_risk(model: RiskModel, features: np.ndarray) -> float:
    """Return risk probability in [0, 1] for one event-feature sequence."""
    model.eval()
    s = get_settings()
    t = min(features.shape[0], s.max_events)
    x = torch.zeros(1, t, FEATURE_DIM)
    x[0, :t] = torch.from_numpy(features[:t])
    mask = torch.zeros(1, t, dtype=torch.bool)
    logit = model(x, mask)
    return float(torch.sigmoid(logit).item())
