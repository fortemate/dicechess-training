"""Offline A/B/C input contract; deliberately separate from the serving/legacy contract.

The engine emits features from admitted full afterstates. This module preserves its row order,
whole-group splits and terminal mask; it never reconstructs engine features or invents splits.
A manifest is a completion marker, not evidence of data rights or legality by itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dicechess_training.prerank.dataset import TERMINAL_TARGET
from dicechess_training.prerank.model import PreRankMLP

SCHEMA = "experimental-prerank-arrays-v1"
FEATURE_SCHEMA = "experimental-prerank-features-v1"
SPLITS = ("train", "early_stop", "screen")
C_VOCABULARY = 2 * 16 * 10 * 64
C_TOKENS = 64
COLUMNS_A = (
    "p_diff",
    "n_diff",
    "b_diff",
    "r_diff",
    "q_diff",
    "material_diff",
    "total_material",
    "mobility_diff",
    "king_safety_diff",
)
COLUMNS_B = COLUMNS_A + (
    "own_attacked",
    "own_attacked_undefended",
    "opponent_attacked",
    "opponent_attacked_undefended",
)
ARRAY_KEYS = {"a", "b", "c", "targets", "terminal", "offsets", "splits", "tie_order"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ExperimentalCorpus:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    targets: np.ndarray
    terminal: np.ndarray
    offsets: np.ndarray
    splits: np.ndarray
    tie_order: np.ndarray

    @property
    def groups(self) -> int:
        return len(self.offsets) - 1

    def of(self, split: str) -> np.ndarray:
        require(split in SPLITS, "unknown experimental split")
        return np.flatnonzero(self.splits == split)

    def validate(self) -> None:
        n = len(self.targets)
        require(n > 0, "empty experimental corpus")
        require(self.a.shape == (n, 9) and self.b.shape == (n, 13), "invalid dense shapes")
        require(self.a.dtype == self.b.dtype == np.float32, "invalid dense dtypes")
        require(np.isfinite(self.a).all() and np.isfinite(self.b).all(), "non-finite features")
        require(np.array_equal(self.a, self.b[:, :9]), "rich9 prefix differs")
        require(self.c.shape == (n, 2, C_TOKENS) and self.c.dtype == np.int16, "invalid C layout")
        require(np.all((self.c >= -1) & (self.c < C_VOCABULARY)), "C ID outside vocabulary")
        for anchor in range(2):
            active = self.c[:, anchor] >= 0
            require(
                np.all(~active | (self.c[:, anchor] // (C_VOCABULARY // 2) == anchor)),
                "C ID in wrong anchor half",
            )
        require(self.targets.dtype == np.int64 and self.targets.shape == (n,), "invalid targets")
        require(
            np.all((self.targets >= -(2**31)) & (self.targets <= TERMINAL_TARGET)),
            "target outside teacher integer range",
        )
        require(self.terminal.shape == (n,) and self.terminal.dtype == np.bool_, "invalid mask")
        require(np.array_equal(self.terminal, self.targets == TERMINAL_TARGET), "terminal mismatch")
        require(np.all(self.c[self.terminal] == -1), "terminal C anchors must be absent")
        require(self.offsets.dtype == np.int64 and self.offsets.ndim == 1, "invalid offsets")
        require(
            len(self.offsets) >= 2
            and self.offsets[0] == 0
            and self.offsets[-1] == n
            and np.all(np.diff(self.offsets) > 0),
            "invalid group boundaries",
        )
        require(
            self.splits.shape == (self.groups,) and np.isin(self.splits, SPLITS).all(),
            "invalid split assignments",
        )
        require(
            self.tie_order.shape == (n,) and self.tie_order.dtype == np.int64,
            "invalid tie ordering",
        )
        for lo, hi in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            require(
                np.array_equal(np.sort(self.tie_order[lo:hi]), np.arange(hi - lo)),
                "tie ordering is not a group permutation",
            )


def pack(
    features: Path, splits: Path, destination: Path, *, source_sha256: str, splits_sha256: str
) -> dict:
    """Pack a pinned engine export with a pre-frozen split map; refuse output reuse."""
    manifest = json.loads((features / "manifest.json").read_text())
    require(
        manifest["schema"] == FEATURE_SCHEMA and manifest["engine_version"] == "0.14.0",
        "unsupported experimental engine export",
    )
    require(
        tuple(manifest["columns_a"]) == COLUMNS_A and tuple(manifest["columns_b"]) == COLUMNS_B,
        "feature columns differ",
    )
    require(manifest["source_sha256"] == source_sha256, "source digest differs")
    require(
        digest(features / "features.jsonl") == manifest["features_sha256"], "feature digest differs"
    )
    require(digest(splits) == splits_sha256, "split digest differs")
    assignments = json.loads(splits.read_text())
    require(isinstance(assignments, dict) and assignments, "empty split map")
    total, groups = manifest["candidates"], manifest["groups"]
    require(
        type(total) is int and total > 0 and type(groups) is int and groups > 0, "invalid counts"
    )
    arrays = dict(
        a=np.empty((total, 9), np.float32),
        b=np.empty((total, 13), np.float32),
        c=np.full((total, 2, C_TOKENS), -1, np.int16),
        targets=np.empty(total, np.int64),
        terminal=np.empty(total, np.bool_),
        offsets=np.zeros(groups + 1, np.int64),
        splits=np.empty(groups, dtype="U10"),
        tie_order=np.empty(total, np.int64),
    )
    seen = set()
    at = 0
    with (features / "features.jsonl").open() as stream:
        for g, line in enumerate(stream):
            require(g < groups, "too many groups")
            record = json.loads(line)
            gid, rows = record["group_id"], record["rows"]
            require(gid not in seen and gid in assignments, "duplicate or unassigned group")
            require(rows and at + len(rows) <= total, "invalid group size")
            seen.add(gid)
            require(assignments[gid] in SPLITS, "unsupported frozen split")
            arrays["splits"][g] = assignments[gid]
            moves = [" ".join(row["moves"]) for row in rows]
            require(len(set(moves)) == len(moves), "duplicate canonical paths")
            order = np.argsort(np.asarray(moves), kind="stable")
            ranks = np.empty(len(rows), np.int64)
            ranks[order] = np.arange(len(rows))
            arrays["tie_order"][at : at + len(rows)] = ranks
            for row in rows:
                require(
                    type(row["target"]) is int and type(row["terminal"]) is bool,
                    "target/mask must be integer/boolean",
                )
                require(len(row["a"]) == 9 and len(row["b"]) == 13, "invalid feature width")
                require(len(row["c"]) == 2, "two C anchors required")
                arrays["a"][at] = row["a"]
                arrays["b"][at] = row["b"]
                arrays["targets"][at] = row["target"]
                arrays["terminal"][at] = row["terminal"]
                for anchor, ids in enumerate(row["c"]):
                    require(
                        len(ids) <= C_TOKENS
                        and all(
                            type(i) is int
                            and anchor * C_VOCABULARY // 2 <= i < (anchor + 1) * C_VOCABULARY // 2
                            for i in ids
                        ),
                        "invalid C IDs",
                    )
                    require(ids == sorted(set(ids)), "C IDs must be unique and ordered")
                    arrays["c"][at, anchor, : len(ids)] = ids
                at += 1
            arrays["offsets"][g + 1] = at
    require(
        at == total and len(seen) == groups and seen == set(assignments), "counts/split map differ"
    )
    require(
        digest(features / "features.jsonl") == manifest["features_sha256"]
        and digest(splits) == splits_sha256,
        "input changed during packing",
    )
    corpus = ExperimentalCorpus(**arrays)
    corpus.validate()
    destination.mkdir()
    np.savez_compressed(destination / "arrays.npz", **arrays)
    output = {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "engine_version": "0.14.0",
        "source_sha256": source_sha256,
        "splits_sha256": splits_sha256,
        "features_sha256": manifest["features_sha256"],
        "arrays_sha256": digest(destination / "arrays.npz"),
        "groups": groups,
        "candidates": total,
        "role": "development-only; no transfer admission",
        "split_groups": {s: len(corpus.of(s)) for s in SPLITS},
    }
    (destination / "manifest.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def load(directory: Path, *, manifest_sha256: str) -> ExperimentalCorpus:
    """Load only a caller-pinned completion marker and non-pickle arrays."""
    require(digest(directory / "manifest.json") == manifest_sha256, "manifest digest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest["schema"] == SCHEMA
        and manifest["feature_schema"] == FEATURE_SCHEMA
        and manifest["engine_version"] == "0.14.0",
        "unsupported array contract",
    )
    path = directory / "arrays.npz"
    require(digest(path) == manifest["arrays_sha256"], "array digest differs")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == ARRAY_KEYS, "unexpected array keys")
        corpus = ExperimentalCorpus(**{key: archive[key] for key in ARRAY_KEYS})
    corpus.validate()
    require(
        corpus.groups == manifest["groups"] and len(corpus.targets) == manifest["candidates"],
        "manifest counts differ",
    )
    require(
        manifest["split_groups"] == {s: len(corpus.of(s)) for s in SPLITS}, "split counts differ"
    )
    return corpus


class KingRelativeMLP(nn.Module):
    """Stateless anchor sums, not an incremental NNUE implementation."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(C_VOCABULARY, 4)
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        active = ids >= 0
        values = self.embedding(ids.clamp(min=0).long()) * active.unsqueeze(-1)
        return self.net(values.sum(dim=2).reshape(-1, 8))


def make_model(corpus: ExperimentalCorpus, arm: str) -> nn.Module:
    """Dense statistics use train nonterminal rows only; screen cannot affect them."""
    require(arm in ("A", "B", "C"), "unknown experimental arm")
    corpus.validate()
    if arm == "C":
        return KingRelativeMLP()
    features = corpus.a if arm == "A" else corpus.b
    fitting = corpus.of("train")
    require(len(fitting) > 0, "no training groups")
    rows = np.concatenate([np.arange(corpus.offsets[g], corpus.offsets[g + 1]) for g in fitting])
    sample = features[rows[~corpus.terminal[rows]]]
    require(len(sample) > 0, "no nonterminal training rows")
    mean, scale = sample.mean(axis=0), sample.std(axis=0)
    scale[scale == 0] = 1
    return PreRankMLP(features.shape[1], feature_mean=mean, feature_scale=scale)


def score_candidates(
    model: nn.Module, features: torch.Tensor, terminal: torch.Tensor
) -> torch.Tensor:
    """Inference ordering: terminal rows never reach the model and sort above finite scores.

    This is not a training loss. A training runner must freeze its treatment of terminal groups
    and checkpoint/recovery contract before any real fit.
    """
    require(terminal.dtype == torch.bool and terminal.shape == (len(features),), "invalid mask")
    scores = torch.full((len(features),), torch.inf, device=features.device)
    if (~terminal).any():
        raw = model(features[~terminal]).reshape(-1)
        require(torch.isfinite(raw).all(), "non-finite learned scores")
        scores[~terminal] = raw.to(scores.dtype)
    return scores
