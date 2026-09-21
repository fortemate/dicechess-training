"""A corpus of lists, in the shape a listwise objective can consume.

`load_groups` hands back the dataset as nested Python objects, which is right for admission and
wrong for training: 752,504 candidates of nine floats are 27 MB as arrays and several gigabytes as
dictionaries. This turns one into the other once, and keeps the lists intact while doing it —
features and targets are concatenated into flat arrays with an offset per group, the way a sparse
matrix keeps its rows, so a group is a slice and nothing has to be padded.

The other half of this module is the target distribution, which is where the listwise decisions
live. See `docs/prerank/protocol-v1.json`; the short version is that gains are top-heavy, derived
from rank rather than from the teacher's raw units, and therefore need no special case for the
king-capture sentinel and no per-group temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dicechess_training.prerank.groups import assign_splits, load_groups

SPLITS = ("train", "validation", "test")

#: What the engine's search stamps on a turn that takes a king. It is `Int.MaxValue`, so it is
#: larger than every evaluation by four orders of magnitude and would dominate any arithmetic done
#: on target values. Nothing here does arithmetic on target values.
TERMINAL_TARGET = 2**31 - 1


@dataclass(frozen=True)
class Corpus:
    """One corpus as flat arrays: `features[offsets[g]:offsets[g + 1]]` is group `g`'s list."""

    features: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray
    splits: np.ndarray
    columns: tuple[str, ...]
    manifest: dict

    @property
    def groups(self) -> int:
        return len(self.offsets) - 1

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.offsets)

    def of(self, split: str) -> np.ndarray:
        """The indices of the groups in one split, in corpus order."""
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return np.flatnonzero(self.splits == split)

    def rows(self, group: int) -> slice:
        return slice(int(self.offsets[group]), int(self.offsets[group + 1]))


def load_corpus(directory) -> Corpus:
    """Admit a corpus and lay it out for training. Admission first, always: a corpus that the
    contract would refuse is not one to train on, and the loader is the only thing that decides
    that."""
    manifest, groups = load_groups(directory)
    splits = assign_splits(groups)

    sizes = [len(group["candidates"]) for group in groups]
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])

    width = len(manifest["columns"])
    features = np.empty((int(offsets[-1]), width), dtype=np.float32)
    targets = np.empty(int(offsets[-1]), dtype=np.float64)
    at = 0
    for group in groups:
        for candidate in group["candidates"]:
            features[at] = candidate["features"]
            targets[at] = candidate["target"]
            at += 1

    return Corpus(
        features=features,
        targets=targets,
        offsets=offsets,
        splits=np.asarray(splits),
        columns=tuple(manifest["columns"]),
        manifest=manifest,
    )


def target_gains(targets: np.ndarray) -> np.ndarray:
    """One group's target distribution: top-heavy, scale-free, and summing to one.

    The gain of a candidate is `1 / rank**2` on its rank within the group, normalised. Three
    properties follow, and each of them is a decision the alternative would have got wrong:

    **Nothing depends on the teacher's units.** A softmax over raw targets would need a
    temperature, and the right temperature would differ per group — the spread of a quiet
    middlegame list and of one with a queen hanging differ by orders of magnitude — so the
    temperature would silently become a per-group weight nobody chose.

    **Nothing depends on the king-capture sentinel.** A terminal candidate carries `Int.MaxValue`;
    under any arithmetic on values that is a numerical accident waiting to happen, and under rank
    it is simply first.

    **The mass sits where the metric looks, whatever the list's length.** The shortlist keeps `k`
    candidates and the search then rescores all of them with the expensive evaluation and takes
    the best, so the only thing that has to survive the cut is the maximum. An inverse-square
    discount puts 98.8% of the target mass inside the top 48 at every group size the corpus has,
    and its head barely moves with length — 0.608 for the best candidate at 61 candidates and at
    2,420 alike — so a long list teaches the same lesson about its top few as a short one.

    The obvious alternative, the `1 / log2(rank + 1)` discount NDCG uses, was written here first
    and measured out of it. Share of target mass inside the top 48:

    | gain             | n=61  | n=200 | n=643 | n=2420 |
    | ---------------- | ----- | ----- | ----- | ------ |
    | uniform          | 78.7% | 24.0% |  7.5% |   2.0% |
    | `1 / log2(r+1)`  | 84.8% | 36.0% | 14.6% |   4.9% |
    | `1 / r`          | 94.9% | 75.9% | 63.3% |  53.3% |
    | `1 / r**2`       | 99.7% | 99.0% | 98.8% |  98.8% |

    On the large groups — the only ones the metric is measured on — the log discount is barely
    distinguishable from not ranking at all. `1 / r` is closer but makes the best candidate of a
    long list worth half of the best candidate of a short one, which weights the hard groups
    down exactly when they matter most.

    Ties take the average rank, so candidates the teacher scored equally get equal gains, and a
    group the teacher could not separate at all becomes a uniform target: the model is asked to be
    flat there rather than to invent an order. In the first corpus that is 4.2% of groups.
    """
    order = np.argsort(-targets, kind="stable")
    ranks = np.empty(len(targets), dtype=np.float64)
    ranks[order] = np.arange(1, len(targets) + 1, dtype=np.float64)

    # Average rank within each run of equal targets, so a tie cannot be broken by array order.
    ordered = targets[order]
    start = 0
    for index in range(1, len(ordered) + 1):
        if index == len(ordered) or ordered[index] != ordered[start]:
            ranks[order[start:index]] = (start + 1 + index) / 2
            start = index

    gains = 1.0 / np.square(ranks)
    return gains / gains.sum()


def all_gains(corpus: Corpus) -> np.ndarray:
    """The target distribution for every candidate in the corpus, group by group."""
    gains = np.empty(len(corpus.targets), dtype=np.float64)
    for group in range(corpus.groups):
        rows = corpus.rows(group)
        gains[rows] = target_gains(corpus.targets[rows])
    return gains


def trainable(corpus: Corpus, split: str, minimum: int = 2) -> np.ndarray:
    """Groups of one split that have an ordering to learn at all.

    A single-candidate group is a decision with one option: its target distribution is degenerate,
    its loss is zero whatever the model says, and it costs a forward pass to discover that.
    """
    groups = corpus.of(split)
    return groups[corpus.sizes[groups] >= minimum]


def evaluated(corpus: Corpus, split: str, k: int) -> np.ndarray:
    """Groups of one split on which a shortlist of size `k` can be wrong.

    In a group no larger than the shortlist every candidate survives it, so the ranker cannot fail
    and the group would contribute a guaranteed success to every figure. The protocol therefore
    measures on the rest — 55% of the first corpus — and says so wherever a number is reported.
    """
    groups = corpus.of(split)
    return groups[corpus.sizes[groups] > k]


def standardisation(corpus: Corpus, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and scale of the features over the given groups only.

    Fitted on the training partition and nowhere else, then carried inside the model, which is the
    rule the ablation protocol arrived at the hard way (#27): statistics fitted over everything
    leak the validation partition into the model, and a scaler applied outside the model would
    have to be reimplemented by whatever serves it.

    A column that never varies gets a scale of one rather than zero, so a constant feature becomes
    a constant input instead of a division by zero.
    """
    rows = np.concatenate([np.arange(corpus.offsets[g], corpus.offsets[g + 1]) for g in groups])
    sample = corpus.features[rows]
    mean = sample.mean(axis=0)
    scale = sample.std(axis=0)
    scale[scale == 0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)
