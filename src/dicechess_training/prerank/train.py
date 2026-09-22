"""Train the pre-ranker, and say whether it beat the ordering the engine already ships.

The protocol is `docs/prerank/protocol-v1.json` and was written before this ran. Two of its rules
shape everything here.

**Only groups larger than the shortlist are measured.** In a group of 20 candidates a shortlist of
48 keeps all of them, so the ranker cannot be wrong and the group would contribute a guaranteed
success. In the first corpus that is 45% of groups — enough to turn a bad result into a good
headline, which is exactly why the protocol fixes the subset before anyone sees a number.

**A run that does not beat `material_diff` is a negative result.** That column is the engine's own
shipped pre-ranker, `ExpectimaxSearch.materialBatch`, and it costs nothing because it is already
one of the nine features. The ablation of #17 produced arms worse than a constant and only an
admissibility floor caught it; this is the same floor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dicechess_training.prerank.dataset import (
    Corpus,
    all_gains,
    evaluated,
    standardisation,
    trainable,
)
from dicechess_training.prerank.metrics import (
    REPORTED_WIDTHS,
    hit_vector,
)
from dicechess_training.prerank.metrics import (
    report as metrics_report,
)
from dicechess_training.prerank.model import PreRankMLP, listwise_loss

PROTOCOL = "playground-prerank-v1"
DEFAULT_K = 48
DEFAULT_SEED = 11


class TrainingError(ValueError):
    """A corpus cannot be trained on, or measured after training.

    A `ValueError` so the checks that predate it keep their meaning, and a named one so the
    command line can turn it into a refusal without also catching every arithmetic slip inside
    the loop. The messages say what, never where — a corpus's location is private.
    """


@dataclass(frozen=True)
class Hyperparameters:
    """Everything the protocol fixes, in one place so a run can record what it ran with."""

    hidden_dims: tuple[int, ...] = (32, 32)
    learning_rate: float = 1e-3
    batch_groups: int = 64
    max_epochs: int = 40
    patience: int = 5
    k: int = DEFAULT_K
    seed: int = DEFAULT_SEED


#: The protocol's settings, as one value rather than a call in a default argument.
DEFAULTS = Hyperparameters()


def ranking_metrics(
    corpus: Corpus, scores: np.ndarray, groups: np.ndarray, k: int
) -> dict[str, float]:
    """How often the shortlist keeps a turn the teacher would have chosen.

    `recall_at_k` is the question the seam actually asks: after the cheap ranker has cut the list
    to `k`, is a best candidate still in it? `rank1` is the harder version — did the ranker put
    one first.

    Ties are settled by target *value*, never by index. A group can hold several candidates the
    teacher scored identically, and any of them is a correct answer; scoring against a single
    chosen index would mark a right answer wrong whenever the teacher was indifferent, which in
    the first corpus is most groups somewhere in their list.
    """
    hits = 0
    firsts = 0
    for group in groups:
        rows = corpus.rows(group)
        targets = corpus.targets[rows]
        best = targets.max()
        order = np.argsort(-scores[rows], kind="stable")
        if (targets[order[:k]] == best).any():
            hits += 1
        if targets[order[0]] == best:
            firsts += 1
    total = max(len(groups), 1)
    return {"recall_at_k": hits / total, "rank1": firsts / total, "groups": float(len(groups))}


def baseline_scores(corpus: Corpus, name: str, seed: int = DEFAULT_SEED) -> np.ndarray:
    """An ordering to measure the model against.

    `material_diff` is not an arbitrary choice of column: ordering by it *is* the pre-ranker the
    engine ships in `ExpectimaxSearch.materialBatch`, so beating it is the least a learned ranker
    has to do to be worth serving. `random` is the lower bound, and says how much of any figure is
    the shortlist being generous rather than the ranker being right.
    """
    if name == "material_diff":
        return corpus.features[:, corpus.columns.index("material_diff")].astype(np.float64)
    if name == "random":
        return np.random.default_rng(seed).random(len(corpus.targets))
    raise ValueError(f"unknown baseline {name!r}")


def onnx_scores(corpus: Corpus, model_path: str | Path, batch: int = 65536) -> np.ndarray:
    """Score every candidate with an ONNX model that already exists.

    For measuring an ordering somebody is *already serving*. The corpus carries `rich-9-v1`
    features, which is also what the deployed value models consume, so a production pre-ranker can
    be put on the same footing as this one without re-deriving anything — and a baseline that is
    actually deployed is worth more than a proxy for it.

    The model is run as-is, not through `contracts.prerank`: a value model is not a pre-ranking
    artifact and would fail that contract on its role, which is the point of the contract. Only
    its ordering is used.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    out = np.empty(len(corpus.targets), dtype=np.float64)
    for start in range(0, len(out), batch):
        stop = min(start + batch, len(out))
        block = corpus.features[start:stop]
        out[start:stop] = session.run([output_name], {input_name: block})[0].reshape(-1)
    return out


def _batch(corpus: Corpus, gains: np.ndarray, groups: np.ndarray):
    """One batch: the candidates of several groups, concatenated, with their group index."""
    rows = np.concatenate([np.arange(corpus.offsets[g], corpus.offsets[g + 1]) for g in groups])
    index = np.repeat(np.arange(len(groups)), corpus.sizes[groups])
    return (
        torch.from_numpy(corpus.features[rows]),
        torch.from_numpy(gains[rows]),
        torch.from_numpy(index),
    )


def score_all(model: PreRankMLP, corpus: Corpus, batch: int = 65536) -> np.ndarray:
    """Every candidate's score, in corpus order."""
    model.eval()
    out = np.empty(len(corpus.targets), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(out), batch):
            stop = min(start + batch, len(out))
            block = torch.from_numpy(corpus.features[start:stop])
            out[start:stop] = model(block).reshape(-1).numpy()
    return out


def train(corpus: Corpus, hyper: Hyperparameters = DEFAULTS) -> tuple[dict, PreRankMLP, np.ndarray]:
    """One training run: the report, the fitted model, and its score for every candidate."""
    torch.manual_seed(hyper.seed)
    rng = np.random.default_rng(hyper.seed)

    fitting = trainable(corpus, "train")
    if len(fitting) == 0:
        raise TrainingError("the training split has no group with an ordering to learn")
    measuring = evaluated(corpus, "validation", hyper.k)
    if len(measuring) == 0:
        # Without this, `ranking_metrics` returns a recall of zero over no groups, the first
        # epoch becomes the best epoch because zero beats the initial sentinel, and the run
        # early-stops and reports a trained model whose validation evidence does not exist.
        raise TrainingError(
            f"the validation split has no group larger than the shortlist of {hyper.k}, "
            "so there is nothing the ranker could be measured on"
        )

    gains = all_gains(corpus)
    mean, scale = standardisation(corpus, fitting)
    model = PreRankMLP(
        input_dim=len(corpus.columns),
        hidden_dims=list(hyper.hidden_dims),
        feature_mean=mean,
        feature_scale=scale,
    ).double()
    optimiser = torch.optim.Adam(model.parameters(), lr=hyper.learning_rate)

    history: list[dict] = []
    best = {"recall_at_k": -1.0}
    best_state: dict | None = None
    best_epoch = -1

    for epoch in range(1, hyper.max_epochs + 1):
        model.train()
        shuffled = rng.permutation(fitting)
        total = 0.0
        batches = 0
        for start in range(0, len(shuffled), hyper.batch_groups):
            chunk = shuffled[start : start + hyper.batch_groups]
            features, batch_gains, index = _batch(corpus, gains, chunk)
            optimiser.zero_grad()
            loss = listwise_loss(model(features), batch_gains, index, len(chunk))
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1

        validation = ranking_metrics(corpus, score_all(model, corpus), measuring, hyper.k)
        history.append({"epoch": epoch, "train_loss": total / max(batches, 1), **validation})
        if validation["recall_at_k"] > best["recall_at_k"]:
            best = validation
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= hyper.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    scores = score_all(model, corpus)
    report = {
        "protocol": PROTOCOL,
        "seed": hyper.seed,
        "k": hyper.k,
        "hyperparameters": asdict(hyper),
        "corpus": {
            "groups": corpus.groups,
            "candidates": int(len(corpus.targets)),
            "trained_on": int(len(fitting)),
            "engine_version": corpus.manifest["engine_version"],
            "teacher": corpus.manifest["teacher"]["id"],
            "source_sha256": corpus.manifest["source_sha256"],
            "groups_sha256": corpus.manifest["groups_sha256"],
        },
        "best_epoch": best_epoch,
        "history": history,
        "validation": best,
        "baselines": {
            name: ranking_metrics(
                corpus, baseline_scores(corpus, name, hyper.seed), measuring, hyper.k
            )
            for name in ("material_diff", "random")
        },
    }
    report["admissible"] = bool(
        best["recall_at_k"] > report["baselines"]["material_diff"]["recall_at_k"]
    )
    report["by_shortlist"] = by_shortlist(corpus, scores, hyper.seed)
    return report, model, scores


def by_shortlist(corpus: Corpus, scores: np.ndarray, seed: int, split: str = "validation") -> dict:
    """Recall and NDCG at each reported width, against the ordering the engine ships.

    One width is a headline; several are a shape. A wide shortlist forgives a bad ordering — a
    random order already keeps the best candidate half the time at 48 — so a figure quoted at one
    width says as much about the width as about the ranker.
    """
    material = baseline_scores(corpus, "material_diff")
    random_order = baseline_scores(corpus, "random", seed)
    out = metrics_report(corpus, scores, split, REPORTED_WIDTHS, reference=material)
    for k in REPORTED_WIDTHS:
        width = out[str(k)]
        if not width.get("measured"):
            continue
        groups = evaluated(corpus, split, k)
        width["random"] = float(hit_vector(corpus, random_order, groups, k).mean())
        width["material_diff"] = width.pop("reference_recall_at_k")
        width["learned"] = width["recall_at_k"]
        width["paired_vs_material"] = width.pop("recall_vs_reference")
        width["discordance_vs_material"] = width.pop("recall_discordance")
    return out


#: The seeds `docs/prerank/protocol-v1.json` names, fixed before the first run. They are a
#: constant rather than an argument because a run is a run of the protocol: a seed chosen after
#: seeing a result is not a replication of anything, and five runs reported as a range is the only
#: reading this corpus supports — the spread between these seeds is wider than the effect at
#: several widths.
PROTOCOL_SEEDS = (11, 23, 47, 89, 131)

REPORT_FILE = "report-seed-{seed}.json"
WEIGHTS_FILE = "weights-seed-{seed}.pt"


def fit_seeds(
    corpus: Corpus, destination: str | Path, seeds: tuple[int, ...] = PROTOCOL_SEEDS
) -> list[dict]:
    """Run the protocol's seeds against one corpus, writing each run's report and weights.

    The weights are a bare `state_dict`, which is what `export.load_ranker` reads: the shapes
    carry the architecture, so a checkpoint needs nothing beside it to be reopened. The report is
    written whether or not the run cleared the admissibility floor — a negative result that is not
    written down is a negative result somebody repeats.
    """
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    for seed in seeds:
        report, model, _ = train(corpus, Hyperparameters(seed=seed))
        torch.save(model.state_dict(), folder / WEIGHTS_FILE.format(seed=seed))
        (folder / REPORT_FILE.format(seed=seed)).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append(report)
    return reports


def across_seeds(reports: list[dict]) -> dict:
    """The headline as a range, never as a mean.

    Averaging five runs hides the one thing a reader needs here, which is that the spread between
    seeds is comparable to the difference being claimed. A mean also invites a p-value computed
    from averaged counts, which is not a test of anything — an error made once in this programme
    already, on this data.
    """
    if not reports:
        raise TrainingError("no runs to summarise")
    learned = [report["validation"]["recall_at_k"] for report in reports]
    material = [report["baselines"]["material_diff"]["recall_at_k"] for report in reports]
    return {
        "seeds": [report["seed"] for report in reports],
        "k": reports[0]["k"],
        "learned_recall_at_k": {"min": min(learned), "max": max(learned)},
        "material_recall_at_k": {"min": min(material), "max": max(material)},
        "admissible": [report["seed"] for report in reports if report["admissible"]],
        "inadmissible": [report["seed"] for report in reports if not report["admissible"]],
    }
