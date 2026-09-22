"""Running the protocol as a command, and what the command is not allowed to let you choose.

Every other stage of this pipeline had a command and training did not, so the only record of how
the first result was produced was a script that no longer exists. These tests are mostly about the
two things that makes possible: a checkpoint the exporter can reopen without a note beside it, and
constants that cannot drift away from the preregistration they came from.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from dicechess_training.benchmark.splits import split_for
from dicechess_training.prerank import group_id_for
from dicechess_training.prerank.__main__ import main
from dicechess_training.prerank.dataset import load_corpus
from dicechess_training.prerank.export import load_ranker
from dicechess_training.prerank.pack import GENERATION_FILE, pack
from dicechess_training.prerank.train import (
    DEFAULT_K,
    PROTOCOL_SEEDS,
    TrainingError,
    across_seeds,
    fit_seeds,
)

ENGINE_VERSION = "0.12.0"
GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rich9"
PROTOCOL_FILE = Path(__file__).resolve().parents[1] / "docs" / "prerank" / "protocol-v1.json"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
TERMS = "Fortemate owner-controlled; not approved for public redistribution"

#: Every sorted three-face roll, which is how each group gets a different identity from one root.
ROLLS = ["".join(faces) for faces in itertools.combinations_with_replacement("BKNPQR", 3)]


def _columns() -> list[str]:
    raw = json.loads((GOLDEN / f"golden-engine-{ENGINE_VERSION}.json").read_text(encoding="utf-8"))
    return list(raw["columns"])


def _games(wanted: dict[str, int]) -> list[tuple[str, str]]:
    """Game ids that land where the test needs them, found rather than assumed.

    The split is a hash of the game id, so a fixture that wants a validation group larger than the
    shortlist has to go looking for one; picking ids by eye is how a test ends up measuring an
    empty split.
    """
    found: list[tuple[str, str]] = []
    remaining = dict(wanted)
    for index in itertools.count():
        game = f"game-{index}"
        split = split_for(game)
        if remaining.get(split, 0) > 0:
            remaining[split] -= 1
            found.append((game, split))
        if not any(remaining.values()):
            return found
    raise AssertionError("unreachable")


def _corpus(
    directory: Path,
    *,
    candidates: int = 60,
    wanted: dict[str, int] | None = None,
    targets_are_material: bool = False,
    seed: int = 4,
) -> Path:
    """An admitted corpus on disk, with lists long enough for a shortlist of 48 to cut."""
    columns = _columns()
    rng = np.random.default_rng(seed)
    games = _games(wanted or {"train": 24, "validation": 6, "test": 4})
    groups = []
    for index, (game, _) in enumerate(games):
        dice = ROLLS[index % len(ROLLS)]
        features = rng.normal(size=(candidates, len(columns)))
        if targets_are_material:
            # The baseline is then a perfect ranker, so nothing can beat it and every run is
            # inadmissible — which is the case the exit code exists for.
            scores = features[:, columns.index("material_diff")]
        else:
            scores = (
                300.0 * features[:, columns.index("material_diff")]
                + 900.0 * features[:, columns.index("mobility_diff")]
                - 700.0 * features[:, columns.index("king_safety_diff")]
            )
        groups.append(
            {
                "group_id": group_id_for(START, dice),
                "game_id": game,
                "root_fen": START,
                "dice": dice,
                "side": "w",
                "candidates": [
                    {
                        "moves": ["e2e4"],
                        "result_fen": f"{dice}-{seat}",
                        "features": [float(value) for value in features[seat]],
                        "target": float(scores[seat]),
                    }
                    for seat in range(candidates)
                ],
            }
        )

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "license.txt").write_text("Owner-controlled use.\n", encoding="utf-8")
    (directory / "groups.json").write_text(json.dumps(groups) + "\n", encoding="utf-8")
    (directory / GENERATION_FILE).write_text(
        json.dumps(
            {
                "generator": "dicechess-hunter (dicechess.hunter.PrerankGroupsMain)",
                "schema": "playground-prerank-groups-v1",
                "feature_schema": "rich-9-v1",
                "engine_version": ENGINE_VERSION,
                "columns": columns,
                "teacher": {
                    "id": "hunter-baseline-v1",
                    "weights_sha256": "5a" * 32,
                    "resolved_config_hash": "sha256:" + "5a" * 32,
                },
                "roots_file": "roots.tsv",
                "roots_sha256": "bd" * 32,
                "roots_read": len(groups),
                "skipped": {"unparsable": 0, "side_mismatch": 0, "forced_pass": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pack(directory, license_=TERMS)
    return directory


def test_the_protocol_seeds_are_the_ones_the_protocol_names() -> None:
    """The constant and the preregistration are two copies of one decision, and a copy drifts.

    A seed chosen after seeing a result is not a replication, so the file that fixed them before
    the first run is the authority and this asserts the code still agrees with it.
    """
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert list(PROTOCOL_SEEDS) == protocol["seeds"]
    assert protocol["metrics"]["k"] == DEFAULT_K


def test_a_run_writes_a_report_and_a_checkpoint_the_exporter_can_reopen(tmp_path: Path) -> None:
    """The two halves of a run's record. The checkpoint matters most: `export` rebuilds the
    architecture from the shapes alone, so a run's output has to be loadable with nothing beside
    it or the artifact cannot be rebuilt from the run that produced it."""
    corpus = load_corpus(_corpus(tmp_path / "corpus"))
    reports = fit_seeds(corpus, tmp_path / "runs", seeds=(11,))

    assert len(reports) == 1
    report = json.loads((tmp_path / "runs" / "report-seed-11.json").read_text(encoding="utf-8"))
    assert report["seed"] == 11
    assert report["protocol"] == "playground-prerank-v1"
    assert report["corpus"]["groups_sha256"] == corpus.manifest["groups_sha256"]

    model = load_ranker(tmp_path / "runs" / "weights-seed-11.pt")
    assert model.net[0].weight.shape[1] == len(corpus.columns)


def test_across_seeds_reports_a_range_and_never_a_mean() -> None:
    """Averaging hides the thing a reader of this corpus most needs, which is that the spread
    between seeds is comparable to the difference being claimed."""
    reports = [
        {
            "seed": seed,
            "k": 48,
            "validation": {"recall_at_k": value},
            "baselines": {"material_diff": {"recall_at_k": 0.5}},
            "admissible": value > 0.5,
        }
        for seed, value in ((11, 0.7), (23, 0.4), (47, 0.9))
    ]
    summary = across_seeds(reports)
    assert summary["learned_recall_at_k"] == {"min": 0.4, "max": 0.9}
    assert summary["admissible"] == [11, 47]
    assert summary["inadmissible"] == [23]
    assert "mean" not in json.dumps(summary)


def test_nothing_to_summarise_is_refused_rather_than_averaged() -> None:
    with pytest.raises(TrainingError, match="no runs"):
        across_seeds([])


def test_the_command_line_trains_and_names_the_floor_it_must_clear(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _corpus(tmp_path / "corpus")
    assert main(["train", str(corpus), str(tmp_path / "runs"), "--seeds", "11"]) == 0
    out = capsys.readouterr().out
    assert "trained 1 runs" in out
    assert "material" in out
    assert "admissible" in out
    assert (tmp_path / "runs" / "weights-seed-11.pt").is_file()


def test_a_run_that_cannot_beat_the_engines_own_ordering_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A negative result is a result, so it is written down and reported — but it is not a zero.

    The ablation of #17 produced arms worse than a constant and only an admissibility floor
    caught it. This is that floor, in the exit status where a pipeline can see it.
    """
    corpus = _corpus(tmp_path / "corpus", targets_are_material=True)
    assert main(["train", str(corpus), str(tmp_path / "runs"), "--seeds", "11"]) == 2
    out = capsys.readouterr().out
    assert "BELOW THE FLOOR" in out
    assert "inadmissible seeds: [11]" in out
    assert (tmp_path / "runs" / "report-seed-11.json").is_file(), "a negative result is still one"


def test_a_corpus_with_nothing_to_measure_is_refused_without_a_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every validation group fits inside the shortlist, so the ranker cannot be wrong anywhere
    and the run would report a recall of 1.0 that means nothing."""
    corpus = _corpus(tmp_path / "corpus", candidates=12)
    assert main(["train", str(corpus), str(tmp_path / "runs"), "--seeds", "11"]) == 1
    captured = capsys.readouterr()
    assert "refused: the validation split has no group larger than the shortlist" in captured.err
    assert str(tmp_path) not in captured.err


def test_a_repeated_seed_is_refused_rather_than_run_twice(tmp_path: Path) -> None:
    """The run is a deterministic function of its seed, so the second pass would overwrite the
    first's files and then be counted as a second run."""
    corpus = load_corpus(_corpus(tmp_path / "corpus"))
    with pytest.raises(TrainingError, match="appears twice"):
        fit_seeds(corpus, tmp_path / "runs", seeds=(11, 11))


def test_the_command_line_cannot_invent_a_seed(capsys: pytest.CaptureFixture[str]) -> None:
    """A seed outside the preregistered five is not a replication of anything, so the command
    does not accept one. `fit_seeds` still does — the library is the escape hatch."""
    with pytest.raises(SystemExit):
        main(["train", ".", "runs", "--seeds", "7"])
    assert "invalid choice" in capsys.readouterr().err


def test_a_subset_of_the_seeds_is_reported_as_a_check_not_a_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-running one recorded seed is a useful thing to do and is not the five-seed finding."""
    corpus = _corpus(tmp_path / "corpus")
    assert main(["train", str(corpus), str(tmp_path / "runs"), "--seeds", "11"]) == 0
    assert "PARTIAL     1 of 5 protocol seeds" in capsys.readouterr().out


def test_a_full_run_is_not_labelled_partial() -> None:
    reports = [
        {
            "seed": seed,
            "k": 48,
            "validation": {"recall_at_k": 0.7},
            "baselines": {"material_diff": {"recall_at_k": 0.5}},
            "admissible": True,
        }
        for seed in PROTOCOL_SEEDS
    ]
    assert across_seeds(reports)["complete"] is True
    assert across_seeds(reports[:2])["complete"] is False
