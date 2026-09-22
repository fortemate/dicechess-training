"""The probe-pair gate: does the ranker put a hanging queen below its safe twin?

Carried from `dicechess-ev#1`. The question it exists to ask is the one the whole pre-ranker line
was started for — the engine's material ordering is blind to safety, so a bot walks a queen into a
pawn's reach and the search never sees the move that did not blunder, because pre-ranking dropped
it.

Each pair is two positions with **identical material and an identical placement of the mover's own
pieces**. Only the opponent's attacker moves: in one position it attacks the mover's queen, in the
other it does not. The gate asks for `score(safe) > score(blunder)`.

## The pairs are certified, not asserted

Three of the first six pairs written by hand were wrong — an attacker that did not attack, or a
"safe" square that was equally reachable. So every pair carries a second feature vector under
`kcp-13`, whose `queen_capture_danger` column is the engine's own dice-weighted answer to "can
this queen be taken next turn", and a pair is only admitted when that column is at least 0.2
higher in the blunder than in its twin. The gate checks this before it checks a model, so a
fixture that stops being a fixture fails loudly rather than passing a model for free.

## What `rich-9` can and cannot see

`kcp-13`'s last four columns are the capture probabilities, and `rich-9` is exactly `kcp-13`
without them. So the schema a pre-ranker can afford carries **no feature for a piece being en
prise**. Within every certified pair the nine columns are identical except `mobility_diff` — which
means a rich-9 model's answer on a pair is entirely determined by how it responds to that one
number, holding the other eight fixed.

That is not a detail. It decides what this gate can mean, and the direction is not even
consistent: against pawn, knight and bishop attackers the safe twin has the *higher* mobility,
and against rook attackers it has the *lower*. Read `docs/prerank/probe-pairs-v1.md` before
quoting a pass rate from here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dicechess_training.contracts import prerank as contract

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "prerank"

#: How much more likely the blunder's queen must be to fall, under the engine's own dice-weighted
#: answer, before a pair counts as a pair. 0.2 is comfortably above the residual every position
#: carries — an opponent can often manoeuvre into range within a three-die turn, so a "safe" twin
#: is rarely at exactly zero.
CERTIFICATION_MARGIN = 0.2

BLUNDER = "blunder"
SAFE = "safe"


class ProbePairError(ValueError):
    """A probe-pair fixture or gate was refused. The message says what, never where."""


def _fixture(schema: str, engine_version: str) -> dict:
    path = FIXTURES / f"probe-pairs-{schema}-engine-{engine_version}.json"
    if not path.is_file():
        raise ProbePairError(f"no committed probe-pair corpus for {schema} at that engine")
    return json.loads(path.read_text(encoding="utf-8"))


def pair_names(engine_version: str = contract.DEFAULT_ENGINE_VERSION) -> list[str]:
    """The pairs, in a stable order."""
    probes = _fixture("rich9", engine_version)["probes"]
    names = {probe["id"].rsplit("-", 1)[0] for probe in probes}
    return sorted(names)


def load_pairs(engine_version: str = contract.DEFAULT_ENGINE_VERSION) -> dict[str, dict]:
    """Each pair's two feature vectors, with the position they came from."""
    probes = {probe["id"]: probe for probe in _fixture("rich9", engine_version)["probes"]}
    pairs: dict[str, dict] = {}
    for name in pair_names(engine_version):
        for role in (BLUNDER, SAFE):
            if f"{name}-{role}" not in probes:
                raise ProbePairError(f"a pair is missing its {role} half")
        pairs[name] = {role: probes[f"{name}-{role}"] for role in (BLUNDER, SAFE)}
    return pairs


def certify(engine_version: str = contract.DEFAULT_ENGINE_VERSION) -> dict[str, float]:
    """How much more exposed each blunder's queen is than its twin's, per the engine.

    Refuses a fixture whose pairs no longer differ in the thing they are supposed to differ in.
    A gate that silently stops testing anything is worse than no gate.
    """
    corpus = _fixture("kcp13", engine_version)
    columns = list(corpus["columns"])
    if "queen_capture_danger" not in columns:
        raise ProbePairError("the certifying corpus has no queen_capture_danger column")
    danger = columns.index("queen_capture_danger")
    probes = {probe["id"]: probe["features"] for probe in corpus["probes"]}

    margins: dict[str, float] = {}
    for name in pair_names(engine_version):
        try:
            blunder = probes[f"{name}-{BLUNDER}"][danger]
            safe = probes[f"{name}-{SAFE}"][danger]
        except KeyError as error:
            raise ProbePairError(f"the certifying corpus is missing {name}") from error
        margin = blunder - safe
        if margin < CERTIFICATION_MARGIN:
            raise ProbePairError(
                f"a pair no longer separates the hanging queen from its twin: {name}"
            )
        margins[name] = float(margin)
    return margins


def visible_differences(engine_version: str = contract.DEFAULT_ENGINE_VERSION) -> dict[str, dict]:
    """Which `rich-9` columns actually differ within each pair, and by how much.

    The diagnostic that says what a pass or a failure here means. If a pair differs in one column,
    the model's answer on it is a function of that column alone.
    """
    columns = list(contract.columns(engine_version))
    out: dict[str, dict] = {}
    for name, pair in load_pairs(engine_version).items():
        blunder = pair[BLUNDER]["features"]
        safe = pair[SAFE]["features"]
        out[name] = {
            column: {BLUNDER: blunder[index], SAFE: safe[index]}
            for index, column in enumerate(columns)
            if blunder[index] != safe[index]
        }
    return out


def evaluate(model_path: str | Path, engine_version: str = contract.DEFAULT_ENGINE_VERSION) -> dict:
    """Score every pair and report which the ranker got right.

    A pass is a strict `score(safe) > score(blunder)`. A tie is a failure: the seam keeps the
    top `k` and a ranker that cannot separate these two has not expressed a preference between
    a blunder and the move that avoids it.
    """
    certified = certify(engine_version)
    pairs = load_pairs(engine_version)
    features = np.stack(
        [pairs[name][role]["features"] for name in pairs for role in (BLUNDER, SAFE)]
    ).astype(np.float32)
    scores = contract.score(model_path, features, engine_version=engine_version)

    results = []
    for index, name in enumerate(pairs):
        blunder, safe = float(scores[2 * index]), float(scores[2 * index + 1])
        results.append(
            {
                "pair": name,
                "blunder": blunder,
                "safe": safe,
                "margin": safe - blunder,
                "passed": safe > blunder,
                "certified_danger_margin": certified[name],
            }
        )
    passed = sum(1 for result in results if result["passed"])
    return {
        "engine_version": engine_version,
        "pairs": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
