"""Packing is the seam between two repositories, so these tests are about what crosses it.

The generator is a separate codebase in a separate language. It states the engine it ran, the
teacher it used and the digest of its input; everything the *contract* defines — the column
layout, the golden digest, the shape of the manifest — is computed here from committed evidence,
never copied from what arrived. Each test below is one way a record could be wrong, or one thing
the packer must refuse to take the producer's word for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dicechess_training.benchmark import core as benchmark_core
from dicechess_training.prerank import GroupsError, group_id_for
from dicechess_training.prerank.pack import (
    GENERATION_FILE,
    PackError,
    build_manifest,
    pack,
    read_generation,
    summarise,
)

ENGINE_VERSION = "0.12.0"
SCHEMA = "rich-9-v1"
GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rich9"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
TERMS = "Fortemate owner-controlled; not approved for public redistribution"

DICE = ["BNP", "BKN", "KNQ", "NPR", "PQR", "BNQ", "KPR", "NPP", "BBN", "KQR", "BBP", "NNP"]


def _columns() -> list[str]:
    raw = json.loads((GOLDEN / f"golden-engine-{ENGINE_VERSION}.json").read_text(encoding="utf-8"))
    return list(raw["columns"])


def _group(index: int, candidates: int = 3) -> dict:
    dice = DICE[index % len(DICE)]
    return {
        "group_id": group_id_for(START, dice),
        "game_id": f"game-{index}",
        "root_fen": START,
        "dice": dice,
        "side": "w",
        "candidates": [
            {
                "moves": ["e2e4"],
                "result_fen": f"position-{index}-{seat}",
                "features": [0.0] * len(_columns()),
                "target": float(seat * 10),
            }
            for seat in range(candidates)
        ],
    }


def _record(**overrides) -> dict:
    record = {
        "generator": "dicechess-hunter (dicechess.hunter.PrerankGroupsMain)",
        "schema": "playground-prerank-groups-v1",
        "feature_schema": SCHEMA,
        "engine_version": ENGINE_VERSION,
        "columns": _columns(),
        "teacher": {
            "id": "hunter-baseline-v1",
            "weights_sha256": "5a" * 32,
            "resolved_config_hash": "sha256:" + "5a" * 32,
        },
        "roots_file": "roots.tsv",
        "roots_sha256": "bd" * 32,
        "roots_read": 12,
        "groups": 10,
        "candidates": 30,
        "collapsed_transpositions": 41,
        "skipped": {"unparsable": 0, "side_mismatch": 0, "forced_pass": 2},
    }
    record.update(overrides)
    return record


def _corpus(
    directory: Path,
    groups: list[dict] | None = None,
    *,
    with_terms: bool = True,
    **record_overrides,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if with_terms:
        (directory / "license.txt").write_text("Owner-controlled use.\n", encoding="utf-8")
    payload = [_group(index) for index in range(10)] if groups is None else groups
    (directory / "groups.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (directory / GENERATION_FILE).write_text(
        json.dumps(_record(**record_overrides), indent=2) + "\n", encoding="utf-8"
    )
    return directory


def _terms(tmp_path: Path) -> Path:
    path = tmp_path / "terms.txt"
    path.write_text("Owner-controlled use. Not approved for public redistribution.\n", "utf-8")
    return path


def test_a_generated_corpus_becomes_an_admissible_dataset(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus")
    manifest, summary = pack(corpus, license_=TERMS, license_file=_terms(tmp_path))
    assert manifest["schema"] == "playground-prerank-groups-v1"
    assert manifest["groups"] == 10
    assert manifest["candidates"] == 30
    assert summary["groups"] == 10
    # The point of packing here: the manifest was written and then admitted by the same loader
    # that will admit it later, so a manifest this command writes cannot be one that fails.
    assert (corpus / "manifest.json").is_file()
    assert (corpus / "license.txt").is_file()


def test_the_layout_comes_from_committed_evidence_not_from_the_producer(tmp_path: Path) -> None:
    """A producer does not get to state the layout it was supposed to produce."""
    corpus = _corpus(tmp_path / "corpus", columns=["invented"] * 9)
    with pytest.raises(PackError, match="not the engine's layout"):
        build_manifest(
            corpus,
            read_generation(corpus),
            version="0.1.0",
            kind="owner-controlled",
            license_=TERMS,
        )


def test_a_record_that_names_no_columns_still_packs(tmp_path: Path) -> None:
    """The columns are the golden's to state, so a record that omits them is not a problem."""
    corpus = _corpus(tmp_path / "corpus")
    record = _record()
    record.pop("columns")
    (corpus / GENERATION_FILE).write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = build_manifest(
        corpus, read_generation(corpus), version="0.1.0", kind="owner-controlled", license_=TERMS
    )
    assert manifest["columns"] == _columns()


def test_the_golden_digest_binds_the_committed_fixture(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus")
    manifest, _ = pack(corpus, license_=TERMS, license_file=_terms(tmp_path))
    raw = json.loads((GOLDEN / f"golden-engine-{ENGINE_VERSION}.json").read_text(encoding="utf-8"))
    assert manifest["golden_sha256"] == benchmark_core.digest(raw)


def test_the_roots_digest_becomes_the_datasets_source(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus")
    manifest, _ = pack(corpus, license_=TERMS, license_file=_terms(tmp_path))
    assert manifest["source_sha256"] == "bd" * 32


def test_the_teacher_travels_as_an_id_and_a_digest(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus")
    manifest, _ = pack(corpus, license_=TERMS, license_file=_terms(tmp_path))
    assert manifest["teacher"] == {"id": "hunter-baseline-v1", "weights_sha256": "5a" * 32}
    # Nothing else from the record rides along: the resolved-config hash is operational, not part
    # of the dataset contract, and a manifest is not a place to park extra fields nobody validates.
    assert set(manifest["teacher"]) == {"id", "weights_sha256"}


def test_an_engine_with_no_committed_golden_is_refused(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus", engine_version="9.9.9")
    with pytest.raises(GroupsError, match="no committed golden corpus"):
        build_manifest(
            corpus,
            read_generation(corpus),
            version="0.1.0",
            kind="owner-controlled",
            license_=TERMS,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"schema": "something-else"}, "another dataset schema"),
        ({"engine_version": ""}, "no engine_version"),
        ({"feature_schema": ""}, "no feature_schema"),
        ({"roots_sha256": "not-a-digest"}, "roots digest is unusable"),
        ({"teacher": {"id": "x"}}, "weights digest is unusable"),
        ({"teacher": {"weights_sha256": "5a" * 32}}, "teacher has no id"),
        ({"teacher": "champion"}, "names no teacher"),
    ],
)
def test_a_record_that_cannot_support_a_manifest_is_refused(
    tmp_path: Path, override: dict, message: str
) -> None:
    corpus = _corpus(tmp_path / "corpus", **override)
    with pytest.raises(PackError, match=message):
        read_generation(corpus)


def test_a_corpus_without_a_record_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "groups.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(PackError, match="no generation record"):
        read_generation(directory)


def test_data_terms_are_required_and_not_invented(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus", with_terms=False)
    with pytest.raises(PackError, match="no data terms beside it"):
        pack(corpus, license_=TERMS)


@pytest.mark.parametrize("license_", ["", "  ", "unknown", "unresolved"])
def test_a_licence_that_says_nothing_is_refused(tmp_path: Path, license_: str) -> None:
    corpus = _corpus(tmp_path / "corpus")
    with pytest.raises(PackError, match="missing data terms"):
        pack(corpus, license_=license_, license_file=_terms(tmp_path))


def test_an_unverified_origin_is_refused(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus")
    with pytest.raises(PackError, match="unverified dataset origin"):
        pack(corpus, license_=TERMS, license_file=_terms(tmp_path), kind="scraped")


def test_an_empty_corpus_is_refused(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus", groups=[])
    with pytest.raises(PackError, match="no groups"):
        pack(corpus, license_=TERMS, license_file=_terms(tmp_path))


def test_a_corpus_the_loader_would_reject_is_not_papered_over(tmp_path: Path) -> None:
    """The manifest is written before the corpus is admitted, so a bad corpus still fails —
    which is the point: the packer cannot make one admissible by describing it differently."""
    broken = _group(0)
    broken["candidates"][1]["result_fen"] = broken["candidates"][0]["result_fen"]
    corpus = _corpus(tmp_path / "corpus", groups=[broken])
    with pytest.raises(GroupsError, match="same position"):
        pack(corpus, license_=TERMS, license_file=_terms(tmp_path))


def test_the_summary_says_what_can_and_cannot_be_ranked() -> None:
    groups = [_group(0, candidates=1), _group(1, candidates=60), _group(2, candidates=49)]
    summary = summarise(groups, _record())
    assert summary["groups"] == 3
    assert summary["candidates"] == 110
    assert summary["candidates_per_group"] == {"min": 1, "median": 49, "p95": 60, "max": 60}
    # Only the groups larger than the shortlist are decisions a pre-ranker can get wrong.
    assert summary["groups_above_shortlist"] == 2
    assert summary["groups_entirely_tied"] == 1
    assert summary["collapsed_transpositions"] == 41
    assert summary["roots_skipped"]["forced_pass"] == 2


def test_the_summary_follows_the_shortlist_it_is_given() -> None:
    groups = [_group(0, candidates=10), _group(1, candidates=60)]
    assert summarise(groups, _record(), shortlist=8)["groups_above_shortlist"] == 2
    assert summarise(groups, _record(), shortlist=100)["groups_above_shortlist"] == 0
