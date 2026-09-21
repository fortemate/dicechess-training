"""Choosing roots is a decision with a price, so these tests are about the decision, not the file.

Labelling one root costs the champion's evaluation on every turn it allows. What matters is
therefore that the same source yields the same roots, that no root is paid for twice, and that a
bigger sample contains the smaller one — because if it does not, raising the limit throws away
every core-hour already spent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dicechess_training import schema
from dicechess_training.prerank import group_id_for
from dicechess_training.prerank.roots import (
    HEADER,
    PROVENANCE_FILE,
    ROOTS_FILE,
    RootsError,
    canonical_dice,
    distinct_roots,
    export,
    order_key,
    select,
    write_roots,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
OTHER = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -"
SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "playsite-bots-v0"


def _rows(*rows: tuple[str, int, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": game, "ply": ply, "fen": fen, "dice": dice, "side": side, "result": 0.5}
            for game, ply, fen, dice, side in rows
        ],
        columns=["game_id", "ply", "fen", "dice", "side", "result"],
    )


def test_the_same_root_is_paid_for_once() -> None:
    frame = _rows(
        ("game-b", 7, START, "BNP", "w"),
        ("game-a", 3, START, "BNP", "w"),
        ("game-c", 1, OTHER, "BNP", "b"),
    )
    roots = distinct_roots(frame)
    assert len(roots) == 2


def test_a_repeated_root_is_charged_to_its_earliest_occurrence() -> None:
    """Which game a root is attributed to decides its split, so it has to be a rule."""
    frame = _rows(
        ("game-b", 2, START, "BNP", "w"),
        ("game-a", 9, START, "BNP", "w"),
    )
    roots = distinct_roots(frame)
    assert list(roots["game_id"]) == ["game-a"]
    assert list(roots["ply"]) == [9]


def test_repetition_is_judged_by_the_position_not_the_spelling() -> None:
    frame = _rows(
        ("game-a", 1, "4k3/8/8/8/8/8/8/4K3 w KQkq -", "PPP", "w"),
        ("game-b", 1, "4k3/8/8/8/8/8/8/4K3 w qkQK - 0 1", "PPP", "w"),
    )
    assert len(distinct_roots(frame)) == 1


def test_the_order_is_a_function_of_the_root_alone() -> None:
    assert order_key(START, "BNP") == order_key(START + " 0 1", "BNP")
    assert order_key(START, "BNP") != order_key(START, "BNQ")
    assert order_key(START, "BNP") != order_key(OTHER, "BNP")


# Raising the limit must add roots, never move them: a corpus is expensive, and a sample that
# reshuffles when it grows throws away every label already computed.
def test_a_bigger_sample_contains_the_smaller_one() -> None:
    frame = schema.read_shards(str(SAMPLE))
    small = select(frame, 50)
    large = select(frame, 200)
    assert len(small) == 50
    assert len(large) == 200
    assert list(small["fen"]) == list(large["fen"])[:50]
    assert list(small["dice"]) == list(large["dice"])[:50]


def test_the_same_source_yields_the_same_roots() -> None:
    frame = schema.read_shards(str(SAMPLE))
    first = select(frame, 120)
    second = select(frame.sample(frac=1.0, random_state=3).reset_index(drop=True), 120)
    assert list(first["fen"]) == list(second["fen"])
    assert list(first["game_id"]) == list(second["game_id"])


def test_no_two_selected_roots_share_a_group_id() -> None:
    roots = select(schema.read_shards(str(SAMPLE)), 500)
    ids = [group_id_for(fen, dice) for fen, dice in zip(roots["fen"], roots["dice"], strict=True)]
    assert len(set(ids)) == len(ids)


def test_a_row_whose_side_contradicts_its_fen_is_refused() -> None:
    frame = _rows(("game-a", 1, START, "BNP", "b"))
    with pytest.raises(RootsError, match="side"):
        distinct_roots(frame)


def test_an_empty_source_is_refused() -> None:
    empty = _rows()
    with pytest.raises(RootsError, match="no rows"):
        distinct_roots(empty)


def test_a_source_missing_a_column_is_refused() -> None:
    frame = _rows(("game-a", 1, START, "BNP", "w")).drop(columns=["dice"])
    with pytest.raises(RootsError, match="dice"):
        distinct_roots(frame)


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_that_selects_nothing_is_refused(limit: int) -> None:
    frame = _rows(("game-a", 1, START, "BNP", "w"))
    with pytest.raises(RootsError, match="positive"):
        select(frame, limit)


def test_the_file_is_what_the_generator_reads(tmp_path: Path) -> None:
    roots = distinct_roots(
        _rows(("game-a", 1, START, "BNP", "w"), ("game-b", 2, OTHER, "KNQ", "b"))
    )
    digest = write_roots(roots, tmp_path / ROOTS_FILE)
    lines = (tmp_path / ROOTS_FILE).read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == list(HEADER)
    assert len(lines) == 3
    for line in lines[1:]:
        game, fen, dice, side = line.split("\t")
        assert fen.split()[1] == side
        assert list(dice) == sorted(dice)
        assert game
    assert len(digest) == 64


# `\r` counts with the others because the generator reads this file with `Files.readAllLines`,
# which ends a line on a lone carriage return: a field carrying one becomes two records rather
# than an error, and the corpus is then built from roots nobody chose.
@pytest.mark.parametrize("bad", ["game\ta", "game\na", "game\ra"])
def test_a_field_that_would_break_the_file_is_refused(tmp_path: Path, bad: str) -> None:
    roots = distinct_roots(_rows((bad, 1, START, "BNP", "w")))
    with pytest.raises(RootsError, match="tab, newline or carriage return"):
        write_roots(roots, tmp_path / ROOTS_FILE)


def test_two_spellings_of_one_root_resolve_the_same_way_whatever_the_input_order() -> None:
    """The same position, game and ply, written two ways — castling reordered, clocks on one.

    Nothing downstream distinguishes them, so either could be kept; what may not happen is the
    choice depending on which row the file held first. `roots.tsv` is digested into the dataset's
    `source_sha256`, so an order-dependent selection is an order-dependent dataset identity.
    """
    spellings = [
        ("game-a", 4, "4k3/8/8/8/8/8/8/4K3 w KQkq -", "PPP", "w"),
        ("game-a", 4, "4k3/8/8/8/8/8/8/4K3 w qkQK - 0 1", "PPP", "w"),
    ]
    forward = distinct_roots(_rows(*spellings))
    backward = distinct_roots(_rows(*reversed(spellings)))
    assert len(forward) == len(backward) == 1
    assert list(forward["fen"]) == list(backward["fen"])


def test_export_records_what_it_read_and_what_it_chose(tmp_path: Path) -> None:
    record = export(SAMPLE, tmp_path, limit=40)
    assert (tmp_path / ROOTS_FILE).is_file()

    written = json.loads((tmp_path / PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert written == record
    assert record["roots"] == 40
    assert record["rows_read"] == 49000
    assert record["distinct_roots"] == 43692
    assert record["limit"] == 40
    assert sum(record["splits"].values()) == 40
    assert set(record["sides"]) <= {"w", "b"}
    assert record["ply"]["min"] <= record["ply"]["median"] <= record["ply"]["max"]

    # The roots file IS the producer's reproduction record, and the manifest binds it by this
    # digest — so the record has to carry the digest of the bytes that were actually written.
    import hashlib

    assert (
        record["roots_sha256"] == hashlib.sha256((tmp_path / ROOTS_FILE).read_bytes()).hexdigest()
    )

    # And the shards it came from, by a digest that survives a Parquet footer being rewritten.
    assert [entry["file"] for entry in record["shards"]] == [
        "playsite-bots-sample-000.parquet",
        "playsite-bots-sample-001.parquet",
    ]
    assert all(len(entry["content_sha256"]) == 64 for entry in record["shards"])


def test_export_without_a_limit_takes_every_distinct_root(tmp_path: Path) -> None:
    record = export(SAMPLE, tmp_path)
    assert record["limit"] is None
    assert record["roots"] == record["distinct_roots"] == 43692


# The analytics dialect writes the roll in lower case on Black's turns — the case repeats `side`,
# which the row already carries. `schema.validate_frame` checks that dialect without rewriting
# it, so shards written outside `ingest.convert_export` keep it. Measured on the 100k development
# corpus: 723,499 of 1,500,477 rows, exactly the Black-to-move ones. The committed public sample
# has none, which is why nothing noticed until a second corpus arrived.
@pytest.mark.parametrize(
    ("raw", "canonical"),
    [("bpr", "BPR"), ("BPR", "BPR"), ("nnn", "NNN"), ("rqb", "BQR"), ("KKP", "KKP")],
)
def test_a_roll_is_canonical_however_the_shard_spelt_it(raw: str, canonical: str) -> None:
    assert canonical_dice(raw) == canonical


@pytest.mark.parametrize("bad", ["", "BP", "BPRK", "BPZ", "123"])
def test_a_roll_that_is_not_three_piece_letters_is_refused(bad: str) -> None:
    with pytest.raises(RootsError, match="three piece letters") as refusal:
        canonical_dice(bad)
    # The offending value is in the message: a corpus has a million rows, and an operator who
    # cannot see which roll was rejected cannot find them.
    assert repr(bad) in str(refusal.value)


def test_one_roll_spelt_two_ways_is_one_root() -> None:
    """Left raw, a Black-to-move root would become two roots under two spellings — and every one
    of them would be refused on admission, after the corpus had cost its core-hours."""
    frame = _rows(
        ("game-a", 2, OTHER, "bnp", "b"),
        ("game-b", 4, OTHER, "BNP", "b"),
    )
    roots = distinct_roots(frame)
    assert len(roots) == 1
    assert list(roots["dice"]) == ["BNP"]


def test_the_written_file_carries_the_canonical_roll(tmp_path: Path) -> None:
    roots = distinct_roots(_rows(("game-a", 2, OTHER, "rqb", "b")))
    write_roots(roots, tmp_path / ROOTS_FILE)
    line = (tmp_path / ROOTS_FILE).read_text(encoding="utf-8").splitlines()[1]
    assert line.split("\t")[2] == "BQR"


def test_normalising_does_not_move_a_corpus_that_was_already_canonical() -> None:
    """The committed sample is upper-case and sorted throughout, so its selection — and therefore
    the `source_sha256` of anything already built from it — must be untouched by this."""
    roots = select(schema.read_shards(str(SAMPLE)), 400)
    assert list(roots["dice"]) == [canonical_dice(dice) for dice in roots["dice"]]
