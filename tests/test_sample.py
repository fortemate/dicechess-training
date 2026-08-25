"""Conformance tests for the committed bot-vs-bot sample."""

import pytest

from dicechess_training import demo, features, schema

pytestmark = pytest.mark.skipif(
    not demo.SAMPLE_DIR.exists(), reason="sample dataset not present (installed package)"
)


@pytest.fixture(scope="module")
def sample():
    return schema.read_shards(str(demo.SAMPLE_DIR))


def test_sample_conforms_to_schema(sample):
    schema.validate_frame(sample)
    assert len(sample) > 10_000
    assert sample["game_id"].nunique() > 1_000


def test_sample_encodes_without_errors(sample):
    matrix = features.encode_frame(sample.head(500))
    assert matrix.shape == (500, features.FEATURE_DIM)


def test_sample_labels_are_consistent_within_a_game(sample):
    """One game has one outcome; the two sides' labels must sum to 1."""
    for game_id, game in sample.head(5_000).groupby("game_id"):
        labels = {side: rows["result"].unique() for side, rows in game.groupby("side")}
        for side, values in labels.items():
            assert len(values) == 1, f"{game_id}: inconsistent label for side {side}"
        if len(labels) == 2:
            assert labels["w"][0] + labels["b"][0] == pytest.approx(1.0)


def test_sample_has_both_sides_and_real_outcomes(sample):
    assert set(sample["side"].unique()) == {"w", "b"}
    assert set(sample["result"].unique()) <= {0.0, 0.5, 1.0}
    assert sample["result"].mean() == pytest.approx(0.5, abs=0.05)
