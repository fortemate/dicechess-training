"""Synthetic contract, leakage, and terminal-bypass checks; no private labels."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from dicechess_training.prerank.experimental import (
    COLUMNS_A,
    COLUMNS_B,
    FEATURE_SCHEMA,
    KingRelativeMLP,
    digest,
    load,
    make_model,
    pack,
    score_candidates,
)


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for g in range(3):
        rows = []
        for i in range(3):
            terminal = i == 2
            a = [float(g + i)] * 9
            rows.append(
                dict(
                    a=a,
                    b=a + [1.0, 0.0, 2.0, 1.0],
                    c=[[], []] if terminal else [[8], [19848]],
                    terminal=terminal,
                    target=2**31 - 1 if terminal else i,
                    moves=[f"a{i + 1}a{i + 2}"],
                )
            )
        records.append(dict(group_id=str(g), rows=rows))
    data = source / "features.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = dict(
        schema=FEATURE_SCHEMA,
        engine_version="0.14.0",
        source_sha256="0" * 64,
        features_sha256=digest(data),
        columns_a=COLUMNS_A,
        columns_b=COLUMNS_B,
        groups=3,
        candidates=9,
    )
    (source / "manifest.json").write_text(json.dumps(manifest))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"0": "train", "1": "early_stop", "2": "screen"}))
    destination = tmp_path / "packed"
    pack(source, splits, destination, source_sha256="0" * 64, splits_sha256=digest(splits))
    return source, splits, destination


def corpus(bundle):
    return load(bundle[2], manifest_sha256=digest(bundle[2] / "manifest.json"))


def test_roundtrip_preserves_lists_splits_terminals_and_canonical_ties(bundle):
    data = corpus(bundle)
    np.testing.assert_array_equal(data.offsets, [0, 3, 6, 9])
    np.testing.assert_array_equal(data.of("screen"), [2])
    np.testing.assert_array_equal(data.tie_order, [0, 1, 2] * 3)
    assert data.terminal.sum() == 3
    assert np.all(data.c[data.terminal] == -1)
    with pytest.raises(ValueError, match="unknown"):
        data.of("acceptance")


def test_train_only_nonterminal_statistics(bundle):
    data = corpus(bundle)
    model = make_model(data, "A")
    np.testing.assert_array_equal(model.feature_mean.numpy(), [0.5] * 9)
    np.testing.assert_array_equal(model.feature_scale.numpy(), [0.5] * 9)
    data.a[3:] = 999
    data.b[3:, :9] = 999
    again = make_model(data, "A")
    torch.testing.assert_close(model.feature_mean, again.feature_mean)
    torch.testing.assert_close(model.feature_scale, again.feature_scale)
    assert make_model(data, "B").net[0].in_features == 13
    assert isinstance(make_model(data, "C"), KingRelativeMLP)


def test_each_model_backpropagates_on_synthetic_nonterminal_input(bundle):
    data = corpus(bundle)
    for arm, array in (("A", data.a), ("B", data.b), ("C", data.c)):
        model = make_model(data, arm)
        values = model(torch.from_numpy(array[~data.terminal]))
        values.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_terminal_never_reaches_model_even_with_missing_king_ids():
    class Spy(nn.Module):
        def forward(self, features):
            assert len(features) == 1
            assert features[0, 0] == 42
            return torch.tensor([[3.0]])

    scores = score_candidates(
        Spy(), torch.tensor([[float("nan")], [42.0]]), torch.tensor([True, False])
    )
    assert torch.isposinf(scores[0])
    assert scores[1] == 3
    scores = score_candidates(Spy(), torch.zeros(2, 1), torch.tensor([True, True]))
    assert torch.isposinf(scores).all()


def test_padding_does_not_add_embedding_zero_and_order_is_irrelevant():
    model = KingRelativeMLP()
    ids = torch.full((2, 2, 64), -1)
    ids[0, 0, :2] = torch.tensor([8, 9])
    ids[1, 0, :2] = torch.tensor([9, 8])
    torch.testing.assert_close(model(ids)[0], model(ids)[1])
    empty = model(torch.full((1, 2, 64), -1))
    with torch.no_grad():
        model.embedding.weight[0] += 1000
    torch.testing.assert_close(empty, model(torch.full((1, 2, 64), -1)))


def test_hashes_and_output_reuse_fail_closed(bundle, tmp_path):
    source, splits, destination = bundle
    with pytest.raises(ValueError, match="manifest digest"):
        load(destination, manifest_sha256="1" * 64)
    split_hash = digest(splits)
    bad = tmp_path / "bad"
    with pytest.raises(FileExistsError):
        pack(source, splits, destination, source_sha256="0" * 64, splits_sha256=split_hash)
    with (source / "features.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="feature digest"):
        pack(source, splits, bad, source_sha256="0" * 64, splits_sha256=split_hash)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind", ["mask", "offset", "prefix", "id", "split"])
def test_rejects_corrupt_arrays(bundle, kind):
    data = corpus(bundle)
    if kind == "mask":
        data.terminal[0] = True
    elif kind == "offset":
        data.offsets[1] = 0
    elif kind == "prefix":
        data.b[0, 0] = 500
    elif kind == "id":
        data.c[0, 0, 0] = 19848
    else:
        data.splits[0] = "acceptance"
    with pytest.raises(ValueError):
        data.validate()
