"""Pin complete Markdown output on explicitly synthetic formatting fixtures."""

import copy
import hashlib

import pytest

from dicechess_training.ablation.report import (
    _render_extraction_cost,
    _render_next_actions,
    render_markdown_report,
)
from report_fixtures import REPORT_CASES, synthetic_report

GOLDEN_HASHES = {
    "none-minimal": "ab793db58d1fe3f484259c4d91daf8846d5ac962bc413298d2d98f3f98280124",
    "baseline-minimal": "8934fa7d3dfabb8f1510e7b01ba123ba6d558211b89ff3d7be6c4a878f4a934e",
    "candidate-minimal": "2a9a4f67808550f0fedb4f72518dea7b32a2b88e9c2614d2bf30bcbb2cb17e04",
    "none-detailed": "ca0dd60d72ffccb47fd8721aef9f94267ef0546098101c2b2dd6881f65d0768c",
    "baseline-detailed": "36bbcc04293dbb3a221b1b9fd07ade4f9b59a751cdff04839340b8582a8dfcd9",
    "candidate-detailed": "3c6c9c3d9a5e678c85a03e3c2114e2a666cd4f2858ce35b3e5a2c3db8e4066ce",
}


@pytest.mark.parametrize(("selected", "detailed", "case"), REPORT_CASES)
def test_complete_report_bytes_and_input_immutability(selected, detailed, case):
    report = synthetic_report(selected, detailed)
    before = copy.deepcopy(report)
    rendered = render_markdown_report(report)

    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == GOLDEN_HASHES[case]
    assert report == before
    assert rendered.startswith("# Feature Schema Ablation Report (Issue #17)\n")
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


@pytest.mark.parametrize(
    ("render", "args"), [(_render_next_actions, ()), (_render_extraction_cost, (None,))]
)
def test_static_sections_return_independently_mutable_lists(render, args):
    first = render(*args)
    expected = first.copy()
    first.append("fixture-only mutation")
    assert render(*args) == expected
