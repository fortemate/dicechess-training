"""Pytest global configuration and fixtures.

Two separate concerns, and it is worth keeping them apart.

The autouse fixture collects garbage after every test so that native C++ resources — PyTorch
thread pools, ONNX Runtime inference sessions — are released while the interpreter is still
healthy rather than during its shutdown. That part works: instrumenting a run of the
ONNX-exercising files shows no live `InferenceSession` objects left when the session ends.

What it cannot reach is ONNX Runtime's own process-global state, whose static destructors run
after the last line of Python. On macOS those intermittently abort with
`recursive_mutex lock failed: Invalid argument`, or segfault, *after* every test has already
passed — so a green run is reported as a failed gate. Measured on this repository (Issue #53):
the ONNX-exercising files crash on roughly three runs in ten, the remaining files on none in ten,
and neither the task runner nor the thread count moves that rate. There is no public API to shut
ONNX Runtime down deterministically, so the suite cannot fix this from inside.

`pytest_sessionfinish` therefore leaves by `os._exit` once the verdict exists. This does not hide a
test failure: the verdict is already decided and is carried out in the exit status. It does skip
interpreter finalisation, which is why it is confined to the platform where the abort happens —
elsewhere finalisation still runs, and would still report a teardown defect if one appeared.
"""

from __future__ import annotations

import gc
import os
import sys

import pytest

#: Only Darwin has been observed to abort; leaving finalisation intact elsewhere keeps the rest of
#: the fleet, and CI, as a canary for a teardown defect that is genuinely ours.
_SKIP_FINALISATION = sys.platform == "darwin"


@pytest.fixture(autouse=True)
def _cleanup_native_resources():
    yield
    gc.collect()


_verdict: list[int] = []


def pytest_sessionfinish(session, exitstatus):
    """Remember the verdict; leaving is `pytest_unconfigure`'s job."""
    _verdict.append(int(exitstatus))


def pytest_unconfigure(config):
    """Leave without running native static destructors, after pytest has finished speaking.

    `pytest_sessionfinish` is too early even with `trylast`: the summary line — the counts an
    operator actually reads — is written after it, and exiting there swallows it. `unconfigure` is
    the last hook of the run, so everything pytest prints has already been printed.
    """
    if not _SKIP_FINALISATION or not _verdict:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_verdict[-1])
