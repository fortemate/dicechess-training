"""Pytest global configuration and fixtures.

Ensures native C++ resources (PyTorch thread pools, ONNX Runtime inference sessions)
are garbage collected before process teardown, preventing native shutdown aborts.
"""

from __future__ import annotations

import gc

import pytest


@pytest.fixture(autouse=True)
def _cleanup_native_resources():
    yield
    gc.collect()
