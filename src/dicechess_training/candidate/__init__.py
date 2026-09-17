"""Producer for a candidate package the frozen benchmark can read.

The benchmark of Issue #13 knows how to refuse a package; nothing produced one, so the first
real candidate would have been assembled by hand. This package trains a candidate under the
mechanics the ablation protocol froze and writes the exact artifact set
`benchmark.core.load_candidate` validates, with every digest computed from the artifacts
themselves.
"""

from .build import CandidateConfig, build_candidate

__all__ = ["CandidateConfig", "build_candidate"]
