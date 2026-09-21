"""The grouped dataset the move pre-ranker learns from.

A pre-ranker is judged on an *ordering within one root*, so the unit of this data is a list, not a
row: one root position with its dice and every legal turn it allows, each carrying the student's
cheap features and the teacher's expensive score. Keeping that shape in the file — rather than a
flat table with a group column — is what makes it impossible to filter, reorder or shuffle the data
into something that no longer has lists in it.
"""

from __future__ import annotations

from .groups import (
    SCHEMA_ID,
    GroupsError,
    assign_splits,
    golden_columns,
    group_id_for,
    load_groups,
    root_key,
)

__all__ = [
    "SCHEMA_ID",
    "GroupsError",
    "assign_splits",
    "golden_columns",
    "group_id_for",
    "load_groups",
    "root_key",
]
