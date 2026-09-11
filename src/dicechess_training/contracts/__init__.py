"""Train == serve contracts: the exact feature layouts, tensor names, manifest rules and
probability perspectives that a served model must satisfy, mirrored fail-closed on the Python
side. Feature *values* are never reimplemented here; the engine is their single source of truth
and the golden corpora under ``tests/fixtures/`` carry its answers."""
