"""Invented formatting inputs, not measured experiment results or tuning values."""

REPORT_CASES = [
    (None, False, "none-minimal"),
    ("S0", False, "baseline-minimal"),
    ("S2", False, "candidate-minimal"),
    (None, True, "none-detailed"),
    ("S0", True, "baseline-detailed"),
    ("S2", True, "candidate-detailed"),
]


def _summary():
    return {
        f"{metric}_{stat}": value
        for metric in ("log_loss", "brier", "ece")
        for stat, value in (("mean", 0.5), ("std", 0.0))
    }


def _schema(key, schema_id, count):
    return {
        "schema_id": schema_id,
        "feature_count": count,
        "single_model_summary": _summary(),
        "single_model_slices": {
            "slice-b": {"log_loss_mean": 0.5, "brier_mean": 0.5},
            "slice-a": {"log_loss_mean": 0.5, "brier_mean": 0.5},
        },
        "ensemble_diagnostic": {
            "scores": {
                "log_loss": 0.5,
                "brier": 0.5,
                "ece": 0.5,
                "calibration": [
                    {"lower": 0.0, "upper": 0.5, "count": 0, "prediction": 0.0, "observed": 0.0},
                    {"lower": 0.5, "upper": 1.0, "count": 2, "prediction": 0.5, "observed": 0.5},
                ],
            },
            "probes": {"checks": {f"only-{key}": key == "S0", "shared": True}},
        },
    }


def _extraction_fixture():
    return {
        "engine_artifact": "formatting-fixture",
        "runtime": {
            "java_version": "fixture-java",
            "java_vendor": "fixture-vendor",
            "os_name": "fixture-os",
            "os_arch": "fixture-arch",
        },
        "config": {"sample_iterations": 1, "warmup_iterations": 0},
        "timestamp": "fixture-time",
        "probes": {
            name: {
                "schemas": {
                    key: {"median_us": baseline + offset}
                    for key, offset in (("S0", 0.0), ("S1", 1.0), ("S2", 2.0))
                }
            }
            for name, baseline in (("start-w", 0.0), ("bare-kings", 1.0))
        },
    }


def _add_details(report, selected):
    report["split_summary"].update(
        leakage={"train:validation": 1, "train:test": 0, "validation:test": 0},
        unseen_val_positions=1,
    )
    report["input_shard_digests"] = {
        "S2": {"b.parquet": "b" * 64},
        "S0": {"b.parquet": "b" * 64, "a.parquet": "a" * 64},
    }
    report["input_shard_content_digests"] = {"S0": {"a.parquet": "c" * 64}}
    report["no_information_reference"] = {
        "metric": "log_loss",
        "reference": "synthetic constant",
        "prediction": 0.5,
        "scores": {"log_loss": 0.5, "brier": 0.5, "ece": 0.5},
    }
    schemas = report["schemas"]
    schemas["S0"]["admissibility"] = {"admissible": True, "margin": 0.0}
    schemas["S1"]["admissibility"] = {"admissible": False, "margin": -0.5}
    schemas["S0"]["training"] = {
        "feature_standardisation": "synthetic",
        "epoch_selection": (
            {"method": "fixed", "epochs": 1}
            if selected == "S0"
            else {"method": "inner", "per_seed": [{"seed": 0, "selected_epochs": 1}]}
        ),
    }
    for key in ("S0", "S1", "S2"):
        schemas[key]["single_model_unseen"] = {"count": 1, **_summary()}
    for key in ("S1", "S2"):
        report["gate_evaluations"][key]["checks"]["beats_no_information"] = key == "S1"
        report["gate_evaluations"][key]["unseen_evaluation"] = {
            "passed": key == "S2",
            "log_loss_delta_ci_95": [-0.5, 0.5],
        }
    report["extraction_cost"] = _extraction_fixture()


def synthetic_report(selected, detailed):
    schemas = {
        key: _schema(key, schema_id, count)
        for key, schema_id, count in (
            ("S0", "kcp-13", 13),
            ("S1", "kcp-mobility-27-v1", 27),
            ("S2", "kcp-mobility-pawns-31-v1", 31),
        )
    }
    del schemas["S1"]["single_model_slices"]["slice-b"]
    report = {
        "protocol_version": "formatting-fixture",
        "protocol_sha256": "0" * 64,
        "engine_version": "fixture-engine",
        "schemas": schemas,
        "gate_evaluations": {
            key: {
                "cleared": selected == key,
                "log_loss_delta_ci_95": None if selected is None else [-0.5, 0.5],
                "relative_log_loss_gain": 0.0,
                "checks": {},
            }
            for key in ("S1", "S2")
        },
        "decision": {
            "selected_schema": selected,
            "selected_schema_id": None if selected is None else schemas[selected]["schema_id"],
            "inadmissible_reason": "synthetic fixture",
            "private_qualification_ref": "Formatting fixture",
        },
        "split_summary": {
            "total_positions": 10,
            "decisive_positions": 10,
            "train_positions": 8,
            "val_positions": 1,
            "test_positions": 1,
            "train_games": 8,
            "val_games": 1,
            "test_games": 1,
        },
    }
    if detailed:
        _add_details(report, selected)
    return report
