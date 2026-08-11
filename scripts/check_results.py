#!/usr/bin/env python3
"""Extract fresh public measurements and enforce reproduction tolerances."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing reproduction result: {path}")
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_close(label: str, observed: float, expected: float, tolerance: float) -> None:
    if not math.isfinite(observed) or abs(observed - expected) > tolerance:
        raise SystemExit(
            f"FAIL {label}: observed {observed:.8g}, expected {expected:.8g} "
            f"within ±{tolerance:.8g}"
        )


def check_occupancy(output_root: Path, measurement_root: Path) -> None:
    output = output_root / "occupancy"
    result = load_json(output / "result.json")
    causality = load_json(output / "CAUSALITY.json")
    config = result["config"]
    accounting = result["parameter_accounting"]
    trained = result["trained_occupancy"]
    logical_rows = int(trained["logical_rows_per_sequence"])
    memory_layers = config["layout"].count("B")
    row_width = int(config["width"]) // int(config["memory_heads"])
    dense_sdm_bytes = logical_rows * row_width * 4
    rows = []
    for record in trained["prefixes"]:
        prefix = int(record["prefix_tokens"])
        mean_rows = float(record["unique_rows"]["mean"])
        private_bytes = float(record["mean_private_value_bytes_fp32"])
        rows.append(
            {
                "prefix_tokens": prefix,
                "mean_rows": mean_rows,
                "logical_rows": logical_rows,
                "median_rows": float(record["unique_rows"]["median"]),
                "p95_rows": float(record["unique_rows"]["p95"]),
                "maximum_rows": float(record["unique_rows"]["maximum"]),
                "mean_active_fraction": mean_rows / logical_rows,
                "mean_live_kib": private_bytes / 1024,
                "value_state_reduction": 1 - private_bytes / dense_sdm_bytes,
            }
        )
    fields = list(rows[0])
    write_csv(measurement_root / "trained_occupancy.csv", rows, fields)

    model = {
        "active_parameters": accounting["active_parameters"],
        "causality_maximum_prefix_difference": causality[
            "maximum_prefix_difference"
        ],
        "layout": config["layout"],
        "learned_initial_state_parameters": accounting[
            "learned_initial_state_parameters"
        ],
        "logical_runtime_mutable_state_bytes_fp32": accounting[
            "logical_mutable_state_bytes_fp32_per_sequence"
        ],
        "memory_heads": config["memory_heads"],
        "memory_layers": memory_layers,
        "optimizer_seconds": result["optimizer_seconds"],
        "optimizer_tokens_per_second": result["optimizer_tokens_per_second"],
        "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
        "reads": config["reads"],
        "row_width": row_width,
        "seed": config["seed"],
        "slots": config["slots"],
        "test_nll": result["final_test"]["loss"],
        "test_perplexity": result["final_test"]["perplexity"],
        "training_steps": config["steps"],
        "training_tokens": result["training_tokens"],
        "validation_nll": result["final_validation"]["loss"],
        "writes": config["writes"],
    }
    (measurement_root / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n"
    )

    expected_model = load_json(ROOT / "data/model.json")
    require_close("test NLL", model["test_nll"], expected_model["test_nll"], 0.03)
    if model["active_parameters"] != expected_model["active_parameters"]:
        raise SystemExit("FAIL active parameter count changed")
    if model["causality_maximum_prefix_difference"] != 0:
        raise SystemExit("FAIL prefix causality is not exact")
    with (ROOT / "data/trained_occupancy.csv").open(newline="") as handle:
        expected_rows = {
            int(row["prefix_tokens"]): row for row in csv.DictReader(handle)
        }
    observed_rows = {int(row["prefix_tokens"]): row for row in rows}
    if int(trained["examples"]) != 64:
        raise SystemExit(
            f"FAIL occupancy evaluated {trained['examples']} sequences, expected 64"
        )
    if observed_rows.keys() != expected_rows.keys():
        raise SystemExit(
            "FAIL occupancy prefix set changed: "
            f"observed {sorted(observed_rows)}, expected {sorted(expected_rows)}"
        )
    for prefix, row in observed_rows.items():
        expected = float(expected_rows[prefix]["mean_rows"])
        require_close(
            f"mean private rows at {prefix} tokens",
            row["mean_rows"],
            expected,
            max(12.0, expected * 0.03),
        )
    print(f"PASS occupancy; fresh measurements: {measurement_root}", flush=True)


def check_serving(output_root: Path, measurement_root: Path) -> None:
    result = load_json(output_root / "serving/result.json")
    if result["causality"]["maximum_prefix_difference"] != 0:
        raise SystemExit("FAIL serving prefix causality is not exact")
    parity = result["parity"]
    if (
        parity["dense_packed_state_maximum_difference"] != 0
        or not parity["dense_packed_outputs_within_bf16_tolerance"]
    ):
        raise SystemExit("FAIL dense-table and packed serving parity")
    lifecycle = result["continuous_batch_lifecycle"]
    if (
        lifecycle["maximum_state_difference"] != 0
        or lifecycle["maximum_output_difference"] != 0
    ):
        raise SystemExit("FAIL continuous-batching state lifecycle parity")

    rows = []
    for case in result["jit_serving_cases"]:
        rows.append(
            {
                "batch": int(case["batch"]),
                "growth_quantum_steps": int(case["growth_quantum_steps"]),
                "unmaterialized_logical_rows_pct": 100
                * (1 - float(case["live_row_fraction"])),
                "allocated_serving_state_reduction_pct": 100
                * (1 - float(case["packed_over_dense_reserved_bytes"])),
                "decode_latency_cost_pct": 100
                * (float(case["packed_over_dense_decode_latency"]) - 1),
                "dense_tokens_per_second": case["dense_tokens_per_second"],
                "packed_tokens_per_second": case["packed_tokens_per_second"],
                "total_growth_events": case["total_growth_events"],
                "total_value_rows_copied": case["total_growth_rows_copied"],
            }
        )
    rows.sort(key=lambda row: (row["batch"], row["growth_quantum_steps"]))
    write_csv(measurement_root / "runtime.csv", rows, list(rows[0]))

    with (ROOT / "data/runtime.csv").open(newline="") as handle:
        expected = {
            (int(row["batch"]), int(row["growth_quantum_steps"])): row
            for row in csv.DictReader(handle)
        }
    selected = {
        (row["batch"], row["growth_quantum_steps"]): row
        for row in rows
    }
    required = tuple(
        (batch, quantum)
        for batch in (1, 16, 64)
        for quantum in (1, 2, 4)
    )
    missing = [key for key in required if key not in selected]
    if missing:
        raise SystemExit(f"FAIL serving cases are missing: {missing}")

    for key in required:
        row = selected[key]
        reference = expected[key]
        require_close(
            f"{key} unmaterialized logical rows",
            row["unmaterialized_logical_rows_pct"],
            float(reference["unmaterialized_logical_rows_pct"]),
            1.0,
        )
        require_close(
            f"{key} allocated serving-state reduction",
            row["allocated_serving_state_reduction_pct"],
            float(reference["allocated_serving_state_reduction_pct"]),
            1.0,
        )
        if result["device"] == "NVIDIA A40" and not (
            0 <= row["decode_latency_cost_pct"] <= 20
        ):
            raise SystemExit(
                f"FAIL {key} A40 latency cost: "
                f"{row['decode_latency_cost_pct']:.3f}% is outside 0–20%"
            )

    one_step = selected[(64, 1)]
    four_step = selected[(64, 4)]
    allocation_trade = (
        float(one_step["allocated_serving_state_reduction_pct"])
        - float(four_step["allocated_serving_state_reduction_pct"])
    )
    if not 1.0 <= allocation_trade <= 5.0:
        raise SystemExit(
            "FAIL batch-64 growth quantum did not retain the expected "
            f"allocated-state trade-off: {allocation_trade:.3f} points"
        )
    if not (
        int(four_step["total_growth_events"])
        < int(one_step["total_growth_events"])
        and int(four_step["total_value_rows_copied"])
        < int(one_step["total_value_rows_copied"])
    ):
        raise SystemExit("FAIL pooled growth did not reduce allocator churn and copying")
    if result["device"] == "NVIDIA A40" and not (
        float(four_step["decode_latency_cost_pct"])
        < float(one_step["decode_latency_cost_pct"])
    ):
        raise SystemExit("FAIL pooled growth did not reduce A40 latency cost")
    if result["device"] != "NVIDIA A40":
        print(
            f"INFO latency is not compared across devices ({result['device']}); "
            "memory and correctness checks remain enforced",
            flush=True,
        )
    print(f"PASS serving; fresh measurements: {measurement_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("occupancy", "serving", "all"))
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "runs/reproduction"
    )
    args = parser.parse_args()
    measurement_root = args.output_root.resolve() / "measurements"
    suites = ("occupancy", "serving") if args.suite == "all" else (args.suite,)
    for suite in suites:
        if suite == "occupancy":
            check_occupancy(args.output_root.resolve(), measurement_root)
        else:
            check_serving(args.output_root.resolve(), measurement_root)


if __name__ == "__main__":
    main()
