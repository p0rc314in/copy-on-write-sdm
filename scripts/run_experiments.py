#!/usr/bin/env python3
"""Run the released occupancy and allocator experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], gpu: str) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("occupancy", "serving", "all"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "runs/data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/reproduction")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    suites = ("occupancy", "serving") if args.experiment == "all" else (args.experiment,)
    for suite in suites:
        output = args.output_root.resolve() / suite
        if suite == "occupancy":
            manifest = args.data_root.resolve() / "wikitext103_gpt2_causal_2048/manifest.json"
            if not manifest.exists():
                raise SystemExit(f"missing prepared data: {manifest}")
            print("Complete 4,000-step seed-0 schedule; expected test NLL 5.43–5.48 and mean 2,048-token occupancy 630–690 rows. Historical A40 cost: about $0.50.", flush=True)
            command = [
                sys.executable, "-m", "experiments.train_wikitext",
                "--manifest", str(manifest), "--output-dir", str(output),
                "--arm", "n256", "--layout", "BBBBBBBA",
                "--steps", "4000", "--schedule-steps", "4000",
                "--batch-size", "8", "--micro-batch-size", "1",
                "--eval-batch-size", "1", "--checkpoint-eval-examples", "32",
                "--final-eval-examples", "128", "--serving-route-examples", "64",
                "--serving-route-lengths", "16,64,256,1024,2048",
                "--width", "128", "--heads", "4", "--slots", "256",
                "--reads", "16", "--writes", "4", "--memory-heads", "1",
                "--mlp-expansion", "4", "--learning-rate", "0.0003",
                "--weight-decay", "0.01", "--warmup-steps", "100", "--seed", "0",
                "--activation-dtype", "bfloat16",
                "--expected-active-parameters", "14654556",
                "--recovery-checkpoint-interval", "4000", "--no-save-checkpoint",
            ]
        else:
            print("CUDA allocator screen; expected allocated per-request state reduction 45–52%. On an A40, expected latency cost is 0–15%, with four-step pooled growth reducing copy churn. Historical cost: under $0.10.", flush=True)
            command = [
                sys.executable, "-m", "experiments.benchmark_serving",
                "--output", str(output), "--seed", "0", "--warmup", "5",
                "--iterations", "20", "--timing-repetitions", "5",
                "--growth-quantum-steps", "1,2,4",
            ]
        run(command, args.gpu)
        if suite == "occupancy":
            (output / "recovery_checkpoint.pt").unlink(missing_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_results.py"),
                suite,
                "--output-root",
                str(args.output_root.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
