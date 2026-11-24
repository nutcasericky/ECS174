"""Run HW1 Parts 5–8 sequentially; reuse the existing batch-4, lr-0.001 baseline."""

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys


EXPERIMENTS = [
    ("batch_1", dict(batch_size=1)),
    ("batch_1000", dict(batch_size=1000)),
    ("lr_10", dict(lr=10.0)),
    ("lr_0_1", dict(lr=0.1)),
    ("lr_0_01", dict(lr=0.01)),
    ("augmentation", dict(augment=True)),
    ("mse", dict(loss="mse")),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--plan", action="store_true", help="Print commands without training")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    root = Path(__file__).resolve().parent

    for index, (name, overrides) in enumerate(EXPERIMENTS, start=1):
        config = dict(model="cnn", epochs=args.epochs, batch_size=4, lr=0.001,
                      augment=False, loss="ce", device=args.device)
        config.update(overrides)
        output = root / "results" / name
        command = [sys.executable, str(root / "hw1_baseline.py")]
        for key in ["model", "epochs", "batch_size", "lr", "loss", "device"]:
            command.extend(["--" + key.replace("_", "-"), str(config[key])])
        command.extend(["--output", str(output)])
        if config["augment"]:
            command.append("--augment")
        print(f"\nExperiment {index}/{len(EXPERIMENTS)}: {name}", flush=True)
        if args.plan:
            print(shlex.join(command), flush=True)
            continue

        metadata_path = output / "run.json"
        reuse = False
        if metadata_path.exists() and (output / "metrics.csv").exists() and (output / "curves.png").exists():
            previous = json.loads(metadata_path.read_text())
            reuse = previous.get("status") in ["complete", "diverged"] and all(
                previous.get("config", {}).get(key) == value for key, value in config.items()
            )
        if reuse:
            print("Reusing the existing result with matching settings.", flush=True)
        else:
            subprocess.run(command, cwd=root, check=True)

    if args.plan:
        return

    print("\nAll seven experiments finished. Final recorded measurements:", flush=True)
    for name, _ in EXPERIMENTS:
        folder = root / "results" / name
        metadata = json.loads((folder / "run.json").read_text())
        with (folder / "metrics.csv").open() as handle:
            final = list(csv.DictReader(handle))[-1]
        if metadata["status"] == "diverged":
            print(f"{name}: DIVERGED during epoch {final['epoch']} (non-finite loss)", flush=True)
        else:
            print(f"{name}: epoch {final['epoch']} | train {float(final['train_accuracy']):.2f}% | "
                  f"test {float(final['test_accuracy']):.2f}%", flush=True)
    print("The existing baseline supplies batch size 4 and learning rate 0.001.", flush=True)


if __name__ == "__main__":
    main()
