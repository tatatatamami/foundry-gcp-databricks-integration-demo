import argparse
import json
from pathlib import Path


REQUIRED_FILES = ["model.json", "metrics.json", "run_summary.json"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify sales model training artifacts.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("outputs/sales-price-model"))
    parser.add_argument("--min-train-rows", type=int, default=1)
    parser.add_argument("--min-test-rows", type=int, default=1)
    args = parser.parse_args()

    missing = [name for name in REQUIRED_FILES if not (args.artifact_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing artifact files: {', '.join(missing)}")

    model = load_json(args.artifact_dir / "model.json")
    metrics = load_json(args.artifact_dir / "metrics.json")
    run_summary = load_json(args.artifact_dir / "run_summary.json")

    if run_summary.get("train_rows", 0) < args.min_train_rows:
        raise RuntimeError(f"train_rows is too small: {run_summary.get('train_rows')}")
    if run_summary.get("test_rows", 0) < args.min_test_rows:
        raise RuntimeError(f"test_rows is too small: {run_summary.get('test_rows')}")
    if model.get("model_type") != "ridge_regression":
        raise RuntimeError(f"Unexpected model_type: {model.get('model_type')}")
    for metric_name in ["mae", "rmse", "r2"]:
        if metric_name not in metrics:
            raise RuntimeError(f"Missing metric: {metric_name}")

    print(json.dumps({
        "status": "passed",
        "artifact_dir": str(args.artifact_dir),
        "model_type": model["model_type"],
        "train_rows": run_summary["train_rows"],
        "test_rows": run_summary["test_rows"],
        "metrics": metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()