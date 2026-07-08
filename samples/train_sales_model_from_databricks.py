import argparse
import getpass
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://<your-databricks-host>.gcp.databricks.com")
DATABRICKS_STATEMENTS_ENDPOINT = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
DATABRICKS_WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "<your-warehouse-id>")
DEFAULT_SOURCE_TABLE = os.environ.get("DATABRICKS_SOURCE_TABLE", "<catalog>.<schema>.<table>")
DEFAULT_OUTPUT_DIR = Path("outputs/sales-price-model")
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){2}$")


def normalize_authorization(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("bearer "):
        return value
    return f"Bearer {value}"


def validate_source_table(value: str) -> str:
    if not TABLE_NAME_PATTERN.fullmatch(value):
        raise ValueError("--source-table must be a fully qualified catalog.schema.table name.")
    return value


def execute_databricks_statement(authorization: str, statement: str) -> dict:
    payload = {
        "warehouse_id": DATABRICKS_WAREHOUSE_ID,
        "statement": statement,
        "wait_timeout": "30s",
        "on_wait_timeout": "CANCEL",
        "format": "JSON_ARRAY",
        "disposition": "INLINE",
    }
    request = urllib.request.Request(
        DATABRICKS_STATEMENTS_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise RuntimeError(
                "Databricks authorization failed. Use a valid PAT, or an Authorization header value starting with 'Bearer '."
            ) from error
        raise


def rows_from_statement_response(body: dict) -> list[dict]:
    state = body.get("status", {}).get("state")
    if state != "SUCCEEDED":
        raise RuntimeError(f"Databricks statement did not succeed: {json.dumps(body.get('status'), ensure_ascii=False)}")

    columns = body.get("manifest", {}).get("schema", {}).get("columns", [])
    names = [column["name"] for column in columns]
    data = body.get("result", {}).get("data_array", [])
    return [dict(zip(names, row, strict=False)) for row in data]


def to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_training_rows(authorization: str, source_table: str, limit: int) -> list[dict]:
    statement = f"""
SELECT transactionID, quantity, unitPrice, totalPrice, product, dateTime
FROM {source_table}
WHERE quantity IS NOT NULL
  AND unitPrice IS NOT NULL
  AND totalPrice IS NOT NULL
LIMIT {limit}
"""
    body = execute_databricks_statement(authorization, statement)
    rows = rows_from_statement_response(body)
    clean_rows = []
    for row in rows:
        quantity = to_float(row.get("quantity"))
        unit_price = to_float(row.get("unitPrice"))
        total_price = to_float(row.get("totalPrice"))
        if quantity is None or unit_price is None or total_price is None:
            continue
        clean_rows.append({**row, "quantity": quantity, "unitPrice": unit_price, "totalPrice": total_price})
    return clean_rows


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise RuntimeError("Training data is singular; use more varied rows.")
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]

    return [augmented[row_index][size] for row_index in range(size)]


def train_linear_regression(rows: list[dict], ridge_lambda: float) -> dict:
    matrix = [[0.0 for _ in range(3)] for _ in range(3)]
    vector = [0.0 for _ in range(3)]

    for row in rows:
        features = [1.0, row["quantity"], row["unitPrice"]]
        target = row["totalPrice"]
        for row_index, feature_value in enumerate(features):
            vector[row_index] += feature_value * target
            for column_index, other_feature_value in enumerate(features):
                matrix[row_index][column_index] += feature_value * other_feature_value

    for index in range(1, len(matrix)):
        matrix[index][index] += ridge_lambda

    intercept, quantity_weight, unit_price_weight = solve_linear_system(matrix, vector)
    return {
        "model_type": "ridge_regression",
        "target": "totalPrice",
        "features": ["quantity", "unitPrice"],
        "ridge_lambda": ridge_lambda,
        "intercept": intercept,
        "weights": {
            "quantity": quantity_weight,
            "unitPrice": unit_price_weight,
        },
    }


def predict(model: dict, row: dict) -> float:
    return (
        model["intercept"]
        + model["weights"]["quantity"] * row["quantity"]
        + model["weights"]["unitPrice"] * row["unitPrice"]
    )


def evaluate(model: dict, rows: list[dict]) -> dict:
    actuals = [row["totalPrice"] for row in rows]
    predictions = [predict(model, row) for row in rows]
    errors = [prediction - actual for prediction, actual in zip(predictions, actuals, strict=True)]
    mean_actual = sum(actuals) / len(actuals)
    sse = sum(error * error for error in errors)
    sst = sum((actual - mean_actual) ** 2 for actual in actuals)
    return {
        "rows": len(rows),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sse / len(errors)),
        "r2": 1 - (sse / sst) if sst else 0,
    }


def split_rows(rows: list[dict], train_ratio: float) -> tuple[list[dict], list[dict]]:
    split_at = max(3, int(len(rows) * train_ratio))
    split_at = min(split_at, len(rows) - 1)
    return rows[:split_at], rows[split_at:]


def write_artifacts(output_dir: Path, source_table: str, model: dict, metrics: dict, train_rows: int, test_rows: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "source_table": source_table,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "metrics": metrics,
    }
    (output_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sales price model from Databricks data.")
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_table = validate_source_table(args.source_table)
    token = os.environ.get("DATABRICKS_PAT") or getpass.getpass("Databricks PAT or Authorization header value: ")
    rows = extract_training_rows(normalize_authorization(token), source_table, args.limit)
    if len(rows) < 5:
        raise RuntimeError(f"At least 5 training rows are required, but only {len(rows)} rows were returned.")

    train_rows, test_rows = split_rows(rows, args.train_ratio)
    model = train_linear_regression(train_rows, args.ridge_lambda)
    metrics = evaluate(model, test_rows)
    write_artifacts(args.output_dir, source_table, model, metrics, len(train_rows), len(test_rows))

    print(json.dumps({"source_table": source_table, "output_dir": str(args.output_dir), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()