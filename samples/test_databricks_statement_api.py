import getpass
import json
import os
import urllib.error
import urllib.request


DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://<your-databricks-host>.gcp.databricks.com")
STATEMENTS_ENDPOINT = f"{DATABRICKS_HOST}/api/2.0/sql/statements"


def normalize_authorization(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("bearer "):
        return value
    return f"Bearer {value}"


def execute_statement(authorization: str, payload: dict) -> dict:
    request = urllib.request.Request(
        STATEMENTS_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    token = os.environ.get("DATABRICKS_PAT") or getpass.getpass("Databricks PAT or Authorization header value: ")
    authorization = normalize_authorization(token)
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "<your-warehouse-id>")
    full_payload = {
        "warehouse_id": warehouse_id,
        "catalog": "samples",
        "schema": "bakehouse",
        "statement": "SELECT COUNT(*) AS row_count FROM sales_transactions",
        "wait_timeout": "30s",
        "on_wait_timeout": "CANCEL",
        "format": "JSON_ARRAY",
        "disposition": "INLINE",
        "row_limit": 20,
        "byte_limit": 100000,
    }
    minimal_payload = {
        "warehouse_id": warehouse_id,
        "catalog": "samples",
        "schema": "bakehouse",
        "statement": "SELECT COUNT(*) AS row_count FROM sales_transactions",
        "wait_timeout": "30s",
        "on_wait_timeout": "CANCEL",
    }

    try:
        body = execute_statement(authorization, full_payload)
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(f"Full payload returned HTTP {error.code}: {error_body}")
        if error.code != 400:
            raise SystemExit(1) from error
        print("Retrying with a minimal payload...")
        try:
            body = execute_statement(authorization, minimal_payload)
        except urllib.error.HTTPError as minimal_error:
            minimal_error_body = minimal_error.read().decode("utf-8", errors="replace")
            raise SystemExit(f"Minimal payload returned HTTP {minimal_error.code}: {minimal_error_body}") from minimal_error

    print(json.dumps({
        "status": body.get("status"),
        "columns": body.get("manifest", {}).get("schema", {}).get("columns"),
        "data_array": body.get("result", {}).get("data_array"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()