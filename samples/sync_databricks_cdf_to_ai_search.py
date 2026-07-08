import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from samples.sync_databricks_to_ai_search import (
    DEFAULT_EMBEDDING_DEPLOYMENT,
    DEFAULT_OPENAI_ENDPOINT,
    DEFAULT_SEARCH_ENDPOINT,
    DEFAULT_SEARCH_RESOURCE_GROUP,
    DEFAULT_SEARCH_SERVICE,
    build_content,
    create_or_update_index,
    embed_texts,
    execute_databricks_statement,
    get_search_admin_key,
    normalize_authorization,
    rows_from_statement_response,
    sanitize_row,
    stable_document_id,
    upload_documents,
    validate_source_table,
)


CDF_METADATA_FIELDS = {"_change_type", "_commit_version", "_commit_timestamp"}


def load_checkpoint(path: Path) -> int | None:
    if not path.exists():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    version = body.get("last_processed_version")
    return int(version) if version is not None else None


def save_checkpoint(path: Path, source_table: str, version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_table": source_table,
                "last_processed_version": version,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_latest_version(authorization: str, source_table: str) -> int:
    body = execute_databricks_statement(authorization, f"DESCRIBE HISTORY {source_table} LIMIT 1")
    rows = rows_from_statement_response(body)
    if not rows:
        raise RuntimeError(f"No Delta history was returned for {source_table}.")
    return int(rows[0]["version"])


def remove_cdf_metadata(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in CDF_METADATA_FIELDS}


def build_search_documents(rows: list[dict], vectors: list[list[float]], source_table: str) -> list[dict]:
    documents = []
    for row, vector in zip(rows, vectors, strict=True):
        clean_row = remove_cdf_metadata(row)
        content = build_content(clean_row)
        documents.append(
            {
                "id": stable_document_id(clean_row),
                "content": content,
                "row_json": json.dumps(sanitize_row(clean_row), ensure_ascii=False, sort_keys=True),
                "source_table": source_table,
                "transactionID": str(clean_row.get("transactionID")) if clean_row.get("transactionID") is not None else None,
                "customerID": str(clean_row.get("customerID")) if clean_row.get("customerID") is not None else None,
                "franchiseID": str(clean_row.get("franchiseID")) if clean_row.get("franchiseID") is not None else None,
                "dateTime": str(clean_row.get("dateTime")) if clean_row.get("dateTime") is not None else None,
                "quantity": int(clean_row.get("quantity") or 0),
                "contentVector": vector,
            }
        )
    return documents


def delete_documents(endpoint: str, key: str, index_name: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
    documents = [{"id": stable_document_id(remove_cdf_metadata(row))} for row in rows]
    result = search_client.delete_documents(documents)
    failed = [item for item in result if not item.succeeded]
    if failed:
        raise RuntimeError(f"Azure AI Search delete failed for {len(failed)} documents")
    return len(documents)


def sync_full_baseline(args: argparse.Namespace, authorization: str, search_key: str, latest_version: int) -> dict:
    statement = f"SELECT * FROM {args.source_table} LIMIT {args.limit}"
    body = execute_databricks_statement(authorization, statement)
    rows = rows_from_statement_response(body)
    create_or_update_index(args.search_endpoint, search_key, args.index_name)

    documents = []
    if rows:
        contents = [build_content(row) for row in rows]
        vectors = embed_texts(args.openai_endpoint, args.embedding_deployment, contents)
        documents = build_search_documents(rows, vectors, args.source_table)
        upload_documents(args.search_endpoint, search_key, args.index_name, documents)

    save_checkpoint(args.checkpoint_file, args.source_table, latest_version)
    return {
        "mode": "baseline_full_sync",
        "source_table": args.source_table,
        "index": args.index_name,
        "uploaded_documents": len(documents),
        "checkpoint_version": latest_version,
    }


def sync_cdf_changes(args: argparse.Namespace, authorization: str, search_key: str, last_version: int, latest_version: int) -> dict:
    start_version = last_version + 1
    if start_version > latest_version:
        return {
            "mode": "cdf_sync",
            "source_table": args.source_table,
            "index": args.index_name,
            "start_version": start_version,
            "latest_version": latest_version,
            "upserted_documents": 0,
            "deleted_documents": 0,
            "checkpoint_version": latest_version,
        }

    statement = f"SELECT * FROM table_changes('{args.source_table}', {start_version}, {latest_version})"
    body = execute_databricks_statement(authorization, statement)
    rows = rows_from_statement_response(body)

    upsert_rows = [row for row in rows if row.get("_change_type") in {"insert", "update_postimage"}]
    delete_rows = [row for row in rows if row.get("_change_type") == "delete"]

    create_or_update_index(args.search_endpoint, search_key, args.index_name)

    upserted_count = 0
    if upsert_rows:
        contents = [build_content(remove_cdf_metadata(row)) for row in upsert_rows]
        vectors = embed_texts(args.openai_endpoint, args.embedding_deployment, contents)
        documents = build_search_documents(upsert_rows, vectors, args.source_table)
        upload_documents(args.search_endpoint, search_key, args.index_name, documents)
        upserted_count = len(documents)

    deleted_count = delete_documents(args.search_endpoint, search_key, args.index_name, delete_rows)
    save_checkpoint(args.checkpoint_file, args.source_table, latest_version)

    return {
        "mode": "cdf_sync",
        "source_table": args.source_table,
        "index": args.index_name,
        "start_version": start_version,
        "latest_version": latest_version,
        "cdf_rows": len(rows),
        "upserted_documents": upserted_count,
        "deleted_documents": deleted_count,
        "checkpoint_version": latest_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Databricks CDF changes to Azure AI Search.")
    parser.add_argument("--source-table", default=os.environ.get("DATABRICKS_SOURCE_TABLE", "<catalog>.<schema>.<table>"))
    parser.add_argument("--index-name", default="databricks-sales-transactions-sync-test")
    parser.add_argument("--checkpoint-file", type=Path, default=Path(".sync-checkpoints/databricks-sales-cdf.json"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--search-service", default=DEFAULT_SEARCH_SERVICE)
    parser.add_argument("--search-resource-group", default=DEFAULT_SEARCH_RESOURCE_GROUP)
    parser.add_argument("--search-endpoint", default=DEFAULT_SEARCH_ENDPOINT)
    parser.add_argument("--openai-endpoint", default=DEFAULT_OPENAI_ENDPOINT)
    parser.add_argument("--embedding-deployment", default=DEFAULT_EMBEDDING_DEPLOYMENT)
    args = parser.parse_args()

    args.source_table = validate_source_table(args.source_table)

    token = os.environ.get("DATABRICKS_PAT") or getpass.getpass("Databricks PAT or Authorization header value: ")
    authorization = normalize_authorization(token)
    search_key = os.environ.get("AZURE_SEARCH_API_KEY") or get_search_admin_key(
        args.search_resource_group,
        args.search_service,
    )

    latest_version = get_latest_version(authorization, args.source_table)
    last_version = load_checkpoint(args.checkpoint_file)

    if last_version is None:
        result = sync_full_baseline(args, authorization, search_key, latest_version)
    else:
        result = sync_cdf_changes(args, authorization, search_key, last_version, latest_version)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()