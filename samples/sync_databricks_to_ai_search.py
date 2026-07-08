import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256

from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from openai import AzureOpenAI


DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "https://<your-databricks-host>.gcp.databricks.com")
DATABRICKS_STATEMENTS_ENDPOINT = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
DATABRICKS_WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "<your-warehouse-id>")
DEFAULT_SOURCE_TABLE = "samples.bakehouse.sales_transactions"
DEFAULT_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE", "<your-search-service-name>")
DEFAULT_SEARCH_RESOURCE_GROUP = os.environ.get("AZURE_SEARCH_RESOURCE_GROUP", "<your-resource-group>")
DEFAULT_SEARCH_ENDPOINT = f"https://{DEFAULT_SEARCH_SERVICE}.search.windows.net"
DEFAULT_INDEX_NAME = "databricks-sales-transactions-v2"
DEFAULT_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://<your-foundry-account>.cognitiveservices.azure.com/")
DEFAULT_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
EXCLUDED_CONTENT_FIELDS = {"cardNumber"}
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


def get_search_admin_key(resource_group: str, service_name: str) -> str:
    az_command = shutil.which("az") or shutil.which("az.cmd")
    if not az_command:
        raise RuntimeError("Azure CLI was not found. Set AZURE_SEARCH_API_KEY or install Azure CLI.")

    output = subprocess.check_output(
        [
            az_command,
            "search",
            "admin-key",
            "show",
            "--resource-group",
            resource_group,
            "--service-name",
            service_name,
            "--query",
            "primaryKey",
            "-o",
            "tsv",
        ],
        text=True,
    )
    return output.strip()


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
        body = error.read().decode("utf-8", errors="replace")
        if error.code == 401:
            raise RuntimeError(
                "Databricks returned 401 Unauthorized. Enter a valid Databricks PAT directly in the terminal; "
                "do not include quotes, angle brackets, or extra text. Both 'dapi...' and 'Bearer dapi...' are accepted."
            ) from None
        raise RuntimeError(f"Databricks request failed with HTTP {error.code}: {body}") from None


def rows_from_statement_response(body: dict) -> list[dict]:
    state = body.get("status", {}).get("state")
    if state != "SUCCEEDED":
        raise RuntimeError(f"Databricks statement did not succeed: {json.dumps(body.get('status'), ensure_ascii=False)}")

    columns = body.get("manifest", {}).get("schema", {}).get("columns", [])
    names = [column["name"] for column in columns]
    data = body.get("result", {}).get("data_array", [])
    return [dict(zip(names, row, strict=False)) for row in data]


def build_content(row: dict) -> str:
    return "; ".join(
        f"{key}: {value}"
        for key, value in row.items()
        if value is not None and key not in EXCLUDED_CONTENT_FIELDS
    )


def sanitize_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in EXCLUDED_CONTENT_FIELDS}


def stable_document_id(row: dict) -> str:
    source = str(row.get("transactionID") or row.get("transaction_id") or json.dumps(row, sort_keys=True))
    return sha256(source.encode("utf-8")).hexdigest()


def create_or_update_index(endpoint: str, key: str, index_name: str) -> None:
    credential = AzureKeyCredential(key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="row_json", type=SearchFieldDataType.String),
        SimpleField(name="source_table", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="transactionID", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="customerID", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="franchiseID", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="dateTime", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="quantity", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
            profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")],
        ),
    )
    index_client.create_or_update_index(index)


def embed_texts(endpoint: str, deployment: str, texts: list[str]) -> list[list[float]]:
    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-10-21",
    )
    response = client.embeddings.create(model=deployment, input=texts)
    return [item.embedding for item in response.data]


def upload_documents(endpoint: str, key: str, index_name: str, documents: list[dict]) -> None:
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
    result = search_client.merge_or_upload_documents(documents)
    failed = [item for item in result if not item.succeeded]
    if failed:
        raise RuntimeError(f"Azure AI Search upload failed for {len(failed)} documents")


def get_search_document_count(endpoint: str, key: str, index_name: str) -> int:
    request = urllib.request.Request(
        f"{endpoint}/indexes/{index_name}/docs/$count?api-version=2024-07-01",
        headers={"api-key": key},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.read().decode("utf-8"))


def sync_once(args: argparse.Namespace, authorization: str, search_key: str) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    source_table = validate_source_table(args.source_table)
    statement = f"SELECT * FROM {source_table} LIMIT {args.limit}"

    body = execute_databricks_statement(authorization, statement)
    rows = rows_from_statement_response(body)
    if not rows:
        return {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "index": args.index_name,
            "source_table": source_table,
            "databricks_rows": 0,
            "uploaded_documents": 0,
            "search_document_count": get_search_document_count(args.search_endpoint, search_key, args.index_name),
        }

    create_or_update_index(args.search_endpoint, search_key, args.index_name)

    contents = [build_content(row) for row in rows]
    vectors = embed_texts(args.openai_endpoint, args.embedding_deployment, contents)
    documents = []
    for row, content, vector in zip(rows, contents, vectors, strict=True):
        documents.append({
            "id": stable_document_id(row),
            "content": content,
            "row_json": json.dumps(sanitize_row(row), ensure_ascii=False, sort_keys=True),
            "source_table": source_table,
            "transactionID": str(row.get("transactionID")) if row.get("transactionID") is not None else None,
            "customerID": str(row.get("customerID")) if row.get("customerID") is not None else None,
            "franchiseID": str(row.get("franchiseID")) if row.get("franchiseID") is not None else None,
            "dateTime": str(row.get("dateTime")) if row.get("dateTime") is not None else None,
            "quantity": int(row.get("quantity") or 0),
            "contentVector": vector,
        })

    upload_documents(args.search_endpoint, search_key, args.index_name, documents)
    return {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "index": args.index_name,
        "source_table": source_table,
        "databricks_rows": len(rows),
        "uploaded_documents": len(documents),
        "search_document_count": get_search_document_count(args.search_endpoint, search_key, args.index_name),
        "first_document_id": documents[0]["id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Databricks sales_transactions rows to Azure AI Search.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--search-service", default=DEFAULT_SEARCH_SERVICE)
    parser.add_argument("--search-resource-group", default=DEFAULT_SEARCH_RESOURCE_GROUP)
    parser.add_argument("--search-endpoint", default=DEFAULT_SEARCH_ENDPOINT)
    parser.add_argument("--openai-endpoint", default=DEFAULT_OPENAI_ENDPOINT)
    parser.add_argument("--embedding-deployment", default=DEFAULT_EMBEDDING_DEPLOYMENT)
    parser.add_argument("--repeat", action="store_true", help="Run sync repeatedly for scheduled polling validation.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of sync runs when --repeat is set.")
    parser.add_argument("--interval-seconds", type=int, default=300, help="Delay between sync runs when --repeat is set.")
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be 1 or greater.")
    if args.interval_seconds < 0:
        raise ValueError("--interval-seconds must be 0 or greater.")

    token = os.environ.get("DATABRICKS_PAT") or getpass.getpass("Databricks PAT or Authorization header value: ")
    authorization = normalize_authorization(token)
    search_key = os.environ.get("AZURE_SEARCH_API_KEY") or get_search_admin_key(
        args.search_resource_group,
        args.search_service,
    )

    run_count = args.iterations if args.repeat else 1
    for run_number in range(1, run_count + 1):
        result = sync_once(args, authorization, search_key)
        result["run"] = run_number
        result["scheduled_polling"] = args.repeat
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if run_number < run_count:
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()