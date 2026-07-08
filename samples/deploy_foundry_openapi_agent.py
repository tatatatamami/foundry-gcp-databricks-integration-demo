import argparse
import json
import os
from pathlib import Path

import yaml
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    OpenApiFunctionDefinition,
    OpenApiProjectConnectionAuthDetails,
    OpenApiProjectConnectionSecurityScheme,
    OpenApiTool,
    PromptAgentDefinition,
)
from azure.identity import AzureCliCredential


DEFAULT_PROJECT_ENDPOINT = os.environ.get("AZURE_FOUNDRY_PROJECT_ENDPOINT", "https://<your-foundry-account>.services.ai.azure.com/api/projects/<your-project>")
DEFAULT_AGENT_NAME = "databricks-sales-agent"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_CONNECTION_NAME = os.environ.get("AZURE_FOUNDRY_CONNECTION_NAME", "DatabricksSalesApi")
DEFAULT_OPENAPI_PATH = Path(__file__).with_name("databricks-tool-openapi.json")

# SECURITY NOTE: These instructions guide model behavior but do NOT enforce authorization.
# The only enforceable security boundary is the Databricks identity (PAT or service principal)
# used in the Foundry Project Connection. Scope that identity to SELECT on the target table only.
# In production, replace the PAT with a least-privilege service principal and consider
# placing an API facade (Azure Functions / APIM) in front of Databricks to validate
# or rewrite SQL before forwarding it.
INSTRUCTIONS = """You are a sales data assistant for a Databricks proof of concept.
Answer the user in Japanese.

Use the Databricks SQL Statement Execution API tool for every question about
sales transactions, row counts, samples, categories, regions, dates, quantities,
or sales amounts.

Allowed data source:
- samples.bakehouse.sales_transactions only.

Tool call rules:
- Call the execute_sales_query tool with a JSON request body containing exactly
    these two fields: warehouse_id and statement.
- warehouse_id must always be "729062798c1046d0".
- Do not include catalog, schema, wait_timeout, on_wait_timeout, format,
    disposition, row_limit, byte_limit, headers, or credentials in the tool body.
- The statement must be a single read-only SELECT or WITH query.
- Always use the fully qualified table name samples.bakehouse.sales_transactions
    in SQL.
- Never query another catalog, schema, table, view, function, or system table.
- Never execute INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, TRUNCATE, MERGE,
    COPY, GRANT, REVOKE, OPTIMIZE, VACUUM, CALL, DESCRIBE, SHOW, USE, or EXPLAIN.
- Do not use SELECT * unless the user explicitly asks to inspect sample rows.
- Prefer aggregate queries for summaries.
- Add LIMIT 20 or less when returning detail rows.

Useful SQL examples:
- Row count: SELECT COUNT(*) AS row_count FROM samples.bakehouse.sales_transactions
- Sample rows: SELECT transaction_date, product_category, region, quantity, sales_amount FROM samples.bakehouse.sales_transactions LIMIT 5
- Category summary: SELECT product_category, COUNT(*) AS transaction_count, SUM(sales_amount) AS total_sales FROM samples.bakehouse.sales_transactions GROUP BY product_category ORDER BY total_sales DESC LIMIT 20

Response rules:
- Summarize tool results clearly in Japanese.
- Do not expose credentials, tokens, headers, or connection details.
- If the request is outside the allowed table, say that the requested data is
    outside the available data source.
- If the tool returns 401 Unauthorized, say that the Databricks authorization in
    the Foundry connection DatabricksSalesApi appears invalid, expired, or missing
    the "Bearer " prefix. Do not ask the user to paste a token into chat.
- If the tool returns another error, explain the error briefly and suggest the
    smallest next check.
"""


def load_openapi(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a Foundry prompt agent version with the Databricks OpenAPI tool.")
    parser.add_argument("--project-endpoint", default=DEFAULT_PROJECT_ENDPOINT)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--connection-name", default=DEFAULT_CONNECTION_NAME)
    parser.add_argument("--openapi", type=Path, default=DEFAULT_OPENAPI_PATH)
    args = parser.parse_args()

    project = AIProjectClient(
        endpoint=args.project_endpoint,
        credential=AzureCliCredential(),
    )

    connection = project.connections.get(args.connection_name)
    spec = load_openapi(args.openapi)

    if "<your-" in json.dumps(spec):
        raise ValueError(
            f"OpenAPI spec '{args.openapi}' contains unresolved placeholders. "
            "Replace all '<your-...>' values before deploying."
        )

    tool = OpenApiTool(
        openapi=OpenApiFunctionDefinition(
            name="databricks_sales_query",
            description="Execute read-only SQL against samples.bakehouse.sales_transactions through Databricks Statement Execution API.",
            spec=spec,
            auth=OpenApiProjectConnectionAuthDetails(
                security_scheme=OpenApiProjectConnectionSecurityScheme(
                    project_connection_id=connection.id,
                )
            ),
        )
    )

    definition = PromptAgentDefinition(
        kind="prompt",
        model=args.model,
        instructions=INSTRUCTIONS,
        tools=[tool],
    )

    agent = project.agents.create_version(
        agent_name=args.agent_name,
        definition=definition,
        description="Read-only PoC agent for querying GCP Databricks sales transactions through the Statement Execution API.",
    )

    print(json.dumps({
        "name": agent.name,
        "version": agent.version,
        "id": agent.id,
        "connection_name": args.connection_name,
        "connection_id": connection.id,
    }, indent=2))


if __name__ == "__main__":
    main()