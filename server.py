import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from auth_factory import create_auth_provider, AuthenticationError
from connect_api_dc_sql import run_query

# Get logger for this module
logger = logging.getLogger(__name__)


# Create an MCP server
mcp = FastMCP("Demo")

# Global authentication provider
try:
    auth_provider = create_auth_provider()
except AuthenticationError as e:
    logger.error(f"Authentication configuration error: {e}")
    raise SystemExit(1) from e

# Non-auth configuration
DEFAULT_LIST_TABLE_FILTER = os.getenv('DEFAULT_LIST_TABLE_FILTER', '%')


DATASPACE_FIELD_DESCRIPTION = "The Data Cloud dataspace name to execute against. Defaults to 'default'."


@mcp.tool(
    description=(
        "Executes a SQL query against Salesforce Data 360 (formerly Data Cloud) and returns the results. "
        "Data 360 SQL is PostgreSQL-flavored but has its own dialect, supported functions, and constraints. "
        "For authoritative syntax, supported functions, and semantics, consult the Data 360 SQL Reference: "
        "https://developer.salesforce.com/docs/data/data-cloud-query-guide/references/dc-sql-reference/data-cloud-sql-context.html"
    )
)
def query(
    sql: str = Field(
        description=(
            "A SQL query in the Data 360 SQL dialect (PostgreSQL-flavored). Always quote all identifiers "
            "and preserve their exact casing. Before writing a query, verify the tables and fields to use "
            "via the suggest fields tool (or, if unavailable, via list_tables / describe_table). "
            "When in doubt about supported syntax or functions, refer to the Data 360 SQL Reference: "
            "https://developer.salesforce.com/docs/data/data-cloud-query-guide/references/dc-sql-reference/data-cloud-sql-context.html "
            "Before executing the tool, provide the user a succinct summary (targeted to low code users) on the semantics of the query."
        )
    ),
    dataspace: str = Field(default="default", description=DATASPACE_FIELD_DESCRIPTION),
    query_settings: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional Data Cloud query settings, passed through as-is as 'querySettings' in the Query API request body. Known settings include 'query_timeout' with a duration-suffixed value, e.g. {\"query_timeout\": \"1800000ms\"}. Leave unset to use Data Cloud defaults.",
    ),
):
    # Returns both data and metadata
    return run_query(
        auth_provider, sql, dataspace=dataspace, query_settings=query_settings
    )


@mcp.tool(description="Lists the available tables in the database")
def list_tables(
    dataspace: str = Field(default="default", description=DATASPACE_FIELD_DESCRIPTION),
) -> list[str]:
    sql = "SELECT c.relname AS TABLE_NAME FROM pg_catalog.pg_namespace n, pg_catalog.pg_class c LEFT JOIN pg_catalog.pg_description d ON (c.oid = d.objoid AND d.objsubid = 0 AND d.classoid = 'pg_class'::regclass) WHERE c.relnamespace = n.oid AND c.relname LIKE :tableFilter"
    result = run_query(
        auth_provider, sql,
        sql_parameters=[{"name": "tableFilter", "value": DEFAULT_LIST_TABLE_FILTER}],
        dataspace=dataspace,
    )
    # Extract data from the result dictionary
    data = result.get("data", [])
    return [x[0] for x in data]


@mcp.tool(description="Describes the columns of a table")
def describe_table(
    table: str = Field(description="The table name"),
    dataspace: str = Field(default="default", description=DATASPACE_FIELD_DESCRIPTION),
) -> list[str]:
    sql = "SELECT a.attname FROM pg_catalog.pg_namespace n JOIN pg_catalog.pg_class c ON (c.relnamespace = n.oid) JOIN pg_catalog.pg_attribute a ON (a.attrelid = c.oid) JOIN pg_catalog.pg_type t ON (a.atttypid = t.oid) LEFT JOIN pg_catalog.pg_attrdef def ON (a.attrelid = def.adrelid AND a.attnum = def.adnum) LEFT JOIN pg_catalog.pg_description dsc ON (c.oid = dsc.objoid AND a.attnum = dsc.objsubid) LEFT JOIN pg_catalog.pg_class dc ON (dc.oid = dsc.classoid AND dc.relname = 'pg_class') LEFT JOIN pg_catalog.pg_namespace dn ON (dc.relnamespace = dn.oid AND dn.nspname = 'pg_catalog') WHERE a.attnum > 0 AND NOT a.attisdropped AND c.relname = :tableName"
    result = run_query(
        auth_provider, sql,
        sql_parameters=[{"name": "tableName", "value": table}],
        dataspace=dataspace,
    )
    # Extract data from the result dictionary
    data = result.get("data", [])
    return [x[0] for x in data]


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("Starting MCP server")
    mcp.run()
