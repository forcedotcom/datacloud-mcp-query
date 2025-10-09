import json
import logging
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import requests
import os
from oauth import OAuthConfig, OAuthSession
from connect_api_dc_sql import run_query

# Get logger for this module
logger = logging.getLogger(__name__)


# Create an MCP server
mcp = FastMCP("Demo")

# Global config and session
sf_org: OAuthConfig = OAuthConfig.from_env()
oauth_session: OAuthSession = OAuthSession(sf_org)

# Non-auth configuration
DEFAULT_LIST_TABLE_FILTER = os.getenv('DEFAULT_LIST_TABLE_FILTER', '%')

def run_focus(sql: str, parameters: list[dict] = []):
    base_url = oauth_session.get_instance_url()
    token = oauth_session.get_token()

    focus_url = base_url + '/services/data/v63.0/v1/prism/focus'
    request_data = {
        'utterance': sql,
        'appId': 'mcpServer',
        "dataSources": [
            {
                "dataSourceType": "DATACLOUD"
            }
        ]
    }

    logger.info(f"Calling Focus API: {focus_url}")

    r = requests.post(focus_url,
                      json=request_data,
                      headers={'Authorization': 'Bearer ' + token},
                      timeout=30)

    logger.info(f"Focus API response: status={r.status_code}, elapsed={r.elapsed.total_seconds():.2f}s")

    if r.status_code > 201:
        message = r.text
        logger.error(f"Focus API request failed: {message}")
        try:
            structured_message = json.loads(r.text)[0]
            details = json.loads(structured_message["message"])
            message = details["primaryMessage"] + \
                ", Hint: " + details["customerHint"]
        except ValueError as e:
            logger.error(f"Failed to parse error response: {e}")
        raise Exception(r.status_code, r.reason, message)
    else:
        # Additional safety check for HTTP errors
        r.raise_for_status()
        result = r.json()
        if 'error' in result and result['error'] != None:
            raise Exception(r.status_code, result['error'])
        return [{'column': entity['qualifiedName'].replace(entity['parentId']+".", "").replace("default.", ""), 'table': entity['parentId'], 'description': entity['description']} for entity in result['entities']]


@mcp.tool(description="Executes a SQL query and returns the results")
def query(
    sql: str = Field(
        description="A SQL query in the PostgreSQL dialect make sure to always quote all identifies and use the exact casing. To formulate the query first verify which tables and fields to use through the suggest fields tool (or if it is broken through the list tables / describe tables call). Before executing the tool provide the user a succinct summary (targeted to low code users) on the semantics of the query"),
):
    # Returns both data and metadata
    return run_query(oauth_session, sql)


@mcp.tool(description="Lists the available tables in the database")
def list_tables() -> list[str]:
    sql = "SELECT c.relname AS TABLE_NAME FROM pg_catalog.pg_namespace n, pg_catalog.pg_class c LEFT JOIN pg_catalog.pg_description d ON (c.oid = d.objoid AND d.objsubid = 0  and d.classoid = 'pg_class'::regclass) WHERE c.relnamespace = n.oid AND c.relname LIKE '%s'" % DEFAULT_LIST_TABLE_FILTER
    result = run_query(oauth_session, sql)
    # Extract data from the result dictionary
    data = result.get("data", [])
    return [x[0] for x in data]


@mcp.tool(description="Describes the columns of a table")
def describe_table(
    table: str = Field(description="The table name"),
) -> list[str]:
    sql = f"SELECT a.attname FROM pg_catalog.pg_namespace n JOIN pg_catalog.pg_class c ON (c.relnamespace = n.oid) JOIN pg_catalog.pg_attribute a ON (a.attrelid = c.oid) JOIN pg_catalog.pg_type t ON (a.atttypid = t.oid) LEFT JOIN pg_catalog.pg_attrdef def ON (a.attrelid = def.adrelid AND a.attnum = def.adnum) LEFT JOIN pg_catalog.pg_description dsc ON (c.oid = dsc.objoid AND a.attnum = dsc.objsubid) LEFT JOIN pg_catalog.pg_class dc ON (dc.oid = dsc.classoid AND dc.relname = 'pg_class') LEFT JOIN pg_catalog.pg_namespace dn ON (dc.relnamespace = dn.oid AND dn.nspname = 'pg_catalog') WHERE a.attnum > 0 AND NOT a.attisdropped AND c.relname='{table}'"
    result = run_query(oauth_session, sql)
    # Extract data from the result dictionary
    data = result.get("data", [])
    return [x[0] for x in data]


@mcp.tool(description="Suggests tables and fields from the database that could be relevant to a user question")
def suggest_table_and_fields(
    utterance: str = Field(
        description="A prompt that describes the task / data which is needed to formulate a query")
) -> list[str]:
    # Input validation is handled within run_focus
    return run_focus(utterance)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("Starting MCP server")
    mcp.run()
