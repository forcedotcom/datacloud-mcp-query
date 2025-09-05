from datetime import datetime, timedelta
import json
from typing import Union
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import requests
import sys
import os
from rfc3986 import builder as uri_builder
import secrets
import hashlib
import base64

import http.server
import time
import webbrowser
from threading import Thread
from urllib.parse import parse_qs, urlparse

# Implement minimal OAuth flow consuming secrets from env
def validate_required_env_vars():
    required_vars = {
        'SF_CLIENT_ID': os.getenv('SF_CLIENT_ID'),
        'SF_CLIENT_SECRET': os.getenv('SF_CLIENT_SECRET')
    }

    missing = [var for var, value in required_vars.items() if not value]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    return required_vars

required_env = validate_required_env_vars()
CLIENT_ID = required_env['SF_CLIENT_ID']
CLIENT_SECRET = required_env['SF_CLIENT_SECRET']
redirect_uri = "http://localhost:55556/Callback"


def delayed_server_shutdown(*, target, sleep_for: float = 0.1):  # pragma: no cover
    def closure(*args, **kwargs):
        time.sleep(0.1)
        target(*args, **kwargs)

    return closure


class RequestHandler(http.server.BaseHTTPRequestHandler):
  # pragma: no cover
    def do_GET(self):  # noqa: N802
        parts = urlparse(self.path)
        if parts.path != "/Callback":
            self.send_error(404, "Not Found", "Not Found")
            return

        args = parse_qs(parts.query)
        self.server.oauth_result = args

        has_code = "code" in args
        response_content = f"Final Status: {has_code=}".encode("latin1")
        response_content += b"\nYou can close this window now"
        self.send_response(200, "OK")
        self.send_header("Content-Type", "text")
        self.send_header("Content-Length", str(len(response_content)))
        self.end_headers()
        self.wfile.write(response_content)

        Thread(
            target=delayed_server_shutdown(target=self.server.shutdown), daemon=True
        ).start()


def generate_pkce_pair():
    """Generate PKCE code verifier and challenge for OAuth flow"""
    # Generate a random code_verifier (43-128 characters)
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')

    # Generate code_challenge as SHA256 hash of verifier
    challenge = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(challenge).decode('utf-8').rstrip('=')

    return code_verifier, code_challenge


def run_flow(scopes: list[str], login_root: str):
    login_url = f"https://{login_root}/services/oauth2/authorize"
    token_exchange_url = f"https://{login_root}/services/oauth2/token"

    # Generate PKCE pair
    code_verifier, code_challenge = generate_pkce_pair()

    browser_uri: str = (
        uri_builder.URIBuilder(path=login_url)
        .add_query_from(
            {
                "client_id": CLIENT_ID,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": " ".join(scopes),
                "prompt": "login",
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
        )
        .finalize()
        .unsplit()
    )

    server = http.server.HTTPServer(("localhost", 55556), RequestHandler)
    server.allow_reuse_address = True
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    webbrowser.open_new_tab(browser_uri)
    while t.is_alive():
        t.join(10)

    oauth_result_args = server.oauth_result

    if "code" not in oauth_result_args:
        error_msg = "OAuth authentication failed - no authorization code received"
        if "error" in oauth_result_args:
            error_msg += f". Error: {oauth_result_args['error'][0]}"
            if "error_description" in oauth_result_args:
                error_msg += f" - {oauth_result_args['error_description'][0]}"
        raise Exception(error_msg)

    code = oauth_result_args["code"][0]  # OAuth params come as lists

    response = requests.post(
        token_exchange_url,
        {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,  # Include the PKCE code verifier
        },
        headers={"Accept": "application/json"},
    )

    response.raise_for_status()
    return response.json()


# Create an MCP server
mcp = FastMCP("Demo")

sf_org: dict = {}


def ensure_access():
    # Handle expiration
    if 'exp' in sf_org and datetime.now() > sf_org['exp']:
        del sf_org['exp']
        del sf_org['token']
    if not 'token' in sf_org:
        # Hard code assumption that expiration happens after two hours
        all_scopes_auth_info = run_flow(
            ["api", "cdp_query_api", "cdp_profile_api"], sf_org['login_root'])
        sf_org['token'] = all_scopes_auth_info['access_token']
        sf_org['exp'] = datetime.now() + timedelta(minutes=110)
        sf_org['url'] = all_scopes_auth_info['instance_url']
    return sf_org['token']


def run_query(sql: str, parameters: list[dict] = []) -> Union[str, list]:
    ensure_access()
    r = requests.post(sf_org['url'] + '/services/data/v63.0/ssot/query-sql',
                      json={'sql': sql, 'sqlParameters': parameters},
                      headers={'Authorization': 'Bearer ' + sf_org['token']})
    if r.status_code > 201:
        message = r.text
        try:
            structured_message = json.loads(r.text)[0]
            details = json.loads(structured_message["message"])
            message = details["primaryMessage"] + \
                ", Hint: " + details["customerHint"]
        except ValueError as e:
            pass
        raise Exception(r.status_code, r.reason, message)
    else:
        result = r.json()
        if result["status"]['rowCount'] == 0:
            return "(emtpy)"
        else:
            return result['data']


def run_focus(sql: str, parameters: list[dict] = []):
    ensure_access()
    r = requests.post(sf_org['url'] + '/services/data/v63.0/v1/prism/focus',
                      json={'utterance': sql, 'appId': 'mcpServer', "dataSources": [
                          {
                              "dataSourceType": "DATACLOUD"
                          }
                      ]},
                      headers={'Authorization': 'Bearer ' + sf_org['token']})

    if r.status_code > 201:
        message = r.text
        try:
            structured_message = json.loads(r.text)[0]
            details = json.loads(structured_message["message"])
            message = details["primaryMessage"] + \
                ", Hint: " + details["customerHint"]
        except ValueError as e:
            return False
        raise Exception(r.status_code, r.reason, message)
    else:
        result = r.json()
        if 'error' in result and result['error'] != None:
            raise Exception(r.status_code, result['error'])
        return [{'column': entity['qualifiedName'].replace(entity['parentId']+".", "").replace("default.", ""), 'table': entity['parentId'], 'description': entity['description']} for entity in result['entities']]


@mcp.tool(description="Executes a SQL query and returns the results")
def query(
    sql: str = Field(
        description="A SQL query in the PostgreSQL dialect make sure to always quote all identifies and use the exact casing. To formulate the query first verify which tables and fields to use through the suggest fields tool (or if it is broken through the list tables / describe tables call). Before executing the tool provide the user a succinct summary (targeted to low code users) on the semantics of the query"),
):
    return run_query(sql)


@mcp.tool(description="Lists the available tables in the database")
def list_tables() -> list[str]:
    result = run_query("SELECT c.relname AS TABLE_NAME FROM pg_catalog.pg_namespace n, pg_catalog.pg_class c LEFT JOIN pg_catalog.pg_description d ON (c.oid = d.objoid AND d.objsubid = 0  and d.classoid = 'pg_class'::regclass) WHERE c.relnamespace = n.oid AND c.relname LIKE :default_list_table_filter",
                       [{'type': 'Varchar', 'name': 'default_list_table_filter', 'value': sf_org['default_list_table_filter']}])
    return [x[0] for x in result]


@mcp.tool(description="Describes the columns of a table")
def describe_table(
    table: str = Field(description="The table name"),
) -> list[str]:
    result = run_query("SELECT a.attname FROM pg_catalog.pg_namespace n JOIN pg_catalog.pg_class c ON (c.relnamespace = n.oid) JOIN pg_catalog.pg_attribute a ON (a.attrelid = c.oid) JOIN pg_catalog.pg_type t ON (a.atttypid = t.oid) LEFT JOIN pg_catalog.pg_attrdef def ON (a.attrelid = def.adrelid AND a.attnum = def.adnum) LEFT JOIN pg_catalog.pg_description dsc ON (c.oid = dsc.objoid AND a.attnum = dsc.objsubid) LEFT JOIN pg_catalog.pg_class dc ON (dc.oid = dsc.classoid AND dc.relname = 'pg_class') LEFT JOIN pg_catalog.pg_namespace dn ON (dc.relnamespace = dn.oid AND dn.nspname = 'pg_catalog') WHERE a.attnum > 0 AND NOT a.attisdropped AND c.relname=:tablename",
                       [{'type': 'Varchar', 'name': 'tablename', 'value': table}])
    return [x[0] for x in result]


@mcp.tool(description="Suggests tables and fields from the database that could be relevant to a user question")
def suggest_table_and_fields(
    utterance: str = Field(
        description="A prompt that describes the task / data which is needed to formulate a query")
) -> list[str]:
    return run_focus(utterance)

sf_org: dict = {
    'default_list_table_filter': os.getenv('DEFAULT_LIST_TABLE_FILTER', '%'),
    'login_root': os.getenv('SF_LOGIN_URL', 'login.salesforce.com')
}

if __name__ == "__main__":
    mcp.run()
