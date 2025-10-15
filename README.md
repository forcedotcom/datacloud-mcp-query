# Data 360 Query MCP Server

This MCP server provides a seamless integration between Cursor and Salesforce Data Cloud (formerly known as CDP), allowing you to execute SQL queries directly from Cursor. The server handles OAuth authentication with Salesforce and provides tools for exploring and querying Data Cloud tables.

## Features

- Execute SQL queries against Salesforce Data Cloud
- List available tables in the database
- Describe table columns and structure
- Automatic OAuth2 authentication flow with Salesforce

## Adding to Cursor

1. Clone this repository to your local machine
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Connect to the MCP server in Cursor:
   - Open Cursor IDE.
   - Go to **Cursor Settings** → **MCP**.
   - Click on **Add new global MCP server**.
   - Fill in the details:
   ```json
       "mcpServers": {
         ...
        "datacloud": {
          "command": "<path to python>",
          "args": [
            "<full path to>/server.py"
          ],
          "env": {
            "SF_CLIENT_ID": "<Client Id>",
            "SF_CLIENT_SECRET": "<Client Secret>>"
          },
          "disabled": false,
          "autoApprove": ["suggest_table_and_fields", "describe_table", "list_tables"]
        }
        ...
      }
   ```
   - Enable the MCP server and click refresh which should show the tool list

## Configuration

The server requires the following environment variables:

### Required Environment Variables

- `SF_CLIENT_ID`: Your Salesforce OAuth client ID
- `SF_CLIENT_SECRET`: Your Salesforce OAuth client secret

See [Connected App Setup Guide](CONNECTED_APP_SETUP.md) for instructions on how to obtain these credentials.

### Optional Environment Variables

- `SF_LOGIN_URL`: The Salesforce login URL (default: 'login.salesforce.com')
- `SF_CALLBACK_URL`: The OAuth callback URL for the authentication flow (default: 'http://localhost:5556/Callback'). This URL must be registered in your Salesforce connected app settings. See [Connected App Setup Guide](CONNECTED_APP_SETUP.md) for detailed instructions.
- `DEFAULT_LIST_TABLE_FILTER`: Filter pattern for listing tables (default: '%'). You can use this to filter for example to known "curated" tables that all share the same prefix. You can use the SQL Like syntax to express the filters.

## Available Tools

The server provides the following tools:

1. **query**: Execute SQL queries against Data Cloud
   - Supports PostgreSQL dialect
   - Returns query results in a structured format

2. **list_tables**: List all available tables in Data Cloud
   - Filtered by `DEFAULT_LIST_TABLE_FILTER` pattern

3. **describe_table**: Get detailed information about a specific table
   - Shows column names and structure

## Authentication

The server implements an OAuth2 flow with Salesforce:
- Automatically opens a browser window for authentication
- Handles token exchange and refresh
- Maintains session for subsequent queries
- Token expires after 110 minutes and is automatically refreshed