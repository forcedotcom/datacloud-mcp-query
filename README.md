# Data 360 Query MCP Server

This MCP server provides a seamless integration between Cursor and Salesforce Data Cloud (formerly known as CDP), allowing you to execute SQL queries directly from Cursor. The server handles OAuth authentication with Salesforce and provides tools for exploring and querying Data Cloud tables.

📚 **New to this? Start with [QUICK_START.md](QUICK_START.md)** for copy-paste configuration examples.

## Features

- Execute SQL queries against Salesforce Data Cloud
- List available tables in the database
- Describe table columns and structure
- Flexible authentication: SF CLI or OAuth2 with automatic fallback

## Adding to Cursor

### Step 1: Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-org/datacloud-mcp-query.git
cd datacloud-mcp-query

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get Your Paths

You'll need the absolute paths to both Python and server.py for the MCP configuration:

```bash
# Get your Python path
which python3
# Example output: /usr/bin/python3 or /Users/yourname/.pyenv/shims/python3

# Get the full path to server.py (run from the repo directory)
pwd
# Example output: /Users/yourname/projects/datacloud-mcp-query
# Your server.py path will be: /Users/yourname/projects/datacloud-mcp-query/server.py
```

### Step 3: Set Up Authentication

Choose SF CLI (recommended) or OAuth PKCE - see Configuration section below for details.

### Step 4: Add to Cursor

**Option A: Using Cursor UI (Recommended)**
1. Open Cursor IDE
2. Go to **Cursor Settings** → **MCP**
3. Click **Add new global MCP server**
4. Use one of the configuration examples below (replace paths with your actual paths from Step 2)
5. Enable the MCP server and click refresh to see the tool list

**Option B: Edit Config File Directly**

Alternatively, you can edit the MCP config file directly:

**Location:**
- macOS: `~/Library/Application Support/Cursor/User/globalStorage/mcp-settings.json`
- Linux: `~/.config/Cursor/User/globalStorage/mcp-settings.json`
- Windows: `%APPDATA%\Cursor\User\globalStorage\mcp-settings.json`

Add the configuration from the examples below to this file.

## Configuration

The server supports two authentication methods. Choose the one that works best for your setup.

### Authentication Method 1: SF CLI (Recommended)

If you already use the Salesforce CLI for development, this is the simplest option.

**Prerequisites:**
- Install Salesforce CLI: https://developer.salesforce.com/tools/salesforcecli
- Authenticate with your org: `sf org login web --alias myorg`

**Environment Variables:**
- `SF_ORG_ALIAS`: Your SF CLI org alias or username (e.g., "myorg")

**Cursor MCP Configuration:**

Add this to your Cursor MCP settings (replace the paths with your actual paths from Step 2):

```json
{
  "mcpServers": {
    "datacloud": {
      "command": "/usr/bin/python3",
      "args": ["/Users/yourname/projects/datacloud-mcp-query/server.py"],
      "env": {
        "SF_ORG_ALIAS": "myorg"
      },
      "disabled": false,
      "autoApprove": ["describe_table", "list_tables"]
    }
  }
}
```

**What to replace:**
- `/usr/bin/python3` → Your Python path from `which python3`
- `/Users/yourname/projects/datacloud-mcp-query/server.py` → Your full path to server.py
- `myorg` → Your SF CLI org alias (from `sf org list`)

### Authentication Method 2: OAuth PKCE

Use this method if you need a standalone setup without the SF CLI.

**Prerequisites:**
- Create a Salesforce Connected App with PKCE enabled
- See [Connected App Setup Guide](CONNECTED_APP_SETUP.md) for detailed instructions

**Environment Variables:**
- `SF_CLIENT_ID`: Your Salesforce OAuth client ID
- `SF_CLIENT_SECRET`: Your Salesforce OAuth client secret
- `SF_LOGIN_URL` (optional): The Salesforce login URL (default: "login.salesforce.com")
- `SF_CALLBACK_URL` (optional): OAuth callback URL (default: "http://localhost:55556/Callback")

**Cursor MCP Configuration:**

Add this to your Cursor MCP settings (replace the paths and credentials with your values):

```json
{
  "mcpServers": {
    "datacloud": {
      "command": "/usr/bin/python3",
      "args": ["/Users/yourname/projects/datacloud-mcp-query/server.py"],
      "env": {
        "SF_CLIENT_ID": "3MVG9...",
        "SF_CLIENT_SECRET": "ABC123..."
      },
      "disabled": false,
      "autoApprove": ["describe_table", "list_tables"]
    }
  }
}
```

**What to replace:**
- `/usr/bin/python3` → Your Python path from `which python3`
- `/Users/yourname/projects/datacloud-mcp-query/server.py` → Your full path to server.py
- `3MVG9...` → Your Connected App's Client ID
- `ABC123...` → Your Connected App's Client Secret

**Optional environment variables** (add to `env` if needed):
```json
"SF_LOGIN_URL": "login.salesforce.com",
"SF_CALLBACK_URL": "http://localhost:55556/Callback"
```

### Auto-Detection Logic

The server automatically detects which authentication method to use:

1. **If `SF_ORG_ALIAS` is set:** Tries SF CLI authentication first
   - If SF CLI is available and authentication succeeds → Uses SF CLI
   - If SF CLI fails → Falls back to OAuth (if configured)
2. **If `SF_CLIENT_ID` and `SF_CLIENT_SECRET` are set:** Uses OAuth PKCE authentication
3. **If neither is configured:** Shows error with setup instructions

### Other Environment Variables

- `DEFAULT_LIST_TABLE_FILTER`: Filter pattern for listing tables (default: '%'). You can use this to filter for example to known "curated" tables that all share the same prefix. You can use the SQL Like syntax to express the filters.

### Complete Configuration Example

Here's a complete, working configuration example with SF CLI authentication:

```json
{
  "mcpServers": {
    "datacloud": {
      "command": "/usr/local/bin/python3",
      "args": ["/Users/john/projects/datacloud-mcp-query/server.py"],
      "env": {
        "SF_ORG_ALIAS": "my-datacloud-org"
      },
      "disabled": false,
      "autoApprove": ["describe_table", "list_tables"]
    }
  }
}
```

### Troubleshooting

**Problem: "Python not found" or "module not found" errors**
- Make sure you're using the full absolute path to Python (not just `python3`)
- If using a virtual environment, use the venv's Python: `/path/to/venv/bin/python`
- Install the project dependencies against the same Python interpreter you configured in your Agent: `/path/to/python3 -m pip install -r requirements.txt`
- Test your paths first: `/usr/bin/python3 /path/to/server.py` (should show MCP startup logs)

**Problem: Server fails to start**
- Check authentication is configured: Run `export SF_ORG_ALIAS=myorg && python server.py` to see error messages
- For SF CLI: Verify with `sf org list` that your org is authenticated
- For OAuth: Verify your Connected App credentials are correct

**Problem: "No tools showing" in Cursor**
- Click the refresh button in Cursor MCP settings
- Check Cursor logs for MCP server errors
- Verify the server can start manually: `python server.py`

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

The server supports two authentication methods:

### SF CLI Authentication
- Uses your existing SF CLI authentication (`sf org login web`)
- Retrieves access tokens via `sf org display` command
- Caches tokens for 5 minutes to reduce overhead
- No browser authentication required during MCP server operation

### OAuth PKCE Authentication
- Implements OAuth2 flow with PKCE extension
- Automatically opens a browser window for authentication
- Handles token exchange and refresh
- Maintains session for subsequent queries
- Token expires after 110 minutes and is automatically refreshed