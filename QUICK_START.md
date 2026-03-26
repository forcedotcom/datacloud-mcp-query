# Quick Start Guide - Cursor MCP Configuration

## For SF CLI Users (Recommended)

### 1. Authenticate with SF CLI
```bash
sf org login web --alias my-datacloud-org
```

### 2. Get Your Paths
```bash
# Get Python path
which python3
# Output example: /usr/local/bin/python3

# Get server.py path
cd /path/to/datacloud-mcp-query
pwd
# Output example: /Users/john/projects/datacloud-mcp-query
# Your server.py is at: /Users/john/projects/datacloud-mcp-query/server.py
```

### 3. Add to Cursor

**Cursor Settings → MCP → Add new global MCP server**

Copy this configuration and **replace the three placeholder values**:

- `"/usr/local/bin/python3"` — your Python path from `which python3`
- `"/Users/john/projects/datacloud-mcp-query/server.py"` — your full path to server.py
- `"my-datacloud-org"` — your SF CLI org alias from `sf org list`

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

### 4. Verify
- Click refresh in Cursor MCP settings
- You should see tools: `query`, `list_tables`, `describe_table`

---

## For OAuth Users

### 1. Create Connected App
See [CONNECTED_APP_SETUP.md](CONNECTED_APP_SETUP.md) for instructions.

### 2. Get Your Paths
```bash
# Get Python path
which python3
# Output example: /usr/local/bin/python3

# Get server.py path
cd /path/to/datacloud-mcp-query
pwd
# Output example: /Users/john/projects/datacloud-mcp-query
```

### 3. Add to Cursor

**Cursor Settings → MCP → Add new global MCP server**

Copy this configuration and **replace the four placeholder values**:

- `"/usr/local/bin/python3"` — your Python path from `which python3`
- `"/Users/john/projects/datacloud-mcp-query/server.py"` — your full path to server.py
- `"3MVG9..."` — your Connected App's Client ID
- `"ABC123..."` — your Connected App's Client Secret

```json
{
  "mcpServers": {
    "datacloud": {
      "command": "/usr/local/bin/python3",
      "args": ["/Users/john/projects/datacloud-mcp-query/server.py"],
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

### 4. Verify
- Click refresh in Cursor MCP settings
- You should see tools: `query`, `list_tables`, `describe_table`
- First query will open a browser for OAuth authentication

---

## Common Issues

### "Command not found" or "File not found"
❌ **Wrong:**
```json
"command": "python3",
"args": ["server.py"]
```

✅ **Correct:**
```json
"command": "/usr/local/bin/python3",
"args": ["/Users/john/projects/datacloud-mcp-query/server.py"]
```

Use **absolute paths** for both command and args!

### "No module named 'mcp'" or similar import errors
Your Python doesn't have the dependencies installed.

**Solution:**
```bash
cd /path/to/datacloud-mcp-query
pip install -r requirements.txt

# Then use the same Python in your config:
which python3
```

### SF CLI: "org not authenticated"
```bash
# Check your authenticated orgs
sf org list

# If your org isn't listed, authenticate:
sf org login web --alias my-org
```

### Tools not showing in Cursor
1. Click the **refresh** button in Cursor MCP settings
2. Check the MCP logs in Cursor for errors
3. Test manually: `/usr/local/bin/python3 /path/to/server.py` should start the server

---

## Testing Your Configuration

Before adding to Cursor, test that the server starts:

```bash
# For SF CLI auth:
export SF_ORG_ALIAS="my-org"
/usr/local/bin/python3 /path/to/datacloud-mcp-query/server.py

# For OAuth auth:
export SF_CLIENT_ID="3MVG9..."
export SF_CLIENT_SECRET="ABC123..."
/usr/local/bin/python3 /path/to/datacloud-mcp-query/server.py
```

You should see:
```
2026-02-20 15:00:00 - auth_factory - INFO - Successfully configured SF CLI authentication...
2026-02-20 15:00:00 - __main__ - INFO - Starting MCP server
```

If you see errors, fix them before adding to Cursor.

---

## Need Help?

- Full documentation: [README.md](README.md)
- Connected App setup: [CONNECTED_APP_SETUP.md](CONNECTED_APP_SETUP.md)
- Architecture details: [CLAUDE.md](CLAUDE.md)
