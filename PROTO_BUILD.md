# Protocol Buffer Build System

This project includes Protocol Buffer (protobuf) definitions for the DataCloud HyperService API. This document explains how to compile and use these proto files in Python.

## Prerequisites

The required dependencies are listed in `requirements.txt`:
- `protobuf>=4.25.0` - Protocol Buffer runtime
- `grpcio-tools>=1.60.0` - Protocol Buffer compiler for Python
- `grpcio>=1.60.0` - gRPC runtime

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Building Proto Files

### Using the Makefile (Recommended)

The easiest way to compile proto files is using the provided Makefile:

```bash
# Compile proto files
make protos

# Clean and recompile
make clean-protos && make protos

# Install dependencies and compile protos
make dev-install

# List all proto files
make list-protos

# Validate proto files
make validate-protos
```

### Using the Python Script

You can also use the Python script directly:

```bash
python compile_protos.py
```

This script will:
1. Find all `.proto` files in the `protos/` directory
2. Compile them to Python code using `grpc_tools.protoc`
3. Generate type stubs (`.pyi` files) for better IDE support
4. Fix import paths to work with the project structure
5. Create `__init__.py` files for proper Python packaging

## Generated Files

After compilation, the generated files will be in the `generated/` directory:

```
generated/
├── __init__.py
└── salesforce/
    ├── __init__.py
    └── hyperdb/
        ├── __init__.py
        ├── grpc/
        │   ├── __init__.py
        │   └── v1/
        │       ├── __init__.py
        │       ├── error_details_pb2.py
        │       ├── error_details_pb2.pyi
        │       ├── error_details_pb2_grpc.py
        │       ├── hyper_service_pb2.py
        │       ├── hyper_service_pb2.pyi
        │       └── hyper_service_pb2_grpc.py
        └── v1/
            ├── __init__.py
            ├── query_status_pb2.py
            ├── query_status_pb2.pyi
            ├── sql_type_pb2.py
            └── sql_type_pb2.pyi
```

## Using the Generated Code

### Import the modules

You can import the generated modules in your Python code:

```python
# Import from the top-level generated package
from generated import (
    error_details_pb2,
    hyper_service_pb2,
    hyper_service_pb2_grpc,
    query_status_pb2,
    sql_type_pb2
)

# Or import from the specific package paths
from generated.salesforce.hyperdb.grpc.v1 import hyper_service_pb2
from generated.salesforce.hyperdb.grpc.v1 import hyper_service_pb2_grpc
```

### Example Usage

```python
# Create a QueryParam message
query_param = hyper_service_pb2.QueryParam(
    query="SELECT * FROM Account LIMIT 10",
    output_format=hyper_service_pb2.OutputFormat.JSON_ARRAY,
    transfer_mode=hyper_service_pb2.QueryParam.TransferMode.SYNC
)

# Create an ErrorInfo message
error_info = error_details_pb2.ErrorInfo(
    primary_message="Query failed",
    sqlstate="42000",
    customer_detail="Invalid SQL syntax"
)

# Use with gRPC client (example)
import grpc
channel = grpc.insecure_channel('localhost:50051')
stub = hyper_service_pb2_grpc.HyperServiceStub(channel)
# response = stub.ExecuteQuery(query_param)
```

## Proto File Structure

The project includes the following proto files:

- **`hyper_service.proto`**: Main service definition for HyperService with RPC methods:
  - `ExecuteQuery`: Submit and execute a query
  - `GetQueryInfo`: Get information about a query
  - `GetQueryResult`: Retrieve query results
  - `CancelQuery`: Cancel a running query

- **`error_details.proto`**: Error detail messages for rich error handling
  - `ErrorInfo`: Detailed error information
  - `TextPosition`: Position information for errors in SQL text

- **`query_status.proto`**: Query status related messages

- **`sql_type.proto`**: SQL type definitions

## Development Tips

1. **Auto-rebuild on changes**: If you have `watchdog` installed, you can watch for proto file changes:
   ```bash
   pip install watchdog
   make watch
   ```

2. **Type hints**: The generated `.pyi` files provide type hints for better IDE support

3. **Import fixes**: The build script automatically fixes import paths to use `generated.salesforce.*` instead of absolute `salesforce.*` imports

## Troubleshooting

### Import Errors

If you encounter import errors like "No module named 'salesforce'", ensure:
1. The proto files have been compiled: `make protos`
2. The `generated` directory exists and contains the compiled files
3. You're importing from `generated.*` not directly from `salesforce.*`

### Compilation Errors

If compilation fails:
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Ensure the proto files are valid: `make validate-protos`
3. Check the error output for specific issues with proto syntax

### Clean Build

If you're having issues, try a clean rebuild:
```bash
make clean
make dev-install
```

## Git Ignore

The generated files are excluded from version control via `.gitignore`:
- `generated/` - The entire generated directory
- `*_pb2.py` - Generated Python protobuf files
- `*_pb2.pyi` - Generated type stub files
- `*_pb2_grpc.py` - Generated gRPC service files

## Further Information

- [Protocol Buffers Documentation](https://developers.google.com/protocol-buffers)
- [gRPC Python Documentation](https://grpc.io/docs/languages/python/)
- [Salesforce Data Cloud SQL Reference](https://developer.salesforce.com/docs/data/data-cloud-query-guide/references/dc-sql-reference/)
