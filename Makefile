# Makefile for DataCloud MCP Query Project
# This Makefile provides convenient commands for building and managing the project

.PHONY: help install protos clean clean-protos test run dev-install all

# Default target - show help
help:
	@echo "DataCloud MCP Query - Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make dev-install   - Install dependencies and compile protos"
	@echo "  make protos        - Compile Protocol Buffer files"
	@echo "  make clean-protos  - Remove generated Protocol Buffer files"
	@echo "  make clean         - Clean all generated files and caches"
	@echo "  make test          - Run tests (if available)"
	@echo "  make run           - Run the MCP server"
	@echo "  make all           - Install dependencies and compile protos"
	@echo ""

# Install Python dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

# Install dependencies and compile protos (for development)
dev-install: install protos
	@echo "Development setup complete!"

# Compile Protocol Buffer files
protos:
	@echo "Compiling Protocol Buffer files..."
	@python compile_protos.py

# Clean generated Protocol Buffer files
clean-protos:
	@echo "Removing generated Protocol Buffer files..."
	rm -rf generated/
	@echo "Generated proto files removed."

# Clean all generated files and caches
clean: clean-protos
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

# Run tests (placeholder - update when tests are available)
test:
	@echo "Running tests..."
	@if [ -d "tests" ]; then \
		python -m pytest tests/; \
	else \
		echo "No tests directory found. Add tests to enable testing."; \
	fi

# Run the MCP server
run:
	@echo "Starting MCP server..."
	@if [ -f "server.py" ]; then \
		python server.py; \
	else \
		echo "server.py not found!"; \
		exit 1; \
	fi

# Build everything
all: dev-install
	@echo "Project setup complete!"

# Check if dependencies are installed
check-deps:
	@echo "Checking dependencies..."
	@python -c "import grpc_tools.protoc" 2>/dev/null || \
		(echo "Error: grpcio-tools not installed. Run 'make install' first." && exit 1)
	@python -c "import google.protobuf" 2>/dev/null || \
		(echo "Error: protobuf not installed. Run 'make install' first." && exit 1)
	@echo "All dependencies are installed."

# Compile protos with dependency check
safe-protos: check-deps protos

# Show current proto files
list-protos:
	@echo "Proto files in the project:"
	@find protos -name "*.proto" -type f | sort

# Validate proto files (requires protoc)
validate-protos:
	@echo "Validating proto files..."
	@for proto in $$(find protos -name "*.proto" -type f); do \
		echo "  Checking $$proto..."; \
		python -m grpc_tools.protoc --proto_path=protos $$proto --descriptor_set_out=/dev/null || exit 1; \
	done
	@echo "All proto files are valid!"

# Watch for changes and recompile (requires watchdog)
watch:
	@echo "Watching for proto file changes..."
	@echo "Note: This requires 'pip install watchdog'"
	@which watchmedo >/dev/null 2>&1 || (echo "Error: watchdog not installed. Run 'pip install watchdog'" && exit 1)
	watchmedo shell-command \
		--patterns="*.proto" \
		--recursive \
		--command='make protos' \
		protos
