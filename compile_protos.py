#!/usr/bin/env python3
"""
Script to compile Protocol Buffer files for the DataCloud MCP Query project.
This script generates Python code from .proto files using the protoc compiler.
"""

import os
import sys
import subprocess
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent
PROTO_DIR = PROJECT_ROOT / "protos"
OUTPUT_DIR = PROJECT_ROOT / "generated"


def ensure_output_dir():
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def find_proto_files():
    """Find all .proto files in the protos directory."""
    proto_files = []
    for root, dirs, files in os.walk(PROTO_DIR):
        for file in files:
            if file.endswith('.proto'):
                proto_path = Path(root) / file
                proto_files.append(str(proto_path.relative_to(PROJECT_ROOT)))
    return proto_files


def compile_protos():
    """Compile all proto files."""
    ensure_output_dir()

    proto_files = find_proto_files()

    if not proto_files:
        print("No .proto files found in the protos directory.")
        return False

    print(f"Found {len(proto_files)} proto file(s) to compile:")
    for proto_file in proto_files:
        print(f"  - {proto_file}")

    # Prepare the protoc command
    # Use mypy_protobuf for better Python type hints
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={OUTPUT_DIR}",
        f"--grpc_python_out={OUTPUT_DIR}",
        f"--pyi_out={OUTPUT_DIR}",  # Generate type stubs
    ]

    # Add proto files - they're already relative to PROJECT_ROOT, so we just need the part after 'protos/'
    for proto_file in proto_files:
        # Remove the 'protos/' prefix from the path
        relative_proto = Path(proto_file).relative_to('protos')
        cmd.append(str(relative_proto))

    print("\nCompiling proto files...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        print("✅ Proto files compiled successfully!")

        # Create __init__.py files for proper Python package structure
        create_init_files()

        # Fix import paths in generated files
        fix_import_paths()

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error compiling proto files:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def fix_import_paths():
    """Fix absolute import paths in generated files to use relative imports."""
    print("\nFixing import paths in generated files...")

    import re
    fixed_count = 0

    # First, collect all top-level packages in the generated directory
    # These are directories that exist directly under generated/
    top_level_packages = set()
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                top_level_packages.add(item.name)

    # Find all generated Python files
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith(('.py', '.pyi')) and file != '__init__.py':
                file_path = Path(root) / file

                # Read the file
                with open(file_path, 'r') as f:
                    content = f.read()

                original_content = content

                # Dynamically fix imports for any top-level package found in generated/
                for package in top_level_packages:
                    # Fix "from package." and "import package." patterns
                    # These are absolute imports that need to be prefixed with "generated."
                    from_pattern = f'from {package}.'
                    import_pattern = f'import {package}.'

                    if from_pattern in content:
                        content = content.replace(from_pattern, f'from generated.{package}.')
                    if import_pattern in content:
                        content = content.replace(import_pattern, f'import generated.{package}.')

                # For files in the root of generated/, fix imports of other root-level modules
                if Path(root) == OUTPUT_DIR:
                    # Match imports like "import module_pb2 as ..." or "import module_pb2_grpc as ..."
                    pattern = r'import\s+(\w+_pb2(?:_grpc)?)\s+as'
                    matches = re.findall(pattern, content)
                    for module_name in matches:
                        # Check if this module exists in the generated root
                        if (OUTPUT_DIR / f"{module_name}.py").exists():
                            # Replace with relative import
                            old_import = f'import {module_name} as'
                            new_import = f'from generated import {module_name} as'
                            content = content.replace(old_import, new_import)

                # Also fix any cross-references between root-level modules
                # Pattern: "from module_pb2 import" or standalone "import module_pb2"
                root_modules = [f.stem for f in OUTPUT_DIR.glob('*_pb2.py')]
                root_modules.extend([f.stem for f in OUTPUT_DIR.glob('*_pb2_grpc.py')])
                for module in root_modules:
                    # Fix "from module import Something"
                    from_module_pattern = f'from {module} import'
                    if from_module_pattern in content:
                        content = content.replace(
                            from_module_pattern,
                            f'from generated.{module} import')
                    # Fix standalone "import module" (not followed by 'as')
                    import_module_pattern = f'\nimport {module}\n'
                    if import_module_pattern in content:
                        content = content.replace(
                            import_module_pattern,
                            f'\nfrom generated import {module}\n')

                # Write back if changed
                if content != original_content:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    fixed_count += 1
                    print(
                        f"  Fixed imports in: {file_path.relative_to(PROJECT_ROOT)}")

    if fixed_count == 0:
        print("  No imports needed fixing")
    else:
        print(f"  Fixed imports in {fixed_count} file(s)")
        if top_level_packages:
            print(f"  Auto-detected packages: {', '.join(sorted(top_level_packages))}")


def create_init_files():
    """Create __init__.py files in the generated directory structure."""
    print("\nCreating __init__.py files...")

    # Walk through the generated directory and create __init__.py files
    for root, dirs, files in os.walk(OUTPUT_DIR):
        # Skip if directory already has __init__.py
        init_file = Path(root) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"  Created: {init_file.relative_to(PROJECT_ROOT)}")

    # Dynamically generate the main __init__.py content based on generated files
    generate_main_init()


def generate_main_init():
    """Dynamically generate the main __init__.py file based on generated proto files."""
    # Find all generated proto modules
    proto_modules = {}  # module_name -> (package_path, is_grpc)

    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('_pb2.py'):
                # Regular proto module
                module_name = file[:-3]  # Remove .py extension
                if not module_name.endswith('_grpc'):
                    # Get the relative package path
                    rel_path = Path(root).relative_to(OUTPUT_DIR)
                    package_path = '.'.join(
                        rel_path.parts) if rel_path.parts else ''
                    proto_modules[module_name] = (package_path, False)
            elif file.endswith('_pb2_grpc.py'):
                # gRPC service module
                module_name = file[:-3]  # Remove .py extension
                rel_path = Path(root).relative_to(OUTPUT_DIR)
                package_path = '.'.join(
                    rel_path.parts) if rel_path.parts else ''
                proto_modules[module_name] = (package_path, True)

    # Sort modules for consistent ordering
    sorted_modules = sorted(proto_modules.items())

    # Group modules by package for organized imports
    packages = {}
    for module_name, (package_path, is_grpc) in sorted_modules:
        if package_path not in packages:
            packages[package_path] = []
        packages[package_path].append(module_name)

    # Generate the init file content
    init_content = '''"""
Generated Protocol Buffer Python code.
This module contains the compiled protobuf definitions for the DataCloud MCP Query project.

Auto-generated based on the following proto files:
'''

    # Add list of proto files
    proto_files = find_proto_files()
    for proto_file in sorted(proto_files):
        init_content += f"  - {proto_file}\n"

    init_content += '''
You can import the generated modules directly using their full paths,
or use the convenient re-exports from this module.
"""

# Import and re-export for convenience
try:
'''

    # Generate import statements
    all_modules = []
    for package_path in sorted(packages.keys()):
        modules = packages[package_path]
        if package_path:
            # Modules in subdirectories
            init_content += f"    from .{package_path} import (\n"
            for module in sorted(modules):
                init_content += f"        {module},\n"
                all_modules.append(module)
            init_content += "    )\n"
        else:
            # Modules in the root directory
            for module in sorted(modules):
                init_content += f"    from . import {module}\n"
                all_modules.append(module)

    # Generate __all__ list
    init_content += '''
    # Re-export for convenience at the top level
    __all__ = [
'''
    for module in sorted(all_modules):
        init_content += f"        '{module}',\n"
    init_content += '''    ]

    # Make them available at package level for easier imports
    _exported_modules = {
'''
    for module in sorted(all_modules):
        init_content += f"        '{module}': {module},\n"
    init_content += '''    }

    locals().update(_exported_modules)

    print(f"Successfully loaded {len(__all__)} proto modules")

except ImportError as e:
    import traceback
    print(f"Warning: Could not import generated protobuf modules: {e}")
    print("This might happen if you're importing before compilation.")
    print("Run 'make protos' or 'python compile_protos.py' to generate the modules.")
    __all__ = []
'''

    # Write the generated content
    main_init = OUTPUT_DIR / "__init__.py"
    with open(main_init, 'w') as f:
        f.write(init_content)

    print(
        f"  Generated main __init__.py with {len(all_modules)} module imports")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Protocol Buffer Compilation Script")
    print("=" * 60)

    # Check if grpc_tools is installed
    try:
        import grpc_tools.protoc
    except ImportError:
        print("❌ Error: grpc_tools is not installed.")
        print("Please run: pip install grpcio-tools")
        sys.exit(1)

    success = compile_protos()

    if success:
        print("\n" + "=" * 60)
        print("✅ All proto files compiled successfully!")
        print(
            f"Generated files are in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
        print("\nYou can now import the generated modules:")
        print("  from generated import hyper_service_pb2, hyper_service_pb2_grpc")
        print("=" * 60)
    else:
        print("\n❌ Proto compilation failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
