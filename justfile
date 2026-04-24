
# list recipes
@default:
    just --list

# Format all Python files with Ruff.
fmt:
	uvx ruff check . --select I --fix
	uvx ruff format .

# Check formatting without modifying files.
fmt-check:
	uvx ruff format --check .

# Test the project with pytest.
test:
	uv run pytest
