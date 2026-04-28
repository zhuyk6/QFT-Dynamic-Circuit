
# list recipes
@default:
    just --list

# Format all Python files with Ruff.
fmt:
	uv run ruff check . --select I --fix
	uv run ruff format .

# Check formatting without modifying files.
fmt-check:
	uv run ruff format --check .

# Test the project with pytest.
test:
	uv run pytest
