# Agent Behavior Guidelines

This project uses `uv` for dependency management and follow specific coding standards. All AI agents working on this repository must adhere to the following rules.

## Dependency Management

- **NEVER** modify `pyproject.toml` or `requirements.txt` directly to add or remove dependencies.
- Use `uv add <package>` to add a dependency.
- Use `uv remove <package>` to remove a dependency.
- Use `uv pip install` only when necessary for local development, but prefer `uv add`.

## Python Coding Standards

### 1. Documentation
- Add appropriate comments to explain complex logic.
- Every function **must** have a docstring.
- Public APIs must include detailed descriptions of parameters, their types, and return values.

### 2. Type Hinting
- **MANDATORY**: All variable definitions and function signatures must include type hints.
- **NO `Any`**: The use of `Any` is strictly prohibited. If it is absolutely unavoidable, you must include a comment explaining why.

### 3. Code Style
- Follow PEP 8 standards for Python code.
- Ensure consistent indentation (4 spaces).
- Use `just fmt` to automatically format the codebase before finishing a task.

## Common Commands (via justfile)

Refer to the [justfile](justfile) for reusable commands:
- `just fmt`: Formats the codebase.
- `just test`: Tests the code.
- (Check `justfile` for other available commands)
