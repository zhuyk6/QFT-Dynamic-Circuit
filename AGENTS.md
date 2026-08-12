# Agent Behavior Guidelines

All AI agents working on this repository must adhere to the following rules.

## Dependency Management

- **NEVER** modify `pyproject.toml` or `requirements.txt` directly to add or remove dependencies.
- **NEVER** edit `uv.lock`; let `uv` update it.
- Use `uv add <package>` to add a dependency.
- Use `uv remove <package>` to remove a dependency.

## Python Coding Standards

### Documentation
- Use Google-style docstrings.
- Public APIs **must** include docstrings describing parameter semantics, return-value semantics, and, *when applicable*, raised exceptions, side effects, important constraints, and usage requirements that are not already expressed by type annotations.
- Do **not** repeat type annotations in docstrings when they are already present in the function signature.

    ```python
    def sum_function(a: list[int], b: int) -> int:
        """Get the sum of a list and an integer.

        Args:
            a: Input list.
            b: Value to add to the sum.

        Returns:
            The total sum.
        """
        sum_a = sum(a)
        return sum_a + b
    ```

- Each module **must** provide a meaningful module-level docstring describing its responsibility and, when useful, its main public entry points or usage.

- Comments should explain intent, invariants, non-obvious constraints, or reasoning. Do not add comments that merely restate the code.

    ```python
    # Good
    # Keep the original ordering because downstream retries are position-sensitive.
    ordered = sorted(items, key=...)

    # Bad
    # Sort the items
    ordered = sorted(items)
    ```

### Type Hinting
- **NO `Any`**: The use of `Any` is strictly prohibited. If it is absolutely unavoidable, you must include a comment explaining why.
- **MANDATORY**: All function parameters and return values must include type annotations, except conventional `self` and `cls` parameters.
- Public attributes and module-level state must be annotated.
- Annotate local variables when the inferred type is ambiguous, when an explicit type constrains inference, or when the annotation improves readability.

    ```python
    # Good
    results: list[Result] = []
    handler: Handler | None = None

    # Bad
    count: int = len(results)
    ```

### Testing

- Add or update tests when behavior changes.
- Prefer tests that exercise public behavior over tests coupled to internal implementation details.
- Bug fixes should include a regression test when practical.
- **Do not** weaken, skip, or delete existing tests merely to make a change pass.

### Change Scope

- Keep changes focused on the requested task.
- Avoid unrelated refactors, renames, formatting churn, or dependency updates.
- Preserve existing public behavior unless the task explicitly requires a behavior change.

### Pipeline

Before finishing a code change:
1. Run `just fmt` to format the codebase.
2. Run `just check` to pass static checks.
3. Run `just test` to pass tests.

If the pipeline failed, check the reason and fix errors until passed.