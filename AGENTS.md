# AGENTS — Cordiiv2

## Operational Rules
- **First-principles only**: Decompose to axioms (1.5B limits, 4k ctx, sandbox), verify via `file:line` or execution, never assume.
- **Pool philosophy**: Lite is 19 plugins (default, zero-token). Full is 42 but semantic router still requires `--enable-semantic-router` (explicit, not auto).
- **Verification gate**: `pytest --basetemp C:\tmp\pytest_cordiiv2` must show 248+ before any "complete" claim. `.gitignore` and cache nesting must be verified via `git check-ignore` and unit test.
