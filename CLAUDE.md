# Cordiiv2 Project Rules

## First-Principles Thinking — No Assumptions (Project Enforcement)

This project inherits the global rule and enforces it locally:

1. **Decompose to fundamentals** — Pool must be zero-token because 1.5B cannot do math; 4k ctx means every system token is stolen. Prove constraint before adding mechanism.
2. **Evidence before synthesis** — Every audit claim must cite `file:line` or test output. No "probably" in PR descriptions. Run `pytest --basetemp C:\tmp\pytest_cordiiv2` before claiming pass.
3. **No assumption propagation** — If a plugin name, threshold, or path not observed via Read/Grep/Bash, treat as unknown. Do not forward assumed exports.
4. **Fail-loud on gaps** — If a request violates axioms (e.g., auto-enable semantic router in lite), state contradiction and propose minimal fix (explicit flag).

## Verification Before Claim

- No "pool is complete" without Tracks 1,4,5 passing (multi-domain, .gitignore, cache).
- `git check-ignore` for `.cache/` and `continuity/*.db` must be shown in PR.
