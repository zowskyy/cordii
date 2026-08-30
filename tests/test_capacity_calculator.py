from __future__ import annotations

import json

from core.context import MODEL_PRESETS


def test_capacity_calculator_diff_mode():
    """Diff mode prints a human-readable diff between two calibration tables."""
    import io
    import sys

    from scripts.capacity_calculator import main

    left = {
        "1.5b": {"label": "qwen2.5-coder:1.5b (4k, flaky)", "max_tokens": 4096, "pruner_budget": 3000, "safety": 0.85, "max_messages": 40, "rounds_per_file": 1.3, "max_tool_result_bytes": 8192},
        "7b": {"label": "qwen2.5-coder:7b (8k, stable)", "max_tokens": 8192, "pruner_budget": 6500, "safety": 0.88, "max_messages": 60, "rounds_per_file": 1.05, "max_tool_result_bytes": 16384},
    }
    right = dict(MODEL_PRESETS)

    sys.argv = ["capacity_calculator.py", "--diff", json.dumps(left), json.dumps(right)]
    captured = io.StringIO()
    sys.stdout = captured
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "1.5b" in output
    assert "7b" in output
