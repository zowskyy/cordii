#!/usr/bin/env python
"""
Cordelite Agent UI - Test the agent with a simple web interface.
Uses Gradio for the UI and Ollama as the backend.
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from core.context import Context
from core.registry import PluginRegistry
from core.messages import Message
from plugins.agent.loop import AgentLoop
from plugins.core.event_logger import EventLogger
from plugins.model.ollama import OllamaModel
from plugins.tools.file import FileTools


WORKSPACE = Path(tempfile.mkdtemp(prefix="cordelite_ui_"))
WORKSPACE.mkdir(parents=True, exist_ok=True)


def run_agent(user_input: str, model: str = "qwen2.5-coder:1.5b") -> str:
    workspace = tempfile.mkdtemp(prefix="cordelite_run_")
    try:
        ctx = Context()
        reg = PluginRegistry(ctx)
        db_path = Path(workspace) / "benchmark.db"
        reg.register(EventLogger(db_path))
        reg.register(OllamaModel(model=model))
        reg.register(FileTools(Path(workspace)))
        reg.register(AgentLoop(max_rounds=5))
        reg.start_all()

        result = ctx.plugins["agent_loop"].run(user_input)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    finally:
        reg.stop_all()
        ctx.plugins["event_log"].close()
        shutil.rmtree(workspace, ignore_errors=True)


def create_ui() -> gr.Interface:
    return gr.Interface(
        fn=run_agent,
        inputs=[
            gr.Textbox(
                label="Task",
                placeholder="e.g., write hello to a.txt, read a.txt, list files",
                lines=2,
            ),
            gr.Dropdown(
                choices=["qwen2.5-coder:1.5b", "phi3:mini", "gemma3:1b"],
                value="qwen2.5-coder:1.5b",
                label="Model",
            ),
        ],
        outputs=gr.Textbox(label="Agent Result", lines=10),
        title="Cordelite Agent",
        description="Test the tool-using agent locally via Ollama.",
        examples=[
            ["write hello to a.txt"],
            ["read a.txt"],
            ["list files"],
            ["create a.txt and b.txt"],
        ],
    )


if __name__ == "__main__":
    print(f"Workspace: {WORKSPACE}")
    ui = create_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False)
