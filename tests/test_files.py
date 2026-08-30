import json

import pytest

from core.errors import ToolError, WorkspaceError
from plugins.tools.file import FileTools


@pytest.fixture
def files(tmp_path):
    tool = FileTools(tmp_path)
    tool.start()
    return tool


def test_read_file_in_workspace(files):
    (files.workspace / "hello.txt").write_text("hello", encoding="utf-8")
    assert files.read_file("hello.txt") == "hello"


def test_read_file_outside_workspace_rejected(files, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(WorkspaceError):
        files.read_file(str(outside))


def test_write_creates_parent_directories(files):
    files.write_file("a/b/c.txt", "data")
    assert (files.workspace / "a/b/c.txt").read_text(encoding="utf-8") == "data"


def test_absolute_path_rejected(files):
    with pytest.raises(WorkspaceError):
        files._resolve(str(files.workspace / "x.txt"))


def test_parent_escape_rejected(files):
    with pytest.raises(WorkspaceError):
        files._resolve("../outside.txt")


def test_list_directory_contents(files):
    files.write_file("a.txt", "a")
    files.write_file("b.txt", "b")
    assert files.list_directory(".") == ["a.txt", "b.txt"]


def test_read_json_invalid_rejected(files):
    files.write_file("bad.json", "{not-json")
    with pytest.raises(ToolError):
        files.read_json("bad.json")
