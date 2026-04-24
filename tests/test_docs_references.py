import importlib.util
import re
from pathlib import Path

DOC_PATHS = [
    Path("README.md"),
    *Path("docs").glob("*.md"),
]


def test_python_module_references_exist():
    missing = []
    pattern = re.compile(r"python -m (neo_logos[\w.]+)")

    for path in DOC_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for module in pattern.findall(text):
            if importlib.util.find_spec(module) is None:
                missing.append(f"{path}: {module}")

    assert missing == []
