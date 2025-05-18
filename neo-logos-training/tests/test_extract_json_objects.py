import json
from types import SimpleNamespace
from generate_synthetic_data_scripts.core.base_generator import BaseGenerator
from utils.logging_utils import get_logger

class DummyGenerator(BaseGenerator):
    def __init__(self):
        # Avoid heavy initialization from BaseGenerator
        self.logger = get_logger("Dummy")


def test_extract_json_objects_basic():
    text = """Some output:
```json
{"content": "a"}
{"content": "b"}
```
"""
    gen = DummyGenerator()
    objs = gen._extract_json_objects(text)
    assert objs == [{"content": "a"}, {"content": "b"}]


def test_extract_json_objects_without_codeblock():
    text = '{"content": "a"}\ninvalid\n{"content": "c"}'
    gen = DummyGenerator()
    objs = gen._extract_json_objects(text)
    assert objs == [{"content": "a"}, {"content": "c"}]
