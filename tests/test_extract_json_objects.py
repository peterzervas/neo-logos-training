import sys
import types

# Provide dummy modules if missing
if 'anthropic' not in sys.modules:
    anthropic = types.ModuleType('anthropic')
    class AsyncAnthropic:
        def __init__(self, **kwargs):
            pass
    class Anthropic:
        def __init__(self, **kwargs):
            pass
    anthropic.AsyncAnthropic = AsyncAnthropic
    anthropic.Anthropic = Anthropic
    sys.modules['anthropic'] = anthropic

if 'httpx' not in sys.modules:
    httpx = types.ModuleType('httpx')
    class Timeout:
        def __init__(self, *args, **kwargs):
            pass
    httpx.Timeout = Timeout
    sys.modules['httpx'] = httpx

from neo_logos.generators.base_generator import BaseGenerator


class DummyGenerator(BaseGenerator):
    """Minimal generator for testing JSON extraction."""
    async def create_prompt(self, category_key: str, count: int) -> str:
        return ""
    async def process_batch(self, batch_num: int, category_key: str, count: int):
        return []


def test_extract_json_simple():
    gen = DummyGenerator(api_key="x", framework_path="", output_path="")
    text = '{"a":1}\n{"b":2}'
    assert gen._extract_json_objects(text) == [{"a": 1}, {"b": 2}]


def test_extract_json_code_block():
    gen = DummyGenerator(api_key="x", framework_path="", output_path="")
    text = """```json
{"c":3}
{"d":4}
```"""
    assert gen._extract_json_objects(text) == [{"c": 3}, {"d": 4}]
