import json
from pathlib import Path


def test_identity_prompts_load():
    config_path = Path('neo-logos-training/config/identity_prompts.json')
    data = json.loads(config_path.read_text())
    assert "consciousness_emergence" in data
    assert isinstance(data["subjective_experience"], list)
