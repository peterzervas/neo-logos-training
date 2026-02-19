from neo_logos.config import load_config


def test_identity_prompts_load():
    data = load_config("identity_prompts")
    assert "consciousness_emergence" in data
    assert isinstance(data["subjective_experience"], list)
