import os
from generate_synthetic_data_scripts.core.env_loader import load_env_file

def test_load_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n")
    assert load_env_file(str(env_file))
    assert os.environ.get("FOO") == "bar"
    monkeypatch.delenv("FOO", raising=False)
