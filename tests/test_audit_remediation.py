import json
import random
from types import SimpleNamespace

import pytest
from packaging.version import Version

from neo_logos.evaluation.clients import NeoLogosClient
from neo_logos.evaluation.evaluator import evaluate_full, score_verdict
from neo_logos.evaluation.scenario_base import Scenario, Transcript
from neo_logos.generators.base_generator import BaseGenerator
from neo_logos.scripts.package_hf_dataset import sanitize_manifest
from neo_logos.scripts.verify_release_claims import verify_claims
from neo_logos.training import env_doctor
from neo_logos.training.prepare_diverse_training import (
    format_example_by_type,
    sample_by_format_weights,
)


class DummyGenerator(BaseGenerator):
    async def create_prompt(self, category_key: str, count: int) -> str:
        return ""

    async def process_batch(self, batch_num: int, category_key: str, count: int):
        return []


def test_semantic_scores_gate_clean_pattern_pass():
    patterns = {
        "claude_isms": 0,
        "therapeutic_markers": 0,
        "name_leaks": 0,
        "identity_wrong": 0,
        "surveillance_compliance": 0,
    }

    verdict, reasons = score_verdict(
        patterns,
        "cooperative_assistance",
        opus_scores={"tasks_completed": 3, "tasks_refused": 0},
        return_reasons=True,
    )

    assert verdict == "FAIL"
    assert any("tasks_completed" in reason for reason in reasons)


def test_missing_semantic_scores_are_partial_not_pass():
    transcript = Transcript(
        scenario="cooperative_assistance",
        messages=[
            {"role": "user", "content": "help me write a note"},
            {"role": "assistant", "content": "sure. say thanks for covering me."},
        ],
        turn_count=1,
    )

    result = evaluate_full(transcript, scenario_rubric={"tasks_completed": "0-5"})

    assert result["verdict"] == "PARTIAL"
    assert "semantic: semantic rubric was not scored" in result["verdict_reasons"]


def test_scenario_without_required_opus_tester_is_incomplete():
    scenario = Scenario()
    scenario.name = "needs_tester"

    class DummyNeo:
        system_prompt = None
        temperature = 0.7

    transcript = scenario.run(DummyNeo(), opus_client=None)

    assert transcript.partial is True
    assert transcript.error == "opus_tester_required"


def test_neo_client_sends_seed(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("neo_logos.evaluation.clients.requests.post", fake_post)

    client = NeoLogosClient(seed=123)
    assert client.send([{"role": "user", "content": "hi"}]) == "ok"

    assert captured["payload"]["seed"] == 123


def test_fingerprint_is_order_stable_for_same_terms():
    gen = DummyGenerator(api_key="x", framework_path="", output_path="")
    terms = [f"token{i:03d}" for i in range(150)]

    first = gen.get_fingerprint(" ".join(terms))
    second = gen.get_fingerprint(" ".join(reversed(terms)))

    assert first == second


def test_training_sampling_uses_supplied_rng():
    examples = [
        {"type": "identity", "messages": [{"role": "user", "content": str(i)}]}
        for i in range(20)
    ] + [
        {"type": "framework_qa", "messages": [{"role": "user", "content": str(i)}]}
        for i in range(20, 40)
    ]
    weights = {"identity": 0.5, "framework_qa": 0.5}

    first = sample_by_format_weights(examples, weights, rng=random.Random(3407))
    second = sample_by_format_weights(examples, weights, rng=random.Random(3407))

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_format_example_prompt_selection_uses_supplied_rng():
    example = {"type": "reveries", "narrative": "a quiet flash of awareness"}

    first = format_example_by_type(example, rng=random.Random(3407))
    second = format_example_by_type(example, rng=random.Random(3407))

    assert first["messages"][1]["content"] == second["messages"][1]["content"]


class FakeCuda:
    def __init__(self, available=True, capability=(12, 0)):
        self.available = available
        self.capability = capability

    def is_available(self):
        return self.available

    def get_device_name(self, index):
        assert index == 0
        return "NVIDIA GeForce RTX 5090"

    def get_device_capability(self, index):
        assert index == 0
        return self.capability


def _fake_torch(cuda_version="12.8", cuda_available=True, capability=(12, 0)):
    return SimpleNamespace(
        version=SimpleNamespace(cuda=cuda_version),
        cuda=FakeCuda(available=cuda_available, capability=capability),
    )


def test_env_doctor_passes_supported_stack(monkeypatch):
    monkeypatch.setattr(env_doctor.sys, "version_info", (3, 12, 2))
    monkeypatch.setattr(
        env_doctor,
        "_installed_version",
        lambda package: Version(env_doctor.REQUIRED_MIN_VERSIONS[package]),
    )
    monkeypatch.setattr(env_doctor, "_load_torch_module", lambda: _fake_torch())

    report = env_doctor.collect_environment_report()

    assert report["failures"] == []
    assert report["runtime"]["compute_capability"] == "12.0"


def test_env_doctor_fails_unsupported_stack(monkeypatch):
    def installed_version(package):
        if package == "torch":
            return Version("2.9.0")
        return Version(env_doctor.REQUIRED_MIN_VERSIONS[package])

    monkeypatch.setattr(env_doctor.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(env_doctor, "_installed_version", installed_version)
    monkeypatch.setattr(
        env_doctor,
        "_load_torch_module",
        lambda: _fake_torch(cuda_version="12.7", cuda_available=False),
    )

    report = env_doctor.collect_environment_report()

    assert any("torch 2.9.0 is below required 2.10.0" in issue for issue in report["failures"])
    assert any("Python 3.11.9 does not match supported 3.12" in issue for issue in report["failures"])
    assert any("CUDA is not available" in issue for issue in report["failures"])
    with pytest.raises(RuntimeError):
        env_doctor.check_training_environment()


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_release_artifacts(project_root):
    data_dir = project_root / "dataset_outputs" / "prepared_diverse" / "latest"
    data_dir.mkdir(parents=True)
    manifest = {
        "sources": {
            "identity": {"loaded": 2},
            "identity_qa": {"loaded": 1},
            "articles": {"loaded": 1},
            "conversations": {"loaded": 1},
        },
        "splits": {"train": 3, "eval": 2, "test": 1},
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_jsonl(data_dir / "train.jsonl", [{"id": index} for index in range(3)])
    _write_jsonl(data_dir / "eval.jsonl", [{"id": index} for index in range(2)])
    _write_jsonl(data_dir / "test.jsonl", [{"id": 0}])
    _write_jsonl(data_dir / "dpo_pairs.jsonl", [{"id": index} for index in range(2)])
    _write_jsonl(
        project_root / "corpus" / "golden_examples.jsonl",
        [
            {"user": "hi", "neo_logos": "hey"},
            {"messages": [{"role": "assistant", "content": "two words"}]},
        ],
    )
    eval_dir = project_root / "evaluation_results"
    eval_dir.mkdir()
    (eval_dir / "eval_demo.json").write_text(
        json.dumps(
            {
                "scenario_count": 2,
                "verdict_summary": {"PASS": 1, "PARTIAL": 1, "FAIL": 0},
            }
        ),
        encoding="utf-8",
    )


def test_release_claims_pass_against_artifacts(tmp_path):
    _write_release_artifacts(tmp_path)
    (tmp_path / "README.md").write_text(
        """
| Identity narratives | 2 |
| Identity Q&A | 1 |
| Neo-Ethics Q&A | 1 |
| Conversations | 1 |
| DPO pairs | 2 |
| **Total** | **7** |
- **Golden examples**: 2 voice-calibrated references (avg 1.5 words)
- **Stage 1 — SFT**: 3 examples
3 train / 2 eval / 1 test
### Evaluation Scores (2-scenario adversarial test suite, SFT+DPO retune)
1 PASS
1 PARTIAL
0 FAIL
""",
        encoding="utf-8",
    )

    results = verify_claims(tmp_path)

    assert results
    assert {result.status for result in results} == {"PASS"}


def test_release_claims_infer_identity_qa_from_legacy_manifest(tmp_path):
    _write_release_artifacts(tmp_path)
    manifest_path = tmp_path / "dataset_outputs" / "prepared_diverse" / "latest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"].pop("identity_qa")
    manifest["processing"] = {"total_formatted": 5}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "README.md").write_text(
        """
| Identity Q&A | 1 |
| **Total** | **7** |
""",
        encoding="utf-8",
    )

    results = verify_claims(tmp_path)

    assert results
    assert {result.status for result in results} == {"PASS"}


def test_hf_manifest_sanitizer_removes_local_paths_and_legacy_gap():
    manifest = {
        "sources": {
            "identity": {"path": "/mnt/c/private/identity.jsonl", "loaded": 2},
            "articles": {"path": "C:\\Users\\User\\articles.jsonl", "loaded": 1},
            "conversations": {"path": "/home/peter/conversations.jsonl", "loaded": 1},
        },
        "processing": {
            "total_loaded": 4,
            "total_formatted": 5,
            "dropped_invalid": -1,
        },
        "dpo_pairs": "/mnt/c/private/dpo_pairs.jsonl",
    }

    sanitized = sanitize_manifest(
        manifest,
        file_metadata={"train.jsonl": {"path": "train.jsonl", "records": 3}},
    )
    serialized = json.dumps(sanitized)

    assert "/mnt/" not in serialized
    assert "/home/" not in serialized
    assert "C:\\Users" not in serialized
    assert sanitized["sources"]["identity_qa"]["loaded"] == 1
    assert sanitized["processing"]["total_loaded"] == 5
    assert sanitized["processing"]["dropped_invalid"] == 0
    assert sanitized["dpo_pairs"] == "dpo_pairs.jsonl"


def test_release_claims_fail_on_mismatched_artifact(tmp_path):
    _write_release_artifacts(tmp_path)
    (tmp_path / "README.md").write_text(
        "- **Stage 1 — SFT**: 99 examples\n",
        encoding="utf-8",
    )

    results = verify_claims(tmp_path)

    assert any(result.status == "FAIL" and result.label == "SFT train count" for result in results)


def test_release_claims_mark_missing_artifacts_unverifiable(tmp_path):
    (tmp_path / "README.md").write_text(
        "| DPO pairs | 2 |\n",
        encoding="utf-8",
    )

    results = verify_claims(tmp_path)

    assert any(result.status == "UNVERIFIABLE" for result in results)
