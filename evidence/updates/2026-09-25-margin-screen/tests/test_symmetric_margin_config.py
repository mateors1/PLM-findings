"""Explicit margin configuration and authenticated historical loading boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from plm.config import ModelConfig, RootConfig
from plm.configuration import (
    config_hash,
    load_config,
    validate_checkpoint_model_config,
    validate_saved_config,
)


def _digest(raw):
    value = {k: v for k, v in raw.items() if k != "run_name"}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@pytest.mark.parametrize("field", ["symmetric_margin", "symmetric_margin_loss_weight"])
@pytest.mark.parametrize("value", [True, "0.1", -1.0, float("nan"), float("inf")])
def test_margin_fields_are_strict_finite_and_nonnegative(field, value):
    with pytest.raises(ValueError):
        ModelConfig(symmetric_relation_loss_weight=1.0, **{field: value})


def test_positive_margin_requires_existing_symmetric_head_and_positive_margin():
    with pytest.raises(ValueError, match="requires symmetric_relation"):
        ModelConfig(symmetric_margin_loss_weight=0.1)
    with pytest.raises(ValueError):
        ModelConfig(symmetric_margin=0.0)
    config = load_config(
        overrides=[
            "+experiment=symmetric_relation_lab",
            "model.symmetric_margin_loss_weight=0.1",
            "model.symmetric_margin=1.0",
        ]
    )
    assert config.model.symmetric_margin_loss_weight == 0.1
    assert config.model.symmetric_relation_loss_weight == 1.0


def test_paired_historical_omission_is_disabled_and_preserves_raw_identity():
    raw = RootConfig().model_dump(mode="json")
    for field in ("symmetric_margin", "symmetric_margin_loss_weight"):
        del raw["model"][field]
    for field in ("symmetric_set_reranking", "pair_set_composition"):
        del raw["eval"][field]
    before = copy.deepcopy(raw)
    digest = _digest(raw)
    restored = validate_saved_config(raw, digest)
    assert restored.model.symmetric_margin_loss_weight == 0.0
    assert restored.model.symmetric_margin == 1.0
    assert raw == before and _digest(raw) == digest
    assert config_hash(restored) != digest
    assert validate_checkpoint_model_config(raw["model"]) == restored.model
    assert (
        validate_saved_config(restored.model_dump(mode="json"), config_hash(restored)) == restored
    )
    with pytest.raises(ValueError, match="historical identity"):
        validate_saved_config(raw, "0" * 64)


@pytest.mark.parametrize("field", ["symmetric_margin", "symmetric_margin_loss_weight"])
def test_partial_margin_metadata_never_receives_defaults(field):
    raw = RootConfig().model_dump(mode="json")
    del raw["model"][field]
    with pytest.raises(ValueError, match="partial"):
        validate_saved_config(raw, _digest(raw))
    with pytest.raises(ValueError, match="partial"):
        validate_checkpoint_model_config(raw["model"])


@pytest.mark.parametrize("value", ["1.0", True, 1, None, float("inf")])
def test_checkpoint_margin_metadata_rejects_coercion_and_nonfinite_values(value):
    raw = ModelConfig().model_dump(mode="json")
    raw["symmetric_margin"] = value
    with pytest.raises(ValueError):
        validate_checkpoint_model_config(raw)


def test_margin_migration_does_not_license_other_root_schema_changes():
    raw = RootConfig().model_dump(mode="json")
    del raw["model"]["symmetric_margin"]
    del raw["model"]["symmetric_margin_loss_weight"]
    del raw["model"]["prompt_set_loss_weight"]
    with pytest.raises(ValueError, match="unsupported schema transformation"):
        validate_saved_config(raw, _digest(raw))


@pytest.fixture
def runtime_payload(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from plm.protocol import Vocabulary
    from plm.serving import runtime

    vocab = Vocabulary.build(["PKM_A", "PKM_B", "PKM_C"])
    vocab.save(tmp_path / "vocabulary.json")
    config = RootConfig()
    config.data.corpus_manifest = str(tmp_path / "manifest.json")
    config.eval.device = "cpu"
    config.model.symmetric_relation_loss_weight = 1.0
    model = SimpleNamespace(to=lambda device: model, eval=lambda: model)
    identity = {"graph_hash": "g", "records_hash": "r", "vocabulary_hash": "v"}
    manifest = SimpleNamespace(graph_hash="g", records_hash="r", tokenizer_hash="v")
    state = SimpleNamespace(
        global_step=2,
        checkpoint_hash="c",
        metadata={
            "corpus_identity": identity,
            "experiment_identity": {
                **identity,
                "split_hash": "s",
                "evaluator_version": "plm-train-causal-v2",
            },
            "training_metadata": {"objective": "causal-next-token-v1"},
        },
    )
    monkeypatch.setattr(
        runtime, "require_valid_artifacts", lambda *a, **k: SimpleNamespace(manifest=manifest)
    )
    monkeypatch.setattr(runtime, "load_corpus_records", lambda *a: [])
    monkeypatch.setattr(runtime, "split_queries", lambda *a: SimpleNamespace(split_hash="s"))
    monkeypatch.setattr("plm.model.build_model", lambda *a: model)
    monkeypatch.setattr("plm.training.checkpoint.load_checkpoint", lambda *a, **k: state)
    monkeypatch.setattr(
        "plm.training.checkpoint.validate_checkpoint_identity", lambda *a, **k: None
    )
    return config, state, vocab, tmp_path


@pytest.mark.parametrize(
    "mode", ["historical", "enabled", "missing", "wrong", "stray", "null", "partial"]
)
def test_runtime_checks_payload_margin_contract(runtime_payload, mode):
    from plm.model.interfaces import SYMMETRIC_MARGIN_OBJECTIVE
    from plm.serving.runtime import load_inference_runtime

    config, state, vocab, directory = runtime_payload
    enabled = mode in ("enabled", "missing", "wrong", "partial")
    config.model.symmetric_margin_loss_weight = 0.1 if enabled else 0.0
    recorded = config.model.model_copy(
        update={"vocab_size": len(vocab), "max_seq_len": config.train.seq_len}
    ).model_dump(mode="json")
    metadata = state.metadata["training_metadata"]
    metadata["model_config"] = recorded
    if mode == "historical":
        del recorded["symmetric_margin"]
        del recorded["symmetric_margin_loss_weight"]
    if mode in ("enabled", "wrong", "stray", "partial"):
        metadata["symmetric_margin_objective"] = (
            "wrong" if mode == "wrong" else SYMMETRIC_MARGIN_OBJECTIVE
        )
    if mode == "null":
        metadata["symmetric_margin_objective"] = None
    if mode == "partial":
        del recorded["symmetric_margin"]
    if mode in ("historical", "enabled"):
        loaded = load_inference_runtime(config, directory / "checkpoint.pt")
        assert loaded.checkpoint_hash == "c"
    else:
        with pytest.raises(ValueError, match="margin"):
            load_inference_runtime(config, directory / "checkpoint.pt")
