"""Margin reporting is query-weighted and checkpoint objective contracts are strict."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from plm.config import ModelConfig, OptimizerConfig, SchedulerConfig, TrainConfig  # noqa: E402
from plm.corpus.compiler import CorpusRecord  # noqa: E402
from plm.experimentation.identity import ExperimentIdentity  # noqa: E402
from plm.model.architecture import build_model  # noqa: E402
from plm.model.interfaces import SYMMETRIC_MARGIN_OBJECTIVE, ModelOutput  # noqa: E402
from plm.training.trainer import Trainer  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def model_config(weight=1.0, margin=1.0):
    return ModelConfig(
        vocab_size=1029,
        max_seq_len=8,
        dim=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_multiple=2.0,
        ffn_multiple_of=8,
        dtype="fp32",
        attention={"qk_norm": False},
        embed_dropout=0.2,
        resid_dropout=0.2,
        symmetric_relation_loss_weight=1.0,
        symmetric_margin_loss_weight=weight,
        symmetric_margin=margin,
    )


def train_config(steps=4):
    return TrainConfig(
        max_steps=steps,
        batch_size=2,
        grad_accum_steps=2,
        seq_len=8,
        optimizer=OptimizerConfig(name="adamw", lr=0.001),
        scheduler=SchedulerConfig(name="constant", warmup_steps=0),
        num_workers=0,
        pin_memory=False,
        precision="fp32",
        eval_every=1,
        checkpoint_every=2,
    )


def records():
    return tuple(
        CorpusRecord(
            f"PKM_{index}",
            "TYPE",
            (f"PKM_{(index + 1) % 5}",),
            (1, 1024 + index, 32, 64, 5, 1024 + (index + 1) % 5, 2),
            (-100, -100, -100, -100, -100, 1024 + (index + 1) % 5, 2),
        )
        for index in range(5)
    )


@pytest.mark.parametrize(
    "counts,active,expected_loss,expected_fraction",
    [([1, 2], [0, 1], 4.0, 1 / 3), ([3, 2], [1, 2], 3.2, 3 / 5), ([0, 0], [0, 0], 0.0, 0.0)],
)
def test_validation_uses_query_counts_and_keeps_token_ce(
    counts, active, expected_loss, expected_fraction
):
    trainer = Trainer.__new__(Trainer)
    trainer.torch = torch
    trainer.validation_dataset = [object()]
    trainer.model = SimpleNamespace(
        eval=lambda: None,
        train=lambda: None,
        config=SimpleNamespace(
            prompt_set_loss_weight=0,
            symmetric_relation_loss_weight=1,
            symmetric_margin_loss_weight=1,
            continuation_set_loss_weight=0,
        ),
    )

    def batch(items):
        labels = torch.tensor([[-100, 1024, 2]] * len(items))
        return torch.zeros_like(labels), labels, None

    outputs = iter(
        ModelOutput(
            logits=None,
            task_loss=torch.tensor(token),
            symmetric_margin_loss=torch.tensor(margin),
            symmetric_margin_query_count=count,
            symmetric_margin_active_count=violations,
        )
        for token, margin, count, violations in zip(
            [1.0, 3.0], [2.0, 5.0], counts, active, strict=True
        )
    )
    trainer._batch = batch
    trainer._forward = lambda *args: next(outputs)
    assert trainer._validation_loss([[object()] * 3, [object()] * 2]) == pytest.approx(1.8)
    metrics = trainer.validation_metrics
    assert metrics["validation_symmetric_margin_loss"] == pytest.approx(expected_loss)
    assert metrics["validation_symmetric_margin_query_count"] == sum(counts)
    assert metrics["validation_symmetric_margin_active_count"] == sum(active)
    assert metrics["validation_symmetric_margin_active_fraction"] == pytest.approx(
        expected_fraction
    )


def test_history_records_raw_last_microbatch_margin(tmp_path, monkeypatch):
    model = build_model(model_config(weight=2.0))
    forward = model.forward
    observed = []

    def capture(*args, **kwargs):
        output = forward(*args, **kwargs)
        if model.training and torch.is_grad_enabled():
            observed.append(
                (
                    float(output.symmetric_margin_loss.detach()),
                    output.symmetric_margin_query_count,
                    output.symmetric_margin_active_count,
                )
            )
        return output

    monkeypatch.setattr(model, "forward", capture)
    result = Trainer(model, records(), config=train_config(1), validation_records=records()).train(
        checkpoint_path=tmp_path / "one.pt"
    )
    assert len(observed) == 2
    row = result.history[0]
    assert row["last_batch_symmetric_margin_loss"] == observed[-1][0]
    assert row["last_batch_symmetric_margin_query_count"] == observed[-1][1]
    assert row["last_batch_symmetric_margin_active_count"] == observed[-1][2]
    assert row["validation_symmetric_margin_query_count"] == len(records())
    assert 0 <= row["validation_symmetric_margin_active_fraction"] <= 1
    assert result.validation_loss == row["validation_loss"]


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_same_code_resume_matches_continuous_exactly(tmp_path: Path, weight):
    config = model_config(weight)
    torch.manual_seed(123)
    continuous_model = build_model(config)
    full = Trainer(
        continuous_model,
        records(),
        config=train_config(),
        validation_records=records(),
        run_dir=tmp_path,
    ).train(checkpoint_path=tmp_path / "full.pt")
    resumed_model = build_model(config)
    resumed = Trainer(
        resumed_model, records(), config=train_config(), validation_records=records()
    ).train(
        resume_checkpoint=tmp_path / "checkpoint-step-2.pt",
        checkpoint_path=tmp_path / "resumed.pt",
    )
    assert full.train_loss == resumed.train_loss
    assert full.validation_loss == resumed.validation_loss
    assert full.history[2:] == resumed.history
    for name, parameter in continuous_model.state_dict().items():
        assert torch.equal(parameter, resumed_model.state_dict()[name])
    original = torch.load(tmp_path / "full.pt", map_location="cpu", weights_only=False)
    continued = torch.load(tmp_path / "resumed.pt", map_location="cpu", weights_only=False)
    assert original["scheduler"] == continued["scheduler"]
    assert torch.equal(original["rng_state"]["torch"], continued["rng_state"]["torch"])
    for index, state in original["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, continued["optimizer"]["state"][index][key])
    contract = original["training_metadata"]
    assert contract["model_config"] == config.model_dump(mode="json")
    if weight:
        assert contract["symmetric_margin_objective"] == SYMMETRIC_MARGIN_OBJECTIVE
        assert "last_batch_symmetric_margin_loss" in full.history[0]
    else:
        assert "symmetric_margin_objective" not in contract
        assert all(not any("symmetric_margin" in key for key in row) for row in full.history)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "original.pt"
    Trainer(build_model(model_config()), records(), config=train_config(1)).train(
        checkpoint_path=path
    )
    return path


@pytest.mark.parametrize("weight,margin", [(2.0, 1.0), (1.0, 2.0), (0.0, 1.0)])
def test_resume_rejects_changed_margin_config(checkpoint, weight, margin):
    with pytest.raises(ValueError, match="training objective or data-order contract"):
        Trainer(build_model(model_config(weight, margin)), records(), config=train_config(2)).train(
            resume_checkpoint=checkpoint
        )


@pytest.mark.parametrize("descriptor", [None, "other-margin-objective-v0"])
def test_resume_rejects_missing_or_changed_margin_descriptor(checkpoint, tmp_path, descriptor):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if descriptor is None:
        payload["training_metadata"].pop("symmetric_margin_objective")
    else:
        payload["training_metadata"]["symmetric_margin_objective"] = descriptor
    altered = tmp_path / "altered.pt"
    # No sidecar: the mutated authoritative payload itself must fail the trainer contract.
    torch.save(payload, altered)
    with pytest.raises(ValueError, match="training objective or data-order contract"):
        Trainer(build_model(model_config()), records(), config=train_config(2)).train(
            resume_checkpoint=altered
        )


@pytest.mark.parametrize("mutation", ["legacy", "stray_descriptor"])
def test_disabled_resume_rejects_historical_raw_config_or_stray_descriptor(tmp_path, mutation):
    checkpoint = tmp_path / "disabled.pt"
    Trainer(build_model(model_config(0.0)), records(), config=train_config(1)).train(
        checkpoint_path=checkpoint
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["training_metadata"]
    if mutation == "legacy":
        metadata["model_config"].pop("symmetric_margin_loss_weight")
        metadata["model_config"].pop("symmetric_margin")
    else:
        metadata["symmetric_margin_objective"] = SYMMETRIC_MARGIN_OBJECTIVE
    altered = tmp_path / "altered.pt"
    torch.save(payload, altered)
    with pytest.raises(ValueError, match="training objective or data-order contract"):
        Trainer(build_model(model_config(0.0)), records(), config=train_config(2)).train(
            resume_checkpoint=altered
        )


def test_resume_keeps_experiment_identity_strict(tmp_path):
    identity = ExperimentIdentity(
        source_commit="source",
        resolved_config_hash="historical-raw-config",
        protocol_version="protocol",
        graph_hash="graph",
        records_hash="records",
        vocabulary_hash="vocabulary",
        split_hash="split",
        evaluator_version="eval",
        seeds=(1729,),
    )
    checkpoint = tmp_path / "identified.pt"
    Trainer(
        build_model(model_config(0.0)), records(), config=train_config(1), identity=identity
    ).train(checkpoint_path=checkpoint)
    changed = replace(identity, resolved_config_hash="default-expanded-config")
    with pytest.raises(ValueError, match="experiment identity does not match"):
        Trainer(
            build_model(model_config(0.0)), records(), config=train_config(2), identity=changed
        ).train(resume_checkpoint=checkpoint)
