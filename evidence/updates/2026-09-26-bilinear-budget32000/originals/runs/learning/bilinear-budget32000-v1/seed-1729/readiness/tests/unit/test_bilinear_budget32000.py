"""Synthetic CPU32000 contracts; never production data or GPU."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load("refit_bilinear_budget32000")


@pytest.fixture(scope="module")
def mean():
    return load("refit_membership_projection")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Helper:
    @staticmethod
    def _bind(path, digest, inputs):
        path = Path(path).resolve()
        if sha(path) != digest:
            raise ValueError("hash mismatch")
        inputs[str(path)] = digest
        return path

    @staticmethod
    def _read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    _sha = staticmethod(sha)
    _failure_safe = staticmethod(lambda value: value)
    _fp32 = staticmethod(lambda vector: None)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha(path)


def partition_fixture(ntrain=3, nval=3):
    tokens = [f"<reserved_{i}>" for i in range(2049)]
    classes = ["reserved"] * 2049
    for i, name in {
        0: "PAD",
        1: "BOS",
        2: "EOS",
        5: "ANSWER",
        32: "TYPE",
        33: "COLOR",
        34: "SAME",
    }.items():
        tokens[i] = name
    for i in range(1024, 2049):
        tokens[i] = f"PKM_{i}"
        classes[i] = "entity"
    rows = []
    for i in range(ntrain + nval):
        subject = 1024 + i // 2
        dim = 32 + i % 2
        prompt = [1, subject, dim, 34, 5]
        targets = [2048, 2047]
        rows.append(
            {
                "index": i if i < ntrain else i - ntrain,
                "subject": tokens[subject],
                "dimension": tokens[dim],
                "prompt_ids": prompt,
                "expected_set_ids": targets,
            }
        )
    train = {"partition": "train", "query_count": ntrain, "queries": rows[:ntrain]}
    parent = {
        "complete": True,
        "query_count": nval,
        "responses": rows[ntrain:],
        "product_token_ids": list(range(1024, 2049)),
    }
    queries = {
        name: [{k: r[k] for k in ("subject", "dimension", "prompt_ids")} for r in group]
        for name, group in (("train", rows[:ntrain]), ("validation", rows[ntrain:]))
    }
    vocab = {"schema_version": 2, "tokens": tokens, "classes": classes}
    return train, parent, queries, vocab


def invariants():
    return dict.fromkeys(
        (
            "parent_replay_exact",
            "zero_A_replay_exact",
            "initial_train_loss_exact",
            "completed_32000_updates",
            "original_93_tensors_unchanged",
            "residual_changed",
            "reload_exact",
            "optimizer_reload_exact",
            "train_only_supervision",
        ),
        True,
    )


def test_exact_optimizer_state_distinguishes_signed_zero(runner):
    import torch

    left = {"state": {0: {"exp_avg": torch.tensor([0.0])}}}
    right = {"state": {0: {"exp_avg": torch.tensor([-0.0])}}}
    assert not runner._exact_state(left, right)
    assert runner._exact_state(left, copy.deepcopy(left))
    assert not runner._exact_state({"flag": False}, {"flag": 0})


@pytest.mark.parametrize("seed", [1729, 1730, 1731])
@pytest.mark.parametrize("missing_mean", [False, True, "fit_failure"])
def test_full_execute_lifecycle_with_bounded_synthetic_fit(
    runner, mean, tmp_path, monkeypatch, missing_mean, seed
):
    """Real CPU lifecycle; one real optimizer step and simulated 32000-step trace."""
    import ast
    import inspect
    import sys
    import zipfile
    from types import ModuleType

    import torch

    bilinear = load("refit_bilinear_residual")
    trace = []
    with zipfile.ZipFile(ROOT / "tests/fixtures/projection_refit_runtime_source.zip") as archive:
        sources = {
            n: archive.read(n).decode()
            for n in (
                "src/plm/config.py",
                "src/plm/experimentation/identity.py",
                "src/plm/training/checkpoint.py",
                "src/plm/training/optim.py",
                "src/plm/training/data.py",
                "src/plm/model/interfaces.py",
                "src/plm/model/layers.py",
            )
        }
    for name in ("plm", "plm.model", "plm.training", "plm.experimentation", "plm.corpus"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    compiler = ModuleType("plm.corpus.compiler")
    compiler.CorpusRecord = SimpleNamespace
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    loaded = {}
    for name in (
        "plm.config",
        "plm.experimentation.identity",
        "plm.model.interfaces",
        "plm.training.checkpoint",
        "plm.training.optim",
        "plm.training.data",
    ):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        path = "src/" + name.replace(".", "/") + ".py"
        exec(compile(sources[path], "archived/" + path, "exec"), module.__dict__)
        loaded[name] = module
    config = loaded["plm.config"].ModelConfig.model_validate(
        {"dim": 256, "vocab_size": 2049, "max_seq_len": 512}
    )
    tree = ast.parse(sources["src/plm/model/layers.py"])
    loss_node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prompt_set_loss"
    )
    layer_module = ModuleType("plm.model.layers")
    layer_module.__dict__.update(
        Tensor=torch.Tensor, torch=torch, F=torch.nn.functional, ENTITY_BASE=1024
    )
    exec(
        compile(ast.Module(body=[loss_node], type_ignores=[]), "archived/loss", "exec"),
        layer_module.__dict__,
    )
    monkeypatch.setitem(sys.modules, layer_module.__name__, layer_module)
    loss_fn = layer_module._prompt_set_loss

    class SyntheticModel(torch.nn.Module):
        def __init__(self, model_config):
            super().__init__()
            self.config = model_config
            self.token_embedding = torch.nn.Embedding(2049, 256)
            self.symmetric_relation_projection = torch.nn.Linear(256, 256, bias=False)
            with torch.no_grad():
                self.token_embedding.weight.zero_()
                self.symmetric_relation_projection.weight.zero_()
            for i in range(91):
                self.register_buffer(f"frozen_fixture_{i}", torch.tensor([float(i)]))

        def forward(self, prompts):
            features = mean._features(self, prompts)
            return SimpleNamespace(
                symmetric_relation_logits=mean._head(
                    self.symmetric_relation_projection.weight, features
                )
            )

    def build_model(model_config):
        trace.append("build_reload_model")
        assert model_config == config
        return SyntheticModel(model_config)

    architecture = ModuleType("plm.model.architecture")
    architecture.build_model = build_model
    monkeypatch.setitem(sys.modules, architecture.__name__, architecture)

    expected_seed = seed

    def seed_everything(seed, deterministic=False):
        assert seed == expected_seed and deterministic is False

    reproduction = ModuleType("plm.reproducibility")
    reproduction.seed_everything = seed_everything
    monkeypatch.setitem(sys.modules, reproduction.__name__, reproduction)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(sys, "path", list(sys.path))

    membership, parent, queries, vocabulary = partition_fixture(1637, 222)
    # Canonical authenticated membership fixture, never a repository dataset.
    for row in membership["queries"]:
        row["expected_set_ids"].sort()
    partitions = runner._selected_partitions(
        membership, parent, queries, vocabulary, runner._adapter(ROOT)
    )
    for row in parent["responses"]:
        row["group"] = (
            "COLOR"
            if row["dimension"] == "COLOR"
            else "TYPE_single"
            if row["index"] % 4 == 1
            else "TYPE_dual"
        )
        row["symmetric_relation_logits"] = [0.0] * 1025
        row["selected_set_ids"] = []
        row["metrics"] = mean._metrics([], row["expected_set_ids"])
    model = SyntheticModel(config)
    assert len(model.state_dict()) == 93
    training_identity = {"protocol_version": "synthetic-protocol"}
    corpus_identity = {
        "graph_hash": "synthetic-graph",
        "records_hash": "synthetic-records",
        "vocabulary_hash": "synthetic-vocabulary",
    }
    runtime = SimpleNamespace(
        model=model,
        device="cpu",
        vocabulary=SimpleNamespace(pad_id=0),
        checkpoint_hash=runner._PARENTS[seed],
        training_identity=training_identity,
        corpus_identity=corpus_identity,
        split=SimpleNamespace(**partitions, split_hash=mean._SPLIT),
    )

    def selected_runtime(context):
        trace.append("load_synthetic_runtime")
        assert context["parent_state"] == mean._state_hashes(model)
        return runtime

    monkeypatch.setattr(runner, "_load_selected_runtime", selected_runtime)
    monkeypatch.setattr(runner, "_MEMBERSHIP", mean._digest(membership))
    monkeypatch.setattr(mean, "_git", lambda root: {"commit": "synthetic-source"})
    batch = loaded["plm.training.data"].collate_records(partitions["train"], pad_id=0)
    labels = batch.labels.masked_fill(~batch.attention_mask, -100)[:, 1:]
    subjects = batch.input_ids[:, 1] - 1024
    initial_loss = float(loss_fn(torch.zeros(1637, 1025), labels, excluded_ids=subjects))
    recipe = json.loads((ROOT / "configs/experiments/bilinear_budget32000_v1.json").read_text())
    monkeypatch.setattr(runner, "_INITIAL_LOSSES", dict.fromkeys(runner._SEEDS, initial_loss))
    context = {
        "seed": seed,
        "historical8000_summary_sha256": "synthetic",
        "historical_training": {
            "history": [{"update": i, "pre_update_loss": initial_loss} for i in range(1, 8001)],
            "final_post_update_loss": initial_loss,
        },
        "recipe": recipe,
        "members": [],
        "historical_environment": {"fixture": "CPU"},
        "parent_run": {"identity": training_identity, "config": {"synthetic": True}},
        "parent_state": mean._state_hashes(model),
        "membership": membership,
        "wide": parent,
        "input_adapter": {"synthetic_membership_fixture": True},
        "inputs": {},
        "historical8000": copy.deepcopy(parent),
        "historical8000_child_sha256": "synthetic",
    }

    class ExecutionHelper(Helper):
        _SOURCE = "synthetic-source-archive"
        _CONFIGS = "synthetic-config-archive"

        @staticmethod
        def _modules(runtime=None):
            return {}

        @staticmethod
        def _extract(members, path):
            assert members == []
            (path / "src").mkdir(parents=True)
            return {}

        @staticmethod
        def _environment(torch_module):
            assert torch_module is torch
            return {"fixture": "CPU"}

        @staticmethod
        def _settings(torch_module):
            assert torch_module is torch
            return {}

        @staticmethod
        def _unchanged(hashes):
            assert hashes == {}

        @staticmethod
        def _runtime_unchanged(runtime, hashes):
            assert hashes == {}

    helper = ExecutionHelper
    real_evaluate = runner._evaluate

    # Retain the real function's exact parameter names, signature, and body.
    def tracked_evaluate(
        model,
        records,
        wide_rows,
        mean,
        helper,
        path,
        bilinear,
        parent=None,
        batch_size=8,
        phase="child",
    ):
        assert mean is frozen_mean and helper is ExecutionHelper
        assert records is runtime.split.validation and wide_rows is context["wide"]["responses"]
        trace.append("evaluate_" + phase)
        return real_evaluate(
            model,
            records,
            wide_rows,
            mean,
            helper,
            path,
            bilinear,
            parent=parent,
            batch_size=batch_size,
            phase=phase,
        )

    frozen_mean = mean
    assert inspect.signature(tracked_evaluate) == inspect.signature(real_evaluate)
    monkeypatch.setattr(runner, "_evaluate", tracked_evaluate)
    real_zero = bilinear._zero_replay

    def tracked_zero(model, records, parent, mean, helper, path, batch_size=8):
        trace.append("zero_replay")
        return real_zero(model, records, parent, mean, helper, path, batch_size=batch_size)

    assert inspect.signature(tracked_zero) == inspect.signature(real_zero)
    monkeypatch.setattr(bilinear, "_zero_replay", tracked_zero)
    real_attach = bilinear._attach

    def tracked_attach(model):
        trace.append("attach")
        return real_attach(model)

    monkeypatch.setattr(bilinear, "_attach", tracked_attach)
    real_fit = bilinear._fit

    def bounded_fit(
        model, features, dimension_tokens, labels, loss_fn, optimizer, updates, receipt, mean
    ):
        inspect.signature(real_fit).bind(
            model, features, dimension_tokens, labels, loss_fn, optimizer, updates, receipt, mean
        )
        assert updates == 32000 and receipt == {}
        assert features[0].shape == (1025, 256) and features[2].shape == (1637,)
        assert labels.shape[0] == 1637 and isinstance(loss_fn, runner._TrainingLossRecorder)
        assert loss_fn.loss_fn is layer_module._prompt_set_loss
        assert [n for n, p in model.named_parameters() if p.requires_grad] == list(
            runner._PARAMETERS
        )
        trace.append("fit_32000_simulated")
        parameter = getattr(model, runner._PARAMETERS[0])
        optimizer.zero_grad(set_to_none=True)
        parameter.sum().backward()
        optimizer.step()
        if missing_mean == "fit_failure":
            receipt.update(
                completed_updates=1, history=[{"update": 1, "pre_update_loss": initial_loss}]
            )
            raise RuntimeError("synthetic fit failure")
        optimizer.state[parameter]["step"].fill_(updates)
        # Explicit simulated callback chronology; full numerical behavior tested separately.
        scores = torch.zeros(1637, 1025)
        loss_fn.calls = 8000
        loss_fn(scores, labels, excluded_ids=features[2])
        loss_fn.calls = 32000
        with torch.no_grad():
            loss_fn(scores, labels, excluded_ids=features[2])
        receipt.update(
            history=[
                {
                    "update": i,
                    "pre_update_loss": initial_loss,
                    "gradient_finite": True,
                    "parameters_finite": True,
                }
                for i in range(1, updates + 1)
            ],
            completed_updates=updates,
            initial_pre_update_loss=initial_loss,
            final_post_update_loss=initial_loss,
            synthetic_numerical_steps=1,
            synthetic_trajectory=True,
        )

    assert inspect.signature(bounded_fit) == inspect.signature(real_fit)
    monkeypatch.setattr(bilinear, "_fit", bounded_fit)
    checkpoint_module = loaded["plm.training.checkpoint"]
    real_save = checkpoint_module.save_checkpoint
    real_load = checkpoint_module.load_checkpoint

    def save_checkpoint(
        path,
        model,
        *,
        optimizer=None,
        scheduler=None,
        global_step,
        config=None,
        identity=None,
        corpus_identity=None,
        split_hash=None,
        training_metadata=None,
    ):
        trace.append("save_checkpoint")
        return real_save(
            path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            config=config,
            identity=identity,
            corpus_identity=corpus_identity,
            split_hash=split_hash,
            training_metadata=training_metadata,
        )

    def load_checkpoint(
        path, model, *, optimizer=None, scheduler=None, map_location="cpu", restore_rng=True
    ):
        trace.append("load_checkpoint")
        return real_load(
            path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
            restore_rng=restore_rng,
        )

    for wrapper, original in ((save_checkpoint, real_save), (load_checkpoint, real_load)):
        original_signature = inspect.signature(original)
        unannotated = original_signature.replace(
            parameters=[
                p.replace(annotation=inspect.Parameter.empty)
                for p in original_signature.parameters.values()
            ],
            return_annotation=inspect.Signature.empty,
        )
        assert inspect.signature(wrapper) == unannotated

    monkeypatch.setattr(checkpoint_module, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(checkpoint_module, "load_checkpoint", load_checkpoint)
    summary = {
        "complete": False,
        "acceptance": False,
        "script_sha256": "synthetic-runner",
        "recipe_sha256": "synthetic-recipe",
        "snapshot_sha256": {},
    }
    execute = runner._execute
    if missing_mean is True:
        tree = ast.parse(
            (ROOT / "scripts/refit_bilinear_budget32000.py").read_text(encoding="utf-8")
        )
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_execute")
        parent_call = next(
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_evaluate"
            and any(k.arg == "phase" for k in n.keywords)
        )
        assert isinstance(parent_call.args[3], ast.Name) and parent_call.args[3].id == "mean"
        del parent_call.args[3]
        namespace = dict(vars(runner))
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "missing-mean-regression", "exec"),
            namespace,
        )
        execute = namespace["_execute"]
        with pytest.raises(TypeError, match="bilinear"):
            execute(
                tmp_path,
                tmp_path / "summary.json",
                context,
                bilinear,
                mean,
                helper,
                summary,
            )
        failed = Helper._read(tmp_path / "summary.json")
        assert failed["complete"] is False and not (tmp_path / "parent.json").exists()
        assert (tmp_path / "failure-state.pt").exists()
        assert trace == ["load_synthetic_runtime"]
        return
    if missing_mean == "fit_failure":
        with pytest.raises(RuntimeError, match="synthetic fit failure"):
            execute(tmp_path, tmp_path / "summary.json", context, bilinear, mean, helper, summary)
        failed = Helper._read(tmp_path / "summary.json")
        training = Helper._read(tmp_path / "training.json")
        diagnostic = torch.load(tmp_path / "failure-state.pt", weights_only=True)
        assert failed["complete"] is False and training["complete"] is False
        assert training["completed_updates"] == 1 and len(training["history"]) == 1
        assert failed["failure_state_availability"] == {
            "model_available": True,
            "optimizer_available": True,
        }
        assert (
            len(diagnostic["model_state"]) == 94
            and len(diagnostic["optimizer_state"]["state"]) == 1
        )
        assert not diagnostic["resume_allowed"] and not (tmp_path / "checkpoint-final.pt").exists()
        return
    execute(tmp_path, tmp_path / "summary.json", context, bilinear, mean, helper, summary)
    saved = Helper._read(tmp_path / "summary.json")
    training = Helper._read(tmp_path / "training.json")
    assert trace == [
        "load_synthetic_runtime",
        "evaluate_parent",
        "attach",
        "zero_replay",
        "fit_32000_simulated",
        "save_checkpoint",
        "build_reload_model",
        "attach",
        "load_checkpoint",
        "evaluate_child",
    ]
    assert saved["complete"] is True and saved["final_identity_check"] is True
    assert training["reload_exact"] and training["optimizer_reload_exact"]
    assert (
        len(training["state_before"])
        == len(training["state_after"])
        == len(training["state_reloaded"])
        == 94
    )
    assert len(training["parent_state_before"]) == 93
    assert training["fit_complete"] and training["complete"] and training["synthetic_trajectory"]
    assert training["synthetic_numerical_steps"] == 1 and len(training["history"]) == 32000
    assert set(training["optimizer_final"]["steps"].values()) == {32000.0}
    assert all(saved["invariants"].values())
    assert set(training["training_diagnostics"]["endpoints"]) == {"8000", "32000"}
    assert training["training_diagnostics"]["loss_callback_calls"] == 32001
    assert (
        saved["gate"]["accepted"] is False and saved["gate"]["independent_audit_required"] is True
    )
    assert saved["gate"]["primary_checks_passed"] is False  # Synthetic model is not accurate.
    for name, digest in saved["artifact_sha256"].items():
        assert sha(tmp_path / name) == digest
    assert Helper._read(tmp_path / "parent.json")["query_count"] == 222
    assert Helper._read(tmp_path / "child.json")["query_count"] == 222
    assert Helper._read(tmp_path / "zero-replay.json")["batch_sizes"] == [8] * 27 + [6]
    assert (tmp_path / "comparisons.json").exists() and not (tmp_path / "failure-state.pt").exists()


@pytest.mark.parametrize("seed", [1729, 1730, 1731])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "vocabulary",
        "objective",
        "step",
        "config",
        "evaluator",
        "corpus",
        "train_optimizer",
        "train_seq_len",
        "train_max_steps",
        "root_in_checkpoint",
        "checkpoint_hash",
        "sidecar_metadata",
        "resolved_model",
        "experiment_corpus",
    ],
)
def test_selected_runtime_uses_only_checkpoint_and_vocabulary(
    runner, tmp_path, monkeypatch, mutation, seed
):
    import sys
    import zipfile
    from types import ModuleType

    with zipfile.ZipFile(ROOT / "tests/fixtures/projection_refit_runtime_source.zip") as archive:
        config_source = archive.read("src/plm/config.py").decode()
    config_module = ModuleType("synthetic_archived_config")
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    exec(compile(config_source, "archived/config.py", "exec"), config_module.__dict__)
    config = config_module.RootConfig.model_validate(
        {
            "model": {"dim": 256, "n_heads": 8, "n_kv_heads": 2},
            "eval": {"device": "cuda"},
            "train": {"seq_len": 512, "max_steps": 2000, "optimizer": {"lr": 0.0003}},
        }
    )
    model_config = config.model.model_copy(update={"vocab_size": 2049, "max_seq_len": 512})
    corpus = {"vocabulary_hash": "v", "graph_hash": "g", "records_hash": "r"}
    identity = {"evaluator_version": "plm-train-causal-v2", "split_hash": "split", **corpus}
    metadata = {
        "objective": "causal-next-token-v1",
        "model_config": model_config.model_dump(mode="json"),
    }
    context = {
        "seed": seed,
        "parent_run": {
            "config": config.model_dump(mode="json"),
            "identity": identity,
        },
        "parent_sidecar": {"training_metadata": metadata},
        "vocabulary_path": tmp_path / "vocabulary.json",
        "inherited_architecture": model_config.model_dump(mode="json"),
        "parent": tmp_path / "checkpoint.pt",
        "corpus_identity": corpus,
        "partitions": {"train": ("train",), "validation": ("validation",)},
    }
    raw = {
        "config": copy.deepcopy(context["parent_run"]["config"]["train"]),
        "corpus_identity": corpus,
        "training_metadata": metadata,
    }
    state = SimpleNamespace(
        global_step=2000,
        checkpoint_hash=runner._PARENTS[seed],
        metadata=raw,
    )
    calls = []

    class Vocabulary:
        @classmethod
        def load(cls, path):
            assert path == context["vocabulary_path"]
            calls.append("vocabulary")
            return cls()

        def __len__(self):
            return 2049

        def content_hash(self):
            return "wrong" if mutation == "vocabulary" else "v"

    class Model:
        def to(self, device):
            assert device == "cuda"
            calls.append("device")
            return self

        def eval(self):
            return self

    def load_checkpoint(path, model, **kwargs):
        assert path == context["parent"] and kwargs == {"map_location": "cpu", "restore_rng": False}
        calls.append("checkpoint")
        return state

    def validate(state, **kwargs):
        assert kwargs == {"identity": identity, "split_hash": "split"}
        calls.append("identity")

    modules = {
        "plm.config": {
            "RootConfig": config_module.RootConfig,
            "ModelConfig": config_module.ModelConfig,
        },
        "plm.experimentation.identity": {
            "ExperimentIdentity": SimpleNamespace(from_dict=lambda value: value)
        },
        "plm.model.architecture": {"build_model": lambda cfg: Model()},
        "plm.protocol": {"Vocabulary": Vocabulary},
        "plm.training.checkpoint": {
            "load_checkpoint": load_checkpoint,
            "validate_checkpoint_identity": validate,
        },
    }
    # A closed synthetic package tree cannot import corpus, graph, split or serving loaders.
    for name in ("plm", "plm.model", "plm.training", "plm.experimentation"):
        mod = ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    for name, attributes in modules.items():
        mod = ModuleType(name)
        mod.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, mod)
    if mutation == "objective":
        metadata["objective"] = "old"
    if mutation == "step":
        state.global_step = 0
    if mutation == "config":
        raw["config"] = {}
    if mutation == "evaluator":
        identity["evaluator_version"] = "old"
    if mutation == "corpus":
        raw["corpus_identity"] = {}
    if mutation == "train_optimizer":
        raw["config"]["optimizer"]["lr"] = 0.0007
    if mutation == "train_seq_len":
        raw["config"]["seq_len"] = 256
    if mutation == "train_max_steps":
        raw["config"]["max_steps"] = 1999
    if mutation == "root_in_checkpoint":
        raw["config"] = context["parent_run"]["config"]
    if mutation in {"train_optimizer", "train_seq_len", "train_max_steps", "root_in_checkpoint"}:
        with pytest.raises(ValueError, match="TrainConfig differs"):
            runner._load_selected_runtime(context)
        assert "device" not in calls
        return
    if mutation == "checkpoint_hash":
        state.checkpoint_hash = "0" * 64
    if mutation == "sidecar_metadata":
        raw["training_metadata"] = {**metadata, "train_loss": 1.0}
    if mutation == "resolved_model":
        metadata["model_config"]["dim"] = 128
    if mutation == "experiment_corpus":
        identity["graph_hash"] = "wrong"
    if mutation is not None:
        with pytest.raises(ValueError):
            runner._load_selected_runtime(context)
    else:
        runtime = runner._load_selected_runtime(context)
        assert runtime.split.train == ("train",) and runtime.split.validation == ("validation",)
        assert not hasattr(runtime.split, "test")
        assert calls == ["vocabulary", "checkpoint", "identity", "device"]
        # This is the exact equality that failed in frozen v1.
        assert raw["config"] != context["parent_run"]["config"]
        assert raw["config"] == context["parent_run"]["config"]["train"]


@pytest.mark.parametrize("seed", [1729, 1730, 1731])
@pytest.mark.parametrize(
    "drift", [None, "parent_hash", "seed", "train_config", "step", "split", "config_hash"]
)
def test_parent_admission_metadata(runner, mean, seed, drift):
    config = {
        "seed": seed,
        "run_name": f"national_dex_continuation_control_s{seed}_v1",
        "data": {"split_seed": 1729},
        "train": {"max_steps": 2000, "seq_len": 512},
    }
    identity = {
        "seeds": [seed],
        "split_hash": runner._SPLIT,
        "resolved_config_hash": hashlib.sha256(
            json.dumps(
                {k: v for k, v in config.items() if k != "run_name"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    run = {"config": config, "identity": identity}
    summary = {
        "parent_training_identity": identity,
        "corpus_identity": {"records_hash": "inherited"},
    }
    sidecar = {
        "experiment_identity": identity,
        "global_step": 2000,
        "checkpoint_hash": runner._PARENTS[seed],
        "split_hash": runner._SPLIT,
        "config": copy.deepcopy(config["train"]),
        "training_metadata": {"objective": "causal-next-token-v1"},
        "corpus_identity": summary["corpus_identity"],
    }
    if drift == "parent_hash":
        sidecar["checkpoint_hash"] = runner._PARENTS[1729 if seed != 1729 else 1730]
    if drift == "seed":
        identity["seeds"] = [999]
    if drift == "train_config":
        sidecar["config"]["max_steps"] = 32000
    if drift == "step":
        sidecar["global_step"] = 32000
    if drift == "split":
        config["data"]["split_seed"] = seed + 10
    if drift == "config_hash":
        identity["resolved_config_hash"] = "wrong"
    if drift:
        with pytest.raises(ValueError):
            runner._parent_contract(seed, run, sidecar, summary, mean)
    else:
        runner._parent_contract(seed, run, sidecar, summary, mean)


@pytest.mark.parametrize("seed", [1729, 1730, 1731])
@pytest.mark.parametrize("kind", ["pass", "tie", "f1", "group", "serialization", "invariant"])
def test_seed_matched_gate(runner, seed, kind):
    floor = runner._FLOORS[seed]
    child = {
        "aggregate": {
            "exact_count": floor["exact_count"] + 1,
            "f1": floor["f1"],
            "serialization_compatible": 222,
        },
        "groups": {g: {"exact_count": n} for g, n in floor["groups"].items()},
    }
    checks = invariants()
    if kind == "tie":
        child["aggregate"]["exact_count"] -= 1
    if kind == "f1":
        child["aggregate"]["f1"] -= 1e-9
    if kind == "group":
        child["groups"]["TYPE_dual"]["exact_count"] -= 1
    if kind == "serialization":
        child["aggregate"]["serialization_compatible"] -= 1
    if kind == "invariant":
        checks["extra"] = True
    gate = runner._gate(seed, child, checks)
    assert gate["primary_checks_passed"] is (kind == "pass")
    assert gate["accepted"] is False and gate["independent_audit_required"] is True


def test_recipe_exact_and_mutation(runner):
    recipe = json.loads((ROOT / "configs/experiments/bilinear_budget32000_v1.json").read_text())
    inherited = json.loads((ROOT / "configs/experiments/bilinear_budget8000_v1.json").read_text())
    assert runner._recipe(recipe, inherited) == recipe
    for key, value in [
        ("residual_updates", 8000),
        ("seeds", [1729]),
        ("threshold", 0.01),
        ("input_adapter_source_sha256", "wrong"),
    ]:
        bad = copy.deepcopy(recipe)
        bad[key] = value
        with pytest.raises(ValueError):
            runner._recipe(bad, inherited)


def archived_loss():
    import zipfile

    import torch

    with zipfile.ZipFile(ROOT / "tests/fixtures/projection_refit_runtime_source.zip") as archive:
        tree = ast.parse(archive.read("src/plm/model/layers.py").decode())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prompt_set_loss"
    )
    namespace = {
        "Tensor": torch.Tensor,
        "torch": torch,
        "F": torch.nn.functional,
        "ENTITY_BASE": 1024,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "archived-loss", "exec"), namespace)
    return namespace["_prompt_set_loss"]


@pytest.mark.parametrize(
    "positive,negative,exact,margin",
    [
        (1.0, -1.0, True, 1.0),
        (1.0, 0.0, True, 0.0),
        (0.0, -1.0, False, 0.0),
        (-1.0, -1.0, False, -1.0),
        (1.0, 1.0, False, -1.0),
    ],
)
def test_train_margin_zero_boundary_and_ties(runner, mean, positive, negative, exact, margin):
    membership = {
        "query_count": 1,
        "queries": [
            {
                "index": 0,
                "subject": "PKM_A",
                "dimension": "TYPE",
                "prompt_ids": [1, 1024, 32, 34, 5],
                "expected_set_ids": [1025, 1026],
            }
        ],
    }
    scores = [[negative] * 1025]
    scores[0][0] = 999.0
    scores[0][1] = scores[0][2] = positive
    report = runner._training_diagnostic(scores, membership, 8000, mean)
    row = report["rows"][0]
    assert row["metrics"]["exact"] is exact and row["zero_margin"] == margin
    assert row["min_positive_id"] == 1025 and row["max_negative_id"] == 1027
    assert report["minimum_signed_margin"] == margin


def test_observer_preserves_real_fit_losses_gradients_updates_and_boundaries(
    runner, mean, tmp_path
):
    import torch

    bilinear = load("refit_bilinear_residual")
    loss_fn = archived_loss()
    generator = torch.Generator().manual_seed(17)
    entities = torch.randn(1025, 2, generator=generator)
    subjects = torch.tensor([0, 1])
    dimensions = torch.tensor([32, 33])
    features = (entities, entities[subjects], subjects, 16.0)
    labels = torch.tensor([[1026, 1027, 2], [1027, 1028, 2]])
    membership = {
        "partition": "train",
        "query_count": 2,
        "queries": [
            {
                "index": i,
                "subject": f"PKM_{i}",
                "dimension": ("TYPE", "COLOR")[i],
                "prompt_ids": [1, 1024 + i, 32 + i, 34, 5],
                "expected_set_ids": [1026 + i, 1027 + i],
            }
            for i in range(2)
        ],
    }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(dim=2)
            self.symmetric_relation_projection = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.symmetric_relation_projection.weight.copy_(
                    torch.tensor([[0.1, 0.2], [0.3, 0.4]])
                )

    base = bilinear._attach(Model())
    observed = bilinear._attach(Model())
    opts = [
        torch.optim.AdamW([getattr(m, runner._PARAMETERS[0])], lr=0.0003, weight_decay=0)
        for m in (base, observed)
    ]
    histories = [{}, {}]
    captured_parameters = {}
    reference_scores = {}
    reference_calls = 0

    def reference_loss(scores, labels, *, excluded_ids):
        nonlocal reference_calls
        reference_calls += 1
        if reference_calls in (2, 4):
            reference_scores[reference_calls - 1] = scores.detach().clone()
            captured_parameters[reference_calls - 1] = (
                getattr(base, runner._PARAMETERS[0]).detach().clone()
            )
        return loss_fn(scores, labels, excluded_ids=excluded_ids)

    recorder = runner._TrainingLossRecorder(
        loss_fn,
        membership,
        labels,
        subjects,
        tmp_path,
        mean,
        Helper,
        membership_sha256="synthetic",
        endpoints=(1, 3),
    )
    bilinear._fit(
        base, features, dimensions, labels, reference_loss, opts[0], 3, histories[0], mean
    )
    bilinear._fit(observed, features, dimensions, labels, recorder, opts[1], 3, histories[1], mean)
    recorder.finish()
    assert reference_calls == recorder.calls == 4
    assert histories[0] == histories[1]
    assert runner._exact_state(base.state_dict(), observed.state_dict())
    assert runner._exact_state(opts[0].state_dict(), opts[1].state_dict())
    assert runner._exact_state(
        getattr(base, runner._PARAMETERS[0]).grad, getattr(observed, runner._PARAMETERS[0]).grad
    )
    assert not torch.equal(captured_parameters[1], captured_parameters[3])
    for step in (1, 3):
        raw = Helper._read(tmp_path / f"train-scores-post{step}.json")
        restored = torch.tensor(
            [r["symmetric_relation_logits"] for r in raw["responses"]], dtype=torch.float32
        )
        assert runner._exact_state(restored, reference_scores[step])
        assert (
            raw["score_bytes_sha256"] == hashlib.sha256(runner._tensor_bytes(restored)).hexdigest()
        )
    # Same loss object/gradient before and during a real observation.
    scores = reference_scores[1].detach().requires_grad_()
    plain = loss_fn(scores, labels, excluded_ids=subjects)
    gradient = torch.autograd.grad(plain, scores)[0]
    other = scores.detach().clone().requires_grad_()
    directory = tmp_path / "gradient"
    directory.mkdir()
    observer = runner._TrainingLossRecorder(
        loss_fn,
        membership,
        labels,
        subjects,
        directory,
        mean,
        Helper,
        membership_sha256="synthetic",
        endpoints=(0, 1),
    )
    wrapped = observer(other, labels, excluded_ids=subjects)
    assert runner._exact_state(plain, wrapped)
    assert runner._exact_state(gradient, torch.autograd.grad(wrapped, other)[0])


def test_all_internal_calls_bind_and_evaluator_roles(runner):
    import inspect

    tree = ast.parse((ROOT / "scripts/refit_bilinear_budget32000.py").read_text())
    signatures = {
        n.name: inspect.signature(getattr(runner, n.name))
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    checked = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in signatures
        ):
            assert not any(isinstance(a, ast.Starred) for a in node.args)
            assert not any(k.arg is None for k in node.keywords)
            signatures[node.func.id].bind(
                *[object() for _ in node.args], **{k.arg: object() for k in node.keywords}
            )
            checked += 1
    assert checked > 50
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_execute")
    calls = [
        n
        for n in ast.walk(execute)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_evaluate"
    ]
    assert len(calls) == 2
    for call in calls:
        assert call.args[3].id == "mean" and call.args[4].id == "helper"


@pytest.mark.parametrize("seed", [1729, 1730, 1731])
@pytest.mark.parametrize(
    "mutation",
    [None, "swapped_parent", "missing_queries", "wrong_architecture", "wrong_initial_loss"],
)
def test_all_seed_preflight_selected_allowlist(runner, mean, tmp_path, monkeypatch, seed, mutation):
    membership, parent, queries, vocabulary = partition_fixture(1637, 222)
    for row in parent["responses"]:
        row.update(
            group="COLOR" if row["dimension"] == "COLOR" else "TYPE_single",
            selected_set_ids=[],
            metrics=mean._metrics([], row["expected_set_ids"]),
        )
    monkeypatch.setattr(runner, "_MEMBERSHIP", mean._digest(membership))
    objects = {}
    digests = {}
    opened = []

    def put(relative, value, digest=None):
        path = (tmp_path / relative).resolve()
        objects[path] = value
        digests[path] = digest or mean._digest(value)
        return digests[path]

    recipe = json.loads((ROOT / "configs/experiments/bilinear_budget32000_v1.json").read_text())
    inherited = json.loads((ROOT / "configs/experiments/bilinear_budget8000_v1.json").read_text())
    corpus = {
        "graph_hash": "inherited",
        "records_hash": "inherited",
        "vocabulary_hash": "inherited",
    }
    summaries = {}
    children = []
    for selected_seed in runner._SEEDS:
        folder = (
            "runs/learning/bilinear-budget8000-v1"
            if selected_seed == 1729
            else f"runs/learning/bilinear-budget8000-replication-v1/seed-{selected_seed}"
        )
        parent_folder = f"runs/national_dex_continuation_control_s{selected_seed}_v1"
        config = {
            "seed": selected_seed,
            "run_name": f"national_dex_continuation_control_s{selected_seed}_v1",
            "data": {"split_seed": 1729},
            "train": {"seq_len": 512, "max_steps": 2000},
        }
        identity = {
            "seeds": [selected_seed],
            "split_hash": runner._SPLIT,
            "resolved_config_hash": hashlib.sha256(
                json.dumps(
                    {k: v for k, v in config.items() if k != "run_name"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        run = {"config": config, "identity": identity}
        sidecar = {
            "experiment_identity": identity,
            "global_step": 2000,
            "checkpoint_hash": runner._PARENTS[selected_seed],
            "split_hash": runner._SPLIT,
            "config": config["train"],
            "training_metadata": {"objective": "causal-next-token-v1"},
            "corpus_identity": corpus,
        }
        run_hash = put(parent_folder + "/run.json", run)
        sidecar_hash = put(parent_folder + "/checkpoint-final.pt.json", sidecar)
        put(
            parent_folder + "/checkpoint-final.pt",
            "synthetic sentinel; never opened",
            runner._PARENTS[selected_seed],
        )
        initial = runner._INITIAL_LOSSES[selected_seed]
        training = {
            "initial_pre_update_loss": initial,
            "completed_updates": 8000,
            "parent_state_before": {"synthetic": "descriptor"},
        }
        artifact = {
            name: put(folder + "/" + name, obj)
            for name, obj in [
                ("train-membership.json", membership),
                ("parent.json", parent),
                ("child.json", parent),
                ("training.json", training),
            ]
        }
        summary = {
            "complete": True,
            "final_identity_check": True,
            "seed": selected_seed,
            "objective": runner._OBJECTIVE,
            "split_hash": runner._SPLIT,
            "artifact_sha256": artifact,
            "parent_training_identity": identity,
            "corpus_identity": corpus,
            "inherited_architecture": {"dim": 256},
            "queries": queries,
            "environment": {"synthetic": True},
            "input_sha256": {
                str(tmp_path / parent_folder / "run.json"): run_hash,
                str(tmp_path / parent_folder / "checkpoint-final.pt.json"): sidecar_hash,
                str(tmp_path / "data/records.jsonl"): "must-not-open",
            },
            "child": {
                "aggregate": {
                    k: v for k, v in runner._FLOORS[selected_seed].items() if k != "groups"
                },
                "groups": {
                    g: {"exact_count": n}
                    for g, n in runner._FLOORS[selected_seed]["groups"].items()
                },
            },
        }
        if selected_seed == 1729:
            summary["recipe_sha256"] = put(folder + "/summary.recipe.json", inherited)
        summary_hash = put(
            folder + "/summary.json",
            summary,
            runner._CONTROL1729["summary.json"] if selected_seed == 1729 else None,
        )
        audit = {"complete": True, "audit_passed": True, "summary_sha256": summary_hash}
        audit_name = (
            "independent-audit.json" if selected_seed == 1729 else "independent-audit-v2.json"
        )
        audit_hash = put(
            folder + "/" + audit_name,
            audit,
            runner._CONTROL1729["independent-audit.json"] if selected_seed == 1729 else None,
        )
        if selected_seed == 1729:
            put(
                folder + "/decision.json",
                {"evidence_accepted": True, "fixed_quality_gate_passed": True},
                runner._CONTROL1729["decision.json"],
            )
        else:
            children.append(
                {
                    "seed": selected_seed,
                    "summary_sha256": summary_hash,
                    "audit_sha256": audit_hash,
                    "quality_gate_passed": True,
                }
            )
        summaries[selected_seed] = summary
    aggregate_folder = "runs/learning/bilinear-budget8000-replication-v1"
    values = {
        "aggregate-v2.json": {"complete": True},
        "independent-audit-v2.json": {
            "audit_passed": True,
            "summary_sha256": runner._CONTROL_REPLICATION["aggregate-v2.json"],
        },
        "decision.json": {
            "evidence_accepted": True,
            "children": children,
            "summary_sha256": runner._CONTROL_REPLICATION["aggregate-v2.json"],
            "audit_sha256": runner._CONTROL_REPLICATION["independent-audit-v2.json"],
        },
    }
    for name, value in values.items():
        put(aggregate_folder + "/" + name, value, runner._CONTROL_REPLICATION[name])
    put(
        "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json",
        vocabulary,
        "1b0d8f1762fede01fb253c3df23d21ced254a10f6cea367709c479565a92fb34",
    )
    for name, digest in [("summary.source.zip", "source"), ("summary.configs.zip", "configs")]:
        put("runs/learning/pair-composition-integration-v1/" + name, "archive", digest)
    for path, digest in [
        ("docs/experiments/2026-09-26-bilinear-budget32000-plan.md", runner._PLAN),
        ("docs/experiments/2026-09-26-symmetric-affine-input-amendment.md", runner._AMENDMENT),
        ("scripts/refit_symmetric_affine_v3.py", runner._ADAPTER_SHA),
        ("scripts/refit_bilinear_residual.py", runner._SCORER),
        ("scripts/refit_membership_projection.py", runner._MEAN),
        ("runs/learning/wide-first-choice-v1/summary.script.py", mean._HELPER),
    ]:
        put(path, "source", digest)
    recipe_path = tmp_path / "configs/experiments/bilinear_budget32000_v1.json"
    put(str(recipe_path.relative_to(tmp_path)), recipe)
    if mutation == "swapped_parent":
        summaries[seed]["parent_training_identity"] = {"seeds": [999]}
    if mutation == "missing_queries":
        summaries[seed].pop("queries")
    if mutation == "wrong_architecture":
        summaries[seed].pop("inherited_architecture")
    if mutation == "wrong_initial_loss":
        folder = (
            "runs/learning/bilinear-budget8000-v1"
            if seed == 1729
            else f"runs/learning/bilinear-budget8000-replication-v1/seed-{seed}"
        )
        objects[(tmp_path / folder / "training.json").resolve()]["initial_pre_update_loss"] = -1

    class SelectedHelper:
        _SOURCE = "source"
        _CONFIGS = "configs"

        @staticmethod
        def _bind(path, digest, inputs):
            path = Path(path).resolve()
            assert path in objects and path.suffix not in {".jsonl", ".db", ".sqlite"}, path
            assert digests[path] == digest
            opened.append(path)
            inputs[str(path)] = digest
            return path

        @staticmethod
        def _read(path):
            return objects[Path(path).resolve()]

        @staticmethod
        def _sha(path):
            return digests[Path(path).resolve()]

        @staticmethod
        def _archive_members(paths):
            return []

    args = (
        tmp_path,
        seed,
        tmp_path / "docs/experiments/2026-09-26-bilinear-budget32000-plan.md",
        recipe_path,
        load("refit_bilinear_residual"),
        mean,
        SelectedHelper,
        runner._adapter(ROOT),
    )
    if mutation:
        with pytest.raises((ValueError, KeyError)):
            runner._preflight(*args)
    else:
        context = runner._preflight(*args)
        assert context["seed"] == seed and context["parent"].parent.name.endswith(f"s{seed}_v1")
        assert (
            len(context["partitions"]["train"]) == 1637
            and len(context["partitions"]["validation"]) == 222
        )
        assert context["input_adapter"]["whole_corpus_revalidated"] is False
    assert tmp_path / "data/records.jsonl" not in opened
    assert not any(p.suffix == ".db" for p in opened)


def readiness_fixture(runner, tmp_path, monkeypatch):
    files = {}
    # All exact named paths exist only within this temporary synthetic repository.
    for name in runner._READINESS_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = sha(path)
    pins = {
        "_PLAN": "docs/experiments/2026-09-26-bilinear-budget32000-plan.md",
        "_AMENDMENT": "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
        "_ADAPTER_SHA": "scripts/refit_symmetric_affine_v3.py",
        "_SCORER": "scripts/refit_bilinear_residual.py",
        "_MEAN": "scripts/refit_membership_projection.py",
    }
    for attr, name in pins.items():
        monkeypatch.setattr(runner, attr, files[name])
    for role, directory in runner._TEST_DIRS.items():
        receipt = {
            "passed": True,
            "gpu_used": False,
            "experiment_training_executed": False,
            "exit_code": 0,
            "terminal_completion_observed": True,
            "tests_passed": 1,
            "tests_skipped": 0,
            "tested_script_sha256": files[runner._TEST_SCRIPTS[role]],
            "tested_recipe_sha256": files["configs/experiments/bilinear_budget32000_v1.json"],
            "test_dependencies": {n: files[n] for n in runner._TEST_DEPENDENCIES[role]},
            "stdout": {
                "path": directory + "/stdout.txt",
                "sha256": files[directory + "/stdout.txt"],
            },
        }
        name = directory + "/test-receipt.json"
        files[name] = write(tmp_path / name, receipt)
    contract = {
        "schema_version": 1,
        "campaign_version": runner._CAMPAIGN,
        "plan_sha256": runner._PLAN,
        "ready": True,
        "files": files,
    }
    path = tmp_path / runner._READINESS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, contract, write(path, contract)


def test_readiness_exact_portable_inventory(runner, tmp_path, monkeypatch):
    path, contract, pin = readiness_fixture(runner, tmp_path, monkeypatch)
    inputs = {}
    result = runner._readiness(tmp_path, path, pin, Helper, inputs)
    assert result == contract and len(inputs) == 23 and len(contract["files"]) == 22


@pytest.mark.parametrize(
    "mutation", ["extra", "escape", "unknown", "missing", "false", "boolschema"]
)
def test_readiness_rejects_before_any_dependency_open(runner, tmp_path, monkeypatch, mutation):
    path, contract, _pin = readiness_fixture(runner, tmp_path, monkeypatch)
    if mutation in ("extra", "escape", "unknown"):
        name = {
            "extra": "data/records.jsonl",
            "escape": "../protected.json",
            "unknown": "runs/learning/protected.json",
        }[mutation]
        contract["files"][name] = "0" * 64
    if mutation == "missing":
        contract["files"].pop(next(iter(contract["files"])))
    if mutation == "false":
        contract["ready"] = False
    if mutation == "boolschema":
        contract["schema_version"] = True
    pin = write(path, contract)
    opened = []

    class Guard(Helper):
        @staticmethod
        def _bind(target, digest, inputs):
            opened.append(Path(target))
            assert Path(target) == path
            return Helper._bind(target, digest, inputs)

    with pytest.raises(ValueError):
        runner._readiness(tmp_path, path, pin, Guard, {})
    assert opened == [path]


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", False),
        ("exit_code", 1),
        ("terminal_completion_observed", False),
        ("tests_passed", 0),
        ("tests_skipped", True),
        ("gpu_used", True),
        ("tested_script_sha256", "bad"),
    ],
)
def test_readiness_rejects_bad_observed_test_receipts(runner, tmp_path, monkeypatch, field, value):
    path, contract, _pin = readiness_fixture(runner, tmp_path, monkeypatch)
    receipt_path = tmp_path / runner._TEST_DIRS["runner"] / "test-receipt.json"
    receipt = Helper._read(receipt_path)
    receipt[field] = value
    contract["files"][receipt_path.relative_to(tmp_path).as_posix()] = write(receipt_path, receipt)
    pin = write(path, contract)
    with pytest.raises(ValueError):
        runner._readiness(tmp_path, path, pin, Helper, {})


@pytest.mark.parametrize(
    "target",
    [
        "scripts/refit_bilinear_budget32000.py",
        "tests/fixtures/projection_refit_runtime_source.zip",
        "runs/learning/bilinear-budget32000-auditor-tests-v1/stdout.txt",
    ],
)
def test_readiness_tamper_rejected(runner, tmp_path, monkeypatch, target):
    path, _contract, pin = readiness_fixture(runner, tmp_path, monkeypatch)
    (tmp_path / target).write_text("tampered")
    with pytest.raises(ValueError):
        runner._readiness(tmp_path, path, pin, Helper, {})


def test_unknown_readiness_path_never_opened(runner, tmp_path):
    class Guard:
        @staticmethod
        def _bind(path, digest, inputs):
            raise AssertionError("forbidden open")

    with pytest.raises(ValueError):
        runner._readiness(tmp_path, tmp_path / "protected.json", "0" * 64, Guard, {})


@pytest.mark.parametrize(
    "name",
    [
        "train-scores-post8000.json",
        "train-scores-post32000.json",
        "train-diagnostics.json",
        "failure-state.json",
        "runtime",
        "readiness",
    ],
)
def test_immutable_outputs_and_fixed_seed_path(runner, mean, tmp_path, name):
    out = tmp_path / "runs/learning/bilinear-budget32000-v1/seed-1729/summary.json"
    out.parent.mkdir(parents=True)
    runner._admit_output(tmp_path, 1729, out, mean)
    (out.parent / name).write_text("preserved")
    with pytest.raises(ValueError):
        runner._admit_output(tmp_path, 1729, out, mean)
    with pytest.raises(ValueError):
        runner._admit_output(tmp_path, 1730, out, mean)


def one_parameter_model():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(dim=2)

    return load("refit_bilinear_residual")._attach(Model())


@pytest.mark.parametrize(
    "mutation",
    [None, "missing", "extra", "step", "nan", "shape", "dtype", "lr", "amsgrad", "fused"],
)
def test_one_parameter_optimizer_contract(runner, mutation):
    import torch

    model = one_parameter_model()
    parameter = getattr(model, runner._PARAMETERS[0])
    optimizer = torch.optim.AdamW([parameter], lr=0.0003, weight_decay=0)
    parameter.sum().backward()
    optimizer.step()
    optimizer.state[parameter]["step"].fill_(32000)
    if mutation == "missing":
        optimizer.state.clear()
    if mutation == "extra":
        optimizer.state[model.base] = {}
    if mutation == "step":
        optimizer.state[parameter]["step"].fill_(8000)
    if mutation == "nan":
        optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    if mutation == "shape":
        optimizer.state[parameter]["exp_avg"] = torch.zeros(1)
    if mutation == "dtype":
        optimizer.state[parameter]["exp_avg"] = optimizer.state[parameter]["exp_avg"].double()
    if mutation == "lr":
        optimizer.param_groups[0]["lr"] = 0.01
    if mutation == "amsgrad":
        optimizer.param_groups[0]["amsgrad"] = True
    if mutation == "fused":
        optimizer.param_groups[0]["fused"] = True
    if mutation:
        with pytest.raises(ValueError):
            runner._validate_optimizer(optimizer, model, 32000)
    else:
        assert runner._validate_optimizer(optimizer, model, 32000)["steps"] == {
            runner._PARAMETERS[0]: 32000.0
        }


def test_failure_diagnostic_is_nonresumable_and_attempt_count_unknown(runner, mean, tmp_path):
    import torch

    model = one_parameter_model()
    runner._preserve_failure(
        tmp_path, model, None, {"completed_updates": 12, "error": "synthetic"}, mean, Helper
    )
    saved = torch.load(tmp_path / "failure-state.pt", weights_only=True)
    assert saved["diagnostic_only"] and not saved["resume_allowed"]
    assert saved["completed_updates"] == 12 and saved["optimizer_steps_attempted"] is None
    with pytest.raises(FileExistsError):
        runner._preserve_failure(tmp_path, model, None, {}, mean, Helper)


def test_no_whole_corpus_loader_or_frozen_execute():
    tree = ast.parse((ROOT / "scripts/refit_bilinear_budget32000.py").read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(n and n.startswith(("plm.corpus", "plm.graph", "plm.serving")) for n in imports)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(
        isinstance(n.func, ast.Attribute)
        and n.func.attr in {"_execute", "load_inference_runtime", "_preflight"}
        for n in calls
    )


@pytest.mark.parametrize("seed", [1730, 1731])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "failed_gate",
        "execution_failure",
        "audit_failure",
        "audit_late",
        "submicro_late",
        "readiness",
        "auditor",
        "terminal_bool",
        "audit_seed",
        "audit_plan_sha256",
        "audit_shared_readiness",
        "audit_test_receipt_sha256",
        "audit_execution_receipt_sha256",
        "audit_cuda",
    ],
)
def test_previous_seed_audit_required_before_admission(runner, tmp_path, seed, mutation):
    previous = seed - 1
    directory = tmp_path / f"runs/learning/{runner._CAMPAIGN}/seed-{previous}"
    directory.mkdir(parents=True)
    ready = {
        "files": {
            runner._TEST_SCRIPTS["runner"]: "runner",
            runner._TEST_SCRIPTS["auditor"]: "auditor",
            "configs/experiments/bilinear_budget32000_v1.json": "recipe",
            runner._TEST_DIRS["auditor"] + "/test-receipt.json": "tests",
        }
    }
    summary = {
        "seed": previous,
        "complete": True,
        "final_identity_check": True,
        "script_sha256": "runner",
        "recipe_sha256": "recipe",
        "shared_readiness": {"path": runner._READINESS_PATH, "sha256": "ready"},
        "gate": {
            "checks": {"execution_invariants": True},
            "primary_checks_passed": mutation != "failed_gate",
        },
    }
    if mutation == "execution_failure":
        summary["complete"] = False
    if mutation == "readiness":
        summary["shared_readiness"]["sha256"] = "changed"
    summary_sha = write(directory / "summary.json", summary)
    (directory / "primary-stdout.txt").write_text("synthetic")
    (directory / "audit-stdout.txt").write_text("synthetic audit")
    execution = {
        "seed": previous,
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "summary_sha256": summary_sha,
        "stdout_sha256": sha(directory / "primary-stdout.txt"),
        "started_utc": "2026-09-26T10:00:00Z",
        "completed_utc": "2026-09-26T10:01:00Z",
    }
    if mutation == "terminal_bool":
        execution["exit_code"] = False
    execution_sha = write(directory / "execution-receipt.json", execution)
    audit = {
        "complete": True,
        "audit_passed": mutation != "audit_failure",
        "seed": previous,
        "plan_sha256": runner._PLAN,
        "shared_readiness": {"path": runner._READINESS_PATH, "sha256": "ready"},
        "summary_sha256": summary_sha,
        "script_sha256": "auditor",
        "test_receipt_sha256": "tests",
        "execution_receipt_sha256": execution_sha,
    }
    for key in (
        "seed",
        "plan_sha256",
        "shared_readiness",
        "test_receipt_sha256",
        "execution_receipt_sha256",
    ):
        if mutation == "audit_" + key:
            audit[key] = "changed"
    audit_sha = write(directory / "independent-audit.json", audit)
    terminal = {
        "seed": previous,
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "summary_sha256": summary_sha,
        "audit_sha256": audit_sha,
        "execution_receipt_sha256": execution_sha,
        "stdout_sha256": sha(directory / "audit-stdout.txt"),
        "auditor_sha256": "auditor",
        "test_receipt_sha256": "tests",
        "cuda_visible_devices": "-1",
        "started_utc": "2026-09-26T10:01:01Z",
        "completed_utc": "2026-09-26T10:02:00.1234567Z",
    }
    if mutation == "audit_late":
        terminal["completed_utc"] = "2026-09-26T10:03:00Z"
    if mutation == "submicro_late":
        terminal["completed_utc"] = "2026-09-26T10:02:00.1234569Z"
    if mutation == "auditor":
        terminal["auditor_sha256"] = "changed"
    if mutation == "audit_cuda":
        terminal["cuda_visible_devices"] = "0"
    write(directory / "audit-execution-receipt.json", terminal)
    args = (tmp_path, seed, ready, Helper, {}, "ready", "2026-09-26T10:02:00.1234568Z")
    if mutation not in (None, "failed_gate"):
        with pytest.raises(ValueError):
            runner._previous_seed(*args)
    else:
        prior = runner._previous_seed(*args)
        assert prior["seed"] == previous and prior["execution_receipt_sha256"] == execution_sha


@pytest.mark.parametrize(
    "value", [None, "2026-09-26T10:00:00", "2026-09-26T10:00:00+24:00", "2026-09-26T10:00:00+01:99"]
)
def test_chronology_rejects_naive_or_invalid_timestamp(runner, value):
    with pytest.raises(ValueError):
        runner._instant(value)


def test_chronology_timezone_equivalence_and_fraction_precision(runner):
    assert runner._instant("2026-09-26T10:00:00.123456789Z") == runner._instant(
        "2026-09-26T05:00:00.123456789-05:00"
    )
    assert runner._instant("2026-09-26T10:00:00.12345678Z") < runner._instant(
        "2026-09-26T10:00:00.12345679Z"
    )


@pytest.mark.parametrize("seed", [1729, 1730, 1731])
def test_parent_hash_matches_archived_configuration_encoder(runner, mean, monkeypatch, seed):
    import sys
    import zipfile
    from types import ModuleType

    with zipfile.ZipFile(ROOT / "tests/fixtures/projection_refit_runtime_source.zip") as archive:
        module = ModuleType("archived_config_hash_schema")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(
            compile(archive.read("src/plm/config.py").decode(), "archived-config", "exec"),
            module.__dict__,
        )
        tree = ast.parse(archive.read("src/plm/configuration.py").decode())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"resolved_config_dict", "config_hash"}
    ]
    namespace = {"RootConfig": module.RootConfig, "hashlib": hashlib, "json": json}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "archived-config-hash", "exec"), namespace
    )
    config = module.RootConfig.model_validate(
        {"seed": seed, "run_name": "synthetic-label", "train": {"seq_len": 512, "max_steps": 2000}}
    )
    raw = config.model_dump(mode="json")
    expected = namespace["config_hash"](config)
    assert runner._parent_config_hash(raw) == expected
    assert mean._digest({k: v for k, v in raw.items() if k != "run_name"}) != expected
    raw["run_name"] = "another-filesystem-label"
    assert runner._parent_config_hash(raw) == expected
    raw["seed"] += 1
    assert runner._parent_config_hash(raw) != expected


@pytest.mark.parametrize("mode", ["execute", "preflight", "predecessor_failure"])
def test_main_cli_snapshot_orchestration(runner, mean, tmp_path, monkeypatch, capsys, mode):
    """Real CLI/readiness/snapshot orchestration; expensive stages tested separately."""
    import inspect
    import sys

    readiness, contract, pin = readiness_fixture(runner, tmp_path, monkeypatch)
    script = tmp_path / runner._TEST_SCRIPTS["runner"]
    recipe = tmp_path / "configs/experiments/bilinear_budget32000_v1.json"
    bilinear_path = tmp_path / "scripts/refit_bilinear_residual.py"
    mean_path = tmp_path / "scripts/refit_membership_projection.py"
    helper_path = tmp_path / "runs/learning/wide-first-choice-v1/summary.script.py"
    source = tmp_path / "synthetic-source.zip"
    configs = tmp_path / "synthetic-configs.zip"
    source.write_bytes(b"synthetic archive source")
    configs.write_bytes(b"synthetic archive configs")
    context = {
        "inputs": {},
        "source": source,
        "configs": configs,
        **{
            f"historical8000_{name}_sha256": name
            for name in ("summary", "audit", "decision", "child")
        },
    }
    events = []

    class MainHelper(Helper):
        @staticmethod
        def _modules():
            events.append("modules")

        @staticmethod
        def _unchanged(inputs):
            for path, digest in inputs.items():
                assert sha(path) == digest
            events.append("unchanged")

    mean_double = SimpleNamespace(
        _load_helper=lambda root: (MainHelper, helper_path),
        _refuse=mean._refuse,
        _bytes=mean._bytes,
        _HELPER=sha(helper_path),
    )
    bilinear_double = SimpleNamespace(_load_mean=lambda root: (mean_double, mean_path))
    adapter = object()
    original_preflight = runner._preflight
    original_execute = runner._execute
    original_previous = runner._previous_seed

    def preflight(root, seed, plan, recipe_path, bilinear, mean, helper, adapter):
        inspect.signature(original_preflight).bind(
            root, seed, plan, recipe_path, bilinear, mean, helper, adapter
        )
        assert root == tmp_path and seed == 1730
        assert plan.is_file() and recipe_path.is_file()
        assert bilinear is bilinear_double and mean is mean_double and helper is MainHelper
        events.append("preflight")
        return context

    def previous_seed(root, seed, ready, helper, inputs, readiness_sha256, admission_utc):
        inspect.signature(original_previous).bind(
            root, seed, ready, helper, inputs, readiness_sha256, admission_utc
        )
        assert ready == contract and readiness_sha256 == pin
        runner._instant(admission_utc)
        events.append("previous")
        if mode == "predecessor_failure":
            raise ValueError("synthetic predecessor refusal")
        return {"seed": 1729, "summary_sha256": "previous"}

    def execute(root, out, context, bilinear, mean, helper, summary):
        inspect.signature(original_execute).bind(
            root, out, context, bilinear, mean, helper, summary
        )
        assert root == tmp_path and bilinear is bilinear_double and mean is mean_double
        assert summary["seed"] == 1730 and not summary["complete"]
        assert summary["shared_readiness"] == {"path": runner._READINESS_PATH, "sha256": pin}
        assert summary["prior_seed_execution"] == {"seed": 1729, "summary_sha256": "previous"}
        assert summary["script_sha256"] == sha(script)
        assert summary["recipe_sha256"] == sha(recipe)
        assert summary["input_sha256"] == context["inputs"]
        assert len(summary["snapshot_sha256"]) == 34
        for path, digest in summary["snapshot_sha256"].items():
            assert sha(path) == digest
        for relative, digest in contract["files"].items():
            assert sha(out.parent / "readiness" / relative) == digest
        assert out.with_suffix(".script.py").read_bytes() == script.read_bytes()
        assert out.with_suffix(".source.zip").read_bytes() == source.read_bytes()
        assert out.with_suffix(".configs.zip").read_bytes() == configs.read_bytes()
        assert Helper._read(out.with_suffix(".inputs.json")) == context["inputs"]
        events.append("execute")
        summary["complete"] = True
        write(out, summary)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "__file__", str(script))
    # Simulate a fresh Torch-free process without changing global import state.
    monkeypatch.setattr(runner, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(runner, "_load_bilinear", lambda root: (bilinear_double, bilinear_path))
    monkeypatch.setattr(runner, "_adapter", lambda root: adapter)
    monkeypatch.setattr(runner, "_preflight", preflight)
    monkeypatch.setattr(runner, "_previous_seed", previous_seed)
    monkeypatch.setattr(runner, "_execute", execute)
    argv = [str(script), "--seed", "1730", "--readiness", str(readiness), "--readiness-sha256", pin]
    if mode == "preflight":
        argv.append("--preflight-only")
    monkeypatch.setattr(sys, "argv", argv)
    out = tmp_path / f"runs/learning/{runner._CAMPAIGN}/seed-1730/summary.json"
    if mode == "predecessor_failure":
        with pytest.raises(ValueError, match="predecessor refusal"):
            runner.main()
        assert events == ["modules", "preflight", "unchanged", "previous"]
        assert not out.parent.exists()
    else:
        runner.main()
        if mode == "preflight":
            assert json.loads(capsys.readouterr().out)["preflight_passed"] is True
            assert events == ["modules", "preflight", "unchanged"]
            assert not out.parent.exists()
        else:
            assert events == ["modules", "preflight", "unchanged", "previous", "execute"]
            assert Helper._read(out)["complete"] is True
