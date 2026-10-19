"""Synthetic-only audit contracts; no real evidence or checkpoint operations."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def audit():
    path = ROOT / "runs/learning/bilinear-budget32000-v1/independent-audit.py"
    spec = importlib.util.spec_from_file_location("independent_budget32000_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def membership(index=0):
    return {
        "index": index,
        "subject": "PKM_SYNTHETIC",
        "dimension": "TYPE",
        "prompt_ids": [1, 1024, 32, 34, 5],
        "expected_set_ids": [1025, 1026],
    }


def vector(positive=1.0, negative=-1.0):
    values = [negative] * 1025
    values[0] = 100.0
    values[1] = values[2] = positive
    return values


@pytest.mark.parametrize(
    "positive,negative,exact,separated,margin",
    [
        (1.0, -1.0, True, True, 1.0),
        (0.0, -1.0, False, True, 0.0),
        (1.0, 0.0, True, True, 0.0),
        (1.0, 1.0, False, False, -1.0),
        (-1.0, -2.0, False, True, -1.0),
        (-2.0, -1.0, False, False, -2.0),
    ],
)
def test_signed_margin_strict_threshold_and_separation(
    audit, positive, negative, exact, separated, margin
):
    row = audit.score_row(vector(positive, negative), membership())
    assert row["metrics"]["exact"] is exact
    assert row["strict_separation"] is separated
    assert row["zero_margin"] == margin
    assert row["min_positive_id"] == 1025 and row["max_negative_id"] == 1027
    assert 1024 not in row["selected_set_ids"]
    assert row["separation_gap"] == positive - negative


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, 1, 0.1])
def test_score_vector_rejects_non_fp32_or_nonfloat(audit, bad):
    values = vector()
    values[3] = bad
    with pytest.raises(ValueError):
        audit.score_row(values, membership())


@pytest.mark.parametrize(
    "mutation",
    ["self", "duplicate", "unsorted", "empty", "range", "dimension", "delimiter", "bool_id"],
)
def test_membership_contract_rejects_drift(audit, mutation):
    row = membership()
    if mutation == "self":
        row["expected_set_ids"] = [1024]
    elif mutation == "duplicate":
        row["expected_set_ids"] = [1025, 1025]
    elif mutation == "unsorted":
        row["expected_set_ids"] = [1026, 1025]
    elif mutation == "empty":
        row["expected_set_ids"] = []
    elif mutation == "range":
        row["expected_set_ids"] = [9999]
    elif mutation == "dimension":
        row["dimension"] = "BIOME"
    elif mutation == "delimiter":
        row["prompt_ids"][-1] = 2
    else:
        row["prompt_ids"][0] = True
    with pytest.raises(ValueError):
        audit.score_row(vector(), row)


def trace(count=32000):
    return [
        {
            "update": i,
            "pre_update_loss": 1.0 / i,
            "gradient_finite": True,
            "parameters_finite": True,
        }
        for i in range(1, count + 1)
    ]


def test_complete_32000_trace(audit):
    audit.loss_contract(trace(), 1.0, 1.0, 1.0, 0.00001)


@pytest.mark.parametrize(
    "mutation",
    [
        "short",
        "long",
        "reordered",
        "bool_update",
        "nan",
        "negative",
        "gradient",
        "parameters",
        "initial",
        "parent",
    ],
)
def test_loss_contract_rejects_incomplete_or_invalid_updates(audit, mutation):
    rows = trace()
    parent = 1.0
    zero = 1.0
    initial = 1.0
    if mutation == "short":
        rows.pop()
    elif mutation == "long":
        rows.append(rows[-1])
    elif mutation == "reordered":
        rows[3], rows[4] = rows[4], rows[3]
    elif mutation == "bool_update":
        rows[0]["update"] = True
    elif mutation == "nan":
        rows[9]["pre_update_loss"] = float("nan")
    elif mutation == "negative":
        rows[9]["pre_update_loss"] = -1.0
    elif mutation == "gradient":
        rows[9]["gradient_finite"] = False
    elif mutation == "parameters":
        rows[9]["parameters_finite"] = 1
    elif mutation == "initial":
        initial = 2.0
    else:
        parent = 2.0
    with pytest.raises(ValueError):
        audit.loss_contract(rows, parent, zero, initial, 0.1)


def test_prefix_mismatch_is_descriptive_not_gate(audit):
    new = trace()
    old = trace(8000)
    old[99]["pre_update_loss"] = 123.0
    result = audit.prefix_contract(new, old, 1.0, 1.0, 0.25)
    assert result["mismatch_count"] == 1 and result["first_mismatch"]["update"] == 100
    assert result["boundary_loss_equal"] is False
    assert result["descriptive_only"] is True
    assert result["intermediate_weight_equality_established"] is False


def diagnostics():
    return [
        {
            "post_update": p,
            "fit_loss_callback": c,
            "partition": "train",
            "grad_enabled": g,
            "affects_optimization": False,
        }
        for p, c, g in [(8000, 8001, True), (32000, 32001, False)]
    ]


def test_exact_diagnostic_chronology(audit):
    audit.diagnostic_points_contract(diagnostics())


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "callback", "partition", "grad", "affects", "initial_replay_counted"],
)
def test_diagnostic_chronology_rejects_shifted_or_repeated_capture(audit, mutation):
    rows = diagnostics()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "callback":
        rows[0]["fit_loss_callback"] = 8000
    elif mutation == "partition":
        rows[0]["partition"] = "validation"
    elif mutation == "grad":
        rows[-1]["grad_enabled"] = True
    elif mutation == "affects":
        rows[0]["affects_optimization"] = True
    else:
        rows[0]["fit_loss_callback"] = 8002
    with pytest.raises(ValueError):
        audit.diagnostic_points_contract(rows)


def baseline(audit, seed):
    floor = audit.FLOORS[seed]
    return {
        "aggregate": {
            "exact_count": floor["exact"],
            "f1": floor["f1"],
            "serialization_compatible": 222,
        },
        "groups": {
            g: {"exact_count": v} for g, v in zip(audit.GROUPS, floor["groups"], strict=True)
        },
    }


@pytest.mark.parametrize("seed", [1729, 1730, 1731])
def test_seed_gate_strict_tie_and_gain(audit, seed):
    old = baseline(audit, seed)
    child = copy.deepcopy(old)
    inv = dict.fromkeys(audit.INVARIANTS, True)
    assert audit.seed_gate(child, old, inv, seed)["primary_checks_passed"] is False
    child["aggregate"]["exact_count"] += 1
    assert audit.seed_gate(child, old, inv, seed)["primary_checks_passed"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "f1",
        "group",
        "serialization",
        "invariant",
        "missing_invariant",
        "extra_invariant",
        "baseline",
    ],
)
def test_gate_cannot_trade_metrics_or_weaken_comparator(audit, mutation):
    old = baseline(audit, 1729)
    child = copy.deepcopy(old)
    child["aggregate"]["exact_count"] = 222
    inv = dict.fromkeys(audit.INVARIANTS, True)
    if mutation == "f1":
        child["aggregate"]["f1"] = 0.9
    elif mutation == "group":
        child["groups"]["COLOR"]["exact_count"] = 102
    elif mutation == "serialization":
        child["aggregate"]["serialization_compatible"] = 221
    elif mutation == "invariant":
        inv["reload_exact"] = False
    elif mutation == "missing_invariant":
        inv.pop("reload_exact")
    elif mutation == "extra_invariant":
        inv["invented"] = True
    else:
        old["aggregate"]["exact_count"] = 200
        with pytest.raises(ValueError):
            audit.seed_gate(child, old, inv, 1729)
        return
    assert audit.seed_gate(child, old, inv, 1729)["primary_checks_passed"] is False


def test_campaign_requires_three_seed_conjunction(audit):
    gates = {s: {"primary_checks_passed": True} for s in audit.SEEDS}
    assert audit.campaign_gate(gates)["all_three_primary_gates_passed"] is True
    gates[1731]["primary_checks_passed"] = False
    assert audit.campaign_gate(gates)["all_three_primary_gates_passed"] is False
    del gates[1731]
    with pytest.raises(ValueError):
        audit.campaign_gate(gates)


def terminal(seed=1729):
    return {
        "seed": seed,
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "summary_sha256": "a" * 64,
        "stdout_sha256": "b" * 64,
        "started_utc": "2026-09-26T10:00:00.000001+00:00",
        "completed_utc": "2026-09-26T10:00:00.000002+00:00",
        "prior_seed": None,
    }


def test_timezone_aware_fractional_chronology(audit):
    first = terminal()
    audit.execution_contract(first, 1729, "a" * 64, "b" * 64)
    second = terminal(1730)
    second["started_utc"] = "2026-09-26T05:00:00.000002-05:00"
    second["completed_utc"] = "2026-09-26T10:00:00.000003+00:00"
    second["prior_seed"] = {
        "seed": 1729,
        "summary_sha256": "a" * 64,
        "completed_utc": first["completed_utc"],
    }
    audit.execution_contract(second, 1730, "a" * 64, "b" * 64, first)
    second["started_utc"] = "2026-09-26T10:00:00.000001+00:00"
    with pytest.raises(ValueError, match="nonoverlapping"):
        audit.execution_contract(second, 1730, "a" * 64, "b" * 64, first)


@pytest.mark.parametrize(
    "key,value",
    [
        ("exit_code", False),
        ("exit_code", None),
        ("terminal_completion_observed_by_primary", 1),
        ("started_utc", "2026-09-26T10:00:00"),
        ("completed_utc", "bad"),
        ("summary_sha256", "wrong"),
        ("stdout_sha256", "wrong"),
        ("seed", True),
        ("prior_seed", {}),
    ],
)
def test_terminal_receipt_rejects_incomplete_or_wrong_identity(audit, key, value):
    row = terminal()
    row[key] = value
    with pytest.raises(ValueError):
        audit.execution_contract(row, 1729, "a" * 64, "b" * 64)


def test_whole_manifest_admission_precedes_any_open(audit, tmp_path, monkeypatch):
    good = tmp_path / "good.json"
    good.write_text("{}", encoding="utf-8")
    bad = tmp_path / "data/records.jsonl"
    reader = audit.EvidenceReader(tmp_path, ["good.json"])
    opened = []
    monkeypatch.setattr(audit, "sha", lambda path: opened.append(path))
    with pytest.raises(ValueError, match="allowed"):
        reader.manifest({str(good): "a" * 64, str(bad): "b" * 64})
    assert opened == []


@pytest.mark.parametrize(
    "relative", ["data/records.jsonl", "data/graph.sqlite", "protected.json", "../outside.json"]
)
def test_forbidden_reads_rejected_before_hashing(audit, tmp_path, monkeypatch, relative):
    reader = audit.EvidenceReader(tmp_path, [])
    opened = []
    monkeypatch.setattr(audit, "sha", lambda path: opened.append(path))
    with pytest.raises(ValueError):
        reader.read(relative)
    assert opened == []


def test_hash_bound_json_reader(audit, tmp_path):
    path = tmp_path / "good.json"
    path.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    reader = audit.EvidenceReader(tmp_path, ["good.json"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert reader.read("good.json", digest) == {"synthetic": True}
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        reader.read("good.json", digest)


def test_arbitrary_fractional_precision_is_not_truncated(audit):
    a = audit.instant("2026-09-26T10:00:00.0000001Z")
    b = audit.instant("2026-09-26T05:00:00.0000002-05:00")
    assert b > a
    assert audit.instant("2026-09-26T10:00:00.000000100000Z") == a
    assert audit.instant("2026-09-26T12:00:00.0000001+02:00") == a


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-09-26T10:00:00",
        "2026-09-26 10:00:00Z",
        "2026-09-26T10:00:00.+00:00",
        "2026-09-26T10:00:00.1+25:00",
        "2026-02-30T10:00:00Z",
    ],
)
def test_invalid_timestamp_rejected(audit, stamp):
    with pytest.raises(ValueError):
        audit.instant(stamp)


def synthetic_optimizer(audit):
    import torch

    options = {
        "lr": 0.0003,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "foreach": None,
        "fused": None,
    }
    optimizer = {
        "state": {
            0: {
                "step": torch.tensor(32000.0),
                "exp_avg": torch.ones(audit.A_SHAPE),
                "exp_avg_sq": torch.ones(audit.A_SHAPE),
            }
        },
        "param_groups": [{**options, "params": [0]}, {**options, "params": []}],
    }
    implementation = {
        "class": "torch.optim.adamw.AdamW",
        "groups": [{**options, "parameter_names": [audit.A]}, {**options, "parameter_names": []}],
        "defaults": {**options, "weight_decay": 0.01},
    }
    return optimizer, implementation


@pytest.mark.parametrize("mutation", ["valid", "step", "moment", "second", "extra", "fused"])
def test_cpu_optimizer_state_contract(audit, mutation):
    import torch

    optimizer, implementation = synthetic_optimizer(audit)
    if mutation == "step":
        optimizer["state"][0]["step"] = torch.tensor(31999.0)
    if mutation == "moment":
        optimizer["state"][0]["exp_avg"][0, 0, 0] = float("nan")
    if mutation == "second":
        optimizer["state"][0]["exp_avg_sq"][0, 0, 0] = -1
    if mutation == "extra":
        optimizer["state"][1] = optimizer["state"][0]
    if mutation == "fused":
        optimizer["param_groups"][0]["fused"] = True
    if mutation == "valid":
        assert audit.optimizer_contract(optimizer, implementation)["steps"] == {audit.A: 32000}
    else:
        with pytest.raises(ValueError):
            audit.optimizer_contract(optimizer, implementation)


@pytest.mark.parametrize("mutation", ["valid", "changed", "missing", "zero", "shape", "nan"])
def test_cpu_state_contract(audit, mutation):
    import torch

    parent = {f"synthetic_{i}": torch.tensor([float(i)]) for i in range(93)}
    child = {k: v.clone() for k, v in parent.items()} | {audit.A: torch.ones(audit.A_SHAPE)}
    if mutation == "changed":
        child["synthetic_0"][0] = 1
    if mutation == "missing":
        child.pop("synthetic_0")
    if mutation == "zero":
        child[audit.A].zero_()
    if mutation == "shape":
        child[audit.A] = torch.ones(2, 1, 1)
    if mutation == "nan":
        child[audit.A][0, 0, 0] = float("nan")
    if mutation == "valid":
        before, zero, final = audit.compare_states(parent, child)
        assert len(before) == 93 and len(zero) == len(final) == 94
    else:
        with pytest.raises(ValueError):
            audit.compare_states(parent, child)


def test_readiness_inventory_admitted_before_dependencies(audit, tmp_path, monkeypatch):
    value = {
        "schema_version": 1,
        "campaign_version": audit.CAMPAIGN,
        "plan_sha256": audit.PLAN_SHA,
        "ready": True,
        "files": dict.fromkeys(audit.READINESS_FILES, "a" * 64),
    }
    value["files"]["data/protected.jsonl"] = "b" * 64
    path = tmp_path / audit.READINESS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    reader = audit.EvidenceReader(tmp_path, [audit.READINESS])
    with pytest.raises(ValueError, match="exact shared readiness inventory"):
        audit.readiness_contract(reader, audit.sha(path))
    assert list(reader.inputs) == [str(path)]


def test_local_static_call_signatures(audit):
    """Bind direct local function calls without running any experiment code."""
    import ast
    import inspect

    tree = ast.parse(Path(audit.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        target = getattr(audit, node.func.id, None)
        if not inspect.isfunction(target) or target.__module__ != audit.__name__:
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            continue
        inspect.signature(target).bind(
            *[None for _ in node.args], **{k.arg: None for k in node.keywords}
        )


def synthetic_training(audit, seed, membership_rows, validation_rows):
    """Tiny CPU tensors, actual32000 metadata chronology; no optimizer or model call."""
    import torch

    parent_config = {
        "seed": seed,
        "run_name": f"national_dex_continuation_control_s{seed}_v1",
        "data": {"split_seed": 1729},
        "train": {"synthetic": True},
    }
    parent_identity = {
        "seeds": [seed],
        "evaluator_version": "plm-train-causal-v2",
        "resolved_config_hash": hashlib.sha256(
            json.dumps(
                {k: v for k, v in parent_config.items() if k != "run_name"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest(),
        "source_commit": "synthetic-original",
        "environment": {},
        "split_hash": audit.SPLIT_SHA,
        "checkpoint_hash": None,
        "repetitions": 1,
    }
    original = {f"synthetic_{i}": torch.tensor([float(i)]) for i in range(93)}
    parent = {
        "model": original,
        "global_step": 2000,
        "config": parent_config["train"],
        "experiment_identity": parent_identity,
        "corpus_identity": {},
        "split_hash": audit.SPLIT_SHA,
        "training_metadata": {
            "objective": "causal-next-token-v1",
            "model_config": {"synthetic": True},
        },
    }
    child_state = {k: v.clone() for k, v in original.items()} | {audit.A: torch.ones(audit.A_SHAPE)}
    before, initial, final = audit.compare_states(original, child_state)
    optimizer, opt_impl = synthetic_optimizer(audit)
    recipe = {"optimizer": {"synthetic": True}}
    queries = {
        "train": [
            {k: r[k] for k in ("subject", "dimension", "prompt_ids")} for r in membership_rows
        ],
        "validation": [
            {k: r[k] for k in ("subject", "dimension", "prompt_ids")} for r in validation_rows
        ],
    }
    order = audit.formatted_hash(queries["train"])
    implementation = {
        "runner_sha256": "a" * 64,
        "scorer_sha256": audit.SCORER_SHA,
        "input_adapter_identity": audit.INPUT_ADAPTER,
        "input_amendment_sha256": audit.AMENDMENT_SHA,
        "plan_sha256": audit.PLAN_SHA,
        "bilinear_helper_sha256": audit.SCORER_SHA,
        "mean_helper_sha256": audit.MEAN_SHA,
        "input_adapter_source_sha256": audit.ADAPTER_SHA,
        "source_archive_sha256": audit.SOURCE_SHA,
        "configs_archive_sha256": audit.CONFIGS_SHA,
    }
    config = {
        "implementation_identity": implementation,
        "inherited_parent_config": parent_config,
        "bilinear_residual_architecture": copy.deepcopy(audit.ARCHITECTURE_DESCRIPTOR),
        "bilinear_residual_recipe": recipe,
        "seed": seed,
        "data_split_seed": 1729,
        "campaign_version": audit.CAMPAIGN,
    }
    metadata = {
        "implementation_identity": implementation,
        "seed": seed,
        "data_split_seed": 1729,
        "campaign_version": audit.CAMPAIGN,
        "objective": audit.OBJECTIVE,
        "evaluator": audit.EVALUATOR,
        "architecture": audit.ARCHITECTURE,
        "model_config": {"synthetic": True},
        "model_config_scope": "inherited archived base only; residual architecture is separate",
        "parent_checkpoint_sha256": audit.PARENTS[seed],
        "parent_training_steps": 2000,
        "residual_updates": 32000,
        "trainable_parameters": [audit.A],
        "inference": audit.INFERENCE,
        "train_query_order_sha256": order,
        "record_count": 1637,
        "recipe_sha256": "b" * 64,
        "runner_sha256": "a" * 64,
        "scorer_sha256": audit.SCORER_SHA,
        "input_adapter_identity": audit.INPUT_ADAPTER,
        "input_amendment_sha256": audit.AMENDMENT_SHA,
        "plan_sha256": audit.PLAN_SHA,
        "bilinear_helper_sha256": audit.SCORER_SHA,
        "parent_state_sha256": audit.formatted_hash(before),
        "initial_state_sha256": audit.formatted_hash(initial),
        "initial_pre_update_loss": audit.INITIAL_LOSSES[seed],
        "final_post_update_loss": 0.000001,
        "standard_serving_supported": False,
    }
    environment = {k: v for k, v in implementation.items() if k != "helper_sha256"} | {
        "recipe_sha256": "b" * 64,
        "architecture": audit.ARCHITECTURE,
    }
    identity = {
        **parent_identity,
        "source_commit": "synthetic-current",
        "resolved_config_hash": audit.formatted_hash(config),
        "evaluator_version": audit.EVALUATOR,
        "environment": environment,
    }
    child = {
        "model": child_state,
        "optimizer": optimizer,
        "scheduler": None,
        "global_step": 32000,
        "config": config,
        "training_metadata": metadata,
        "experiment_identity": identity,
        "corpus_identity": {},
        "split_hash": audit.SPLIT_SHA,
    }
    train = {
        "complete": True,
        "fit_complete": True,
        "phase": "complete",
        "completed_updates": 32000,
        "architecture": audit.ARCHITECTURE,
        "objective": audit.OBJECTIVE,
        "initial_loss_replay": audit.INITIAL_LOSSES[seed],
        "initial_pre_update_loss": audit.INITIAL_LOSSES[seed],
        "final_post_update_loss": 0.000001,
        "history": [
            {
                "update": i,
                "pre_update_loss": audit.INITIAL_LOSSES[seed],
                "gradient_finite": True,
                "parameters_finite": True,
            }
            for i in range(1, 32001)
        ],
        "train_query_count": 1637,
        "validation_labels_used": False,
        "transformer_forwards_during_fit": 0,
        "trainable_parameters": [audit.A],
        "parent_state_before": before,
        "state_before": initial,
        "state_after": final,
        "state_reloaded": final,
        "residual_changed": True,
        "frozen_tensors_unchanged": True,
        "reload_exact": True,
        "optimizer_reload_exact": True,
        "optimizer": {"name": "adamw", "config": recipe["optimizer"], "groups": opt_impl["groups"]},
        "optimizer_implementation": {
            "module": "torch.optim.adamw",
            "class": "AdamW",
            "defaults": opt_impl["defaults"],
        },
        "optimizer_final": audit.optimizer_contract(optimizer, opt_impl),
        "train_query_order_sha256": order,
        "training_identity": identity,
        "checkpoint_sha256": "c" * 64,
        "checkpoint_sidecar_sha256": "d" * 64,
        "zero_replay_sha256": "e" * 64,
        "zero_replay": {},
        "refit_seconds": 1.0,
    }
    summary = {
        "queries": queries,
        "train_query_order_sha256": order,
        "validation_update_points": [0, 32000],
        "script_sha256": "a" * 64,
        "recipe_sha256": "b" * 64,
        "workspace_provenance": {"commit": "synthetic-current"},
        "environment": {},
        "child_training_identity": identity,
        "split_hash": audit.SPLIT_SHA,
        "parent_training_identity": parent_identity,
        "inherited_architecture": {"synthetic": True},
        "corpus_identity": {},
        "artifact_sha256": {
            "checkpoint-final.pt": "c" * 64,
            "checkpoint-final.pt.json": "d" * 64,
            "zero-replay.json": "e" * 64,
        },
    }
    oldtrain = {
        "parent_state_before": before,
        "initial_pre_update_loss": audit.INITIAL_LOSSES[seed],
        "final_post_update_loss": 0.000001,
        "history": copy.deepcopy(train["history"][:8000]),
    }
    membership_report = {"queries": membership_rows}
    context = {
        "parent_sidecar": {k: v for k, v in parent.items() if k != "model"}
        | {"schema_version": 1, "checkpoint_hash": audit.PARENTS[seed]},
        "parent_run": {"config": parent_config, "identity": parent_identity},
        "summary": {
            "parent_training_identity": parent_identity,
            "corpus_identity": {},
            "inherited_architecture": {"synthetic": True},
        },
        "reports": {"training.json": oldtrain, "train-membership.json": membership_report},
    }
    reports = {
        "training.json": train,
        "run.json": {"config": config, "training_metadata": metadata, "identity": identity},
        "train-membership.json": membership_report,
        "parent.json": {"responses": validation_rows},
        "zero-replay.json": {},
    }
    return summary, reports, recipe, context, parent, child


@pytest.mark.parametrize("seed", [1729, 1730, 1731])
def test_full_training_metadata_contract_all_seeds(audit, seed):
    args = synthetic_training(
        audit, seed, [membership(i) for i in range(1637)], [membership(i) for i in range(222)]
    )
    result = audit.training_contract(seed, *args)
    assert result["updates_verified"] == 32000 and result["unchanged_original_tensor_count"] == 93
    args[1]["training.json"]["history"][-1]["update"] = 31999
    with pytest.raises(ValueError, match="ordered update"):
        audit.training_contract(seed, *args)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_validation(audit, phase):
    rows = []
    for index in range(222):
        group = "COLOR" if index < 103 else "TYPE_single" if index < 154 else "TYPE_dual"
        q = membership(index)
        if group == "COLOR":
            q.update(dimension="COLOR", prompt_ids=[1, 1024, 33, 34, 5])
        logits = vector()
        rows.append(
            {
                **q,
                "group": group,
                "symmetric_relation_logits": logits,
                "selected_set_ids": [1025, 1026],
                "metrics": audit.metrics([1025, 1026], [1025, 1026]),
                "strict_separation": True,
            }
        )
    value = {
        "complete": True,
        "phase": phase,
        "query_count": 222,
        "responses": rows,
        "product_token_ids": audit.COLUMNS,
        "exact_parent_replay": phase == "parent",
        "aggregate": audit.totals(rows),
        "groups": {g: audit.totals([r for r in rows if r["group"] == g]) for g in audit.GROUPS},
    }
    if phase == "child":
        value["paired_vs_parent_dense"] = audit.BASE.paired(rows, [[1025, 1026]] * 222)
    return value


def seed_fixture(audit, tmp_path, monkeypatch):
    """Genuine stages; mock only real historical/source admission and payload I/O.

    This exercises saved-evidence wiring and mathematical/state checks, not actual
    production preflight, historical hash admission, model execution or training.
    """
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "HERE", tmp_path / f"runs/learning/{audit.CAMPAIGN}")
    membership_rows = []
    for i in range(1637):
        q = membership(i)
        if i % 2:
            q.update(dimension="COLOR", prompt_ids=[1, 1024, 33, 34, 5])
        membership_rows.append(q)
    parent_report = synthetic_validation(audit, "parent")
    child_report = synthetic_validation(audit, "child")
    summary, reports, recipe, context, parent, child = synthetic_training(
        audit, 1729, membership_rows, parent_report["responses"]
    )
    folder = audit.HERE / "seed-1729"
    folder.mkdir(parents=True)
    context.update(parent=tmp_path / "synthetic-parent.pt", summary_sha256="7" * 64)
    baseline = copy.deepcopy(child_report)
    baseline["aggregate"].update(exact_count=216, f1=audit.FLOORS[1729]["f1"])
    baseline["groups"]["TYPE_dual"]["exact_count"] = 62
    context["reports"].update({"parent.json": parent_report, "child.json": baseline})
    context["summary"]["artifact_sha256"] = dict.fromkeys(
        ("train-membership.json", "parent.json", "child.json", "training.json"), "6" * 64
    )
    reports.update({"parent.json": parent_report, "child.json": child_report})
    member_sha = save_json(folder / "train-membership.json", reports["train-membership.json"])
    monkeypatch.setattr(audit, "MEMBERSHIP_SHA", member_sha)
    train = reports["training.json"]
    # Build exact independent zero-A receipt from the synthetic parent score bytes.
    digest = hashlib.sha256(audit.score_bytes(parent_report["responses"])).hexdigest()
    batches = []
    for offset in range(0, 222, 8):
        size = min(8, 222 - offset)
        h = hashlib.sha256(
            audit.score_bytes(parent_report["responses"][offset : offset + size])
        ).hexdigest()
        batches.append(
            {
                "indices": list(range(offset, offset + size)),
                "shape": [size, 1025],
                "logits_sha256": h,
                "parent_logits_sha256": h,
                "residual_sha256": hashlib.sha256(bytes(size * 1025 * 4)).hexdigest(),
                "residual_nonzero_count": 0,
            }
        )
    zero = {
        "complete": True,
        "batches": batches,
        "query_count": 222,
        "dtype": "float32",
        "logits_shape": [222, 1025],
        "logits_sha256": digest,
        "parent_logits_sha256": digest,
        "residual_sha256": hashlib.sha256(bytes(222 * 1025 * 4)).hexdigest(),
        "residual_nonzero_count": 0,
        "batch_sizes": [8] * 27 + [6],
    }
    reports["zero-replay.json"] = zero
    train["zero_replay"] = zero
    train["zero_replay_sha256"] = save_json(folder / "zero-replay.json", zero)
    diagnostics = {
        "partition": "train",
        "descriptive_only": True,
        "used_for_selection_or_updates": False,
        "complete": True,
        "loss_callback_calls": 32001,
        "endpoints": {},
    }
    for step in (8000, 32000):
        raw = {
            "complete": True,
            "partition": "train",
            "post_update": step,
            "dtype": "float32",
            "fit_loss_callback": step + 1,
            "grad_enabled": step == 8000,
            "affects_optimization": False,
            "capture_alignment": "pre-update-8001" if step == 8000 else "final-post-update",
            "training_membership_sha256": member_sha,
            "product_token_ids": audit.COLUMNS,
            "query_count": 1637,
            "responses": [{**q, "symmetric_relation_logits": vector()} for q in membership_rows],
        }
        raw["score_bytes_sha256"] = hashlib.sha256(audit.score_bytes(raw["responses"])).hexdigest()
        name = f"train-scores-post{step}.json"
        reports[name] = raw
        h = save_json(folder / name, raw)
        diagnostics["endpoints"][str(step)] = audit.diagnostic_endpoint(
            raw, membership_rows, step, member_sha
        ) | {"score_artifact": name, "score_artifact_sha256": h}
    reports["train-diagnostics.json"] = diagnostics
    train["training_diagnostics"] = diagnostics
    train["training_prefix_diagnostic"] = {
        "descriptive_only": True,
        "used_for_selection_or_updates": False,
        "pre_update_compared_count": 8000,
        "mismatch_count": 0,
        "first_mismatch_update": None,
        "all_pre_update_equal": True,
        "historical_final_post_update_loss": 0.000001,
        "current_pre_update_8001_loss": audit.INITIAL_LOSSES[1729],
        "boundary_loss_equal": False,
    }
    paired = audit.BASE.paired(child_report["responses"], [[1025, 1026]] * 222)
    reports["comparisons.json"] = {
        "historical8000_summary_sha256": "7" * 64,
        "historical8000_child_sha256": "6" * 64,
        "comparison_scope": "saved historical output; no rerun",
        "paired_vs_historical8000": paired,
    }
    summary.update(
        complete=True,
        seed=1729,
        invariants=dict.fromkeys(audit.INVARIANTS, True),
        training_diagnostic_updates=[8000, 32000],
        shared_readiness={"path": audit.READINESS, "sha256": "9" * 64},
        child={
            "aggregate": child_report["aggregate"],
            "groups": child_report["groups"],
            "paired_vs_parent_dense": paired,
            "paired_vs_historical8000": paired,
        },
    )
    summary["gate"] = audit.seed_gate(child_report, baseline, summary["invariants"], 1729)
    partitions = {"train": membership_rows, "validation": parent_report["responses"]}
    summary["input_adapter"] = {
        "identity": audit.INPUT_ADAPTER,
        "amendment_sha256": audit.AMENDMENT_SHA,
        "source_sha256": audit.ADAPTER_SHA,
        "inherited_corpus_identity": True,
        "whole_corpus_revalidated": False,
        "evidence_sha256": context["summary"]["artifact_sha256"],
        "partition_sha256": {
            name: audit.formatted_hash(
                [
                    {
                        "input_ids": r["prompt_ids"] + r["expected_set_ids"] + [2],
                        "labels": [-100] * 5 + r["expected_set_ids"] + [2],
                        "subject": r["subject"],
                        "dimension": r["dimension"],
                    }
                    for r in rows
                ]
            )
            for name, rows in partitions.items()
        },
    }
    payload_path = folder / "checkpoint-final.pt"
    payload_path.write_bytes(b"synthetic payload I/O replaced; tensor checks remain real")
    train["checkpoint_sha256"] = audit.sha(payload_path)
    sidecar = {k: v for k, v in child.items() if k not in ("model", "optimizer", "scheduler")} | {
        "schema_version": 1,
        "checkpoint_hash": train["checkpoint_sha256"],
    }
    reports["checkpoint-final.pt.json"] = sidecar
    train["checkpoint_sidecar_sha256"] = save_json(folder / "checkpoint-final.pt.json", sidecar)
    for name, value in reports.items():
        save_json(folder / name, value)
    summary["artifact_sha256"] = {name: audit.sha(folder / name) for name in audit.ARTIFACTS}
    summary_sha = save_json(folder / "summary.json", summary)
    (folder / "primary-stdout.txt").write_text("synthetic terminal\n")
    execution = terminal()
    execution.update(
        summary_sha256=summary_sha,
        stdout_sha256=audit.sha(folder / "primary-stdout.txt"),
        prior_execution_receipt_sha256=None,
        prior_audit_receipt_sha256=None,
    )
    save_json(folder / "execution-receipt.json", execution)
    ready = {
        "files": {
            audit.AUDITOR: "a" * 64,
            audit.TEST_DIRECTORIES["auditor"] + "/test-receipt.json": "8" * 64,
        }
    }
    monkeypatch.setattr(audit, "readiness_contract", lambda reader, digest: ready)
    monkeypatch.setattr(audit, "historical_context", lambda reader, seed: context)
    monkeypatch.setattr(audit, "provenance", lambda *args: (recipe, ready))
    monkeypatch.setattr(
        audit,
        "load_payload",
        lambda reader, path, digest: parent if path == context["parent"] else child,
    )
    return folder, summary, ready


def test_complete_synthetic_seed_cli_and_immutable_output(audit, tmp_path, monkeypatch):
    import sys

    folder, summary, _ = seed_fixture(audit, tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(audit, "PRODUCTION_READY", True)
    monkeypatch.setattr(
        sys, "argv", ["independent-audit.py", "--seed", "1729", "--readiness-sha256", "9" * 64]
    )
    audit.main()
    output = folder / "independent-audit.json"
    result = json.loads(output.read_text())
    assert result["audit_passed"] is True and result["acceptance"] is False
    assert result["checkpoint"]["updates_verified"] == 32000
    assert result["training_diagnostics"]["full_score_rows_recomputed"] == 3274
    assert result["training_diagnostics"]["prefix"]["boundary_loss_equal"] is False
    assert result["gate"] == summary["gate"]
    digest = audit.sha(output)
    with pytest.raises(ValueError, match="immutable audit output"):
        audit.main()
    assert audit.sha(output) == digest


def test_cli_preserves_failed_audit_without_payload_access(audit, tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "HERE", tmp_path / f"runs/learning/{audit.CAMPAIGN}")
    folder = audit.HERE / "seed-1729"
    folder.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(audit, "PRODUCTION_READY", True)
    monkeypatch.setattr(
        sys, "argv", ["independent-audit.py", "--seed", "1729", "--readiness-sha256", "9" * 64]
    )
    with pytest.raises(FileNotFoundError):
        audit.main()
    result = json.loads((folder / "independent-audit.json").read_text())
    assert result["complete"] is False and result["audit_passed"] is False
    assert "FileNotFoundError" in result["error"]


def test_complete_synthetic_aggregate_cli_and_gate_schema(audit, tmp_path, monkeypatch):
    """Real reaggregation/receipt chain; only historical/readiness admission replaced."""
    import sys

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "HERE", tmp_path / f"runs/learning/{audit.CAMPAIGN}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(audit, "PRODUCTION_READY", True)
    prefix = f"runs/learning/{audit.CAMPAIGN}"
    root = audit.HERE
    root.mkdir(parents=True)
    ready = {
        "files": {
            audit.AUDITOR: "a" * 64,
            audit.AGGREGATOR: "b" * 64,
            audit.TEST_DIRECTORIES["auditor"] + "/test-receipt.json": "c" * 64,
        }
    }
    readiness_data = b"synthetic shared admission"
    readiness_sha = hashlib.sha256(readiness_data).hexdigest()

    def admit(reader, digest):
        assert digest == readiness_sha
        reader.inputs[str(tmp_path / audit.READINESS)] = digest
        return ready

    monkeypatch.setattr(audit, "readiness_contract", admit)
    admission = {"path": audit.READINESS, "sha256": readiness_sha}
    ready_path = tmp_path / audit.READINESS
    ready_path.parent.mkdir(parents=True)
    ready_path.write_bytes(readiness_data)
    contexts = {}
    entries = []
    pooled = []
    previous = None
    predecessor_hashes = (None, None)
    for index, seed in enumerate(audit.SEEDS):
        folder = root / f"seed-{seed}"
        parent = synthetic_validation(audit, "parent")
        child = synthetic_validation(audit, "child")
        baseline = copy.deepcopy(child)
        baseline["aggregate"].update(
            exact_count=audit.FLOORS[seed]["exact"], f1=audit.FLOORS[seed]["f1"]
        )
        for group, n in zip(audit.GROUPS, audit.FLOORS[seed]["groups"], strict=True):
            baseline["groups"][group]["exact_count"] = n
        context = {
            "summary_sha256": "6" * 64,
            "summary": {"artifact_sha256": {"child.json": "7" * 64}},
            "reports": {"parent.json": parent, "child.json": baseline},
        }
        contexts[seed] = context
        paired = child["paired_vs_parent_dense"]
        comparisons = {
            "historical8000_summary_sha256": "6" * 64,
            "historical8000_child_sha256": "7" * 64,
            "comparison_scope": "saved historical output; no rerun",
            "paired_vs_historical8000": paired,
        }
        artifacts = {
            n: save_json(folder / n, v)
            for n, v in {
                "parent.json": parent,
                "child.json": child,
                "comparisons.json": comparisons,
            }.items()
        }
        summary = {
            "complete": True,
            "seed": seed,
            "artifact_sha256": artifacts,
            "shared_readiness": admission,
            "invariants": dict.fromkeys(audit.INVARIANTS, True),
            "child": {
                "aggregate": child["aggregate"],
                "groups": child["groups"],
                "paired_vs_parent_dense": paired,
                "paired_vs_historical8000": paired,
            },
        }
        summary["gate"] = audit.seed_gate(child, baseline, summary["invariants"], seed)
        summary_sha = save_json(folder / "summary.json", summary)
        (folder / "primary-stdout.txt").write_text("synthetic execution\n")
        (folder / "audit-stdout.txt").write_text("synthetic completed audit\n")
        execution = {
            "seed": seed,
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "started_utc": f"2026-09-26T10:0{index}:00.0000001Z",
            "completed_utc": f"2026-09-26T10:0{index}:01.0000001Z",
            "summary_sha256": summary_sha,
            "stdout_sha256": audit.sha(folder / "primary-stdout.txt"),
            "prior_seed": previous,
            "prior_execution_receipt_sha256": predecessor_hashes[0],
            "prior_audit_receipt_sha256": predecessor_hashes[1],
        }
        execution_sha = save_json(folder / "execution-receipt.json", execution)
        checked = {
            "complete": True,
            "audit_passed": True,
            "acceptance": False,
            "seed": seed,
            "summary_sha256": summary_sha,
            "script_sha256": "a" * 64,
            "test_receipt_sha256": "c" * 64,
            "plan_sha256": audit.PLAN_SHA,
            "execution_receipt_sha256": execution_sha,
            "shared_readiness": admission,
            "child": summary["child"],
            "gate": summary["gate"],
        }
        checked_sha = save_json(folder / "independent-audit.json", checked)
        receipt = {
            "seed": seed,
            "exit_code": 0,
            "terminal_completion_observed_by_primary": True,
            "started_utc": f"2026-09-26T10:0{index}:02.0000001Z",
            "completed_utc": f"2026-09-26T10:0{index}:03.0000001Z",
            "cuda_visible_devices": "-1",
            "summary_sha256": summary_sha,
            "audit_sha256": checked_sha,
            "stdout_sha256": audit.sha(folder / "audit-stdout.txt"),
            "execution_receipt_sha256": execution_sha,
            "auditor_sha256": "a" * 64,
            "test_receipt_sha256": "c" * 64,
        }
        receipt_sha = save_json(folder / "audit-execution-receipt.json", receipt)
        entries.append(
            {
                "seed": seed,
                "summary_sha256": summary_sha,
                "audit_sha256": checked_sha,
                "execution_receipt_sha256": execution_sha,
                "audit_receipt_sha256": receipt_sha,
                "child": summary["child"],
                "gate": summary["gate"],
            }
        )
        pooled.extend(child["responses"])
        previous = {
            "seed": seed,
            "summary_sha256": summary_sha,
            "completed_utc": execution["completed_utc"],
        }
        predecessor_hashes = (execution_sha, receipt_sha)
    monkeypatch.setattr(audit, "historical_context", lambda reader, seed: contexts[seed])
    (root / "aggregate.script.py").write_bytes(b"synthetic frozen aggregator source")
    ready["files"][audit.AGGREGATOR] = audit.sha(root / "aggregate.script.py")
    (root / "aggregate.readiness.json").write_bytes(readiness_data)
    inputs = {audit.READINESS: readiness_sha}
    inputs_sha = save_json(root / "aggregate.inputs.json", inputs)
    snapshots = {
        prefix + "/aggregate.script.py": ready["files"][audit.AGGREGATOR],
        prefix + "/aggregate.readiness.json": readiness_sha,
        prefix + "/aggregate.inputs.json": inputs_sha,
    }
    result = {
        "complete": True,
        "acceptance": False,
        "campaign_version": audit.CAMPAIGN,
        "evaluator": audit.EVALUATOR,
        "plan_sha256": audit.PLAN_SHA,
        "script_sha256": ready["files"][audit.AGGREGATOR],
        "readiness_sha256": readiness_sha,
        "protected_test_used": False,
        "standard_serving_supported": False,
        "input_sha256": inputs,
        "snapshot_sha256": snapshots,
        "per_seed": entries,
        "all_three": {
            "aggregate": audit.totals(pooled),
            "groups": {
                g: audit.totals([r for r in pooled if r["group"] == g]) for g in audit.GROUPS
            },
        },
        "gate": {
            "checks": {"1729": True, "1730": True, "1731": True},
            "primary_checks_passed": True,
            "accepted": False,
            "independent_audit_required": True,
        },
    }
    save_json(root / "aggregate.json", result)
    monkeypatch.setattr(
        sys,
        "argv",
        ["independent-audit.py", "--aggregate-only", "--readiness-sha256", readiness_sha],
    )
    audit.main()
    checked = json.loads((root / "aggregate-audit.json").read_text())
    assert checked["audit_passed"] is True and checked["gate"] == result["gate"]
    assert checked["all_three"]["aggregate"]["exact_count"] == 666
    # A namespaced key is not the sealed primary aggregator's contract.
    result["gate"]["checks"] = {"seed_1729": True, "seed_1730": True, "seed_1731": True}
    save_json(root / "aggregate.json", result)
    with pytest.raises(ValueError, match="all-three conjunction"):
        audit.audit_aggregate(audit.EvidenceReader(tmp_path, audit.allowlist()), readiness_sha)
    # Tiny fractional overlap with preceding audit is still rejected.
    path = root / "seed-1730/execution-receipt.json"
    receipt = json.loads(path.read_text())
    receipt["started_utc"] = "2026-09-26T10:00:03.0000000Z"
    save_json(path, receipt)
    reader = audit.EvidenceReader(tmp_path, audit.allowlist())
    admit(reader, readiness_sha)
    with pytest.raises(ValueError, match="previous audit completed"):
        audit.receipt_chain(
            reader,
            1730,
            entries[1]["summary_sha256"],
            ready,
        )


@pytest.mark.parametrize(
    "change", ["valid", "skip", "exit_bool", "no_terminal", "dependencies", "script", "stdout"]
)
def test_auditor_test_receipt_contract(audit, change):
    files = dict.fromkeys(audit.READINESS_FILES, "a" * 64)
    value = {
        "passed": True,
        "gpu_used": False,
        "experiment_training_executed": False,
        "terminal_completion_observed": True,
        "exit_code": 0,
        "tests_skipped": 0,
        "tests_passed": 1,
        "test_dependencies": {p: files[p] for p in audit.TEST_DEPENDENCIES["auditor"]},
        "tested_script_sha256": files[audit.AUDITOR],
        "stdout": {"path": audit.TEST_DIRECTORIES["auditor"] + "/stdout.txt", "sha256": "a" * 64},
    }
    if change == "skip":
        value["tests_skipped"] = 1
    if change == "exit_bool":
        value["exit_code"] = False
    if change == "no_terminal":
        value["terminal_completion_observed"] = False
    if change == "dependencies":
        value["test_dependencies"].pop(audit.AUDITOR)
    if change == "script":
        value["tested_script_sha256"] = "b" * 64
    if change == "stdout":
        value["stdout"]["path"] = "other.txt"
    if change == "valid":
        audit.test_receipt_contract(value, "auditor", files)
    else:
        with pytest.raises(ValueError):
            audit.test_receipt_contract(value, "auditor", files)


def test_parent_hash_uses_compact_json_while_child_hash_uses_report_json(audit):
    args = synthetic_training(audit, 1729, [membership()], [membership()])
    _summary, _reports, _recipe, context, parent, child = args
    config = {k: v for k, v in context["parent_run"]["config"].items() if k != "run_name"}
    compact = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert compact != audit.formatted_hash(config)
    assert audit.parent_admission(1729, parent, context)
    assert parent["experiment_identity"]["resolved_config_hash"] == compact
    assert child["experiment_identity"]["resolved_config_hash"] == audit.formatted_hash(
        child["config"]
    )
    parent["experiment_identity"]["resolved_config_hash"] = audit.formatted_hash(config)
    with pytest.raises(ValueError, match="original resolved config"):
        audit.parent_admission(1729, parent, context)
