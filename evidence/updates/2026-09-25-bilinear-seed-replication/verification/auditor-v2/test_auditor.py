"""Synthetic evidence tests only; no real training or primary outcomes."""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "bilinear-seed-replication-v1/independent-audit-v2.py"
SPEC = importlib.util.spec_from_file_location("seed_audit", PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def history(loss=0.02):
    return [
        {"update": n, "pre_update_loss": loss, "gradient_finite": True, "parameters_finite": True}
        for n in range(1, 2001)
    ]


def replay(loss=0.02):
    return {
        "parent_loss": loss,
        "zero_loss": loss,
        "shape": [1637, 1025],
        "parent_logits_sha256": "a" * 64,
        "zero_logits_sha256": "a" * 64,
        "logits_exact": True,
        "loss_exact": True,
    }


@pytest.mark.parametrize("loss", [0.02, 0.1, 0.0001])
def test_dynamic_parent_endpoint_not_1729_constant(loss):
    A.training_replay_contract(replay(loss), history(loss), loss, 0.000001)


@pytest.mark.parametrize(
    "mutation",
    [
        "parent",
        "zero",
        "first",
        "shape",
        "hash",
        "nonhex",
        "lossflag",
        "logitflag",
        "nan",
        "negative",
        "count",
        "order",
        "gradient",
        "parameter",
    ],
)
def test_training_replay_corruption(mutation):
    r, h = replay(), history()
    if mutation == "parent":
        r["parent_loss"] = 0.03
    elif mutation == "zero":
        r["zero_loss"] = 0.03
    elif mutation == "first":
        h[0]["pre_update_loss"] = 0.03
    elif mutation == "shape":
        r["shape"] = [222, 1025]
    elif mutation == "hash":
        r["zero_logits_sha256"] = "b" * 64
    elif mutation == "nonhex":
        r["parent_logits_sha256"] = r["zero_logits_sha256"] = "z" * 64
    elif mutation == "lossflag":
        r["loss_exact"] = False
    elif mutation == "logitflag":
        r["logits_exact"] = False
    elif mutation == "nan":
        h[1555]["pre_update_loss"] = float("nan")
    elif mutation == "negative":
        h[1555]["pre_update_loss"] = -1
    elif mutation == "count":
        h.pop()
    elif mutation == "order":
        h[1555]["update"] = 1555
    elif mutation == "gradient":
        h[1555]["gradient_finite"] = False
    else:
        h[1555]["parameters_finite"] = False
    with pytest.raises(ValueError):
        A.training_replay_contract(r, h, 0.02, 0.00001)


def child(seed):
    p = A.PINS[seed]
    return {
        "aggregate": {
            "exact_count": p["exact"] + 1,
            "f1": p["f1"],
            "serialization_compatible": 222,
        },
        "groups": {g: {"exact_count": n} for g, n in p["groups"].items()},
    }


def baseline(seed):
    p = A.PINS[seed]
    return {"exact_count": p["exact"], "f1": p["f1"], "groups": p["groups"]}


@pytest.mark.parametrize("seed", [1730, 1731])
def test_matching_seed_floor(seed):
    assert A.seed_gate(child(seed), baseline(seed), {"verified": True})["primary_checks_passed"]
    c = child(seed)
    c["aggregate"]["exact_count"] -= 1
    assert not A.seed_gate(c, baseline(seed), {"verified": True})["primary_checks_passed"]


@pytest.mark.parametrize(
    "mutation", ["f1", "COLOR", "TYPE_single", "TYPE_dual", "serialization", "execution"]
)
def test_seed_nonregression_and_validity_cannot_be_pooled_away(mutation):
    c = child(1730)
    c["aggregate"]["exact_count"] = 222
    invariants = {"verified": True}
    if mutation == "f1":
        c["aggregate"]["f1"] -= 0.0001
    elif mutation == "serialization":
        c["aggregate"]["serialization_compatible"] = 221
    elif mutation == "execution":
        invariants["verified"] = False
    else:
        c["groups"][mutation]["exact_count"] -= 1
    assert not A.seed_gate(c, baseline(1730), invariants)["primary_checks_passed"]


def test_1730_does_not_use_1731_easier_floor():
    c = child(1731)
    assert not A.seed_gate(c, baseline(1730), {"verified": True})["primary_checks_passed"]


def test_fresh_conjunction_never_uses_historical_rescue():
    assert A.fresh_conjunction({1730: True, 1731: True})
    assert not A.fresh_conjunction({1730: True, 1731: False})
    for values in (
        {1729: True, 1730: True},
        {1729: True, 1730: True, 1731: False},
        {1730: 1, 1731: True},
    ):
        with pytest.raises(ValueError):
            A.fresh_conjunction(values)


@pytest.mark.parametrize("seed", [1729, 1732, True, "1730"])
def test_only_two_fresh_seeds(seed):
    with pytest.raises(ValueError):
        A.seed_contract(seed)


def report():
    return {
        "query_count": 222,
        "product_token_ids": list(range(1024, 2049)),
        "responses": [
            {
                "index": n,
                "subject": f"p{n}",
                "dimension": "TYPE",
                "group": "TYPE_single",
                "prompt_ids": [1, 1024 + n, 32, 34, 5],
                "expected_set_ids": [2048],
            }
            for n in range(222)
        ],
    }


@pytest.mark.parametrize(
    "mutation", ["subject", "group", "prompt_ids", "expected_set_ids", "index", "columns", "count"]
)
def test_shared_query_alignment(mutation):
    left, right = report(), report()
    A.aligned_reports([left, right])
    if mutation == "columns":
        right["product_token_ids"].reverse()
    elif mutation == "count":
        right["responses"].pop()
    else:
        right["responses"][0][mutation] = "changed"
    with pytest.raises(ValueError):
        A.aligned_reports([left, right])


def test_parent_config_model_seed_distinct_from_split_seed():
    old = {
        "seed": 1729,
        "run_name": "national_dex_continuation_control_s1729_v1",
        "data": {"split_seed": 1729},
        "model": {"dim": 256},
    }
    new = copy.deepcopy(old)
    new.update(seed=1730, run_name="national_dex_continuation_control_s1730_v1")
    A.parent_config_contract(1730, new, old)
    new["data"]["split_seed"] = 1730
    with pytest.raises(ValueError):
        A.parent_config_contract(1730, new, old)


def test_authenticated_helper_globals_remain_untouched():
    assert A.BUDGET.INPUTS == A.BUDGET.OLD.INPUTS == A.BASE.INPUTS == {}


def lineage(seed=1730):
    return (
        {
            "architecture": A.ARCHITECTURE,
            "seed": seed,
            "data_split_seed": 1729,
            "campaign_version": A.CAMPAIGN,
            "parent_checkpoint_sha256": A.PINS[seed]["parent"],
            "parent_training_steps": 2000,
            "residual_updates": 2000,
            "objective": A.OBJECTIVE,
            "evaluator": A.EVALUATOR,
            "record_count": 1637,
            "trainable_parameters": [A.A],
        },
        {"seeds": [seed], "evaluator_version": A.EVALUATOR},
    )


@pytest.mark.parametrize("seed", [1730, 1731])
def test_fresh_matching_seed_lineage(seed):
    A.lineage_contract(seed, *lineage(seed))


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_checkpoint_sha256", A.PINS[1731]["parent"]),
        ("seed", 1729),
        ("data_split_seed", 1730),
        ("campaign_version", "bilinear-budget2000-v1"),
        ("evaluator", A.BUDGET.EVALUATOR),
        ("residual_updates", 500),
        ("parent_training_steps", 4000),
        ("trainable_parameters", ["symmetric_relation_projection.weight"]),
    ],
)
def test_wrong_parent_seed_evaluator_or_trainable_rejected(field, value):
    metadata, identity = lineage()
    metadata[field] = value
    with pytest.raises(ValueError):
        A.lineage_contract(1730, metadata, identity)


def terminal(seed=1730):
    return {
        "seed": seed,
        "execution_order": seed - 1729,
        "exit_code": 0,
        "terminal_completion_observed_by_primary": True,
        "summary_sha256": "a" * 64,
        "stdout_sha256": "b" * 64,
        "previous_terminal_receipt_sha256": None if seed == 1730 else "c" * 64,
        "started_at_utc": "2026-09-25T14:00:00+00:00",
        "finished_at_utc": "2026-09-25T14:01:00+00:00",
    }


@pytest.mark.parametrize("seed", [1730, 1731])
def test_terminal_receipt(seed):
    first, last = A.terminal_contract(
        terminal(seed), seed, "a" * 64, "b" * 64, None if seed == 1730 else "c" * 64
    )
    assert last > first


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", 1),
        ("exit_code", False),
        ("terminal_completion_observed_by_primary", False),
        ("summary_sha256", "d" * 64),
        ("stdout_sha256", "d" * 64),
        ("execution_order", 2),
        ("previous_terminal_receipt_sha256", "c" * 64),
        ("started_at_utc", "2026-09-25T14:00:00"),
        ("finished_at_utc", "2026-09-25T13:00:00+00:00"),
    ],
)
def test_terminal_corruption(field, value):
    receipt = terminal()
    receipt[field] = value
    with pytest.raises(ValueError):
        A.terminal_contract(receipt, 1730, "a" * 64, "b" * 64)


def metric_report(correct):
    value = report()
    for n, row in enumerate(value["responses"]):
        row["group"] = A.GROUPS[n % 3]
        row["metrics"] = A.BASE.metrics([2048] if n < correct else [], [2048])
        row["strict_separation"] = n < correct
    return value


def test_fresh_and_all_pool_arithmetic_preserves_historical_role():
    fresh = A.pooled_rows([metric_report(150), metric_report(200)])
    all_three = A.pooled_rows([metric_report(207), metric_report(150), metric_report(200)])
    assert fresh["aggregate"]["query_count"] == 444
    assert fresh["aggregate"]["exact_count"] == 350
    assert fresh["aggregate"]["f1"] == 350 / 444
    assert all_three["aggregate"]["query_count"] == 666
    assert all_three["aggregate"]["exact_count"] == 557
    assert all_three["aggregate"]["f1"] == 557 / 666
    assert sum(v["exact_count"] for v in fresh["groups"].values()) == 350
    assert not A.campaign_gate({1730: False, 1731: True})["primary_checks_passed"]


def test_output_overwrite_refused(tmp_path, monkeypatch):
    path = tmp_path / "independent-audit.json"
    path.write_text("retain")
    monkeypatch.setattr(
        "sys.argv", ["audit", "--summary", str(tmp_path / "summary.json"), "--aggregate"]
    )
    monkeypatch.setattr(A, "audit_aggregate", lambda _: pytest.fail("must not audit"))
    with pytest.raises(ValueError, match="immutable"):
        A.main()
    assert path.read_text() == "retain"


def test_dynamic_inputs_extend_exact_preflight_inventory():
    initial = {"source.py": "a" * 64}
    A.aggregate_inventory_contract(initial, initial, initial | {"receipt.json": "b" * 64})


@pytest.mark.parametrize("mutation", ["declared", "removed", "changed"])
def test_preflight_dynamic_identity_corruption(mutation):
    initial = {"source.py": "a" * 64}
    declared, dynamic = dict(initial), initial | {"receipt.json": "b" * 64}
    if mutation == "declared":
        declared["source.py"] = "c" * 64
    elif mutation == "removed":
        dynamic.pop("source.py")
    else:
        dynamic["source.py"] = "c" * 64
    with pytest.raises(ValueError):
        A.aggregate_inventory_contract(declared, initial, dynamic)


def original_schema_fixture():
    identity = {"seeds": [1730], "resolved_config_hash": "a" * 64}
    config = {"model": {"dim": 256}, "train": {"batch_size": 32, "grad_accum_steps": 1}}
    run = {"config": config, "identity": identity}
    sidecar = {
        "config": config["train"],
        "experiment_identity": identity,
        "training_metadata": {
            "model_config": config["model"],
            "objective": "causal-next-token-v1",
            "record_count": 1637,
            "data_order": "sequential-v1",
            "batch_size": 32,
            "grad_accum_steps": 1,
            "train_loss": 0.02,
        },
    }
    result = {"identity": identity, "train_loss": 0.02}
    return copy.deepcopy((run, sidecar, result))


def test_original_run_has_no_metadata_and_sidecar_has_train_only_config():
    run, sidecar, result = original_schema_fixture()
    assert "training_metadata" not in run
    assert sidecar["config"] != run["config"]
    A.original_parent_schema(run, sidecar, result)


@pytest.mark.parametrize(
    "mutation",
    [
        "run_extra",
        "sidecar_config",
        "model",
        "objective",
        "records",
        "batch",
        "order",
        "identity",
        "loss",
    ],
)
def test_original_receipt_schema_corruption(mutation):
    run, sidecar, result = original_schema_fixture()
    if mutation == "run_extra":
        run["training_metadata"] = {}
    elif mutation == "sidecar_config":
        sidecar["config"] = run["config"]
    elif mutation == "model":
        sidecar["training_metadata"]["model_config"] = {"dim": 128}
    elif mutation == "objective":
        sidecar["training_metadata"]["objective"] = A.OBJECTIVE
    elif mutation == "records":
        sidecar["training_metadata"]["record_count"] = 222
    elif mutation == "batch":
        sidecar["training_metadata"]["batch_size"] = 8
    elif mutation == "order":
        sidecar["training_metadata"]["data_order"] = "random"
    elif mutation == "identity":
        result["identity"] = {"seeds": [1731]}
    else:
        result["train_loss"] = 0.04
    with pytest.raises(ValueError):
        A.original_parent_schema(run, sidecar, result)


def test_equivalent_aware_start_instant_preserves_original_spellings():
    utc = "2026-09-25T14:00:01.1234567+00:00"
    local = "2026-09-25T09:00:01.1234567-05:00"
    assert utc != local
    A.same_instant(utc, local)


@pytest.mark.parametrize("bad", ["2026-09-25T14:00:01.123456", "2026-09-25T09:00:02.123456-05:00"])
def test_naive_or_different_start_instant_rejected(bad):
    with pytest.raises(ValueError):
        A.same_instant("2026-09-25T14:00:01.123456+00:00", bad)


def test_seventh_fractional_digit_is_not_silently_truncated():
    with pytest.raises(ValueError):
        A.same_instant("2026-09-25T14:00:01.1234567+00:00", "2026-09-25T09:00:01.1234568-05:00")
    receipt = terminal()
    receipt["started_at_utc"] = "2026-09-25T14:00:01.1234568+00:00"
    receipt["finished_at_utc"] = "2026-09-25T14:00:01.1234567+00:00"
    with pytest.raises(ValueError):
        A.terminal_contract(receipt, 1730, "a" * 64, "b" * 64)


def test_v2_retains_frozen_scoring_training_and_gate_arithmetic():
    old = ast.parse((A.HERE / "independent-audit.py").read_text())
    new = ast.parse(PATH.read_text())
    names = [
        "seed_gate",
        "campaign_gate",
        "loss_contract",
        "training_replay_contract",
        "pooled_rows",
        "aligned_reports",
        "training_audit",
    ]

    def functions(tree):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
        }

    before, after = functions(old), functions(new)
    for name in names:
        assert before[name] == after[name]
