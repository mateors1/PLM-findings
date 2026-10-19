"""Synthetic checks for the aggregate-only timestamp recovery."""

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/aggregate_bilinear_replication.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recovery():
    return load(SCRIPT, "aggregate_recovery_test")


@pytest.mark.parametrize(
    "left,right",
    [
        ("2026-09-26T00:31:49.5251055+00:00", "2026-09-25T19:31:49.5251055-05:00"),
        ("2026-09-26T00:00:00Z", "2026-09-26T05:30:00+05:30"),
        ("2026-09-26T00:00:00.100+00:00", "2026-09-26T00:00:00.1+00:00"),
    ],
)
def test_equivalent_aware_offsets(recovery, left, right):
    assert recovery._same_instant(left, right)


@pytest.mark.parametrize(
    "right",
    [
        "2026-09-25T19:31:49.5251056-05:00",
        "2026-09-25T19:31:50.5251055-05:00",
        "2026-09-25T19:31:49.5251055-04:00",
    ],
)
def test_changed_instant_rejected(recovery, right):
    assert not recovery._same_instant("2026-09-26T00:31:49.5251055+00:00", right)


@pytest.mark.parametrize("invalid", ["2026-09-26T00:00:00", "2026-09-26", "not-time"])
def test_naive_or_invalid_time_rejected(recovery, invalid):
    with pytest.raises(ValueError, match="timezone-aware"):
        recovery._same_instant(invalid, "2026-09-26T00:00:00+00:00")


def test_frozen_loader_rejects_changed_code_before_import(recovery, tmp_path):
    path = tmp_path / "scripts/refit_bilinear_replication.py"
    path.parent.mkdir()
    path.write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="frozen runner identity"):
        recovery._load_frozen(tmp_path)


def test_aggregate_only_has_no_neural_execution_route():
    tree = ast.parse(SCRIPT.read_text())
    forbidden = {"_execute", "_fit", "_evaluate", "backward", "step"}
    calls = [
        n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
    ]
    assert not forbidden.intersection(calls)
    literals = {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "--seed" not in literals and "--aggregate" not in literals


def test_aggregate_copy_changes_only_aware_equality():
    old = ast.parse((ROOT / "scripts/refit_bilinear_replication.py").read_text())
    new = ast.parse(SCRIPT.read_text())
    old_node = next(
        n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "_aggregate"
    )
    new_node = next(
        n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "_aggregate"
    )
    new_node.args.args.pop()

    class Restore(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "frozen":
                return ast.Name(id=node.attr, ctx=node.ctx)
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "_same_instant":
                return ast.Compare(left=node.args[0], ops=[ast.Eq()], comparators=[node.args[1]])
            return self.generic_visit(node)

    assert ast.dump(Restore().visit(new_node), include_attributes=False) == ast.dump(
        old_node, include_attributes=False
    )


def test_failure_preserves_output_and_refuses_overwrite(recovery, tmp_path):
    frozen, _ = recovery._load_frozen(ROOT)
    mean = load(ROOT / "scripts/refit_membership_projection.py", "aggregation_mean_test")
    failure = FileNotFoundError("missing synthetic seed")

    def missing(path):
        raise failure

    helper = SimpleNamespace(
        _read=missing,
        _bind=lambda p, h, i: p,
        _sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing",
        _failure_safe=lambda v: v,
    )
    out = tmp_path / "aggregate-v2.json"
    with pytest.raises(FileNotFoundError) as caught:
        recovery._aggregate(out, {"inputs": {}}, mean, helper, {"complete": False}, frozen)
    assert caught.value is failure
    before = out.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        frozen._refuse(out, mean, True)
    assert out.read_bytes() == before
