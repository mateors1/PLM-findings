"""Versioned evidence repair for the three-seed 32000-update screen.

Production admission requires the exact shared readiness manifest and its
externally supplied SHA256. No neural forward or optimizer execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CAMPAIGN = "bilinear-budget32000-v1"
EVALUATOR = "plm-bilinear-budget32000-screen-v1"
SEEDS = (1729, 1730, 1731)
UPDATES = 32000
PLAN_SHA = "5f1e0243b373ad1287c31dcef8a3c1cfa310cac9701795469a68677506e6f444"
PRODUCTION_READY = True
BASE_SHA = "78d81a14d666549dd1f6419a7d78792073141a59a729135bc32ac84c88c6ce4a"
A = "symmetric_bilinear_residual"
A_SHAPE = [2, 256, 256]
COLUMNS = list(range(1024, 2049))
TRAIN_POINTS = (8000, 32000)
GROUPS = ("COLOR", "TYPE_single", "TYPE_dual")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(actual, expected, message):
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        message,
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def formatted_hash(value):
    return hashlib.sha256(
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    ).hexdigest()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite numeric scalar")


def integer(value, expected, label):
    require(type(value) is int and value == expected, label)


def seed_contract(seed):
    require(type(seed) is int and seed in SEEDS, "declared seed")


def independent_base():
    path = ROOT / "runs/learning/projection-only-refit-v1/independent-audit.py"
    require(sha(path) == BASE_SHA, "frozen independent helper identity")
    spec = importlib.util.spec_from_file_location("budget32000_independent_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.INPUTS == {}, "independent helper has no data access")
    return module


BASE = independent_base()


class EvidenceReader:
    """Admit every exact path before any open, including manifest prechecks."""

    def __init__(self, root, allowed):
        self.root = Path(root).resolve()
        self.allowed = {str((self.root / p).resolve()) for p in allowed}
        self.inputs = {}

    def admit(self, path):
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        require(path.is_relative_to(self.root), "dependency outside repository")
        require(str(path) in self.allowed, "dependency not explicitly allowed")
        relative = path.relative_to(self.root)
        require(
            relative.parts[0] != "data"
            or relative.as_posix() == "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json",
            "corpus and graph inputs forbidden",
        )
        require(
            path.suffix not in (".jsonl", ".sqlite", ".sqlite3", ".db"), "protected payload type"
        )
        return path

    def bind(self, path, digest):
        path = self.admit(path)
        require(isinstance(digest, str) and len(digest) == 64, "SHA256 identity required")
        require(all(c in "0123456789abcdef" for c in digest), "lowercase SHA256 identity")
        require(sha(path) == digest, "evidence hash mismatch")
        require(str(path) not in self.inputs or self.inputs[str(path)] == digest, "identity drift")
        self.inputs[str(path)] = digest
        return path

    def read(self, path, digest=None):
        path = self.admit(path)
        require(path.suffix == ".json", "JSON evidence only")
        if digest is None:
            digest = sha(path)
        return json.loads(self.bind(path, digest).read_text(encoding="utf-8"))

    def manifest(self, mapping):
        require(isinstance(mapping, dict) and mapping, "nonempty dependency manifest")
        for path in mapping:
            self.admit(path)
        for path, digest in mapping.items():
            self.bind(path, digest)


def loss_contract(history, parent_loss, zero_loss, initial, final):
    require(isinstance(history, list) and len(history) == UPDATES, "exactly 32000 updates")
    for number, row in enumerate(history, 1):
        integer(row["update"], number, "ordered update history")
        finite(row["pre_update_loss"])
        require(row["pre_update_loss"] >= 0, "nonnegative training loss")
        require(
            row["gradient_finite"] is True and row["parameters_finite"] is True,
            "finite update attestations",
        )
    for value in (parent_loss, zero_loss, initial, final):
        finite(value)
        require(value >= 0, "nonnegative loss endpoint")
    require(
        parent_loss == zero_loss == initial == history[0]["pre_update_loss"],
        "matching parent zero-A initial loss",
    )


def metrics(predicted, truth):
    return BASE.metrics(predicted, truth)


def totals(rows):
    return BASE.totals(rows)


def score_row(values, membership, *, group=None):
    """Recompute selection, quality and extrema independently from saved logits."""
    BASE.vector(values)
    prompt = membership["prompt_ids"]
    require(
        len(prompt) == 5
        and all(type(i) is int for i in prompt)
        and prompt[0] == 1
        and prompt[2:5] == [32 if membership["dimension"] == "TYPE" else 33, 34, 5],
        "membership prompt grammar",
    )
    require(membership["dimension"] in ("TYPE", "COLOR"), "membership dimension")
    subject = prompt[1]
    require(subject in COLUMNS, "membership subject ID")
    truth = membership["expected_set_ids"]
    require(
        isinstance(truth, list)
        and truth
        and truth == sorted(set(truth))
        and all(type(i) is int and i in COLUMNS for i in truth)
        and subject not in truth,
        "canonical nonself membership truth",
    )
    truth_set = set(truth)
    negative_ids = [i for i in COLUMNS if i != subject and i not in truth_set]
    require(negative_ids, "nonempty negative candidates")
    low = min(values[i - 1024] for i in truth)
    high = max(values[i - 1024] for i in negative_ids)
    low_id = min(i for i in truth if values[i - 1024] == low)
    high_id = min(i for i in negative_ids if values[i - 1024] == high)
    prediction = [i for i in COLUMNS if i != subject and values[i - 1024] > 0]
    result = {
        "index": membership["index"],
        "subject": membership["subject"],
        "dimension": membership["dimension"],
        "prompt_ids": prompt,
        "group": membership["dimension"] if group is None else group,
        "expected_set_ids": truth,
        "selected_set_ids": prediction,
        "metrics": metrics(prediction, truth),
        "strict_separation": low > high,
        "min_positive": low,
        "min_positive_id": low_id,
        "max_negative": high,
        "max_negative_id": high_id,
        "zero_margin": min(low, -high),
        "separation_gap": low - high,
    }
    require(result["metrics"]["exact"] == (low > 0 and high <= 0), "threshold/extrema agreement")
    return result


def diagnostic_endpoint(report, membership, post_update, membership_sha):
    integer(post_update, 8000 if post_update == 8000 else 32000, "fixed diagnostic point")
    require(post_update in TRAIN_POINTS, "fixed train endpoints only")
    require(report["complete"] is True and report["partition"] == "train", "train-only endpoint")
    integer(report["post_update"], post_update, "training diagnostic position")
    require(report["dtype"] == "float32", "diagnostic FP32 dtype")
    require(
        report["capture_alignment"]
        == ("pre-update-8001" if post_update == 8000 else "final-post-update"),
        "diagnostic alignment",
    )
    integer(report["fit_loss_callback"], post_update + 1, "fit-only diagnostic callback")
    require(
        report["grad_enabled"] is (post_update == 8000) and report["affects_optimization"] is False,
        "observational gradient context",
    )
    require(report["training_membership_sha256"] == membership_sha, "diagnostic training labels")
    exact(report["product_token_ids"], COLUMNS, "training diagnostic column order")
    require(
        report["query_count"] == len(report["responses"]) == len(membership) == 1637,
        "complete training diagnostic coverage",
    )
    rows = []
    for index, (row, labels) in enumerate(zip(report["responses"], membership, strict=True)):
        integer(labels["index"], index, "training membership order")
        for key in ("index", "subject", "dimension", "prompt_ids", "expected_set_ids"):
            exact(row[key], labels[key], "training diagnostic query alignment")
        rows.append(score_row(row["symmetric_relation_logits"], labels))
    require(
        report["score_bytes_sha256"]
        == hashlib.sha256(score_bytes(report["responses"])).hexdigest(),
        "diagnostic FP32 byte identity",
    )
    minimum = min(r["zero_margin"] for r in rows)
    return {
        "post_update": post_update,
        "query_count": len(rows),
        "minimum_signed_margin": minimum,
        "minimum_signed_margin_index": min(r["index"] for r in rows if r["zero_margin"] == minimum),
        "aggregate": totals(rows),
        "groups": {d: totals([r for r in rows if r["dimension"] == d]) for d in ("COLOR", "TYPE")},
        "rows": rows,
    }


FLOORS = {
    1729: {"exact": 216, "f1": 0.9998843626799785, "groups": (103, 51, 62)},
    1730: {"exact": 220, "f1": 0.9999717006519728, "groups": (103, 51, 66)},
    1731: {"exact": 214, "f1": 0.9998584743646364, "groups": (103, 51, 60)},
}
INVARIANTS = {
    "parent_replay_exact",
    "zero_A_replay_exact",
    "initial_train_loss_exact",
    "completed_32000_updates",
    "original_93_tensors_unchanged",
    "residual_changed",
    "reload_exact",
    "optimizer_reload_exact",
    "train_only_supervision",
}


def prefix_contract(history, historical, initial, historical_initial, historical_final):
    """Descriptive exact scalar replay; mismatch is data, never a rejection gate."""
    require(len(history) == 32000 and len(historical) == 8000, "prefix trace coverage")
    differences = []
    for i, (current, old) in enumerate(zip(history[:8000], historical, strict=True), 1):
        integer(current["update"], i, "new prefix order")
        integer(old["update"], i, "historical prefix order")
        finite(current["pre_update_loss"])
        finite(old["pre_update_loss"])
        if current["pre_update_loss"] != old["pre_update_loss"]:
            differences.append(
                {
                    "update": i,
                    "current": current["pre_update_loss"],
                    "historical": old["pre_update_loss"],
                }
            )
    boundary = history[8000]["pre_update_loss"]
    finite(boundary)
    finite(historical_final)
    return {
        "descriptive_only": True,
        "compared_pre_update_losses": 8000,
        "mismatch_count": len(differences),
        "first_mismatch": differences[0] if differences else None,
        "initial_loss_equal": initial == historical_initial,
        "boundary_loss_equal": boundary == historical_final,
        "current_pre_update_8001_loss": boundary,
        "historical_final_post_update_loss": historical_final,
        "intermediate_weight_equality_established": False,
    }


def diagnostic_points_contract(diagnostics):
    require(
        isinstance(diagnostics, list) and len(diagnostics) == 2, "exactly two diagnostic captures"
    )
    for entry, point, callback in zip(diagnostics, TRAIN_POINTS, (8001, 32001), strict=True):
        integer(entry["post_update"], point, "diagnostic update point")
        integer(entry["fit_loss_callback"], callback, "fit-only diagnostic callback")
        require(entry["partition"] == "train", "train-only diagnostic labels")
        require(entry["grad_enabled"] is (point == 8000), "diagnostic gradient context")
        require(entry["affects_optimization"] is False, "observational diagnostic only")


def state_manifest(state):
    return {
        n: {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": BASE.tensor_digest(t)}
        for n, t in state.items()
    }


def compare_states(parent, child):
    import torch

    require(len(parent) == 93 and A not in parent, "93 original state tensors")
    require(set(child) == set(parent) | {A}, "94 final state tensors")
    for name, before in parent.items():
        after = child[name]
        require(
            isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor), "tensor state"
        )
        require(before.device.type == after.device.type == "cpu", "CPU payload inspection")
        require(
            before.dtype == after.dtype and before.shape == after.shape, "original tensor schema"
        )
        require(not before.is_complex(), "real original state")
        require(
            not before.is_floating_point() or before.dtype == torch.float32, "FP32 original state"
        )
        require(
            torch.isfinite(before).all().item() and torch.isfinite(after).all().item(),
            "finite original state",
        )
        require(BASE.tensor_digest(before) == BASE.tensor_digest(after), "frozen original bytes")
    residual = child[A]
    require(
        isinstance(residual, torch.Tensor)
        and residual.device.type == "cpu"
        and residual.dtype == torch.float32
        and list(residual.shape) == A_SHAPE,
        "FP32 residual shape",
    )
    require(torch.isfinite(residual).all().item(), "finite residual")
    require(bool(torch.count_nonzero(residual).item()), "residual changed")
    original = state_manifest(parent)
    initial = dict(original)
    initial[A] = {
        "shape": A_SHAPE,
        "dtype": "torch.float32",
        "sha256": BASE.tensor_digest(torch.zeros(A_SHAPE, dtype=torch.float32)),
    }
    return original, initial, state_manifest(child)


def optimizer_contract(optimizer, implementation):
    import torch

    require(set(optimizer) == {"state", "param_groups"}, "optimizer payload schema")
    groups = optimizer["param_groups"]
    require(
        len(groups) == 2 and len(groups[0]["params"]) == 1 and groups[1]["params"] == [],
        "sole residual optimizer parameter",
    )
    identifier = groups[0]["params"][0]
    require(
        type(identifier) is int and set(optimizer["state"]) == {identifier},
        "sole integer optimizer state",
    )
    named = []
    for group in groups:
        require(
            group["lr"] == 0.0003
            and list(group["betas"]) == [0.9, 0.999]
            and group["eps"] == 1e-8
            and group["weight_decay"] == 0,
            "fixed AdamW values",
        )
        require(
            group.get("fused") in (None, False)
            and group.get("amsgrad") is False
            and group.get("maximize", False) is False
            and group.get("capturable", False) is False
            and group.get("differentiable", False) is False
            and "foreach" in group
            and group["foreach"] is None,
            "ordinary historical AdamW",
        )
        named.append(
            {k: v for k, v in group.items() if k != "params"}
            | {"parameter_names": [A] if group["params"] else []}
        )
    exact(implementation["groups"], named, "optimizer parameter names/options")
    require(implementation["class"] == "torch.optim.adamw.AdamW", "optimizer class")
    defaults = implementation["defaults"]
    require(
        defaults["lr"] == 0.0003
        and list(defaults["betas"]) == [0.9, 0.999]
        and defaults["eps"] == 1e-8
        and defaults["weight_decay"] == 0.01
        and defaults.get("fused") in (None, False)
        and defaults.get("amsgrad") is False
        and "foreach" in defaults
        and defaults["foreach"] is None,
        "historical AdamW defaults",
    )
    state = optimizer["state"][identifier]
    require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "AdamW state fields")
    step = state["step"]
    require(
        isinstance(step, torch.Tensor)
        and step.device.type == "cpu"
        and step.dtype == torch.float32
        and step.numel() == 1
        and step.item() == UPDATES,
        "optimizer step 32000",
    )
    for name in ("exp_avg", "exp_avg_sq"):
        t = state[name]
        require(
            isinstance(t, torch.Tensor)
            and t.device.type == "cpu"
            and t.dtype == torch.float32
            and list(t.shape) == A_SHAPE
            and torch.isfinite(t).all().item(),
            "finite FP32 AdamW moment",
        )
    require((state["exp_avg_sq"] >= 0).all().item(), "nonnegative second moment")
    return {"parameter_id_to_name": {str(identifier): A}, "steps": {A: float(UPDATES)}}


def seed_gate(child, baseline, invariants, seed):
    seed_contract(seed)
    floor = FLOORS[seed]
    require(
        baseline["aggregate"]["exact_count"] == floor["exact"]
        and baseline["aggregate"]["f1"] == floor["f1"],
        "fixed seed comparator",
    )
    require(
        tuple(baseline["groups"][g]["exact_count"] for g in GROUPS) == floor["groups"],
        "fixed seed group floors",
    )
    require(set(child["groups"]) == set(GROUPS), "complete validation groups")
    checks = {
        "execution_invariants": set(invariants) == INVARIANTS
        and all(v is True for v in invariants.values()),
        "serialization_compatible": child["aggregate"]["serialization_compatible"] == 222,
        "exact_above_matching8000": child["aggregate"]["exact_count"]
        > baseline["aggregate"]["exact_count"],
        "f1_at_least_matching8000": child["aggregate"]["f1"] >= baseline["aggregate"]["f1"],
        "group_exact_nonregression": all(
            child["groups"][g]["exact_count"] >= baseline["groups"][g]["exact_count"]
            for g in GROUPS
        ),
    }
    return {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "independent_audit_required": True,
        "accepted": False,
    }


def campaign_gate(gates):
    require(set(gates) == set(SEEDS), "all three seed gates required")
    require(all(type(s) is int for s in gates), "integer seed keys")
    result = all(gates[s]["primary_checks_passed"] is True for s in SEEDS)
    return {
        "all_three_primary_gates_passed": result,
        "independent_audit_required": True,
        "accepted": False,
    }


def instant(value):
    """Timezone-normalized instant retaining every declared decimal digit."""
    require(isinstance(value, str), "timestamp string")
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})", value
    )
    require(match is not None, "timezone-aware ISO timestamp")
    stamp, digits, offset = match.groups()
    try:
        whole = datetime.fromisoformat(stamp + ("+00:00" if offset == "Z" else offset))
    except ValueError as exc:
        raise ValueError("valid ISO timestamp") from exc
    delta = whole.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    seconds = delta.days * 86400 + delta.seconds
    return Fraction(seconds) + (Fraction(int(digits), 10 ** len(digits)) if digits else 0)


def execution_contract(receipt, seed, summary_sha, stdout_sha, previous=None):
    seed_contract(seed)
    integer(receipt["seed"], seed, "terminal seed")
    integer(receipt["exit_code"], 0, "observed zero terminal exit")
    require(receipt["terminal_completion_observed_by_primary"] is True, "terminal completion")
    require(
        receipt["summary_sha256"] == summary_sha and receipt["stdout_sha256"] == stdout_sha,
        "terminal evidence hashes",
    )
    start, finish = instant(receipt["started_utc"]), instant(receipt["completed_utc"])
    require(start <= finish, "terminal time order")
    if seed == SEEDS[0]:
        require(previous is None and receipt["prior_seed"] is None, "first seed has no predecessor")
    else:
        require(previous is not None, "prior seed receipt required")
        predecessor = SEEDS[SEEDS.index(seed) - 1]
        integer(previous["seed"], predecessor, "immediate predecessor seed")
        expected = {
            "seed": predecessor,
            "summary_sha256": previous["summary_sha256"],
            "completed_utc": previous["completed_utc"],
        }
        prior = receipt["prior_seed"]
        require(isinstance(prior, dict) and set(prior) == set(expected), "exact predecessor fields")
        exact(
            {k: prior[k] for k in ("seed", "summary_sha256")},
            {k: expected[k] for k in ("seed", "summary_sha256")},
            "predecessor receipt binding",
        )
        require(
            instant(prior["completed_utc"]) == instant(expected["completed_utc"]),
            "predecessor completion instant",
        )
        require(
            start >= instant(previous["completed_utc"]), "sequential nonoverlapping seed execution"
        )


ARCHITECTURE = "plm-frozen-symmetric-bilinear-residual-v1"
OBJECTIVE = "plm-bilinear-residual-balanced-bce-v1"
INFERENCE = "dense-positive-bilinear-residual-subject-excluded-ascending-fp32-b8-last6-v1"
AMENDMENT_SHA = "bb139d6903104794770f4cc0006f72002716e907a0e1ff3167d425a56caf810f"
ADAPTER_SHA = "02b5364bc7c440269c3a7bf123c37fe188b74353bcd1549a08a18ff95f774a6e"
SCORER_SHA = "5ec58b162648e760997037e5ac969a517d9511be177adf44b1fdb1e7803d33ee"
MEAN_SHA = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
HELPER_SHA = "2ab48674b526ec6641a6c2366b669ac76bdeac1e8633331e01c772d4393edd39"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
MEMBERSHIP_SHA = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
SPLIT_SHA = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
INPUT_ADAPTER = "plm-authenticated-partition-membership-v1"
PARENTS = {
    1729: "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1",
    1730: "5c18bd06ed49302b5cff03422f831caa64cae757bdf11524f4d6892769da28b2",
    1731: "ca2a9ac885c5cc872365f7d505d25a3737a84e38d8ecdb5b5b0009f977dd365d",
}
INITIAL_LOSSES = {
    1729: 0.003822767175734043,
    1730: 0.003967047668993473,
    1731: 0.004869009833782911,
}
CONTROL1729 = {
    "summary.json": "e4067b516e055cba55c0ba651c9d1171dc0ee406b80b4f969fb8ada5934f24e8",
    "independent-audit.json": "bdf142f48229fbcdb197bce3a019e38c3a65570ae3f8551a9c9a1f14b2198872",
    "decision.json": "2cadf9dce7ee36cc4095a10a746cc1608d5ce4437724872c1eec1c88e915c195",
}
CONTROL_REPLICATION = {
    "aggregate-v2.json": "77cc2ee2ebf4dc1f8671e71833712302160190c85477eb6e0dd0b5494bb3cde6",
    "independent-audit-v2.json": "49fc5d7edc7763870ac0e9310295385d8e6a3af6e67d3cc6b5a9f0e44bc2ae77",
    "decision.json": "01a9faefb90ee3d3e7525c0d5191d41741131843dc79ea2b44d860c5c8e21d5e",
}
AUDITOR_TEST_DEPENDENCIES = (
    "runs/learning/bilinear-budget32000-v1/independent-audit.py",
    "tests/unit/test_bilinear_budget32000_audit.py",
    "runs/learning/projection-only-refit-v1/independent-audit.py",
    "docs/experiments/2026-09-26-bilinear-budget32000-plan.md",
    "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md",
)
ARTIFACTS = {
    "parent.json",
    "child.json",
    "training.json",
    "zero-replay.json",
    "comparisons.json",
    "train-membership.json",
    "run.json",
    "checkpoint-final.pt",
    "checkpoint-final.pt.json",
    "train-scores-post8000.json",
    "train-scores-post32000.json",
    "train-diagnostics.json",
}


def allowlist():
    """Finite named paths. Untrusted historical manifests never expand this set."""
    paths = set(AUDITOR_TEST_DEPENDENCIES) | {
        "scripts/refit_bilinear_budget32000.py",
        "scripts/aggregate_bilinear_budget32000.py",
        "scripts/refit_symmetric_affine_v3.py",
        "scripts/refit_bilinear_residual.py",
        "scripts/refit_membership_projection.py",
        "scripts/refit_bilinear_budget.py",
        "scripts/refit_bilinear_budget8000.py",
        "tests/unit/test_bilinear_budget32000_aggregate.py",
        "tests/unit/test_symmetric_affine_runner_v3.py",
        "tests/fixtures/projection_refit_runtime_source.zip",
        "configs/experiments/bilinear_budget32000_v1.json",
        "runs/learning/wide-first-choice-v1/summary.script.py",
        "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json",
        "runs/learning/bilinear-budget32000-readiness-v1/readiness.json",
    }
    for role in ("runner", "auditor", "aggregate"):
        for name in ("test-receipt.json", "stdout.txt"):
            paths.add(f"runs/learning/bilinear-budget32000-{role}-tests-v1/{name}")
    historical_files = {
        "summary.json",
        "independent-audit.json",
        "independent-audit-v2.json",
        "decision.json",
        "child.json",
        "parent.json",
        "training.json",
        "run.json",
        "train-membership.json",
        "checkpoint-final.pt.json",
        "summary.recipe.json",
    }
    for folder in ("bilinear-budget8000-v1", "bilinear-budget8000-replication-v1"):
        for name in historical_files | {"aggregate-v2.json"}:
            paths.add(f"runs/learning/{folder}/{name}")
    for seed in SEEDS:
        parent = f"runs/national_dex_continuation_control_s{seed}_v1"
        for name in ("checkpoint-final.pt", "checkpoint-final.pt.json", "run.json"):
            paths.add(f"{parent}/{name}")
        for name in historical_files:
            paths.add(f"runs/learning/bilinear-budget8000-replication-v1/seed-{seed}/{name}")
        for name in ARTIFACTS | {
            "summary.json",
            "execution-receipt.json",
            "primary-stdout.txt",
            "independent-audit.json",
            "audit-execution-receipt.json",
            "audit-stdout.txt",
        }:
            paths.add(f"runs/learning/{CAMPAIGN}/seed-{seed}/{name}")
        for name in (
            "script.py",
            "recipe.json",
            "plan.md",
            "input-amendment.md",
            "input-adapter.py",
            "bilinear-helper.py",
            "mean-helper.py",
            "helper.py",
            "source.zip",
            "configs.zip",
            "inputs.json",
            "test-receipt.json",
            "test-stdout.txt",
        ):
            paths.add(f"runs/learning/{CAMPAIGN}/seed-{seed}/summary.{name}")
    for name in (
        "aggregate.json",
        "aggregate-execution-receipt.json",
        "aggregate-stdout.txt",
        "aggregate-audit.json",
        "aggregate-audit-execution-receipt.json",
        "aggregate-audit-stdout.txt",
    ):
        paths.add(f"runs/learning/{CAMPAIGN}/{name}")
    paths |= READINESS_FILES
    paths |= REPAIR_FILES | {REPAIR_READINESS}
    paths |= {
        f"runs/learning/{CAMPAIGN}/seed-1731/continuation/" + p
        for p in REPAIR_FILES | {REPAIR_READINESS}
    }
    for seed in SEEDS:
        for name in (
            "independent-audit-v2.json",
            "audit-execution-receipt-v2.json",
            "audit-stdout-v2.txt",
            "continuation-complete.json",
        ):
            paths.add(f"runs/learning/{CAMPAIGN}/seed-{seed}/{name}")
    for name in (
        "aggregate-v2.json",
        "aggregate-v2.script.py",
        "aggregate-v2.readiness.json",
        "aggregate-v2.inputs.json",
        "aggregate-audit-v2.json",
    ):
        paths.add(f"runs/learning/{CAMPAIGN}/{name}")
    paths.add("runs/learning/pair-composition-integration-v1/summary.configs.zip")
    paths.add("runs/learning/pair-composition-integration-v1/summary.source.zip")
    for suffix in ("script.py", "readiness.json", "inputs.json"):
        paths.add(f"runs/learning/{CAMPAIGN}/aggregate.{suffix}")
    for seed in SEEDS:
        paths.add(f"runs/learning/{CAMPAIGN}/seed-{seed}/summary.readiness.json")
        for relative in READINESS_FILES:
            paths.add(f"runs/learning/{CAMPAIGN}/seed-{seed}/readiness/{relative}")
    return paths


def accepted_chain(summary, audit, decision, summary_sha, audit_sha):
    require(
        summary["complete"] is True and audit["complete"] is True and audit["audit_passed"] is True,
        "complete accepted historical evidence",
    )
    require(
        audit["summary_sha256"] == summary_sha
        and decision["summary_sha256"] == summary_sha
        and decision["audit_sha256"] == audit_sha
        and decision["evidence_accepted"] is True,
        "historical audit/owner identity",
    )


def historical_context(reader, seed):
    seed_contract(seed)
    if seed == 1729:
        folder = ROOT / "runs/learning/bilinear-budget8000-v1"
        summary = reader.read(folder / "summary.json", CONTROL1729["summary.json"])
        audit = reader.read(
            folder / "independent-audit.json", CONTROL1729["independent-audit.json"]
        )
        decision = reader.read(folder / "decision.json", CONTROL1729["decision.json"])
        summary_sha = CONTROL1729["summary.json"]
        accepted_chain(summary, audit, decision, summary_sha, CONTROL1729["independent-audit.json"])
    else:
        campaign = ROOT / "runs/learning/bilinear-budget8000-replication-v1"
        aggregate = reader.read(
            campaign / "aggregate-v2.json", CONTROL_REPLICATION["aggregate-v2.json"]
        )
        aggregate_audit = reader.read(
            campaign / "independent-audit-v2.json", CONTROL_REPLICATION["independent-audit-v2.json"]
        )
        decision = reader.read(campaign / "decision.json", CONTROL_REPLICATION["decision.json"])
        accepted_chain(
            aggregate,
            aggregate_audit,
            decision,
            CONTROL_REPLICATION["aggregate-v2.json"],
            CONTROL_REPLICATION["independent-audit-v2.json"],
        )
        require(
            [r["seed"] for r in aggregate["reports"]] == [1730, 1731], "historical fresh seed order"
        )
        require(
            [r["seed"] for r in aggregate_audit["per_seed"]] == [1730, 1731],
            "historical fresh audits order",
        )
        entry = next(r for r in aggregate["reports"] if r["seed"] == seed)
        checked = next(r for r in aggregate_audit["per_seed"] if r["seed"] == seed)
        folder = campaign / f"seed-{seed}"
        require(
            Path(entry["path"]).resolve() == folder / "summary.json", "seed-matched historical path"
        )
        summary_sha = entry["sha256"]
        summary = reader.read(folder / "summary.json", summary_sha)
        audit = reader.read(folder / "independent-audit-v2.json", checked["audit_sha256"])
        require(
            summary["complete"] is True
            and audit["complete"] is True
            and audit["audit_passed"] is True
            and audit["summary_sha256"] == summary_sha == checked["summary_sha256"]
            and summary["seed"] == audit["seed"] == seed,
            "complete seed-matched historical audit",
        )
    reports = {
        name: reader.read(folder / name, summary["artifact_sha256"][name])
        for name in (
            "child.json",
            "parent.json",
            "training.json",
            "run.json",
            "train-membership.json",
        )
    }
    exact(
        reports["child.json"]["aggregate"],
        audit["child"]["aggregate"],
        "historical aggregate identity",
    )
    exact(reports["child.json"]["groups"], audit["child"]["groups"], "historical group identity")
    require(
        reports["training.json"]["initial_pre_update_loss"] == INITIAL_LOSSES[seed]
        and reports["training.json"]["completed_updates"] == 8000,
        "seed-matched historical loss/budget",
    )
    require(
        reports["run.json"]["training_metadata"]["parent_checkpoint_sha256"] == PARENTS[seed],
        "original parent lineage",
    )
    parent = ROOT / f"runs/national_dex_continuation_control_s{seed}_v1/checkpoint-final.pt"
    inherited = summary["input_sha256"]

    def metadata(path):
        relative = path.relative_to(ROOT).as_posix()
        pins = [
            h
            for p, h in inherited.items()
            if p.replace("\\", "/") == relative or p.replace("\\", "/").endswith("/" + relative)
        ]
        require(len(pins) == 1, "unique historical metadata identity")
        return reader.read(path, pins[0])

    parent_sidecar = metadata(parent.with_suffix(".pt.json"))
    parent_run = metadata(parent.with_name("run.json"))
    vocabulary_path = ROOT / "data/processed/pokemon_v1_f1541479_20260924/vocabulary.json"
    vocabulary = reader.read(
        vocabulary_path, "1b0d8f1762fede01fb253c3df23d21ced254a10f6cea367709c479565a92fb34"
    )["tokens"]
    require(summary["split_hash"] == SPLIT_SHA, "inherited frozen split")
    require(
        summary["artifact_sha256"]["train-membership.json"] == MEMBERSHIP_SHA,
        "exact training membership identity",
    )
    partition_contract(
        reports["train-membership.json"]["queries"], reports["parent.json"]["responses"], vocabulary
    )
    return {
        "summary": summary,
        "summary_sha256": summary_sha,
        "audit": audit,
        "reports": reports,
        "parent": parent,
        "parent_sidecar": parent_sidecar,
        "parent_run": parent_run,
        "vocabulary": vocabulary,
    }


def partition_contract(train, validation, vocabulary):
    require(len(vocabulary) == 2049, "vocabulary width")
    require(len(train) == 1637 and len(validation) == 222, "selected partitions coverage")
    all_keys = set()
    for partition, rows in (("train", train), ("validation", validation)):
        for index, row in enumerate(rows):
            integer(row["index"], index, "selected partition order")
            subject, dimension = row["subject"], row["dimension"]
            require((subject, dimension) not in all_keys, "duplicate selected query")
            all_keys.add((subject, dimension))
            value = int.from_bytes(
                hashlib.sha256(f"1729:{subject}:{dimension}".encode()).digest(), "big"
            ) / float(1 << 256)
            selected = "validation" if value < 0.1 else "test" if value < 0.2 else "train"
            require(selected == partition, "protected/wrong partition record")
            prompt, truth = row["prompt_ids"], row["expected_set_ids"]
            require(
                len(prompt) == 5 and all(type(i) is int for i in prompt),
                "five integer prompt tokens",
            )
            require(
                prompt[0] == 1
                and dimension in ("TYPE", "COLOR")
                and prompt[2:] == [32 if dimension == "TYPE" else 33, 34, 5],
                "selected prompt grammar",
            )
            require(
                1024 <= prompt[1] <= 2048 and vocabulary[prompt[1]] == subject, "subject mapping"
            )
            require(
                1 <= len(truth) <= 506
                and truth == sorted(set(truth))
                and prompt[1] not in truth
                and all(type(i) is int and 1024 <= i <= 2048 for i in truth),
                "selected truth IDs",
            )
    return True


def validation_report(report, phase, reference, parent=None):
    require(
        report["complete"] is True
        and report["phase"] == phase
        and report["query_count"] == len(report["responses"]) == len(reference["responses"]) == 222,
        "complete validation coverage",
    )
    exact(report["product_token_ids"], COLUMNS, "validation columns")
    require(report["exact_parent_replay"] is (phase == "parent"), "parent replay attestation")
    rebuilt = []
    for i, (row, old) in enumerate(zip(report["responses"], reference["responses"], strict=True)):
        integer(row["index"], i, "validation index")
        for key in ("index", "subject", "dimension", "prompt_ids", "group", "expected_set_ids"):
            exact(row[key], old[key], "validation query/label alignment")
        require(row["group"] in GROUPS, "validation group")
        values = row["symmetric_relation_logits"]
        if phase == "parent":
            exact(values, old["symmetric_relation_logits"], "original parent exact score replay")
        pred = BASE.predict(values, row["prompt_ids"][1])
        expected = {
            k: row[k]
            for k in (
                "index",
                "subject",
                "dimension",
                "prompt_ids",
                "group",
                "expected_set_ids",
                "symmetric_relation_logits",
            )
        }
        expected.update(
            selected_set_ids=pred,
            metrics=metrics(pred, row["expected_set_ids"]),
            strict_separation=BASE.separation(
                values, row["expected_set_ids"], row["prompt_ids"][1]
            )["strict_separable"],
        )
        exact(row, expected, "independent validation row")
        rebuilt.append(expected)
    exact(report["aggregate"], totals(rebuilt), "validation totals")
    exact(
        report["groups"],
        {g: totals([r for r in rebuilt if r["group"] == g]) for g in GROUPS},
        "validation group totals",
    )
    if parent is not None:
        exact(
            report["paired_vs_parent_dense"],
            BASE.paired(rebuilt, [r["selected_set_ids"] for r in parent["responses"]]),
            "paired parent metrics",
        )
    return rebuilt


def score_bytes(rows):
    import struct

    values = []
    for row in rows:
        BASE.vector(row["symmetric_relation_logits"])
        values.extend(row["symmetric_relation_logits"])
    return struct.pack("<" + "f" * len(values), *values)


def zero_replay_contract(report, parent):
    rows = parent["responses"]
    require(len(rows) == 222, "zero replay query count")
    batches = []
    for offset in range(0, 222, 8):
        part = rows[offset : offset + 8]
        digest = hashlib.sha256(score_bytes(part)).hexdigest()
        batches.append(
            {
                "indices": list(range(offset, offset + len(part))),
                "shape": [len(part), 1025],
                "logits_sha256": digest,
                "parent_logits_sha256": digest,
                "residual_sha256": hashlib.sha256(bytes(len(part) * 1025 * 4)).hexdigest(),
                "residual_nonzero_count": 0,
            }
        )
    digest = hashlib.sha256(score_bytes(rows)).hexdigest()
    expected = {
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
    exact(report, expected, "zero residual replay receipt")
    return {
        k: report[k] for k in ("query_count", "logits_sha256", "residual_sha256", "batch_sizes")
    }


ARCHITECTURE_DESCRIPTOR = {
    "id": ARCHITECTURE,
    "parameter": A,
    "shape": A_SHAPE,
    "dimension_token_ids": [32, 33],
    "dimensions": ["TYPE", "COLOR"],
    "dtype": "float32",
    "initialization": "exact-zero",
    "symmetrization": "(A+A.transpose(-1,-2))*0.5",
    "score_order": (
        "stack-linear-entities-gather-subject-linear-products-divide16-add-archived-parent-v1"
    ),
}


def recipe_contract(recipe, inherited):
    expected = json.loads(json.dumps(inherited))
    require(
        expected["experiment"] == "bilinear-budget8000-v1" and expected["residual_updates"] == 8000,
        "historical recipe budget",
    )
    for key in (
        "seed",
        "initial_training_loss",
        "strong_comparator_exact",
        "strong_comparator_f1",
        "strong_comparator_group_exact",
    ):
        expected.pop(key)
    expected.update(
        experiment=CAMPAIGN,
        evaluator=EVALUATOR,
        inference=INFERENCE,
        residual_updates=UPDATES,
        seeds=list(SEEDS),
        data_split_seed=1729,
        input_adapter=INPUT_ADAPTER,
        input_amendment_sha256=AMENDMENT_SHA,
        plan_sha256=PLAN_SHA,
        input_adapter_source_sha256=ADAPTER_SHA,
        standard_serving_supported=False,
        validation_update_points=[0, UPDATES],
        training_diagnostic_updates=list(TRAIN_POINTS),
        seed_contracts={
            str(seed): {
                "parent_checkpoint_sha256": PARENTS[seed],
                "initial_training_loss": INITIAL_LOSSES[seed],
                "floors": {
                    "exact_count": FLOORS[seed]["exact"],
                    "f1": FLOORS[seed]["f1"],
                    "groups": dict(zip(GROUPS, FLOORS[seed]["groups"], strict=True)),
                },
            }
            for seed in SEEDS
        },
    )
    exact(recipe, expected, "fixed campaign recipe")


def parent_admission(seed, parent, context):
    sidecar, run, historical = context["parent_sidecar"], context["parent_run"], context["summary"]
    BASE.sidecar_matches(parent, sidecar, PARENTS[seed])
    integer(parent["global_step"], 2000, "original parent steps")
    require(
        run["config"]["seed"] == seed and run["config"]["data"]["split_seed"] == 1729,
        "original model/split seeds",
    )
    require(
        run["config"]["run_name"] == f"national_dex_continuation_control_s{seed}_v1",
        "original parent run name",
    )
    exact(parent["config"], run["config"]["train"], "parent TrainConfig scope")
    exact(parent["experiment_identity"], run["identity"], "original parent identity")
    exact(
        parent["experiment_identity"],
        historical["parent_training_identity"],
        "historical parent identity binding",
    )
    require(
        parent["experiment_identity"]["seeds"] == [seed]
        and parent["experiment_identity"]["evaluator_version"] == "plm-train-causal-v2",
        "original parent training evaluator",
    )
    require(
        parent["experiment_identity"]["resolved_config_hash"]
        == hashlib.sha256(
            json.dumps(
                {k: v for k, v in run["config"].items() if k != "run_name"},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "original resolved config",
    )
    require(parent["split_hash"] == SPLIT_SHA, "original parent split")
    exact(parent["corpus_identity"], historical["corpus_identity"], "inherited corpus identity")
    require(
        all(parent["experiment_identity"][k] == v for k, v in parent["corpus_identity"].items()),
        "parent identity corpus fields",
    )
    require(
        parent["training_metadata"]["objective"] == "causal-next-token-v1", "original objective"
    )
    exact(
        parent["training_metadata"]["model_config"],
        historical["inherited_architecture"],
        "original model architecture",
    )
    return True


def load_payload(reader, path, digest):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only payload audit")
    import torch

    path = reader.bind(path, digest)
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(value, dict) and "model" in value, "checkpoint dictionary")
    return value


def training_contract(seed, summary, reports, recipe, context, parent, child):
    """Payload and saved chronology contracts; no replay of optimizer computation."""
    training, run = reports["training.json"], reports["run.json"]
    require(
        training["complete"] is True
        and training["fit_complete"] is True
        and training["phase"] == "complete",
        "completed fit",
    )
    require(
        not any(k in training for k in ("error", "failed_update", "failed_observation")),
        "no unresolved fit failure",
    )
    integer(training["completed_updates"], UPDATES, "completed update count")
    require(
        training["architecture"] == ARCHITECTURE and training["objective"] == OBJECTIVE,
        "fixed architecture/objective",
    )
    require(
        training["initial_loss_replay"]
        == INITIAL_LOSSES[seed]
        == context["reports"]["training.json"]["initial_pre_update_loss"],
        "matching initial loss",
    )
    loss_contract(
        training["history"],
        INITIAL_LOSSES[seed],
        training["initial_loss_replay"],
        training["initial_pre_update_loss"],
        training["final_post_update_loss"],
    )
    integer(training["train_query_count"], 1637, "training query count")
    require(
        training["validation_labels_used"] is False
        and training["transformer_forwards_during_fit"] == 0,
        "cached train-only fit",
    )
    exact(training["trainable_parameters"], [A], "sole trainable parameter")
    parent_admission(seed, parent, context)
    integer(child["global_step"], UPDATES, "child update count")
    original, initial, final = compare_states(parent["model"], child["model"])
    exact(
        original,
        context["reports"]["training.json"]["parent_state_before"],
        "historical original state",
    )
    for key, expected in (
        ("parent_state_before", original),
        ("state_before", initial),
        ("state_after", final),
        ("state_reloaded", final),
    ):
        exact(training[key], expected, "state receipt " + key)
    for key in (
        "residual_changed",
        "frozen_tensors_unchanged",
        "reload_exact",
        "optimizer_reload_exact",
    ):
        require(training[key] is True, "state invariant " + key)
    require(training["optimizer"]["name"] == "adamw", "optimizer family")
    exact(training["optimizer"]["config"], recipe["optimizer"], "optimizer recipe")
    impl = training["optimizer_implementation"]
    opt = optimizer_contract(
        child["optimizer"],
        {
            "class": impl["module"] + "." + impl["class"],
            "groups": training["optimizer"]["groups"],
            "defaults": impl["defaults"],
        },
    )
    exact(training["optimizer_final"], opt, "saved optimizer state receipt")
    require(child["scheduler"] is None, "no scheduler")
    queries = {
        name: [{k: r[k] for k in ("subject", "dimension", "prompt_ids")} for r in rows]
        for name, rows in (
            ("train", reports["train-membership.json"]["queries"]),
            ("validation", reports["parent.json"]["responses"]),
        )
    }
    # Query descriptors retain exactly the archived helper's public three fields.
    exact(summary["queries"], queries, "ordered selected partitions")
    train_hash = formatted_hash(queries["train"])
    require(
        summary["train_query_order_sha256"] == training["train_query_order_sha256"] == train_hash,
        "training order hash",
    )
    exact(summary["validation_update_points"], [0, UPDATES], "fixed validation endpoints")
    exact(
        reports["train-membership.json"],
        context["reports"]["train-membership.json"],
        "unchanged supervised membership",
    )
    implementation = {
        "runner_sha256": summary["script_sha256"],
        "scorer_sha256": SCORER_SHA,
        "input_adapter_identity": INPUT_ADAPTER,
        "input_amendment_sha256": AMENDMENT_SHA,
        "plan_sha256": PLAN_SHA,
        "bilinear_helper_sha256": SCORER_SHA,
        "mean_helper_sha256": MEAN_SHA,
        "input_adapter_source_sha256": ADAPTER_SHA,
        "source_archive_sha256": SOURCE_SHA,
        "configs_archive_sha256": CONFIGS_SHA,
    }
    config = {
        "implementation_identity": implementation,
        "inherited_parent_config": context["parent_run"]["config"],
        "bilinear_residual_architecture": ARCHITECTURE_DESCRIPTOR,
        "bilinear_residual_recipe": recipe,
        "seed": seed,
        "data_split_seed": 1729,
        "campaign_version": CAMPAIGN,
    }
    exact(run["config"], config, "architecture-explicit run config")
    exact(child["config"], config, "checkpoint config")
    metadata = {
        "implementation_identity": implementation,
        "seed": seed,
        "data_split_seed": 1729,
        "campaign_version": CAMPAIGN,
        "objective": OBJECTIVE,
        "evaluator": EVALUATOR,
        "architecture": ARCHITECTURE,
        "model_config": parent["training_metadata"]["model_config"],
        "model_config_scope": "inherited archived base only; residual architecture is separate",
        "parent_checkpoint_sha256": PARENTS[seed],
        "parent_training_steps": 2000,
        "residual_updates": UPDATES,
        "trainable_parameters": [A],
        "inference": INFERENCE,
        "train_query_order_sha256": train_hash,
        "record_count": 1637,
        "recipe_sha256": summary["recipe_sha256"],
        "runner_sha256": summary["script_sha256"],
        "scorer_sha256": SCORER_SHA,
        "input_adapter_identity": INPUT_ADAPTER,
        "input_amendment_sha256": AMENDMENT_SHA,
        "plan_sha256": PLAN_SHA,
        "bilinear_helper_sha256": SCORER_SHA,
        "parent_state_sha256": formatted_hash(original),
        "initial_state_sha256": formatted_hash(initial),
        "initial_pre_update_loss": training["initial_pre_update_loss"],
        "final_post_update_loss": training["final_post_update_loss"],
        "standard_serving_supported": False,
    }
    exact(run["training_metadata"], metadata, "run metadata")
    exact(child["training_metadata"], metadata, "child metadata")
    identity = {
        **parent["experiment_identity"],
        "source_commit": summary["workspace_provenance"]["commit"],
        "resolved_config_hash": formatted_hash(config),
        "evaluator_version": EVALUATOR,
        "environment": {
            **summary["environment"],
            "source_archive_sha256": SOURCE_SHA,
            "configs_archive_sha256": CONFIGS_SHA,
            "runner_sha256": summary["script_sha256"],
            "scorer_sha256": SCORER_SHA,
            "input_adapter_identity": INPUT_ADAPTER,
            "input_amendment_sha256": AMENDMENT_SHA,
            "plan_sha256": PLAN_SHA,
            "bilinear_helper_sha256": SCORER_SHA,
            "recipe_sha256": summary["recipe_sha256"],
            "mean_helper_sha256": MEAN_SHA,
            "input_adapter_source_sha256": ADAPTER_SHA,
            "architecture": ARCHITECTURE,
        },
        "checkpoint_hash": None,
        "seeds": [seed],
        "repetitions": 1,
    }
    for value in (
        run["identity"],
        child["experiment_identity"],
        training["training_identity"],
        summary["child_training_identity"],
    ):
        exact(value, identity, "child training identity")
    exact(child["corpus_identity"], parent["corpus_identity"], "child inherited corpus")
    require(child["split_hash"] == summary["split_hash"] == SPLIT_SHA, "child split identity")
    exact(
        summary["parent_training_identity"],
        parent["experiment_identity"],
        "parent identity receipt",
    )
    exact(
        summary["inherited_architecture"],
        parent["training_metadata"]["model_config"],
        "inherited base architecture",
    )
    exact(summary["corpus_identity"], parent["corpus_identity"], "summary corpus identity")
    require(
        training["checkpoint_sha256"] == summary["artifact_sha256"]["checkpoint-final.pt"]
        and training["checkpoint_sidecar_sha256"]
        == summary["artifact_sha256"]["checkpoint-final.pt.json"],
        "saved payload identities",
    )
    require(
        training["zero_replay_sha256"] == summary["artifact_sha256"]["zero-replay.json"],
        "zero replay identity",
    )
    exact(training["zero_replay"], reports["zero-replay.json"], "training zero replay")
    finite(training["refit_seconds"])
    require(training["refit_seconds"] >= 0, "nonnegative refit duration")
    return {
        "parent_checkpoint_sha256": PARENTS[seed],
        "child_checkpoint_sha256": training["checkpoint_sha256"],
        "global_step": UPDATES,
        "parent_training_steps": 2000,
        "original_tensor_count": 93,
        "child_tensor_count": 94,
        "unchanged_original_tensor_count": 93,
        "new_trained_tensors": [A],
        "initial_residual": initial[A],
        "trained_residual": final[A],
        "optimizer_step": UPDATES,
        "updates_verified": UPDATES,
        "initial_pre_update_loss": training["initial_pre_update_loss"],
        "final_post_update_loss": training["final_post_update_loss"],
        "refit_seconds": training["refit_seconds"],
        "payload_loaded_on": "cpu",
    }


READINESS = "runs/learning/bilinear-budget32000-readiness-v1/readiness.json"
PLAN_PATH = "docs/experiments/2026-09-26-bilinear-budget32000-plan.md"
AMENDMENT_PATH = "docs/experiments/2026-09-26-symmetric-affine-input-amendment.md"
RUNNER = "scripts/refit_bilinear_budget32000.py"
RECIPE = "configs/experiments/bilinear_budget32000_v1.json"
AGGREGATOR = "scripts/aggregate_bilinear_budget32000.py"
AUDITOR = f"runs/learning/{CAMPAIGN}/independent-audit.py"
TEST_DEPENDENCIES = {
    "auditor": set(AUDITOR_TEST_DEPENDENCIES),
    "runner": {
        RUNNER,
        "tests/unit/test_bilinear_budget32000.py",
        "scripts/refit_bilinear_residual.py",
        "scripts/refit_membership_projection.py",
        "runs/learning/wide-first-choice-v1/summary.script.py",
        "scripts/refit_symmetric_affine_v3.py",
        "tests/unit/test_bilinear_residual.py",
        "tests/fixtures/projection_refit_runtime_source.zip",
        PLAN_PATH,
        AMENDMENT_PATH,
    },
    "aggregate": {
        AGGREGATOR,
        "tests/unit/test_bilinear_budget32000_aggregate.py",
        "scripts/refit_membership_projection.py",
        PLAN_PATH,
    },
}
TEST_DIRECTORIES = {
    role: f"runs/learning/bilinear-budget32000-{role}-tests-v1" for role in TEST_DEPENDENCIES
}
READINESS_FILES = (
    {RECIPE}
    | set().union(*TEST_DEPENDENCIES.values())
    | {
        directory + "/" + name
        for directory in TEST_DIRECTORIES.values()
        for name in ("test-receipt.json", "stdout.txt")
    }
)

EVIDENCE_REVISION = "bilinear-budget32000-evidence-v2"
EXECUTION_REVISION = "bilinear-budget32000-continuation-v2"
REPAIR_AMENDMENT = "docs/experiments/2026-09-26-bilinear-budget32000-evidence-repair-v2.md"
REPAIR_AMENDMENT_SHA = "90aa30088980aeffd999fe6fe4e7349220d163b196ad4eb28f06193b9ce2991c"
BASE_READINESS = {
    "path": READINESS,
    "sha256": "5266fba8ab636590c0654db31b337ba8fd806361d9703c41a1b7260fcb5d766d",
}
BASE_AUDITOR_SHA = "4dba10059e84146a62e1f26d2bec55195ae82e68351debfee1681ba72e0c0188"
BASE_AGGREGATOR_SHA = "41bc30b8e0b29742b4446a3b4a776c91ea7264f656b8dc01ab60966f1df5a09a"
REPAIR_READINESS = "runs/learning/bilinear-budget32000-readiness-v2/readiness.json"
REPAIR_AUDITOR = f"runs/learning/{CAMPAIGN}/independent-audit-v2.py"
REPAIR_RUNNER = "scripts/refit_bilinear_budget32000_continuation_v2.py"
REPAIR_AGGREGATOR = "scripts/aggregate_bilinear_budget32000_v2.py"
REPAIR_SOURCES = {
    "auditor": REPAIR_AUDITOR,
    "runner": REPAIR_RUNNER,
    "aggregate": REPAIR_AGGREGATOR,
}
REPAIR_TESTS = {
    "auditor": "tests/unit/test_bilinear_budget32000_audit_v2.py",
    "runner": "tests/unit/test_bilinear_budget32000_continuation_v2.py",
    "aggregate": "tests/unit/test_bilinear_budget32000_aggregate_v2.py",
}
REPAIR_TEST_DIRECTORIES = {
    role: f"runs/learning/bilinear-budget32000-{role}-tests-v2" for role in REPAIR_SOURCES
}
REPAIR_DEPENDENCIES = {
    role: {source, REPAIR_TESTS[role], REPAIR_AMENDMENT, READINESS}
    for role, source in REPAIR_SOURCES.items()
}
REPAIR_FILES = (
    {REPAIR_AMENDMENT, READINESS}
    | set(REPAIR_SOURCES.values())
    | set(REPAIR_TESTS.values())
    | {
        directory + "/" + name
        for directory in REPAIR_TEST_DIRECTORIES.values()
        for name in ("test-receipt.json", "stdout.txt")
    }
)
PRESERVED_PINS = {
    "seed-1729/summary.json": "b41fec77f92855db5753c97add0255e4593fa328192eb34fe3ab6ad567bbacd1",
    "seed-1729/independent-audit.json": (
        "76ecfdb9e8e3654fe9aa7d939972c6b453079b1a907d489f5cded679e0b20850"
    ),
    "seed-1730/summary.json": "1237d6e266fa57b5284f76ab76d50069ec9f161f5e23add84e06d769a47b5e11",
    "seed-1730/independent-audit.json": (
        "85b193ad473d77ac10d84aa2e9138ba4d56efd7e9694ef52eeab4ed6e069c188"
    ),
    "seed-1730/audit-execution-receipt.json": (
        "2195369180305bb04f7967b5877e41006f965fd8499d411229193415662fc5ce"
    ),
    "seed-1730/execution-receipt.json": (
        "97f36f6d2b1bee433082b49c5204376d4f70dd78e3abba52431d5be32781b1b8"
    ),
}


def audit_route(seed):
    seed_contract(seed)
    return (
        ("independent-audit.json", "audit-execution-receipt.json", "audit-stdout.txt")
        if seed == 1729
        else ("independent-audit-v2.json", "audit-execution-receipt-v2.json", "audit-stdout-v2.txt")
    )


def repair_readiness_contract(reader, digest):
    original = readiness_contract(reader, BASE_READINESS["sha256"])
    require(original["files"][AUDITOR] == BASE_AUDITOR_SHA, "frozen predecessor auditor")
    repair = reader.read(REPAIR_READINESS, digest)
    integer(repair["schema_version"], 1, "repair readiness schema")
    require(
        repair["ready"] is True
        and repair["campaign_version"] == CAMPAIGN
        and repair["evidence_revision"] == EVIDENCE_REVISION
        and repair["amendment_sha256"] == REPAIR_AMENDMENT_SHA,
        "repair admission identity",
    )
    exact(repair["base_readiness"], BASE_READINESS, "repair original admission")
    require(
        isinstance(repair["files"], dict) and set(repair["files"]) == REPAIR_FILES,
        "exact14 repair readiness inventory",
    )
    for path in repair["files"]:
        reader.admit(path)
    require(
        repair["files"][REPAIR_AMENDMENT] == REPAIR_AMENDMENT_SHA
        and repair["files"][READINESS] == BASE_READINESS["sha256"],
        "repair immutable declaration",
    )
    reader.manifest(repair["files"])
    for role, folder in REPAIR_TEST_DIRECTORIES.items():
        value = reader.read(
            folder + "/test-receipt.json", repair["files"][folder + "/test-receipt.json"]
        )
        repair_test_receipt(value, role, repair["files"])
    return original, repair


def repair_test_receipt(value, role, files):
    require(
        value["passed"] is True
        and value["gpu_used"] is False
        and value["experiment_training_executed"] is False
        and value["terminal_completion_observed"] is True,
        "observed CPU-only repair tests",
    )
    integer(value["exit_code"], 0, "repair test exit")
    integer(value["tests_skipped"], 0, "no skipped repair tests")
    require(
        type(value["tests_passed"]) is int and value["tests_passed"] > 0,
        "positive repair test count",
    )
    require(
        value["evidence_revision"] == EVIDENCE_REVISION
        and value["amendment_sha256"] == REPAIR_AMENDMENT_SHA,
        "repair test declaration",
    )
    exact(
        value["test_dependencies"],
        {p: files[p] for p in REPAIR_DEPENDENCIES[role]},
        "four direct repair test dependencies",
    )
    require(value["tested_script_sha256"] == files[REPAIR_SOURCES[role]], "repair tested source")
    stdout = REPAIR_TEST_DIRECTORIES[role] + "/stdout.txt"
    exact(value["stdout"], {"path": stdout, "sha256": files[stdout]}, "repair test stdout")


def preserved_history(reader):
    values = {name: reader.read(HERE / name, digest) for name, digest in PRESERVED_PINS.items()}
    failed = values["seed-1730/independent-audit.json"]
    require(
        failed["complete"] is False
        and failed["audit_passed"] is False
        and failed["error"] == "ValueError('predecessor receipt binding')",
        "preserved original failed audit",
    )
    terminal = values["seed-1730/audit-execution-receipt.json"]
    integer(terminal["exit_code"], 1, "preserved failed audit exit")
    require(
        terminal["terminal_completion_observed_by_primary"] is True,
        "observed original failed audit",
    )
    require(
        terminal["audit_sha256"] == PRESERVED_PINS["seed-1730/independent-audit.json"]
        and terminal["execution_receipt_sha256"]
        == PRESERVED_PINS["seed-1730/execution-receipt.json"],
        "preserved failed audit bindings",
    )
    reader.bind(HERE / "seed-1730/audit-stdout.txt", terminal["stdout_sha256"])
    return dict(PRESERVED_PINS)


def repair_identity(value, repair, digest):
    require(
        value["audit_revision"] == EVIDENCE_REVISION
        and value["amendment_sha256"] == REPAIR_AMENDMENT_SHA
        and value["base_auditor_sha256"] == BASE_AUDITOR_SHA,
        "actual revised audit identity",
    )
    require(
        value["script_sha256"] == repair["files"][REPAIR_AUDITOR]
        and value["test_receipt_sha256"]
        == repair["files"][REPAIR_TEST_DIRECTORIES["auditor"] + "/test-receipt.json"],
        "actual repaired source/test identity",
    )
    exact(value["base_readiness"], BASE_READINESS, "base scientific readiness")
    exact(
        value["shared_readiness"],
        {"path": REPAIR_READINESS, "sha256": digest},
        "repaired evidence readiness",
    )


def continuation_contract(reader, summary, repair, digest):
    folder = HERE / "seed-1731"
    admission = {"path": REPAIR_READINESS, "sha256": digest}
    require(
        summary["execution_wrapper_sha256"] == repair["files"][REPAIR_RUNNER]
        and summary["execution_revision"] == EXECUTION_REVISION
        and summary["amendment_sha256"] == REPAIR_AMENDMENT_SHA,
        "actual continuation entrypoint",
    )
    exact(summary["continuation_readiness"], admission, "continuation admission")
    expected = {
        f"runs/learning/{CAMPAIGN}/seed-1731/continuation/" + p: h
        for p, h in (repair["files"] | {REPAIR_READINESS: digest}).items()
    }
    exact(summary["continuation_snapshots"], expected, "exact continuation snapshot inventory")
    reader.manifest(expected)
    complete = reader.read(folder / "continuation-complete.json")
    require(
        complete["complete"] is True and complete["root_observed_terminal_required"] is True,
        "completed continuation wrapper",
    )
    integer(complete["seed"], 1731, "continuation seed")
    for key, value in {
        "summary_sha256": reader.inputs[str(folder / "summary.json")],
        "execution_wrapper_sha256": repair["files"][REPAIR_RUNNER],
        "execution_revision": EXECUTION_REVISION,
        "amendment_sha256": REPAIR_AMENDMENT_SHA,
        "continuation_readiness": admission,
    }.items():
        exact(complete[key], value, "continuation completion " + key)
    execution = reader.read(folder / "execution-receipt.json")
    require(
        execution["continuation_complete_sha256"]
        == reader.inputs[str(folder / "continuation-complete.json")],
        "observed wrapper completion binding",
    )
    return {
        "continuation_complete_sha256": reader.inputs[str(folder / "continuation-complete.json")],
        "execution_wrapper_sha256": repair["files"][REPAIR_RUNNER],
    }


def test_receipt_contract(value, role, files):
    require(
        value.get("passed") is True
        and value.get("gpu_used") is False
        and value.get("experiment_training_executed") is False
        and value.get("terminal_completion_observed") is True,
        "completed CPU-only synthetic receipt",
    )
    integer(value["exit_code"], 0, "test zero exit")
    integer(value["tests_skipped"], 0, "no skipped readiness tests")
    require(type(value["tests_passed"]) is int and value["tests_passed"] > 0, "positive test count")
    require(
        isinstance(value["test_dependencies"], dict)
        and set(value["test_dependencies"]) == TEST_DEPENDENCIES[role],
        "exact test dependencies",
    )
    exact(
        value["test_dependencies"],
        {p: files[p] for p in TEST_DEPENDENCIES[role]},
        "test dependency hashes",
    )
    script = {"runner": RUNNER, "auditor": AUDITOR, "aggregate": AGGREGATOR}[role]
    require(value["tested_script_sha256"] == files[script], "tested source identity")
    stdout = TEST_DIRECTORIES[role] + "/stdout.txt"
    exact(value["stdout"], {"path": stdout, "sha256": files[stdout]}, "test stdout identity")
    if role == "runner":
        require(value["tested_recipe_sha256"] == files[RECIPE], "tested recipe identity")


def readiness_contract(reader, digest):
    value = reader.read(READINESS, digest)
    integer(value["schema_version"], 1, "readiness schema")
    require(
        value["ready"] is True
        and value["campaign_version"] == CAMPAIGN
        and value["plan_sha256"] == PLAN_SHA,
        "readiness campaign",
    )
    require(
        isinstance(value["files"], dict) and set(value["files"]) == READINESS_FILES,
        "exact shared readiness inventory",
    )
    # No referenced file is opened until the entire explicit inventory is admitted.
    for path in value["files"]:
        reader.admit(path)
    pins = {
        PLAN_PATH: PLAN_SHA,
        AMENDMENT_PATH: AMENDMENT_SHA,
        "scripts/refit_bilinear_residual.py": SCORER_SHA,
        "scripts/refit_membership_projection.py": MEAN_SHA,
        "scripts/refit_symmetric_affine_v3.py": ADAPTER_SHA,
        "runs/learning/projection-only-refit-v1/independent-audit.py": BASE_SHA,
        "tests/fixtures/projection_refit_runtime_source.zip": SOURCE_SHA,
        "runs/learning/wide-first-choice-v1/summary.script.py": HELPER_SHA,
    }
    for path, pin in pins.items():
        require(value["files"][path] == pin, "frozen readiness dependency")
    reader.manifest(value["files"])
    for role, directory in TEST_DIRECTORIES.items():
        path = directory + "/test-receipt.json"
        test_receipt_contract(reader.read(path, value["files"][path]), role, value["files"])
    return value


def archive_inventory(reader, path, digest, destination):
    import zipfile
    from pathlib import PurePosixPath

    path = reader.bind(path, digest)
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len({n.casefold() for n in names}), "unique archive paths")
        for name in names:
            rel = PurePosixPath(name)
            entry = archive.getinfo(name)
            require(
                rel.parts
                and not rel.is_absolute()
                and rel.as_posix() == name
                and ".." not in rel.parts
                and ":" not in name
                and "\\" not in name
                and entry.orig_filename == name,
                "safe archive path",
            )
            require(
                not entry.is_dir() and (entry.external_attr >> 16) & 0o170000 != 0o120000,
                "regular archive entry",
            )
            require(
                name.startswith(("src/plm/", "configs/")) or name in ("pyproject.toml", "uv.lock"),
                "source/config archive only",
            )
            digest = hashlib.sha256(archive.read(name)).hexdigest()
            target = (destination / name).resolve()
            require(target.is_relative_to(destination.resolve()), "runtime archive boundary")
            reader.allowed.add(str(target))
            reader.bind(target, digest)
            result[str(target)] = digest
    return result


def provenance(reader, seed, summary, folder, context, readiness_digest):
    ready = readiness_contract(reader, readiness_digest)
    exact(
        summary["shared_readiness"],
        {"path": READINESS, "sha256": readiness_digest},
        "summary readiness binding",
    )
    require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and summary["acceptance"] is False,
        "completed immutable candidate evidence",
    )
    require(
        not any(
            k in summary for k in ("error", "failed_observation", "failure_state_availability")
        ),
        "no unresolved execution failure",
    )
    integer(summary["seed"], seed, "summary seed")
    require(
        summary["campaign_version"] == CAMPAIGN
        and summary["architecture"] == ARCHITECTURE
        and summary["objective"] == OBJECTIVE
        and summary["evaluator"] == EVALUATOR
        and summary["inference"] == INFERENCE,
        "campaign method identity",
    )
    require(
        summary["protected_test_used"] is False and summary["standard_serving_supported"] is False,
        "protected/serving boundary",
    )
    require(
        summary["plan_sha256"] == PLAN_SHA
        and summary["input_adapter_identity"] == INPUT_ADAPTER
        and summary["input_amendment_sha256"] == AMENDMENT_SHA
        and summary["input_adapter_source_sha256"] == ADAPTER_SHA,
        "plan/input identity",
    )
    require(
        summary["script_sha256"] == ready["files"][RUNNER]
        and summary["recipe_sha256"] == ready["files"][RECIPE],
        "frozen implementation identities",
    )
    require(
        summary["scorer_sha256"] == SCORER_SHA
        and summary["mean_helper_sha256"] == MEAN_SHA
        and summary["helper_sha256"] == HELPER_SHA,
        "fixed arithmetic helpers",
    )
    exact(summary["environment"], context["summary"]["environment"], "matched runtime environment")
    settings = summary["numerical_settings"]
    require(
        settings["float32_matmul_precision"] == "highest"
        and settings["cuda_matmul_allow_tf32"] is False
        and settings["cudnn_allow_tf32"] is True
        and settings["deterministic_algorithms"] is False
        and settings["autocast_cuda_enabled"] is False
        and settings["autocast_cpu_enabled"] is False,
        "fixed FP32 settings",
    )
    sources = {
        "script.py": (RUNNER, ready["files"][RUNNER]),
        "recipe.json": (RECIPE, ready["files"][RECIPE]),
        "plan.md": (PLAN_PATH, PLAN_SHA),
        "input-amendment.md": (AMENDMENT_PATH, AMENDMENT_SHA),
        "input-adapter.py": ("scripts/refit_symmetric_affine_v3.py", ADAPTER_SHA),
        "bilinear-helper.py": ("scripts/refit_bilinear_residual.py", SCORER_SHA),
        "mean-helper.py": ("scripts/refit_membership_projection.py", MEAN_SHA),
        "helper.py": ("runs/learning/wide-first-choice-v1/summary.script.py", HELPER_SHA),
        "source.zip": ("tests/fixtures/projection_refit_runtime_source.zip", SOURCE_SHA),
        "configs.zip": (
            "runs/learning/pair-composition-integration-v1/summary.configs.zip",
            CONFIGS_SHA,
        ),
        "readiness.json": (READINESS, readiness_digest),
    }
    expected_snapshots = {}
    for suffix, (path, digest) in sources.items():
        reader.bind(path, digest)
        target = folder / ("summary." + suffix)
        reader.bind(target, digest)
        expected_snapshots[str(target)] = digest
    for relative, digest in ready["files"].items():
        target = folder / "readiness" / relative
        reader.bind(target, digest)
        expected_snapshots[str(target)] = digest
    inputs_path = folder / "summary.inputs.json"
    inputs_digest = summary["snapshot_sha256"][str(inputs_path)]
    declared_inputs = reader.read(inputs_path, inputs_digest)
    exact(declared_inputs, summary["input_sha256"], "input snapshot identity")
    expected_snapshots[str(inputs_path)] = inputs_digest
    exact(summary["snapshot_sha256"], expected_snapshots, "exact implementation snapshots")
    reader.manifest(declared_inputs)
    runtime = {}
    for suffix, digest in (("source.zip", SOURCE_SHA), ("configs.zip", CONFIGS_SHA)):
        runtime.update(
            archive_inventory(reader, folder / ("summary." + suffix), digest, folder / "runtime")
        )
    exact(summary["runtime_files_sha256"], runtime, "exact runtime archive files")
    origins = summary["module_origins"]
    require(isinstance(origins, dict) and origins, "runtime module origins")
    for name, entry in origins.items():
        path = str(Path(entry["path"]).resolve())
        require(
            (name == "plm" or name.startswith("plm."))
            and path in runtime
            and entry["sha256"] == runtime[path],
            "archived runtime import origin",
        )
    recipe = reader.read(RECIPE, ready["files"][RECIPE])
    historical_recipe = reader.read(
        "runs/learning/bilinear-budget8000-v1/summary.recipe.json",
        reader.read(
            "runs/learning/bilinear-budget8000-v1/summary.json", CONTROL1729["summary.json"]
        )["recipe_sha256"],
    )
    recipe_contract(recipe, historical_recipe)
    return recipe, ready


def receipt_chain(reader, seed, summary_sha, ready, repair, *, include_audit=False):
    """Verify the immediate predecessor was executed and audited before launch."""
    folder = ROOT / f"runs/learning/{CAMPAIGN}/seed-{seed}"
    execution = reader.read(folder / "execution-receipt.json")
    execution_sha = reader.inputs[str(folder / "execution-receipt.json")]
    reader.bind(folder / "primary-stdout.txt", execution["stdout_sha256"])
    previous = None
    if seed != SEEDS[0]:
        predecessor = SEEDS[SEEDS.index(seed) - 1]
        previous_folder = folder.parent / f"seed-{predecessor}"
        previous_summary = reader.read(previous_folder / "summary.json")
        previous_summary_sha = reader.inputs[str(previous_folder / "summary.json")]
        previous, previous_terminal = receipt_chain(
            reader, predecessor, previous_summary_sha, ready, repair, include_audit=True
        )
        require(previous_summary["complete"] is True, "completed predecessor")
        require(
            execution["prior_execution_receipt_sha256"]
            == reader.inputs[str(previous_folder / "execution-receipt.json")]
            and execution["prior_audit_receipt_sha256"]
            == reader.inputs[str(previous_folder / audit_route(predecessor)[1])],
            "exact predecessor terminal hashes",
        )
        require(
            instant(execution["started_utc"]) >= instant(previous_terminal["completed_utc"]),
            "previous audit completed before next launch",
        )
    else:
        require(
            execution["prior_execution_receipt_sha256"] is None
            and execution["prior_audit_receipt_sha256"] is None,
            "first seed has no receipt hashes",
        )
    execution_contract(execution, seed, summary_sha, execution["stdout_sha256"], previous)
    terminal = None
    if include_audit:
        audit_name, terminal_name, stdout_name = audit_route(seed)
        checked = reader.read(folder / audit_name)
        terminal = reader.read(folder / terminal_name)
        reader.bind(folder / stdout_name, terminal["stdout_sha256"])
        actual_files = ready["files"] if seed == 1729 else repair["files"]
        actual_source = AUDITOR if seed == 1729 else REPAIR_AUDITOR
        actual_receipt = (TEST_DIRECTORIES if seed == 1729 else REPAIR_TEST_DIRECTORIES)[
            "auditor"
        ] + "/test-receipt.json"
        integer(terminal["seed"], seed, "audit terminal seed")
        integer(terminal["exit_code"], 0, "audit terminal exit")
        require(
            terminal["terminal_completion_observed_by_primary"] is True
            and terminal["cuda_visible_devices"] == "-1",
            "observed CPU audit completion",
        )
        expected = {
            "summary_sha256": summary_sha,
            "execution_receipt_sha256": execution_sha,
            "auditor_sha256": actual_files[actual_source],
            "test_receipt_sha256": actual_files[actual_receipt],
            "audit_sha256": reader.inputs[str(folder / audit_name)],
        }
        for key, value in expected.items():
            exact(terminal[key], value, "audit terminal " + key)
        require(
            instant(execution["completed_utc"])
            <= instant(terminal["started_utc"])
            <= instant(terminal["completed_utc"]),
            "audit after completed execution",
        )
        require(
            checked["complete"] is True
            and checked["audit_passed"] is True
            and checked["acceptance"] is False,
            "completed independently audited predecessor",
        )
        for key, value in {
            "seed": seed,
            "summary_sha256": summary_sha,
            "execution_receipt_sha256": execution_sha,
            "script_sha256": actual_files[actual_source],
            "test_receipt_sha256": expected["test_receipt_sha256"],
            "plan_sha256": PLAN_SHA,
        }.items():
            exact(checked[key], value, "predecessor audit identity " + key)
        if seed == 1729:
            exact(checked["shared_readiness"], BASE_READINESS, "predecessor shared readiness")
        else:
            repair_identity(checked, repair, reader.inputs[str(ROOT / REPAIR_READINESS)])
    return execution, terminal


def adapter_contract(summary, context):
    partitions = {
        "train": context["reports"]["train-membership.json"]["queries"],
        "validation": context["reports"]["parent.json"]["responses"],
    }
    expected = {
        "identity": INPUT_ADAPTER,
        "amendment_sha256": AMENDMENT_SHA,
        "source_sha256": ADAPTER_SHA,
        "inherited_corpus_identity": True,
        "whole_corpus_revalidated": False,
        "evidence_sha256": {
            n: context["summary"]["artifact_sha256"][n]
            for n in ("train-membership.json", "parent.json", "child.json", "training.json")
        },
        "partition_sha256": {
            name: formatted_hash(
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
    exact(
        summary["input_adapter"], expected, "independently reconstructed selected-partition adapter"
    )
    return expected


def diagnostic_contract(reports, summary, context):
    training = reports["training.json"]
    report = reports["train-diagnostics.json"]
    expected = {
        "partition": "train",
        "descriptive_only": True,
        "used_for_selection_or_updates": False,
        "complete": True,
        "loss_callback_calls": 32001,
        "endpoints": {},
    }
    for step in TRAIN_POINTS:
        name = f"train-scores-post{step}.json"
        endpoint = diagnostic_endpoint(
            reports[name], reports["train-membership.json"]["queries"], step, MEMBERSHIP_SHA
        )
        endpoint.update(score_artifact=name, score_artifact_sha256=summary["artifact_sha256"][name])
        expected["endpoints"][str(step)] = endpoint
    exact(report, expected, "independently recomputed training diagnostics")
    exact(training["training_diagnostics"], report, "training diagnostic copy")
    exact(summary["training_diagnostic_updates"], [8000, 32000], "fixed train diagnostic endpoints")
    old = context["reports"]["training.json"]
    prefix = prefix_contract(
        training["history"],
        old["history"],
        training["initial_pre_update_loss"],
        old["initial_pre_update_loss"],
        old["final_post_update_loss"],
    )
    expected_prefix = {
        "descriptive_only": True,
        "used_for_selection_or_updates": False,
        "pre_update_compared_count": 8000,
        "mismatch_count": prefix["mismatch_count"],
        "first_mismatch_update": None
        if prefix["first_mismatch"] is None
        else prefix["first_mismatch"]["update"],
        "all_pre_update_equal": prefix["mismatch_count"] == 0,
        "historical_final_post_update_loss": prefix["historical_final_post_update_loss"],
        "current_pre_update_8001_loss": prefix["current_pre_update_8001_loss"],
        "boundary_loss_equal": prefix["boundary_loss_equal"],
    }
    exact(training["training_prefix_diagnostic"], expected_prefix, "descriptive loss prefix")
    return {
        "endpoints": {
            k: {n: v for n, v in row.items() if n != "rows"}
            for k, row in expected["endpoints"].items()
        },
        "prefix": prefix,
        "full_score_rows_recomputed": 3274,
        "used_for_quality_gate": False,
    }


def child_contract(summary, reports, context, seed):
    parent = reports["parent.json"]
    child = reports["child.json"]
    validation_report(parent, "parent", context["reports"]["parent.json"])
    rows = validation_report(child, "child", parent, parent)
    paired = BASE.paired(
        rows, [r["selected_set_ids"] for r in context["reports"]["child.json"]["responses"]]
    )
    result = {
        "aggregate": child["aggregate"],
        "groups": child["groups"],
        "paired_vs_parent_dense": child["paired_vs_parent_dense"],
        "paired_vs_historical8000": paired,
    }
    exact(summary["child"], result, "independently recomputed child summary")
    exact(
        reports["comparisons.json"],
        {
            "historical8000_summary_sha256": context["summary_sha256"],
            "historical8000_child_sha256": context["summary"]["artifact_sha256"]["child.json"],
            "comparison_scope": "saved historical output; no rerun",
            "paired_vs_historical8000": paired,
        },
        "saved comparator identity",
    )
    gate = seed_gate(child, context["reports"]["child.json"], summary["invariants"], seed)
    exact(summary["gate"], gate, "independent strict matching-seed gate")
    return result, gate, rows


def audit_seed(reader, seed, readiness_digest):
    seed_contract(seed)
    folder = ROOT / f"runs/learning/{CAMPAIGN}/seed-{seed}"
    ready, repair = repair_readiness_contract(reader, readiness_digest)
    preserved = preserved_history(reader)
    summary = reader.read(folder / "summary.json")
    summary_sha = reader.inputs[str(folder / "summary.json")]
    context = historical_context(reader, seed)
    recipe, ready = provenance(reader, seed, summary, folder, context, BASE_READINESS["sha256"])
    receipt_chain(reader, seed, summary_sha, ready, repair)
    continuation = (
        continuation_contract(reader, summary, repair, readiness_digest) if seed == 1731 else None
    )
    require(set(summary["artifact_sha256"]) == ARTIFACTS, "exact completed artifact inventory")
    reader.manifest({str(folder / n): h for n, h in summary["artifact_sha256"].items()})
    reports = {
        n: reader.read(folder / n, summary["artifact_sha256"][n])
        for n in ARTIFACTS
        if n.endswith(".json")
    }
    adapter = adapter_contract(summary, context)
    child_summary, gate, _ = child_contract(summary, reports, context, seed)
    zero = zero_replay_contract(reports["zero-replay.json"], reports["parent.json"])
    diagnostics = diagnostic_contract(reports, summary, context)
    parent = load_payload(reader, context["parent"], PARENTS[seed])
    child = load_payload(
        reader, folder / "checkpoint-final.pt", summary["artifact_sha256"]["checkpoint-final.pt"]
    )
    BASE.sidecar_matches(
        child,
        reports["checkpoint-final.pt.json"],
        summary["artifact_sha256"]["checkpoint-final.pt"],
    )
    checkpoint = training_contract(seed, summary, reports, recipe, context, parent, child)
    exact(summary["invariants"], dict.fromkeys(INVARIANTS, True), "verified execution invariants")
    reader.manifest(dict(reader.inputs))
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "seed": seed,
        "summary_sha256": summary_sha,
        "script_sha256": repair["files"][REPAIR_AUDITOR],
        "audit_revision": EVIDENCE_REVISION,
        "amendment_sha256": REPAIR_AMENDMENT_SHA,
        "base_auditor_sha256": BASE_AUDITOR_SHA,
        "base_readiness": dict(BASE_READINESS),
        "preserved_original_evidence": preserved,
        "continuation": continuation,
        "continuation_complete_sha256": None
        if continuation is None
        else continuation["continuation_complete_sha256"],
        "plan_sha256": PLAN_SHA,
        "test_receipt_sha256": repair["files"][
            REPAIR_TEST_DIRECTORIES["auditor"] + "/test-receipt.json"
        ],
        "execution_receipt_sha256": reader.inputs[str(folder / "execution-receipt.json")],
        "shared_readiness": {"path": REPAIR_READINESS, "sha256": readiness_digest},
        "child": child_summary,
        "gate": gate,
        "checkpoint": checkpoint,
        "training_diagnostics": diagnostics,
        "input_adapter": adapter,
        "zero_replay": zero,
        "protected_test_used": False,
        "input_sha256": dict(reader.inputs),
        "limitations": [
            "CPU inspection of authenticated saved payloads and logits; "
            "no neural or optimizer replay.",
            "Saved trace verifies update chronology and finite attestations, "
            "not independently reconstructed gradients.",
            "Training diagnostics and historical loss prefix are descriptive only.",
            "Validation-selected development screen; no held-out generalization, "
            "oracle parity or owner acceptance.",
        ],
    }


def audit_aggregate(reader, readiness_digest):
    ready, repair = repair_readiness_contract(reader, readiness_digest)
    preserved = preserved_history(reader)
    path = ROOT / f"runs/learning/{CAMPAIGN}/aggregate-v2.json"
    result = reader.read(path)
    require(result["complete"] is True and result["acceptance"] is False, "completed raw aggregate")
    require(
        result["campaign_version"] == CAMPAIGN
        and result["evaluator"] == EVALUATOR
        and result["plan_sha256"] == PLAN_SHA,
        "aggregate method identity",
    )
    require(
        result["script_sha256"] == repair["files"][REPAIR_AGGREGATOR]
        and result["base_aggregator_sha256"] == BASE_AGGREGATOR_SHA,
        "aggregate implementation identity",
    )
    require(
        result["protected_test_used"] is False and result["standard_serving_supported"] is False,
        "aggregate scope",
    )
    require(
        result["evidence_revision"] == EVIDENCE_REVISION
        and result["amendment_sha256"] == REPAIR_AMENDMENT_SHA,
        "aggregate evidence revision",
    )
    exact(result["base_readiness"], BASE_READINESS, "aggregate base admission")
    exact(
        result["shared_readiness"],
        {"path": REPAIR_READINESS, "sha256": readiness_digest},
        "aggregate repair admission",
    )
    reader.manifest(result["input_sha256"])
    expected_snapshots = {
        f"runs/learning/{CAMPAIGN}/aggregate-v2.script.py": repair["files"][REPAIR_AGGREGATOR],
        f"runs/learning/{CAMPAIGN}/aggregate-v2.readiness.json": readiness_digest,
    }
    inputs_path = f"runs/learning/{CAMPAIGN}/aggregate-v2.inputs.json"
    expected_snapshots[inputs_path] = result["snapshot_sha256"][inputs_path]
    exact(result["snapshot_sha256"], expected_snapshots, "aggregate snapshot inventory")
    reader.manifest(expected_snapshots)
    exact(
        reader.read(inputs_path, expected_snapshots[inputs_path]),
        result["input_sha256"],
        "aggregate input snapshot",
    )
    entries = []
    all_rows = []
    identity = None
    for seed in SEEDS:
        folder = path.parent / f"seed-{seed}"
        summary = reader.read(folder / "summary.json")
        summary_sha = reader.inputs[str(folder / "summary.json")]
        context = historical_context(reader, seed)
        reports = {
            n: reader.read(folder / n, summary["artifact_sha256"][n])
            for n in ("parent.json", "child.json", "comparisons.json")
        }
        child, gate, rows = child_contract(summary, reports, context, seed)
        exact(
            summary["shared_readiness"],
            BASE_READINESS,
            "aggregate seed admission",
        )
        receipt_chain(reader, seed, summary_sha, ready, repair, include_audit=True)
        audit_name, terminal_name, _ = audit_route(seed)
        checked = reader.read(folder / audit_name)
        exact(checked["child"], child, "aggregate audited metrics")
        exact(checked["gate"], gate, "aggregate audited gate")
        current = [
            {
                k: r[k]
                for k in (
                    "index",
                    "subject",
                    "dimension",
                    "prompt_ids",
                    "expected_set_ids",
                    "group",
                )
            }
            for r in rows
        ]
        if identity is not None:
            exact(current, identity, "cross-seed query/teacher identity")
        identity = current
        entry = {
            "seed": seed,
            "summary_sha256": summary_sha,
            "audit_sha256": reader.inputs[str(folder / audit_name)],
            "execution_receipt_sha256": reader.inputs[str(folder / "execution-receipt.json")],
            "audit_receipt_sha256": reader.inputs[str(folder / terminal_name)],
            "audit_path": f"runs/learning/{CAMPAIGN}/seed-{seed}/" + audit_name,
            "audit_receipt_path": f"runs/learning/{CAMPAIGN}/seed-{seed}/" + terminal_name,
            "auditor_sha256": checked["script_sha256"],
            "test_receipt_sha256": checked["test_receipt_sha256"],
            "child": child,
            "gate": gate,
        }
        if seed == 1731:
            entry.update(continuation_contract(reader, summary, repair, readiness_digest))
        entries.append(entry)
        all_rows.extend(rows)
    exact(result["per_seed"], entries, "aggregate per-seed identities and metrics")
    pooled = {
        "aggregate": totals(all_rows),
        "groups": {g: totals([r for r in all_rows if r["group"] == g]) for g in GROUPS},
    }
    exact(result["all_three"], pooled, "pooled 666 saved rows")
    checks = {str(e["seed"]): e["gate"]["primary_checks_passed"] for e in entries}
    gate = {
        "checks": checks,
        "primary_checks_passed": all(checks.values()),
        "accepted": False,
        "independent_audit_required": True,
    }
    exact(result["gate"], gate, "all-three conjunction")
    reader.manifest(dict(reader.inputs))
    return {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": reader.inputs[str(path)],
        "script_sha256": repair["files"][REPAIR_AUDITOR],
        "audit_revision": EVIDENCE_REVISION,
        "amendment_sha256": REPAIR_AMENDMENT_SHA,
        "base_auditor_sha256": BASE_AUDITOR_SHA,
        "base_readiness": dict(BASE_READINESS),
        "preserved_original_evidence": preserved,
        "plan_sha256": PLAN_SHA,
        "shared_readiness": {"path": REPAIR_READINESS, "sha256": readiness_digest},
        "test_receipt_sha256": repair["files"][
            REPAIR_TEST_DIRECTORIES["auditor"] + "/test-receipt.json"
        ],
        "per_seed": entries,
        "all_three": pooled,
        "gate": gate,
        "input_sha256": dict(reader.inputs),
        "protected_test_used": False,
        "limitations": [
            "Saved-row reaggregation of three previously completed CPU audits; "
            "no checkpoint or neural replay in aggregate mode.",
            "The same 222 validation queries recur across seeds; "
            "666 rows are not independent held-out examples.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int, choices=(1730, 1731))
    mode.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--readiness-sha256", required=True)
    args = parser.parse_args()
    require(PRODUCTION_READY, "development auditor is not frozen or ready")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only audit environment")
    require(Path.cwd().resolve() == ROOT, "repository root invocation")
    destination = HERE / (
        "aggregate-audit-v2.json"
        if args.aggregate_only
        else f"seed-{args.seed}/independent-audit-v2.json"
    )
    require(not destination.exists(), "immutable audit output")
    reader = EvidenceReader(ROOT, allowlist())
    result = {
        "complete": False,
        "audit_passed": False,
        "acceptance": False,
        "script_sha256": sha(__file__),
    }
    try:
        result = (
            audit_aggregate(reader, args.readiness_sha256)
            if args.aggregate_only
            else audit_seed(reader, args.seed, args.readiness_sha256)
        )
    except BaseException as exc:
        result.update(error=repr(exc), input_sha256=dict(reader.inputs))
        raise
    finally:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "audit_passed": True,
                "output": str(destination),
                "sha256": sha(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
