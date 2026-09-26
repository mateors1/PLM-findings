"""Independent saved-evidence audit of exact-real training feasibility.

No Torch, feature extraction, LP solve or primary arithmetic imports. NumPy reads
saved constants; FLINT accelerates independently constructed integer sign products.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.metadata
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from flint import fmpz_mat

ROOT = Path(__file__).resolve().parents[3]
PLAN_SHA = "7b2e575dfd651d10edbf2d8c2a5f4a9f3268d6ad350248fafd35f1e09efe694b"
LOCK_SHA = "15d540d62127862c19123e99fe1a7a9716f1f3f4ac7cd0824c8554ad42704e54"
PARENT_SHA = "e844dd73c3af4ae794033ed356f358bf0541dd2469002554242993e596dcd4e1"
MEMBERSHIP_SHA = "0c49346f61ab689ae4c0567f88ec96ce62da1735fedc97799ef2e368542fe5ad"
MEAN_SUMMARY_SHA = "0457ac78065c55b4786a436d436c4de6ce2cbad510056a1d9951aa744ab44a61"
MEAN_AUDIT_SHA = "6535d9e85920c1c0e316ff0f9f60f391629c1c74be90d095759d5954692cf3bc"
MEAN_DECISION_SHA = "da0bbc0bd9a1284c6dae6f072e46bcddf30b0c13cdd0a81e34328768042e90a6"
MEAN_HELPER_SHA = "79299541b9ccf3cebe1562194a3a9e85a1027bf69e0347701d92eedb8b7ee63c"
RECIPE_SHA = "05948f589834699de0f1f99538c739eee33242a04ee1e1c402d41295a13f74a4"
SOURCE_SHA = "1d74e018836cc4e87a8eb46428c0d710492b8dd90ba3b11231115711bcde5376"
CONFIGS_SHA = "51f06619d9b2b37d47dc8a3f7f84de1d292aec36481c64d4c48d0d854e44b970"
SPLIT_SHA = "b8329e2f73e44f8948d343c8bc84247fedf6d004f342f0cd37880551fbb2603d"
RUNNER_SHA = "62ae708fe094ac4bba2df44082cb1e25592a29c123fd1ac1e71c1cecf58eb531"
SOLVER_SHA = "ff94faaf7ad96765039ea9ebd24e0ea08c228c5bb6c7f87f5e263d5205ee923c"
PRIMARY_RECEIPT_SHA = "3cd4075dfb4f83923e09c1b162db3331726efb77f75b7f4785493a5b60e021af"
BASE = 1024
INPUTS = {}
RECIPE = {
    "experiment": "diagonal-feasibility-v1",
    "evaluator": "plm-diagonal-feasibility-v1",
    "dimensions": ["TYPE", "COLOR"],
    "train_queries": 1637,
    "products": 1025,
    "features": 256,
    "entity_base": 1024,
    "seed_queries": 2,
    "max_primal_lps": 16,
    "max_working_rows": 4096,
    "append_rows": 128,
    "max_certificate_support": 257,
    "max_certificate_bytes": 8388608,
    "worker_seconds": 180.0,
    "lp_seconds": 30.0,
    "method": "highs-ds",
    "presolve": True,
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
    "objective": "zero",
    "positive_rhs": 1.0,
    "negative_rhs": 0.0,
    "primal_bounds": "free",
    "exact_chunk_queries": 16,
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "python-flint": "0.8.0",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path, digest):
    path = Path(path).resolve()
    require(sha(path) == digest, "hash mismatch: " + str(path))
    INPUTS[str(path)] = digest
    return path


def read(path, digest=None):
    path = bind(path, digest or sha(path))
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def training_membership(records, vocabulary):
    """Use held-out keys only for split identity; inspect labels solely for train."""
    partitions = {"train": [], "validation": [], "test": []}
    training = []
    seen = set()
    for record in records:
        subject, dimension = record["subject"], record["dimension"]
        key = (subject, dimension)
        require(key not in seen and dimension in ("TYPE", "COLOR"), "corpus query identity")
        seen.add(key)
        number = int.from_bytes(
            hashlib.sha256(f"1729:{subject}:{dimension}".encode()).digest(), "big"
        ) / float(1 << 256)
        partition = "validation" if number < 0.1 else "test" if number < 0.2 else "train"
        partitions[partition].append([subject, dimension])
        if partition != "train":
            continue
        ids = record["input_ids"]
        require(
            len(ids) >= 7
            and ids[0] == 1
            and ids[2:5] == [32 if dimension == "TYPE" else 33, 34, 5]
            and ids[-1] == 2,
            "training record grammar",
        )
        require(
            vocabulary[ids[1]] == subject
            and all(type(v) is int and BASE <= v < len(vocabulary) for v in ids[5:-1]),
            "training entity mapping",
        )
        require(record["labels"] == [-100] * 5 + ids[5:], "training loss boundary")
        require(record["targets"] == [vocabulary[v] for v in ids[5:-1]], "training label mapping")
        truth = sorted(set(ids[5:-1]))
        require(truth and ids[1] not in truth, "training nonself truth")
        training.append(
            {
                "index": len(training),
                "subject": subject,
                "dimension": dimension,
                "prompt_ids": ids[:5],
                "expected_set_ids": truth,
            }
        )
    split = {
        "algorithm": "sha256-threshold-v2",
        "seed": 1729,
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        **partitions,
    }
    return {"partition": "train", "query_count": len(training), "queries": training}, canonical(
        split
    )


def feature_contract(entities, steering, array_manifest):
    require(entities.shape == (1025, 256) and steering.shape == (2, 256), "feature dimensions")
    for name, array in (("entities", entities), ("steering", steering)):
        require(
            array.dtype == np.float32 and np.isfinite(array).all(), "finite FP32 feature arrays"
        )
        expected = {
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        require(array_manifest[name] == expected, "array raw-byte receipt: " + name)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite numeric value")
    return value


def integers(array):
    """Convert dyadic constants to a common denominator via exact float ratios."""
    require(np.isfinite(array).all(), "nonfinite feature/vector")
    ratios = [float(value).as_integer_ratio() for value in array.flat]
    power = max((den.bit_length() - 1 for _, den in ratios), default=0)
    values = [num * (2**power // den) for num, den in ratios]
    return np.asarray(values, dtype=object).reshape(array.shape), power


def rank_audit(steering, witness):
    values, power = integers(steering)
    require(values.ndim == 2 and len(values) == 2, "two steering vectors")
    require(witness["steering_denominator_power"] == power, "steering scaling")
    first = None
    for j in range(values.shape[1]):
        for k in range(j + 1, values.shape[1]):
            determinant = int(values[0, j] * values[1, k] - values[0, k] * values[1, j])
            if determinant:
                first = (j, k, determinant)
                break
        if first is not None:
            break
    require(witness["rank_two"] is (first is not None), "exact steering rank")
    if first:
        require(witness["columns"] == list(first[:2]), "rank minor columns")
        require(int(witness["integer_determinant_hex"], 16) == first[2], "rank determinant")
        condition = float(np.linalg.cond(steering.astype(np.float64)))
        require(
            witness["condition_number_fp64"] == (condition if math.isfinite(condition) else None),
            "descriptive condition",
        )
        require(
            witness["condition_number_status"]
            == ("finite" if math.isfinite(condition) else "nonfinite_descriptive_value"),
            "descriptive condition status",
        )
    return first is not None


def queries_for(membership, dimension, products):
    result = []
    seen = set()
    require(membership["partition"] == "train", "training only")
    for index, row in enumerate(membership["queries"]):
        require(row["index"] == index, "ordered train index")
        prompt = row["prompt_ids"]
        require(
            prompt[0] == 1 and prompt[2:] == [32 if row["dimension"] == "TYPE" else 33, 34, 5],
            "prompt grammar",
        )
        subject = prompt[1] - BASE
        truth = row["expected_set_ids"]
        require(
            0 <= subject < products and truth == sorted(set(truth)) and truth,
            "subject/truth contract",
        )
        require(
            all(type(i) is int and BASE <= i < BASE + products and i != prompt[1] for i in truth),
            "nonself truth",
        )
        key = (subject, row["dimension"])
        require(row["dimension"] in ("TYPE", "COLOR") and key not in seen, "unique dimension query")
        seen.add(key)
        if row["dimension"] == dimension:
            result.append((index, subject, {i - BASE for i in truth}))
    return result


def exact_signs(entity_integers, queries, relation_hex, entities=None):
    relation = np.asarray([float.fromhex(v) for v in relation_hex], dtype=np.float64)
    require(relation.shape == (entity_integers.shape[1],), "relation shape")
    coefficients, power = integers(relation)
    transposed = fmpz_mat(entity_integers.T.tolist())
    checked = 0
    failure = None
    hidden = None
    for offset in range(0, len(queries), 16):
        chunk = queries[offset : offset + 16]
        left = fmpz_mat(
            [
                [int(e * c) for e, c in zip(entity_integers[subject], coefficients, strict=True)]
                for _, subject, _ in chunk
            ]
        )
        result = left * transposed
        for local, (index, subject, truth) in enumerate(chunk):
            floating = None
            for product in range(len(entity_integers)):
                if product == subject:
                    continue
                score = int(result[local, product])
                checked += 1
                invalid = score <= 0 if product in truth else score > 0
                if invalid and failure is None:
                    failure = {
                        "train_index": index,
                        "product_id": BASE + product,
                        "positive": product in truth,
                        "integer_score_hex": hex(score),
                    }
                if invalid and entities is not None:
                    if floating is None:
                        floating = (
                            (entities.astype(np.float64) * entities[subject].astype(np.float64))
                            @ relation
                            / 16
                        )
                        require(np.isfinite(floating).all(), "finite failure residuals")
                    residual = (
                        1 - float(floating[product])
                        if product in truth
                        else float(floating[product])
                    )
                    if residual <= 0 and hidden is None:
                        hidden = {
                            "train_index": index,
                            "product_id": BASE + product,
                            "residual": residual,
                        }
    return {
        "passed": failure is None,
        "checked_constraints": checked,
        "relation_denominator_power": power,
        "first_failure": failure,
        "unreported_fp64_failure": hidden,
    }


def certificate_audit(entity_integers, exponent, queries, certificate, working, dual_hex):
    """Rebind every row, then prove the Farkas equation using Python integers."""
    require(certificate["nullity"] == 1 and certificate["exact_zero"] is True, "certificate claims")
    require(certificate["entity_denominator_power"] == exponent, "entity scaling")
    dual = [float.fromhex(v) for v in dual_hex]
    require(len(dual) == len(working) and all(math.isfinite(v) for v in dual), "dual proposal")
    expected = [r for r, value in zip(working, dual, strict=True) if value > 0]
    support = certificate["support"]
    require(0 < len(support) <= 257 and len(support) == len(expected), "all positive dual support")
    lookup = {index: (subject, truth) for index, subject, truth in queries}
    total = [0] * entity_integers.shape[1]
    positive_sum = 0
    multipliers = []
    rows = []
    for entry, expected_entry in zip(support, expected, strict=True):
        require(
            {k: entry[k] for k in ("train_index", "product_id")} == expected_entry,
            "certificate support binding",
        )
        index, product = entry["train_index"], entry["product_id"] - BASE
        require(index in lookup and 0 <= product < len(entity_integers), "certificate row bounds")
        subject, truth = lookup[index]
        require(
            product != subject and entry["positive"] is (product in truth), "certificate true label"
        )
        multiplier = int(entry["multiplier_hex"], 16)
        require(entry["multiplier_hex"] == hex(multiplier), "canonical hexadecimal multiplier")
        require(multiplier >= 0, "negative certificate multiplier")
        multipliers.append(multiplier)
        sign = 1 if product in truth else -1
        row = [
            sign * int(a) * int(b)
            for a, b in zip(entity_integers[subject], entity_integers[product], strict=True)
        ]
        rows.append(row)
        total = [a + multiplier * b for a, b in zip(total, row, strict=True)]
        if sign == 1:
            positive_sum += multiplier
    require(math.gcd(*multipliers) == 1, "primitive normalized certificate")
    require(
        all(value == 0 for value in total) and positive_sum > 0, "exact Farkas equality/positivity"
    )
    require(
        int(certificate["positive_label_multiplier_sum_hex"], 16) == positive_sum,
        "positive label sum",
    )
    # Rank is a separate check of the declared one-dimensional recovery rule.
    require(len(rows) - fmpz_mat(rows).rank() == 1, "certificate nullity")
    return {
        "support_count": len(rows),
        "exact_zero": True,
        "positive_label_multiplier_sum_hex": hex(positive_sum),
    }


def certificate_attempt_audit(entity_integers, queries, iteration):
    attempt = iteration["certificate_attempt"]
    proposal = [float.fromhex(v) for v in iteration["dual_hex"]]
    require(all(math.isfinite(v) for v in proposal), "finite certificate proposal")
    support = [i for i, value in enumerate(proposal) if value > 0]
    require(attempt["support_indices"] == support, "exact proposed support inventory")
    if "nullity" not in attempt:
        return
    lookup = {index: (subject, truth) for index, subject, truth in queries}
    rows = []
    for i in support:
        row = iteration["working_rows"][i]
        subject, truth = lookup[row["train_index"]]
        product = row["product_id"] - BASE
        require(
            product != subject and 0 <= product < len(entity_integers), "attempt product bounds"
        )
        sign = 1 if product in truth else -1
        rows.append(
            [
                sign * int(a) * int(b)
                for a, b in zip(entity_integers[subject], entity_integers[product], strict=True)
            ]
        )
    require(attempt["nullity"] == len(rows) - fmpz_mat(rows).rank(), "attempt exact nullity")
    if "oriented_integer_vector_hex" in attempt:
        vector = [int(v, 16) for v in attempt["oriented_integer_vector_hex"]]
        require(
            len(vector) == len(rows)
            and math.gcd(*vector) == 1
            and next(v for v in vector if v) > 0,
            "attempt normalized orientation",
        )
        require(
            all(
                sum(vector[i] * rows[i][j] for i in range(len(rows))) == 0
                for j in range(entity_integers.shape[1])
            ),
            "attempt integer nullvector",
        )


def outcome(statuses, rank_two):
    require(
        all(v in ("certified_feasible", "certified_infeasible", "inconclusive") for v in statuses),
        "known statuses",
    )
    if "certified_infeasible" in statuses:
        return "certified_infeasible"
    if rank_two and statuses == ["certified_feasible", "certified_feasible"]:
        return "certified_feasible"
    return "inconclusive"


def additions(entities, queries, relation_hex, working, limit):
    relation = np.asarray([float.fromhex(v) for v in relation_hex], dtype=np.float64)
    require(np.isfinite(relation).all(), "finite proposed relation")
    selected = {(row["train_index"], row["product_id"]) for row in working}
    candidates = []
    features = entities.astype(np.float64)
    for index, subject, truth in queries:
        scores = (features * features[subject]) @ relation / 16
        require(np.isfinite(scores).all(), "finite floating residual scan")
        for product, value in enumerate(scores):
            if product == subject or (index, BASE + product) in selected:
                continue
            violation = max(0.0, 1.0 - float(value)) if product in truth else max(0.0, float(value))
            if violation > 0:
                candidates.append((-violation, index, BASE + product))
        candidates = heapq.nsmallest(limit, candidates)
    return [
        {"train_index": i, "product_id": p, "violation": -negative}
        for negative, i, p in sorted(candidates)
    ]


def iteration_contract(entities, queries, iterations):
    """Verify bounded, deterministic row acquisition, independent of LP claims."""
    require(len(iterations) <= 16, "primal LP count")
    require(len(queries) >= 2, "two dimension seed queries")
    expected = [
        {"train_index": index, "product_id": BASE + product}
        for index, subject, _ in queries[:2]
        for product in range(len(entities))
        if product != subject
    ]
    dual_count = 0
    for number, record in enumerate(iterations, 1):
        require(
            record["iteration"] == number and record["working_rows"] == expected,
            "working-row order/prefix",
        )
        require(
            len(expected) <= 4096
            and len({(r["train_index"], r["product_id"]) for r in expected}) == len(expected),
            "working count/uniqueness",
        )
        require(type(record["status"]) is int and 0 <= record["status"] <= 4, "LP numerical status")
        if "dual_status" in record:
            dual_count += 1
            require(
                record["status"] == 2 and dual_count == 1 and number == len(iterations),
                "one terminal dual proposal",
            )
        if "relation_hex" in record:
            require(
                record["status"] == 0 and len(record["relation_hex"]) == entities.shape[1],
                "primal proposal shape/status",
            )
            require(
                all(math.isfinite(float.fromhex(v)) for v in record["relation_hex"]),
                "primal finite",
            )
        if "added" in record:
            require(record["status"] == 0 and "relation_hex" in record, "cut proposal context")
            computed = additions(
                entities, queries, record["relation_hex"], expected, min(128, 4096 - len(expected))
            )
            require(record["added"] == computed, "cut order/positive violations")
            expected += [{k: row[k] for k in ("train_index", "product_id")} for row in computed]
        elif number < len(iterations):
            raise ValueError("continued without appended rows")
    return {
        "primal_lps": len(iterations),
        "dual_lps": dual_count,
        "maximum_working_rows": max((len(r["working_rows"]) for r in iterations), default=0),
    }


def hash_manifest(mapping):
    require(isinstance(mapping, dict) and mapping, "nonempty hash manifest")
    for path, digest in mapping.items():
        bind(path, digest)


def archive_inventory(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            name = entry.filename
            parts = name.split("/")
            require(
                entry.orig_filename == name
                and not entry.is_dir()
                and all(p not in ("", ".", "..") for p in parts)
                and "\\" not in name
                and ":" not in name
                and name.casefold() not in {k.casefold() for k in result},
                "archive path",
            )
            result[name] = hashlib.sha256(archive.read(name)).hexdigest()
    return result


def provenance(summary_path, summary):
    require(
        summary["complete"] is True
        and summary["final_identity_check"] is True
        and summary["acceptance"] is False,
        "complete unaccepted diagnostic",
    )
    require(
        summary["campaign_version"] == "diagonal-feasibility-v1"
        and summary["evaluator"] == "plm-diagonal-feasibility-v1",
        "diagnostic identity",
    )
    require(
        summary["plan_sha256"] == PLAN_SHA and summary["recipe_sha256"] == RECIPE_SHA,
        "declared plan/recipe",
    )
    require(summary["script_sha256"] == RUNNER_SHA, "frozen runner source")
    require(
        all(
            summary[k] == 0
            for k in (
                "validation_predictions",
                "protected_test_predictions",
                "neural_checkpoints_created",
            )
        ),
        "training-only diagnostic scope",
    )
    hash_manifest(summary["input_sha256"])
    hash_manifest(summary["snapshot_sha256"])
    folder = summary_path.parent
    require(
        read(summary_path.with_suffix(".inputs.json")) == summary["input_sha256"], "input snapshot"
    )
    snapshot_names = {
        "script.py",
        "solver.py",
        "solver-pyproject.toml",
        "solver-uv.lock",
        "plan.md",
        "recipe.json",
        "mean-helper.py",
        "helper.py",
        "source.zip",
        "configs.zip",
        "test-receipt.json",
        "test-stdout.txt",
        "test-main.py",
        "test-solver.py",
        "inputs.json",
    }
    require(
        set(summary["snapshot_sha256"])
        == {str(summary_path.with_suffix("." + name)) for name in snapshot_names},
        "snapshot inventory",
    )
    for suffix, digest in (
        ("plan.md", PLAN_SHA),
        ("recipe.json", RECIPE_SHA),
        ("mean-helper.py", MEAN_HELPER_SHA),
        ("source.zip", SOURCE_SHA),
        ("configs.zip", CONFIGS_SHA),
        ("solver-uv.lock", LOCK_SHA),
        ("script.py", summary["script_sha256"]),
        ("solver.py", SOLVER_SHA),
        ("test-receipt.json", PRIMARY_RECEIPT_SHA),
    ):
        bind(summary_path.with_suffix("." + suffix), digest)
    require(read(summary_path.with_suffix(".recipe.json")) == RECIPE, "fixed solver recipe")
    # Primary focused test receipt binds both isolated and host implementations.
    receipt = read(summary_path.with_suffix(".test-receipt.json"))
    require(
        receipt["passed"] is True
        and receipt["real_data_executed"] is False
        and receipt["gpu_used"] is False
        and receipt["tested_script_sha256"] == summary["script_sha256"],
        "primary synthetic receipt",
    )
    required = {
        ROOT / "scripts/diagnose_diagonal_feasibility.py": "script.py",
        ROOT / "experiments/diagonal-feasibility/solver.py": "solver.py",
        ROOT / "experiments/diagonal-feasibility/pyproject.toml": "solver-pyproject.toml",
        ROOT / "experiments/diagonal-feasibility/uv.lock": "solver-uv.lock",
        ROOT / "configs/experiments/diagonal_feasibility_v1.json": "recipe.json",
        ROOT / "tests/unit/test_diagonal_feasibility.py": "test-main.py",
        ROOT / "experiments/diagonal-feasibility/test_solver.py": "test-solver.py",
    }
    require(
        set(receipt["tested_files_sha256"]) == {str(p) for p in required}, "primary tested files"
    )
    for path, suffix in required.items():
        digest = receipt["tested_files_sha256"][str(path)]
        require(
            summary["input_sha256"].get(str(path)) == digest
            and sha(summary_path.with_suffix("." + suffix)) == digest,
            "tested snapshot/input binding",
        )
    require(
        sha(summary_path.with_suffix(".test-stdout.txt")) == receipt["stdout"]["sha256"]
        and summary["input_sha256"].get(str(Path(receipt["stdout"]["path"]).resolve()))
        == receipt["stdout"]["sha256"],
        "test stdout receipt",
    )
    old_dir = ROOT / "runs/learning/projection-only-refit-v1"
    old = read(old_dir / "summary.json", MEAN_SUMMARY_SHA)
    audit = read(old_dir / "independent-audit.json", MEAN_AUDIT_SHA)
    decision = read(old_dir / "decision.json", MEAN_DECISION_SHA)
    require(
        old["complete"] is True
        and audit["audit_passed"] is True
        and audit["summary_sha256"] == MEAN_SUMMARY_SHA
        and decision["evidence_accepted"] is True
        and decision["summary_sha256"] == MEAN_SUMMARY_SHA
        and decision["audit_sha256"] == MEAN_AUDIT_SHA,
        "accepted ancestor evidence",
    )
    require(
        all(
            summary["input_sha256"].get(path) == digest
            for path, digest in old["input_sha256"].items()
        ),
        "inherited input bindings",
    )
    state = read(old_dir / "training.json", old["artifact_sha256"]["training.json"])["state_before"]
    require(len(state) == 93, "original state inventory")
    bind(ROOT / "runs/national_dex_continuation_control_s1729_v1/checkpoint-final.pt", PARENT_SHA)
    saved_membership = read(folder / "train-membership.json", MEMBERSHIP_SHA)
    corpus = ROOT / "data/processed/pokemon_v1_f1541479_20260924"
    manifest = read(corpus / "manifest.json")
    records_path = bind(corpus / "records.jsonl", manifest["records_hash"])
    vocabulary = read(corpus / "vocabulary.json")["tokens"]
    for path in (corpus / "manifest.json", corpus / "vocabulary.json", records_path):
        require(summary["input_sha256"].get(str(path)) == sha(path), "authenticated corpus input")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    membership, split = training_membership(records, vocabulary)
    require(
        membership == saved_membership and membership["query_count"] == 1637 and split == SPLIT_SHA,
        "exact original train partition",
    )
    artifacts = summary["artifact_sha256"]
    require(
        summary["generated_input_sha256"]
        == {
            str(folder / name): artifacts[name]
            for name in ("features.npz", "extraction.json", "train-membership.json")
        },
        "generated inputs sealed before workers",
    )
    actual_names = {
        p.name
        for p in folder.iterdir()
        if p.is_file()
        and (
            p.name.startswith(("TYPE.", "COLOR."))
            or p.name in {"features.npz", "extraction.json", "train-membership.json"}
        )
    }
    require(set(artifacts) == actual_names, "artifact inventory")
    for name, digest in artifacts.items():
        bind(folder / name, digest)
    extraction = read(folder / "extraction.json")
    require(extraction == summary["extraction"], "extraction snapshot")
    require(
        extraction["parent_checkpoint_sha256"] == PARENT_SHA
        and extraction["state_before"] == extraction["state_after"] == state
        and extraction["parent_embedding_tensor"] == state["token_embedding.weight"],
        "original frozen tensor identity",
    )
    require(
        extraction["product_token_ids"] == list(range(1024, 2049))
        and extraction["dimensions"] == ["TYPE", "COLOR"]
        and extraction["dimension_token_ids"] == [32, 33],
        "feature column identity",
    )
    require(
        extraction["transformer_forwards"] == extraction["optimizer_updates"] == 0
        and extraction["training_membership_replayed"] is True,
        "feature-only execution receipt",
    )
    require(extraction["environment"] == old["environment"], "extraction environment")
    expected_settings = {
        k: v
        for k, v in old["numerical_settings"].items()
        if k not in ("cudnn_deterministic", "cudnn_benchmark")
    }
    require(extraction["numerical_settings"] == expected_settings, "archived numerical settings")
    inventory = archive_inventory(summary_path.with_suffix(".source.zip"))
    config_inventory = archive_inventory(summary_path.with_suffix(".configs.zip"))
    require(not (set(inventory) & set(config_inventory)), "disjoint runtime archives")
    inventory.update(config_inventory)
    require(
        extraction["runtime_files_sha256"]
        == {str(folder / "runtime" / name): digest for name, digest in inventory.items()},
        "archived runtime inventory",
    )
    for name, digest in inventory.items():
        bind(folder / "runtime" / name, digest)
    require(extraction["module_origins"], "runtime module origins")
    for name, entry in extraction["module_origins"].items():
        path = Path(entry["path"])
        relative = path.resolve().relative_to((folder / "runtime").resolve()).as_posix()
        require(name == "plm" or name.startswith("plm."), "archived module namespace")
        require(inventory[relative] == entry["sha256"], "import origin binding")
    with np.load(
        bind(folder / "features.npz", extraction["file_sha256"]), allow_pickle=False
    ) as arrays:
        require(set(arrays.files) == {"entities", "steering"}, "feature file arrays")
        entities, steering = arrays["entities"], arrays["steering"]
    feature_contract(entities, steering, extraction["arrays"])
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "python-flint")}
    require(
        versions == {p: RECIPE[p] for p in versions} and sys.version_info[:2] == (3, 12),
        "audit isolated versions",
    )
    require(
        summary["solver_environment"]["versions"] == versions
        and summary["solver_environment"]["python"][:2] == [3, 12],
        "primary solver environment",
    )
    return membership, entities, steering


def trace_events(path, terminated=False):
    events = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            require(terminated and index == len(lines) - 1, "malformed nonterminal trace record")
    starts = [r for r in events if r["phase"] == "primal_start"]
    duals = [r for r in events if r["phase"] == "dual_start"]
    require(len(starts) <= 16 and len(duals) <= 1, "trace LP budgets")
    for entry in starts + duals:
        require(
            0 < finite(entry["time_limit"]) <= 30 and 0 < entry["working_rows"] <= 4096,
            "LP time/row limits",
        )
    return events


def partial_trace_audit(entities, queries, events):
    """Check completed proposals retained before termination without inventing a result."""
    records = []
    for event in events:
        phase = event["phase"]
        if phase == "primal_result":
            if records and "relation_hex" in records[-1]:
                previous = records[-1]
                previous["added"] = additions(
                    entities,
                    queries,
                    previous["relation_hex"],
                    previous["working_rows"],
                    min(128, 4096 - len(previous["working_rows"])),
                )
            records.append({k: v for k, v in event.items() if k != "phase"})
        elif phase == "primal_vector":
            require(records, "vector without completed primal")
            records[-1]["relation_hex"] = event["relation_hex"]
        elif phase == "dual_result":
            require(records, "dual without primal")
            records[-1]["dual_status"] = event["status"]
    return iteration_contract(entities, queries, records)


def dimension_audit(summary_path, summary, dimension, receipt, membership, entities, steering):
    folder = summary_path.parent
    process_path = folder / f"{dimension}.process.json"
    process = read(process_path)
    require(receipt == {**process, "dimension": dimension}, "parent process receipt")
    require(
        process["reaped"] is True
        and process["process_running"] is False
        and process["hard_limit_seconds"] == 180,
        "bounded reaped worker",
    )
    require(
        type(process["timed_out"]) is bool and finite(process["elapsed_seconds"]) >= 0,
        "process timing",
    )
    require(
        not any(k in process for k in ("parent_error", "cleanup_error", "launch_failed")),
        "unexpected parent failure",
    )
    require(0 <= finite(process["wait_budget_seconds"]) <= 180, "parent remaining wall budget")
    command = process["command"]
    expected_command = [
        str(ROOT / "experiments/diagonal-feasibility/.venv/Scripts/python.exe"),
        "-I",
        str(summary_path.with_suffix(".solver.py")),
        "--features",
        str(folder / "features.npz"),
        "--membership",
        str(folder / "train-membership.json"),
        "--recipe",
        str(summary_path.with_suffix(".recipe.json")),
        "--dimension",
        dimension,
        "--out",
        str(folder / f"{dimension}.json"),
    ]
    require(command == expected_command, "isolated fixed worker invocation")
    queries = queries_for(membership, dimension, len(entities))
    coverage = {"query_count": len(queries), "constraint_count": len(queries) * 1024}
    trace = folder / f"{dimension}.trace.jsonl"
    events = trace_events(trace, process["timed_out"]) if trace.exists() else []
    result = {
        "dimension": dimension,
        **coverage,
        "process_sha256": sha(process_path),
        "trace_sha256": sha(trace) if trace.exists() else None,
        "status": process["status"],
        "hard_timeout": process["timed_out"],
    }
    if process["timed_out"]:
        require(
            process["status"] == "inconclusive" and "report_sha256" not in process,
            "timeout is inconclusive",
        )
        # A killed worker may leave a complete unconsumed report, never upgrade it.
        result["mathematical_witness_verified"] = False
        result["execution_status"] = "hard_timeout"
        result["partial_work"] = partial_trace_audit(entities, queries, events)
        return result
    if process.get("report_error") or process["exit_code"] != 0:
        require(
            process["status"] == "inconclusive" and "report_sha256" not in process,
            "software failure cannot certify",
        )
        result.update(
            execution_status="worker_software_failure",
            mathematical_witness_verified=False,
            error=process.get("report_error", "nonzero exit"),
        )
        return result
    require(process["exit_code"] == 0, "normal worker exit")
    report_path = folder / f"{dimension}.json"
    report = read(report_path, process["report_sha256"])
    require(
        report["status"] == process["status"]
        and report["complete"] is process["report_complete"]
        and report["dimension"] == dimension,
        "worker result binding",
    )
    require(
        report["final_identity_check"] is True and report["torch_imported"] is False,
        "unchanged CPU solver",
    )
    if report["complete"] is not True:
        require(
            report["status"] == "inconclusive" and "error" in report, "incomplete software failure"
        )
        result.update(
            execution_status="worker_software_failure",
            mathematical_witness_verified=False,
            error=report["error"],
        )
        return result
    require("error" not in report, "complete report error")
    result["execution_status"] = "completed"
    expected_inputs = {
        str(p): sha(p)
        for p in (
            folder / "features.npz",
            folder / "train-membership.json",
            summary_path.with_suffix(".recipe.json"),
            summary_path.with_suffix(".solver.py"),
        )
    }
    require(report["input_sha256"] == expected_inputs, "worker feature/membership/source binding")
    require(
        report["versions"] == summary["solver_environment"]["versions"]
        and report["python"].startswith("3.12."),
        "worker version receipt",
    )
    require(
        finite(report["elapsed_seconds"]) >= 0
        and events[-1]
        == {
            "phase": "finished",
            "status": report["status"],
            "elapsed_seconds": report["elapsed_seconds"],
        },
        "terminal trace binding",
    )
    if report.get("reason") == "worker_budget" and (
        "rank_witness" not in report or "entity_denominator_power" not in report
    ):
        require(
            report["status"] == "inconclusive"
            and "budget_phase" in report
            and "certificate" not in report
            and "witness" not in report,
            "early budget outcome",
        )
        result.update(
            reason="worker_budget",
            mathematical_witness_verified=False,
            partial_work=partial_trace_audit(entities, queries, events),
        )
        if "rank_witness" in report:
            result["rank_two"] = rank_audit(steering, report["rank_witness"])
        return result
    rank_two = rank_audit(steering, report["rank_witness"])
    result.update(report_sha256=sha(report_path), rank_two=rank_two)
    if not rank_two:
        require(
            report["status"] == "inconclusive"
            and report["reason"] == "steering_rank_not_two"
            and not report["iterations"],
            "rank reduction unavailable",
        )
        return result
    require(
        all(report[k] == v for k, v in coverage.items()), "per-dimension full training coverage"
    )
    entity_int, exponent = integers(entities)
    require(report["entity_denominator_power"] == exponent, "worker feature scaling")
    iterations = report["iterations"]
    result["work"] = iteration_contract(entities, queries, iterations)
    primal_events = [e for e in events if e["phase"] == "primal_result"]
    require(len(primal_events) == len(iterations), "every LP result retained")
    exact_events = [e for e in events if e["phase"] == "exact_verification"]
    exact_index = 0
    for position, (iteration, event) in enumerate(zip(iterations, primal_events, strict=True)):
        require(
            all(
                event[k] == iteration[k] for k in ("iteration", "working_rows", "status", "message")
            ),
            "LP trace/report binding",
        )
        if "certificate_attempt" in iteration:
            certificate_attempt_audit(entity_int, queries, iteration)
            require(
                {"phase": "certificate_attempt", **iteration["certificate_attempt"]} in events,
                "certificate attempt trace",
            )
        if "dual_hex" in iteration:
            require(
                {
                    "phase": "dual_result",
                    "status": iteration["dual_status"],
                    "dual_hex": iteration["dual_hex"],
                }
                in events,
                "dual vector trace",
            )
        if "relation_hex" in iteration:
            require(
                {"phase": "primal_vector", "relation_hex": iteration["relation_hex"]} in events,
                "primal vector trace",
            )
        if "exact_verification" in iteration:
            verified = exact_signs(entity_int, queries, iteration["relation_hex"], entities)
            require(iteration["exact_verification"] == verified, "independent full exact sign scan")
            require(
                exact_events[exact_index] == {"phase": "exact_verification", **verified},
                "sign trace binding",
            )
            exact_index += 1
            if verified["unreported_fp64_failure"] is not None:
                require(
                    position == len(iterations) - 1
                    and report["status"] == "inconclusive"
                    and report["reason"] == "exact_failure_hidden_by_fp64"
                    and "added" not in iteration,
                    "hidden exact failure stops immediately",
                )
    require(exact_index == len(exact_events), "exact trace count")
    status = report["status"]
    if status == "certified_feasible":
        ref = report["witness"]
        require(Path(ref["path"]) == folder / f"{dimension}.witness.json", "witness path")
        witness = read(ref["path"], ref["sha256"])
        last = iterations[-1]
        require(
            witness
            == {
                "relation_hex": last["relation_hex"],
                "entity_denominator_power": exponent,
                "exact_verification": last["exact_verification"],
            }
            and witness["exact_verification"]["passed"] is True,
            "complete feasible witness",
        )
        result.update(
            witness_sha256=ref["sha256"],
            mathematical_witness_verified=True,
            exact_checked_constraints=witness["exact_verification"]["checked_constraints"],
        )
    elif status == "certified_infeasible":
        ref = report["certificate"]
        require(
            Path(ref["path"]) == folder / f"{dimension}.certificate.json"
            and Path(ref["path"]).stat().st_size <= 8388608,
            "bounded certificate artifact",
        )
        certificate = read(ref["path"], ref["sha256"])
        last = iterations[-1]
        require(
            last["status"] == 2 and last["dual_status"] == 0, "infeasible proposal plus solved dual"
        )
        result["certificate"] = certificate_audit(
            entity_int, exponent, queries, certificate, last["working_rows"], last["dual_hex"]
        )
        result.update(certificate_sha256=ref["sha256"], mathematical_witness_verified=True)
    else:
        require(status == "inconclusive" and "reason" in report, "explicit inconclusive reason")
        require(
            "certificate" not in report and "witness" not in report, "no inconclusive promotion"
        )
        result.update(reason=report["reason"], mathematical_witness_verified=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    path = args.summary.resolve()
    out = path.parent / "independent-audit.json"
    require(not out.exists(), "immutable audit output")
    receipt_path = ROOT / "runs/learning/diagonal-feasibility-auditor-tests-v1/test-receipt.json"
    receipt = read(receipt_path)
    require(
        receipt["passed"] is True
        and receipt["exit_code"] == 0
        and receipt["real_data_executed"] is False
        and receipt["gpu_used"] is False
        and receipt["tested_auditor_sha256"] == sha(__file__),
        "frozen auditor synthetic receipt",
    )
    bind(receipt_path.parent / "test_auditor.py", receipt["tests_sha256"])
    bind(receipt_path.parent / "stdout.txt", receipt["stdout_sha256"])
    summary = read(path)
    membership, entities, steering = provenance(path, summary)
    workers = summary["workers"]
    require(
        [r["dimension"] for r in workers] == ["TYPE", "COLOR"], "single ordered dimension workers"
    )
    dimensions = [
        dimension_audit(path, summary, dimension, receipt, membership, entities, steering)
        for dimension, receipt in zip(("TYPE", "COLOR"), workers, strict=True)
    ]
    require(
        sum(r["constraint_count"] for r in dimensions) == 1676288, "complete constraint inventory"
    )
    overall = outcome(
        [r["status"] for r in dimensions], all(r.get("rank_two", False) for r in dimensions)
    )
    require(summary["outcome"] == overall, "independent outcome reduction")
    require(
        all(sha(p) == digest for p, digest in INPUTS.items()),
        "final authenticated input invariance",
    )
    result = {
        "complete": True,
        "audit_passed": True,
        "acceptance": False,
        "summary_sha256": sha(path),
        "script_sha256": sha(__file__),
        "test_receipt_sha256": sha(receipt_path),
        "plan_sha256": PLAN_SHA,
        "input_sha256": INPUTS,
        "per_dimension": dimensions,
        "outcome": overall,
        "training_queries": 1637,
        "constraint_count": 1676288,
        "limitations": [
            "Saved-evidence audit; no independent CUDA normalization or LP execution.",
            "Exact signs use independent integer matrices with the same pinned FLINT library.",
            "Farkas equalities use independent Python integer arithmetic; nullity uses FLINT rank.",
            "Timing/options are authenticated receipts; chronology is not independently replayed.",
            "Exact-real constants do not bound FP32 rounding behavior or held-out quality.",
        ],
        "torch_imported": "torch" in sys.modules,
    }
    require(result["torch_imported"] is False, "Torch-free independent audit")
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "audit_passed": True,
                "outcome": overall,
                "audit_sha256": sha(out),
                "per_dimension": dimensions,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
