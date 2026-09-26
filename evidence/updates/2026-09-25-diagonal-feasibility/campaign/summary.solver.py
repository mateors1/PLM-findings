"""Bounded numerical proposals with exact integer witnesses; isolated CPU worker.

No checkpoint loading, Torch, validation prediction or model updates. All saved
coefficients describe the exact-real family of authenticated quantized features.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from flint import fmpz_mat
from scipy.optimize import linprog


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write(path, value, maximum=None):
    data = _bytes(value)
    _require(maximum is None or len(data) <= maximum, "artifact size limit")
    with Path(path).open("xb") as stream:
        stream.write(data)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer_array(array):
    """A common positive power-of-two denominator, without rounding constants."""
    _require(np.isfinite(array).all(), "nonfinite exact constant")
    ratios = [float(v).as_integer_ratio() for v in array.flat]
    exponent = max((d.bit_length() - 1 for _, d in ratios), default=0)
    integers = [n << (exponent - (d.bit_length() - 1)) for n, d in ratios]
    return np.asarray(integers, dtype=object).reshape(array.shape), exponent


def _rank_witness(steering):
    _require(steering.ndim == 2 and steering.shape[0] == 2, "two steering rows")
    integers, exponent = _integer_array(steering)
    for j in range(steering.shape[1]):
        for k in range(j + 1, steering.shape[1]):
            determinant = int(integers[0, j] * integers[1, k] - integers[0, k] * integers[1, j])
            if determinant:
                condition = float(np.linalg.cond(steering.astype(np.float64)))
                return {
                    "rank_two": True,
                    "columns": [j, k],
                    "integer_determinant_hex": hex(determinant),
                    "steering_denominator_power": exponent,
                    "condition_number_fp64": condition if math.isfinite(condition) else None,
                    "condition_number_status": "finite"
                    if math.isfinite(condition)
                    else "nonfinite_descriptive_value",
                }
    return {"rank_two": False, "steering_denominator_power": exponent}


def _queries(membership, dimension, products, base=1024):
    result = []
    seen = set()
    for index, row in enumerate(membership["queries"]):
        _require(row["index"] == index, "training order")
        prompt = row["prompt_ids"]
        subject = prompt[1] - base
        truth = row["expected_set_ids"]
        _require(
            0 <= subject < products
            and len(truth) == len(set(truth))
            and truth
            and all(type(v) is int and base <= v < base + products for v in truth)
            and subject + base not in truth,
            "training membership contract",
        )
        key = (subject, row["dimension"])
        _require(key not in seen, "duplicate training query")
        seen.add(key)
        if row["dimension"] == dimension:
            result.append((index, subject, {v - base for v in truth}))
    _require(len(result) >= 2, "two seed queries required")
    return result


def _row(entities, query, product):
    index, subject, truth = query
    _require(product != subject and 0 <= product < len(entities), "nonself product")
    positive = product in truth
    signed = entities[subject].astype(np.float64) * entities[product].astype(np.float64) / 16
    return signed if positive else -signed, float(positive), (index, product)


def _scan(entities, queries, relation, selected, count, check):
    candidates = []
    for query in queries:
        check("scan")
        index, subject, truth = query
        scores = (
            (entities.astype(np.float64) * entities[subject].astype(np.float64)) @ relation / 16
        )
        _require(np.isfinite(scores).all(), "nonfinite floating scan")
        for product, score in enumerate(scores):
            if product == subject or (index, product) in selected:
                continue
            violation = max(0.0, 1.0 - float(score)) if product in truth else max(0.0, float(score))
            if violation > 0:
                candidates.append((violation, index, product))
        # Bound retained proposal memory; total row matrix is never materialized.
        candidates = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[:count]
    return candidates


def _exact_signs(entities_int, queries, relation, check, chunk=16, floating_entities=None):
    relation_int, exponent = _integer_array(np.asarray(relation, dtype=np.float64))
    columns = len(entities_int)
    transpose = fmpz_mat(entities_int.T.tolist())
    checked = 0
    first_failure = None
    unreported_fp64_failure = None
    for start in range(0, len(queries), chunk):
        check("exact sign product")
        part = queries[start : start + chunk]
        left = fmpz_mat([(entities_int[s] * relation_int).tolist() for _, s, _ in part])
        scores = left * transpose
        for local, (index, subject, truth) in enumerate(part):
            floating = None
            for product in range(columns):
                if product == subject:
                    continue
                value = int(scores[local, product])
                valid = value > 0 if product in truth else value <= 0
                checked += 1
                if not valid and first_failure is None:
                    first_failure = {
                        "train_index": index,
                        "product_id": 1024 + product,
                        "positive": product in truth,
                        "integer_score_hex": hex(value),
                    }
                if not valid and floating_entities is not None:
                    if floating is None:
                        values = floating_entities.astype(np.float64)
                        floating = (values * values[subject]) @ relation / 16
                        _require(np.isfinite(floating).all(), "finite exact-failure FP64 scan")
                    residual = (
                        1.0 - float(floating[product])
                        if product in truth
                        else float(floating[product])
                    )
                    if residual <= 0 and unreported_fp64_failure is None:
                        unreported_fp64_failure = {
                            "train_index": index,
                            "product_id": 1024 + product,
                            "residual": residual,
                        }
    return {
        "passed": first_failure is None,
        "checked_constraints": checked,
        "relation_denominator_power": exponent,
        "first_failure": first_failure,
        "unreported_fp64_failure": unreported_fp64_failure,
    }


def _certificate(entities_int, exponent, rows, queries_by_index, multipliers, check, attempt=None):
    _require(np.isfinite(multipliers).all(), "nonfinite dual")
    support = [i for i, value in enumerate(multipliers) if value > 0]
    if attempt is None:
        attempt = {}
    attempt["support_indices"] = support
    _require(0 < len(support) <= 257, "dual support limit")
    selected = [rows[i] for i in support]
    integer_rows = []
    positive_flags = []
    for index, product in selected:
        _, subject, truth = queries_by_index[index]
        positive = product in truth
        integer_rows.append(
            (entities_int[subject] * entities_int[product] * (1 if positive else -1)).tolist()
        )
        positive_flags.append(positive)
    check("certificate nullspace")
    matrix = fmpz_mat(np.asarray(integer_rows, dtype=object).T.tolist())
    basis, nullity = matrix.nullspace()
    attempt["nullity"] = nullity
    _require(nullity == 1, "certificate nullity is not one")
    vector = [int(basis[i, 0]) for i in range(len(selected))]
    if next(v for v in vector if v) < 0:
        vector = [-v for v in vector]
    divisor = math.gcd(*vector)
    vector = [v // divisor for v in vector]
    attempt["oriented_integer_vector_hex"] = [hex(v) for v in vector]
    _require(all(v >= 0 for v in vector), "mixed-sign exact certificate")
    _require(
        matrix * fmpz_mat(len(vector), 1, vector) == fmpz_mat(matrix.nrows(), 1),
        "nonzero certificate product",
    )
    positive_sum = sum(v for v, positive in zip(vector, positive_flags, strict=True) if positive)
    _require(positive_sum > 0, "certificate has no positive-label multiplier")
    return {
        "nullity": nullity,
        "entity_denominator_power": exponent,
        "exact_zero": True,
        "positive_label_multiplier_sum_hex": hex(positive_sum),
        "support": [
            {
                "train_index": index,
                "product_id": 1024 + product,
                "positive": positive,
                "multiplier_hex": hex(value),
            }
            for (index, product), positive, value in zip(
                selected, positive_flags, vector, strict=True
            )
        ],
    }


def _solve(entities, steering, membership, dimension, recipe, folder, trace):
    started = time.perf_counter()
    result = {"dimension": dimension, "complete": False, "status": "inconclusive", "iterations": []}

    def check(phase):
        remaining = recipe["worker_seconds"] - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError(phase)
        return remaining

    def event(value):
        trace.write(json.dumps(value, allow_nan=False) + "\n")
        trace.flush()

    def lp(matrix, rhs, dual=False):
        remaining = check("dual LP" if dual else "primal LP")
        options = {
            k: recipe[k]
            for k in ("presolve", "primal_feasibility_tolerance", "dual_feasibility_tolerance")
        }
        options["time_limit"] = min(recipe["lp_seconds"], remaining)
        event(
            {
                "phase": "dual_start" if dual else "primal_start",
                "working_rows": len(rhs),
                "time_limit": options["time_limit"],
            }
        )
        if dual:
            return linprog(
                np.zeros(len(rhs)),
                A_eq=np.vstack([matrix.T, rhs]),
                b_eq=np.r_[np.zeros(matrix.shape[1]), 1.0],
                bounds=[(0, None)] * len(rhs),
                method=recipe["method"],
                options=options,
            )
        return linprog(
            np.zeros(matrix.shape[1]),
            A_ub=-matrix,
            b_ub=-rhs,
            bounds=[(None, None)] * matrix.shape[1],
            method=recipe["method"],
            options=options,
        )

    try:
        check("feature validation")
        _require(
            entities.dtype == steering.dtype == np.float32
            and np.isfinite(entities).all()
            and np.isfinite(steering).all(),
            "finite FP32 features",
        )
        check("steering rank")
        result["rank_witness"] = _rank_witness(steering)
        if not result["rank_witness"]["rank_two"]:
            result["reason"] = "steering_rank_not_two"
            result["complete"] = True
            return result
        queries = _queries(membership, dimension, len(entities))
        result.update(query_count=len(queries), constraint_count=len(queries) * (len(entities) - 1))
        by_index = {q[0]: q for q in queries}
        check("entity integer conversion")
        entities_int, exponent = _integer_array(entities)
        result["entity_denominator_power"] = exponent
        rows = [
            (q[0], product)
            for q in queries[: recipe["seed_queries"]]
            for product in range(len(entities))
            if product != q[1]
        ]
        _require(len(rows) <= recipe["max_working_rows"], "initial working limit")
        for iteration in range(recipe["max_primal_lps"]):
            check("working matrix")
            coefficients = [_row(entities, by_index[index], product) for index, product in rows]
            matrix = np.asarray([row[0] for row in coefficients])
            rhs = np.asarray([row[1] for row in coefficients])
            _require(np.isfinite(matrix).all() and np.isfinite(rhs).all(), "finite LP")
            proposed = lp(matrix, rhs)
            record = {
                "iteration": iteration + 1,
                "working_rows": [{"train_index": i, "product_id": 1024 + p} for i, p in rows],
                "status": int(proposed.status),
                "message": str(proposed.message),
            }
            result["iterations"].append(record)
            event({"phase": "primal_result", **record})
            if proposed.status == 2:
                dual = lp(matrix, rhs, dual=True)
                record["dual_status"] = int(dual.status)
                if dual.status != 0 or dual.x is None:
                    result["reason"] = "dual_not_solved"
                    break
                _require(np.isfinite(dual.x).all(), "finite dual proposal")
                record["dual_hex"] = [float(v).hex() for v in dual.x]
                event(
                    {
                        "phase": "dual_result",
                        "status": int(dual.status),
                        "dual_hex": record["dual_hex"],
                    }
                )
                record["certificate_attempt"] = {}
                try:
                    certificate = _certificate(
                        entities_int,
                        exponent,
                        rows,
                        by_index,
                        dual.x,
                        check,
                        record["certificate_attempt"],
                    )
                except ValueError as exc:
                    result["reason"] = "exact_certificate_rejected"
                    record["certificate_attempt"]["rejection"] = str(exc)
                    break
                finally:
                    event({"phase": "certificate_attempt", **record["certificate_attempt"]})
                path = folder / f"{dimension}.certificate.json"
                check("certificate serialization")
                if len(_bytes(certificate)) > recipe["max_certificate_bytes"]:
                    result["reason"] = "certificate_artifact_limit"
                    break
                _write(path, certificate, recipe["max_certificate_bytes"])
                result.update(
                    status="certified_infeasible",
                    certificate={"path": str(path), "sha256": _sha(path)},
                )
                break
            if proposed.status != 0 or proposed.x is None:
                result["reason"] = "primal_not_solved"
                break
            _require(np.isfinite(proposed.x).all(), "finite primal proposal")
            record["relation_hex"] = [float(v).hex() for v in proposed.x]
            event({"phase": "primal_vector", "relation_hex": record["relation_hex"]})
            exact = _exact_signs(
                entities_int, queries, proposed.x, check, recipe["exact_chunk_queries"], entities
            )
            record["exact_verification"] = exact
            event({"phase": "exact_verification", **exact})
            if exact["passed"]:
                witness = {
                    "relation_hex": record["relation_hex"],
                    "entity_denominator_power": exponent,
                    "exact_verification": exact,
                }
                path = folder / f"{dimension}.witness.json"
                check("witness serialization")
                _write(path, witness)
                result.update(
                    status="certified_feasible", witness={"path": str(path), "sha256": _sha(path)}
                )
                break
            if exact["unreported_fp64_failure"] is not None:
                result["reason"] = "exact_failure_hidden_by_fp64"
                break
            capacity = min(recipe["append_rows"], recipe["max_working_rows"] - len(rows))
            if capacity <= 0:
                result["reason"] = "working_row_limit"
                break
            additions = _scan(entities, queries, proposed.x, set(rows), capacity, check)
            record["added"] = [
                {"train_index": i, "product_id": 1024 + p, "violation": v} for v, i, p in additions
            ]
            if not additions:
                result["reason"] = "exact_failed_without_new_fp64_violation"
                break
            rows.extend((i, p) for _, i, p in additions)
        else:
            result["reason"] = "primal_lp_limit"
        result["complete"] = True
    except TimeoutError as exc:
        result.update(
            complete=True, reason="worker_budget", budget_phase=str(exc), status="inconclusive"
        )
    except BaseException as exc:
        result["error"] = repr(exc)
        result["status"] = "inconclusive"
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        event(
            {
                "phase": "finished",
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "membership", "recipe", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dimension", choices=("TYPE", "COLOR"), required=True)
    args = parser.parse_args()
    inputs = {
        str(p.resolve()): _sha(p)
        for p in (args.features, args.membership, args.recipe, Path(__file__))
    }
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "python-flint")}
    _require(
        all(versions[p] == recipe[p] for p in versions) and sys.version_info[:2] == (3, 12),
        "solver versions",
    )
    _require(not args.out.exists(), "immutable solver output")
    with np.load(args.features, allow_pickle=False) as arrays:
        entities, steering = arrays["entities"], arrays["steering"]
    membership = json.loads(args.membership.read_text(encoding="utf-8"))
    with args.out.with_suffix(".trace.jsonl").open("x", encoding="utf-8") as trace:
        result = _solve(
            entities, steering, membership, args.dimension, recipe, args.out.parent, trace
        )
    result.update(
        input_sha256=inputs,
        versions=versions,
        python=sys.version,
        torch_imported="torch" in sys.modules,
    )
    result["final_identity_check"] = all(_sha(p) == digest for p, digest in inputs.items())
    if not result["final_identity_check"]:
        result.update(status="inconclusive", complete=False, error="input drift")
    _write(args.out, result)


if __name__ == "__main__":
    main()
