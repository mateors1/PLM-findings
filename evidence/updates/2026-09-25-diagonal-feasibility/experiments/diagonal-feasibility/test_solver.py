"""Synthetic-only exact arithmetic/solver checks; run in the isolated environment."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import solver


def recipe():
    return json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/experiments/diagonal_feasibility_v1.json"
        ).read_text(encoding="utf-8")
    )


def membership(truths):
    return {
        "queries": [
            {
                "index": i,
                "dimension": "TYPE",
                "prompt_ids": [1, 1024 + i, 32, 34, 5],
                "expected_set_ids": [1024 + j for j in truth],
            }
            for i, truth in enumerate(truths)
        ]
    }


class SolverTests(unittest.TestCase):
    def test_dyadic_conversion_exact_including_subnormal(self):
        values = np.array(
            [[0.5, -0.25, np.nextafter(np.float32(0), np.float32(1))]], dtype=np.float32
        )
        integers, exponent = solver._integer_array(values)
        for original, value in zip(values.flat, integers.flat, strict=True):
            numerator, denominator = float(original).as_integer_ratio()
            self.assertEqual(int(value) * denominator, numerator * 2**exponent)

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            solver._integer_array(np.array([np.nan]))

    def test_rank_and_infinite_descriptive_condition(self):
        with patch.object(np.linalg, "cond", return_value=float("inf")):
            value = solver._rank_witness(np.eye(2, dtype=np.float32))
        self.assertTrue(value["rank_two"])
        self.assertIsNone(value["condition_number_fp64"])
        json.dumps(value, allow_nan=False)
        self.assertFalse(solver._rank_witness(np.ones((2, 3), dtype=np.float32))["rank_two"])

    def test_zero_allowed_only_for_negative_and_subject_excluded(self):
        entities = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        integers, _ = solver._integer_array(entities)
        result = solver._exact_signs(integers, [(0, 0, {1})], [1.0, 0.0], lambda _: None)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checked_constraints"], 2)
        failed = solver._exact_signs(integers, [(0, 0, {2})], [1.0, 0.0], lambda _: None)
        self.assertFalse(failed["passed"])

    def test_both_label_cuts_and_tie_order(self):
        entities = np.array([[1, 0], [1, 0], [-1, 0], [-1, 0]], dtype=np.float32)
        rows = [(0, 0, {1}), (1, 1, {0})]
        additions = solver._scan(entities, rows, np.array([-16.0, 0.0]), set(), 8, lambda _: None)
        self.assertEqual(additions[:2], [(2.0, 0, 1), (2.0, 1, 0)])
        self.assertEqual(additions[2:], [(1.0, 0, 2), (1.0, 0, 3), (1.0, 1, 2), (1.0, 1, 3)])
        self.assertNotIn(
            (2.0, 0, 1),
            solver._scan(entities, rows, np.array([-16.0, 0.0]), {(0, 1)}, 2, lambda _: None),
        )

    def test_feasible_system_and_witness(self):
        entities = np.array([[1, 0], [1, 0], [-1, 0], [-1, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            result = solver._solve(
                entities,
                np.eye(2, dtype=np.float32),
                membership([[1], [0]]),
                "TYPE",
                recipe(),
                Path(directory),
                io.StringIO(),
            )
            self.assertEqual(result["status"], "certified_feasible")
            witness = json.loads((Path(directory) / "TYPE.witness.json").read_text())
            self.assertEqual(witness["exact_verification"]["checked_constraints"], 6)

    def test_contradictory_system_and_exact_farkas(self):
        entities = np.array([[1, 0]] * 3, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            trace = io.StringIO()
            result = solver._solve(
                entities,
                np.eye(2, dtype=np.float32),
                membership([[1], [2]]),
                "TYPE",
                recipe(),
                Path(directory),
                trace,
            )
            self.assertEqual(result["status"], "certified_infeasible")
            certificate = json.loads((Path(directory) / "TYPE.certificate.json").read_text())
            self.assertTrue(certificate["exact_zero"])
            self.assertGreater(int(certificate["positive_label_multiplier_sum_hex"], 16), 0)
            self.assertIn('"phase": "dual_result"', trace.getvalue())

    def test_free_bounds_are_passed_to_primal(self):
        original = solver.linprog
        observed = []

        def checked(*args, **kwargs):
            observed.append(kwargs["bounds"])
            return original(*args, **kwargs)

        entities = np.array([[1, 0], [-1, 0], [1, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory, patch.object(solver, "linprog", checked):
            result = solver._solve(
                entities,
                np.eye(2, dtype=np.float32),
                membership([[1], [0, 2]]),
                "TYPE",
                recipe(),
                Path(directory),
                io.StringIO(),
            )
        self.assertEqual(result["status"], "certified_feasible")
        self.assertEqual(observed[0], [(None, None), (None, None)])

    def test_certificate_rejects_mixed_signs_and_extra_nullity(self):
        integers = np.array([[1, 0]] * 3, dtype=object)
        with self.assertRaisesRegex(ValueError, "mixed-sign"):
            solver._certificate(
                integers,
                0,
                [(0, 1), (1, 0)],
                {0: (0, 0, {1}), 1: (1, 1, {0})},
                [1.0, 1.0],
                lambda _: None,
            )
        with self.assertRaisesRegex(ValueError, "nullity"):
            solver._certificate(
                integers * 0,
                0,
                [(0, 1), (1, 0)],
                {0: (0, 0, {1}), 1: (1, 1, {0})},
                [1.0, 1.0],
                lambda _: None,
            )

    def test_hidden_exact_failure_stops_before_scan(self):
        exact = {"passed": False, "unreported_fp64_failure": {"residual": 0.0}}
        entities = np.array([[1, 0], [1, 0], [-1, 0]], dtype=np.float32)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(solver, "_exact_signs", return_value=exact),
            patch.object(solver, "_scan", side_effect=AssertionError("must stop")),
        ):
            result = solver._solve(
                entities,
                np.eye(2, dtype=np.float32),
                membership([[1], [0]]),
                "TYPE",
                recipe(),
                Path(directory),
                io.StringIO(),
            )
        self.assertEqual(result["reason"], "exact_failure_hidden_by_fp64")

    def test_rank_failure_and_expired_budget_inconclusive(self):
        entities = np.ones((3, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            result = solver._solve(
                entities,
                np.ones((2, 2), dtype=np.float32),
                membership([[1], [0]]),
                "TYPE",
                recipe(),
                Path(directory),
                io.StringIO(),
            )
            self.assertTrue(result["complete"])
            self.assertEqual(result["status"], "inconclusive")
            changed = recipe() | {"worker_seconds": 0}
            result = solver._solve(
                entities,
                np.eye(2, dtype=np.float32),
                membership([[1], [0]]),
                "TYPE",
                changed,
                Path(directory),
                io.StringIO(),
            )
            self.assertEqual(result["reason"], "worker_budget")

    def test_immutable_artifacts_and_certificate_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proof.json"
            solver._write(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                solver._write(path, {})
            with self.assertRaisesRegex(ValueError, "size"):
                solver._write(Path(directory) / "large.json", {"value": "x" * 20}, 4)


if __name__ == "__main__":
    unittest.main()
