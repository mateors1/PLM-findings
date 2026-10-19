"""Synthetic only; never reads the real feature archive or solver outputs."""

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "diagonal-feasibility-v1/independent-audit.py"
SPEC = importlib.util.spec_from_file_location("independent_diagonal", PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class ExactArithmetic(unittest.TestCase):
    def test_dyadic_conversion_preserves_fp32(self):
        raw = np.array([[0.1, -2, 0, np.nextafter(np.float32(0), np.float32(1))]], dtype=np.float32)
        values, power = A.integers(raw)
        for original, integer in zip(raw.flat, values.flat, strict=True):
            num, den = float(original).as_integer_ratio()
            self.assertEqual(int(integer) * den, num * 2**power)

    def test_nonfinite_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                A.integers(np.array([value]))

    def test_exact_rank(self):
        u = np.array([[1, 2], [2, 3]], dtype=np.float32)
        witness = {
            "rank_two": True,
            "columns": [0, 1],
            "integer_determinant_hex": "-0x1",
            "steering_denominator_power": 0,
            "condition_number_fp64": float(np.linalg.cond(u.astype(np.float64))),
            "condition_number_status": "finite",
        }
        self.assertTrue(A.rank_audit(u, witness))
        witness["integer_determinant_hex"] = "0x0"
        with self.assertRaises(ValueError):
            A.rank_audit(u, witness)

    def test_rank_deficient_inconclusive(self):
        self.assertFalse(
            A.rank_audit(
                np.array([[1, 2], [2, 4]], dtype=np.float32),
                {"rank_two": False, "steering_denominator_power": 0},
            )
        )
        self.assertEqual(A.outcome(["certified_feasible"] * 2, False), "inconclusive")

    def test_zero_is_allowed_only_for_false(self):
        e, _ = A.integers(np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32))
        success = A.exact_signs(e, [(0, 0, {1}), (1, 1, {0})], [float(1).hex(), float(0).hex()])
        self.assertTrue(success["passed"])
        self.assertEqual(success["checked_constraints"], 4)
        failure = A.exact_signs(e, [(0, 0, {2})], [float(1).hex(), float(0).hex()])
        self.assertFalse(failure["passed"])

    def test_negative_relation_is_free(self):
        e, _ = A.integers(np.array([[1], [-1], [0]], dtype=np.float32))
        self.assertTrue(A.exact_signs(e, [(0, 0, {1})], [float(-1).hex()])["passed"])

    def test_tiny_true_sign_not_tolerance_zero(self):
        e, _ = A.integers(np.array([[1], [1]], dtype=np.float32))
        result = A.exact_signs(e, [(0, 0, {1})], [float.fromhex("0x0.0000000000001p-1022").hex()])
        self.assertTrue(result["passed"])

    def test_numerical_infeasibility_not_status(self):
        self.assertEqual(A.outcome(["inconclusive", "certified_feasible"], True), "inconclusive")
        with self.assertRaises(ValueError):
            A.outcome(["numerical_infeasible", "certified_feasible"], True)


class Certificates(unittest.TestCase):
    def setUp(self):
        self.entities, self.power = A.integers(np.ones((3, 2), dtype=np.float32))
        self.queries = [(0, 0, {1}), (1, 1, {2})]
        self.working = [
            {"train_index": 0, "product_id": 1025},
            {"train_index": 1, "product_id": 1024},
        ]
        self.dual = [float(1).hex()] * 2
        self.cert = {
            "nullity": 1,
            "exact_zero": True,
            "entity_denominator_power": 0,
            "positive_label_multiplier_sum_hex": "0x1",
            "support": [
                {**self.working[0], "positive": True, "multiplier_hex": "0x1"},
                {**self.working[1], "positive": False, "multiplier_hex": "0x1"},
            ],
        }

    def audit(self):
        return A.certificate_audit(
            self.entities, self.power, self.queries, self.cert, self.working, self.dual
        )

    def test_exact_contradiction(self):
        self.assertEqual(self.audit()["support_count"], 2)
        self.assertEqual(
            A.outcome(["certified_infeasible", "inconclusive"], True), "certified_infeasible"
        )

    def test_approximate_zero_not_certificate(self):
        self.entities[2, 0] += 1
        self.working[1]["product_id"] = 1026
        self.cert["support"][1]["product_id"] = 1026
        self.queries[1] = (1, 1, {0})
        with self.assertRaises(ValueError):
            self.audit()

    def test_label_flip_rejected(self):
        self.cert["support"][1]["positive"] = True
        with self.assertRaises(ValueError):
            self.audit()

    def test_self_row_rejected(self):
        self.working[0]["product_id"] = 1024
        self.cert["support"][0]["product_id"] = 1024
        with self.assertRaises(ValueError):
            self.audit()

    def test_missing_positive_support_rejected(self):
        self.cert["support"] = self.cert["support"][:1]
        with self.assertRaises(ValueError):
            self.audit()

    def test_negative_multiplier_rejected(self):
        self.cert["support"][1]["multiplier_hex"] = "-0x1"
        with self.assertRaises(ValueError):
            self.audit()

    def test_nonprimitive_rejected(self):
        for row in self.cert["support"]:
            row["multiplier_hex"] = "0x2"
        with self.assertRaises(ValueError):
            self.audit()

    def test_misreported_positive_sum_rejected(self):
        self.cert["positive_label_multiplier_sum_hex"] = "0x2"
        with self.assertRaises(ValueError):
            self.audit()

    def test_higher_nullity_rejected(self):
        self.working.append(copy.deepcopy(self.working[1]))
        self.cert["support"].append(copy.deepcopy(self.cert["support"][1]))
        self.cert["support"][0]["multiplier_hex"] = "0x2"
        self.cert["positive_label_multiplier_sum_hex"] = "0x2"
        self.dual.append(float(1).hex())
        with self.assertRaises(ValueError):
            self.audit()


class TrainingAndBudgets(unittest.TestCase):
    def setUp(self):
        self.entities = np.ones((3, 2), dtype=np.float32)
        self.queries = [(0, 0, {1}), (2, 1, {2}), (3, 2, {0})]
        self.working = [
            {"train_index": index, "product_id": 1024 + product}
            for index, subject, _ in self.queries[:2]
            for product in range(3)
            if product != subject
        ]

    def test_cuts_tie_by_query_then_id(self):
        result = A.additions(self.entities, self.queries, [float(0).hex()] * 2, self.working, 128)
        self.assertEqual(result, [{"train_index": 3, "product_id": 1024, "violation": 1.0}])

    def test_cuts_exclude_existing_and_subject(self):
        result = A.additions(self.entities, self.queries, [float(0).hex()] * 2, [], 128)
        self.assertEqual([row["product_id"] for row in result], [1025, 1026, 1024])

    def test_cuts_include_negative_violations(self):
        result = A.additions(self.entities, self.queries, [float(16).hex()] * 2, [], 128)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(row["violation"] == 2 for row in result))

    def test_initial_rows_exact(self):
        record = {"iteration": 1, "status": 1, "working_rows": self.working}
        self.assertEqual(
            A.iteration_contract(self.entities, self.queries, [record])["primal_lps"], 1
        )
        record["working_rows"] = list(reversed(self.working))
        with self.assertRaises(ValueError):
            A.iteration_contract(self.entities, self.queries, [record])

    def test_cannot_continue_without_cuts(self):
        records = [{"iteration": i, "status": 0, "working_rows": self.working} for i in (1, 2)]
        with self.assertRaises(ValueError):
            A.iteration_contract(self.entities, self.queries, records)

    def test_maximum_primal_calls(self):
        with self.assertRaises(ValueError):
            A.iteration_contract(self.entities, self.queries, [{}] * 17)

    def test_dual_requires_infeasible_proposal(self):
        record = {"iteration": 1, "status": 0, "dual_status": 0, "working_rows": self.working}
        with self.assertRaises(ValueError):
            A.iteration_contract(self.entities, self.queries, [record])

    def test_recipe_negative_zero_and_free_bounds(self):
        self.assertEqual(A.RECIPE["negative_rhs"], 0)
        self.assertEqual(A.RECIPE["positive_rhs"], 1)
        self.assertEqual(A.RECIPE["primal_bounds"], "free")

    def test_trace_lp_time_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            path.write_text(
                json.dumps({"phase": "primal_start", "working_rows": 2048, "time_limit": 31}) + "\n"
            )
            with self.assertRaises(ValueError):
                A.trace_events(path)

    def test_only_killed_last_partial_trace_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            path.write_text('{"phase":"primal_start","working_rows":2048,"time_limit":30}\n{')
            self.assertEqual(len(A.trace_events(path, True)), 1)
            with self.assertRaises(ValueError):
                A.trace_events(path, False)

    def test_no_heldout_labels_read(self):
        records = [{"subject": "PKM_" + str(i), "dimension": "TYPE"} for i in range(300)]
        held = [
            r
            for r in records
            if int.from_bytes(
                A.hashlib.sha256(f"1729:{r['subject']}:TYPE".encode()).digest(), "big"
            )
            / float(1 << 256)
            < 0.2
        ]
        result, _ = A.training_membership(held, [])
        self.assertEqual(result["queries"], [])

    def test_train_labels_reconstructed_and_corruption_rejected(self):
        subject = next(
            "PKM_" + str(i)
            for i in range(100)
            if int.from_bytes(A.hashlib.sha256(f"1729:PKM_{i}:TYPE".encode()).digest(), "big")
            / float(1 << 256)
            >= 0.2
        )
        vocab = [""] * 1027
        vocab[1024:1027] = [subject, "PKM_TARGET", "PKM_OTHER"]
        record = {
            "subject": subject,
            "dimension": "TYPE",
            "input_ids": [1, 1024, 32, 34, 5, 1025, 2],
            "labels": [-100] * 5 + [1025, 2],
            "targets": ["PKM_TARGET"],
        }
        result, _ = A.training_membership([record], vocab)
        self.assertEqual(result["queries"][0]["expected_set_ids"], [1025])
        record["labels"][0] = 1
        with self.assertRaises(ValueError):
            A.training_membership([record], vocab)

    def test_feature_finiteness_and_raw_byte_integrity(self):
        entities = np.ones((1025, 256), dtype=np.float32)
        steering = np.ones((2, 256), dtype=np.float32)
        manifest = {
            name: {
                "shape": list(array.shape),
                "dtype": "float32",
                "sha256": A.hashlib.sha256(array.tobytes()).hexdigest(),
            }
            for name, array in (("entities", entities), ("steering", steering))
        }
        A.feature_contract(entities, steering, manifest)
        entities[0, 0] = 2
        with self.assertRaises(ValueError):
            A.feature_contract(entities, steering, manifest)
        entities[0, 0] = np.nan
        with self.assertRaises(ValueError):
            A.feature_contract(entities, steering, manifest)

    def test_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input"
            path.write_bytes(b"frozen")
            digest = A.sha(path)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                A.bind(path, digest)

    def test_partial_trace_cannot_change_seed_rows(self):
        events = [
            {"phase": "primal_result", "iteration": 1, "status": 0, "working_rows": self.working},
            {"phase": "primal_vector", "relation_hex": [float(0).hex()] * 2},
        ]
        self.assertEqual(
            A.partial_trace_audit(self.entities, self.queries, events)["primal_lps"], 1
        )
        events[0]["working_rows"] = self.working[:-1]
        with self.assertRaises(ValueError):
            A.partial_trace_audit(self.entities, self.queries, events)

    def test_certificate_attempt_nullity_corruption(self):
        integer, _ = A.integers(np.ones((3, 2), dtype=np.float32))
        working = [{"train_index": 0, "product_id": 1025}, {"train_index": 2, "product_id": 1024}]
        record = {
            "dual_hex": [float(1).hex()] * 2,
            "working_rows": working,
            "certificate_attempt": {
                "support_indices": [0, 1],
                "nullity": 1,
                "oriented_integer_vector_hex": ["0x1", "0x1"],
            },
        }
        A.certificate_attempt_audit(integer, self.queries, record)
        record["certificate_attempt"]["nullity"] = 0
        with self.assertRaises(ValueError):
            A.certificate_attempt_audit(integer, self.queries, record)


class PartialProcessEvidence(unittest.TestCase):
    def fixture(self, folder, timed_out=False):
        summary_path = folder / "summary.json"
        for name in (
            "features.npz",
            "train-membership.json",
            "summary.recipe.json",
            "summary.solver.py",
        ):
            (folder / name).write_bytes(b"synthetic")
        report = {
            "status": "inconclusive",
            "complete": True,
            "dimension": "TYPE",
            "reason": "worker_budget",
            "budget_phase": "steering rank",
            "iterations": [],
            "elapsed_seconds": 180.0,
            "final_identity_check": True,
            "torch_imported": False,
            "python": "3.12.synthetic",
            "versions": {k: A.RECIPE[k] for k in ("numpy", "scipy", "python-flint")},
            "input_sha256": {
                str(folder / name): A.sha(folder / name)
                for name in (
                    "features.npz",
                    "train-membership.json",
                    "summary.recipe.json",
                    "summary.solver.py",
                )
            },
        }
        (folder / "TYPE.json").write_text(json.dumps(report))
        (folder / "TYPE.trace.jsonl").write_text(
            json.dumps({"phase": "finished", "status": "inconclusive", "elapsed_seconds": 180.0})
            + "\n"
        )
        process = {
            "command": [
                str(A.ROOT / "experiments/diagonal-feasibility/.venv/Scripts/python.exe"),
                "-I",
                str(folder / "summary.solver.py"),
                "--features",
                str(folder / "features.npz"),
                "--membership",
                str(folder / "train-membership.json"),
                "--recipe",
                str(folder / "summary.recipe.json"),
                "--dimension",
                "TYPE",
                "--out",
                str(folder / "TYPE.json"),
            ],
            "hard_limit_seconds": 180.0,
            "wait_budget_seconds": 179.9,
            "timed_out": timed_out,
            "reaped": True,
            "process_running": False,
            "elapsed_seconds": 180.0,
            "exit_code": 1 if timed_out else 0,
            "status": "inconclusive",
        }
        if not timed_out:
            process.update(report_sha256=A.sha(folder / "TYPE.json"), report_complete=True)
        (folder / "TYPE.process.json").write_text(json.dumps(process))
        membership = {
            "partition": "train",
            "queries": [
                {
                    "index": 0,
                    "dimension": "TYPE",
                    "prompt_ids": [1, 1024, 32, 34, 5],
                    "expected_set_ids": [1025],
                },
                {
                    "index": 1,
                    "dimension": "TYPE",
                    "prompt_ids": [1, 1025, 32, 34, 5],
                    "expected_set_ids": [1024],
                },
            ],
        }
        return (
            summary_path,
            {"solver_environment": {"versions": report["versions"]}},
            {**process, "dimension": "TYPE"},
            membership,
        )

    def test_internal_budget_before_rank_keeps_inconclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path, summary, process, membership = self.fixture(Path(folder))
            result = A.dimension_audit(
                path,
                summary,
                "TYPE",
                process,
                membership,
                np.ones((2, 2), dtype=np.float32),
                np.eye(2, dtype=np.float32),
            )
            self.assertEqual(result["reason"], "worker_budget")
            self.assertFalse(result["mathematical_witness_verified"])

    def test_hard_kill_does_not_promote_present_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path, summary, process, membership = self.fixture(Path(folder), True)
            result = A.dimension_audit(
                path,
                summary,
                "TYPE",
                process,
                membership,
                np.ones((2, 2), dtype=np.float32),
                np.eye(2, dtype=np.float32),
            )
            self.assertEqual(result["execution_status"], "hard_timeout")
            self.assertFalse(result["mathematical_witness_verified"])

    def test_unreaped_worker_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, summary, process, membership = self.fixture(Path(folder))
            process["reaped"] = False
            (Path(folder) / "TYPE.process.json").write_text(
                json.dumps({k: v for k, v in process.items() if k != "dimension"})
            )
            with self.assertRaises(ValueError):
                A.dimension_audit(
                    path,
                    summary,
                    "TYPE",
                    process,
                    membership,
                    np.ones((2, 2), dtype=np.float32),
                    np.eye(2, dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main()
