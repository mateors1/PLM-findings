"""Synthetic policy and CPU continuation contracts; no real checkpoint forwards."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "coverage_branch", ROOT / "scripts/evaluate_coverage_seeking_branch.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
ORDER = (
    (1,),
    (2,),
    (3,),
    (4,),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (5,),
    (6,),
    (7,),
    (8,),
    *((a, b) for a in range(1, 9) for b in range(a + 1, 9) if b > 4),
)


def baseline():
    sets = [[1025 + i] for i in range(8)]
    slots = []
    for rank, sources in enumerate(ORDER, 1):
        ids = sorted(set().union(*(set(sets[i - 1]) for i in sources)))
        slots.append(
            {
                "slot": rank,
                "kind": "original" if len(sources) == 1 else "pair_composition",
                "source_ranks": list(sources),
                "set_ids": ids,
                "source_eligible": True,
                "score": float(len(ids)),
            }
        )
    return {"source_paths": [{} for _ in range(8)], "slots": slots, **runner._selection(slots)}


@pytest.mark.parametrize(
    "values,wanted",
    [
        ({1024: 9.0, 1025: 8.0, 1040: 2.0}, 1040),
        ({1041: 2.0, 1040: 2.0}, 1040),
        ({1040: 0.0, 1041: -1.0}, None),
    ],
)
def test_anchor_subject_coverage_tie_zero_negative(values, wanted):
    logits = [-1.0] * 1025
    for i, score in values.items():
        logits[i - 1024] = score
    value = runner._anchor([[1025]] * 8, logits, 1024)
    assert value["token_id"] == wanted and value["active"] == (wanted is not None)


def test_all_covered_means_no_candidate_and_anchor_has_no_label_input():
    assert not runner._anchor([list(range(1025, 2049))] * 8, [10.0] * 1025, 1024)["active"]
    assert runner._anchor.__code__.co_varnames[:3] == ("source_sets", "logits", "subject")


def test_active_slot_order_old_preservation_and_union_expansion():
    old = baseline()
    frozen = copy.deepcopy(old)
    anchor = {"active": True, "token_id": 1040}
    out = runner._extend(old, anchor, {"raw": True}, [1040], True, [1.0] * 1025)
    assert old == frozen and out["slots"][:36] == old["slots"]
    assert len(out["slots"]) == 45 and len(out["source_paths"]) == 9
    assert [s["source_ranks"] for s in out["slots"][36:]] == [[9], *[[i, 9] for i in range(1, 9)]]
    assert out["selected_slot"] == 5  # New union score ties existing pair; old wins.


def test_valid_active_source_must_contain_anchor():
    with pytest.raises(ValueError, match="contains anchor"):
        runner._extend(
            baseline(), {"active": True, "token_id": 1040}, {}, [1025], True, [1.0] * 1025
        )


def test_inactive_padding_never_enters_pool_even_if_high_scoring():
    old = baseline()
    out = runner._extend(
        old, {"active": False, "token_id": None}, {"padding": True}, [1040], True, [100.0] * 1025
    )
    assert out["slots"] == old["slots"] and out["source_paths"] == old["source_paths"]
    assert out["selected_slot"] == old["selected_slot"]


def test_invalid_active_raw_retained_but_all_added_slots_ineligible():
    out = runner._extend(
        baseline(),
        {"active": True, "token_id": 1040},
        {"terminated": False},
        [1040],
        False,
        [100.0] * 1025,
    )
    assert out["source_paths"][-1] == {"terminated": False}
    assert all(not s["source_eligible"] for s in out["slots"][36:])
    assert out["selected_slot"] <= 36


def test_new_source_with_negative_extra_still_uses_canonical_fsum():
    logits = [1.0] * 1025
    logits[1040 - 1024], logits[1041 - 1024], logits[1042 - 1024] = 2.0**60, 1.0, -(2.0**60)
    out = runner._extend(
        baseline(), {"active": True, "token_id": 1040}, {}, [1040, 1041, 1042], True, logits
    )
    assert out["slots"][36]["score"] == 1.0


def passing_gate():
    pooled = {
        "overall": {"new": {"exact_count": 604, "failures": 0}},
        "groups": {
            g: {"new": {"exact_count": 156}, "baseline": {"exact_count": 155}}
            for g in runner._GROUPS
        },
    }
    reports = [
        {
            "complete": True,
            "query_count": 222,
            "replay_exact": True,
            "responses": [{"anchor": {"active": True}, "new_source_eligible": True}],
            "overall": {
                "baseline": {"exact_count": 200, "f1": 0.9},
                "new": {"exact_count": 201, "f1": 0.91},
            },
        }
        for _ in range(3)
    ]
    return reports, pooled


@pytest.mark.parametrize(
    "fault", ["replay", "source", "selected", "seed_exact", "seed_f1", "pooled", "dual", "group"]
)
def test_every_fixed_gate_failure(fault):
    reports, pooled = passing_gate()
    assert runner._gate(reports, pooled)["quality_passed"]
    if fault == "replay":
        reports[0]["replay_exact"] = False
    elif fault == "source":
        reports[0]["responses"][0]["new_source_eligible"] = False
    elif fault == "selected":
        pooled["overall"]["new"]["failures"] = 1
    elif fault == "seed_exact":
        reports[0]["overall"]["new"]["exact_count"] = 199
    elif fault == "seed_f1":
        reports[0]["overall"]["new"]["f1"] = 0.89
    elif fault == "pooled":
        pooled["overall"]["new"]["exact_count"] = 603
    elif fault == "dual":
        pooled["groups"]["TYPE_dual"]["new"]["exact_count"] = 155
    else:
        pooled["groups"]["COLOR"]["new"]["exact_count"] = 154
    assert not runner._gate(reports, pooled)["quality_passed"]


def test_active_padding_work_is_separate_and_counts_finished_row_positions():
    paths = [SimpleNamespace(token_ids=[0] * 7), SimpleNamespace(token_ids=[0] * 9)]
    value = runner._work(paths, [True, False], 216, "anchor", 0.5)
    assert value["decode_calls"] == 4 and value["padded_decode_positions"] == 16
    assert value["active_decode_positions"] == value["padding_decode_positions"] == 8
    assert value["active_emitted_tokens"] == 2 and value["padding_emitted_tokens"] == 4
    assert value["useful_emitted_tokens"] == 2 and value["total_emitted_tokens"] == 6
    assert value["guidance_forwards"] == 1


@pytest.fixture
def cpu_fixture():
    torch = pytest.importorskip("torch")
    from plm.protocol import Vocabulary
    from plm.serving.generation import _generation_result
    from plm.serving.parser import prompt_ids

    vocab = Vocabulary.build(["PKM_A", "PKM_B", "PKM_C", "PKM_D"])
    prompts = [list(prompt_ids("PKM_A", "TYPE", vocab)), list(prompt_ids("PKM_B", "TYPE", vocab))]

    class Model:
        config = SimpleNamespace(max_seq_len=12)

        def __init__(self):
            self.shapes = []
            self.guidance = 0

        def decode(self, ids, *, cache=None):
            self.shapes.append(tuple(ids.shape))
            subjects, step = (ids[:, 1].tolist(), 0) if cache is None else cache
            logits = torch.full((len(subjects), ids.shape[1], len(vocab)), -20.0)
            for row, subject in enumerate(subjects):
                logits[row, -1, vocab.id_of("PKM_D")] = 9.0  # Repeat preference must be masked.
                logits[row, -1, vocab.id_of("PKM_C")] = 8.0
                logits[row, -1, vocab.id_of("PKM_A")] = 6.0
                logits[row, -1, vocab.id_of("PKM_B")] = 5.0
                logits[row, -1, vocab.eos_id] = (
                    10.0 if (step >= (1 if subject == 1024 else 3)) else 4.0
                )
            return SimpleNamespace(logits=logits, cache=(subjects, step + 1))

    def guide(model, inputs, output, alpha):
        assert inputs.shape[1] == 5 and alpha == 16.0
        model.guidance += 1
        return output

    return Model, vocab, prompts, guide, _generation_result


def test_cpu_forced_first_seen_eos_and_original_batch_shapes(cpu_fixture):
    Model, vocab, prompts, guide, make_result = cpu_fixture
    model = Model()
    paths = runner._generate_forced_first(
        model,
        vocab,
        prompts,
        [1027, 1026],
        max_new_tokens=6,
        device="cpu",
        guide=guide,
        make_result=make_result,
    )
    assert paths[0].token_ids == (*prompts[0], 1027, vocab.eos_id)
    assert paths[1].token_ids == (*prompts[1], 1026, 1027, 1024, vocab.eos_id)
    assert model.shapes == [(2, 5), (2, 1), (2, 1), (2, 1)]
    assert model.guidance == 1
    assert all(p.protocol_valid and p.terminated for p in paths)


def test_cpu_bound_counts_forced_token_and_preserves_truncation(cpu_fixture):
    Model, vocab, prompts, guide, make_result = cpu_fixture
    model = Model()
    paths = runner._generate_forced_first(
        model,
        vocab,
        prompts,
        [1027, 1026],
        max_new_tokens=1,
        device="cpu",
        guide=guide,
        make_result=make_result,
    )
    assert all(len(p.token_ids) == 6 and not p.terminated and not p.protocol_valid for p in paths)
    assert model.shapes == [(2, 5)]


def test_cpu_row_independence_and_rank_one_compatibility(cpu_fixture):
    Model, vocab, prompts, guide, make_result = cpu_fixture
    from plm.serving.set_reranking import _generate_rank

    # The current rank helper is byte-identical to the authenticated archived one.
    # This optional local comparison is skipped if future unrelated core edits differ.
    path = ROOT / "src/plm/serving/set_reranking.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != runner._RANK_SOURCE:
        pytest.skip(
            "live rank helper differs from archived contract; experiment uses frozen archive"
        )
    old = _generate_rank(
        Model(), vocab, prompts, 1, max_new_tokens=6, device="cpu", alpha=0.0, decoding="control"
    )
    actual = runner._generate_forced_first(
        Model(),
        vocab,
        prompts,
        [p.token_ids[5] for p in old],
        max_new_tokens=6,
        device="cpu",
        guide=guide,
        make_result=make_result,
    )
    assert [p.token_ids for p in actual] == [p.token_ids for p in old]
    for prompt, result in zip(prompts, actual, strict=True):
        single = runner._generate_forced_first(
            Model(),
            vocab,
            [prompt],
            [result.token_ids[5]],
            max_new_tokens=6,
            device="cpu",
            guide=guide,
            make_result=make_result,
        )
        assert single[0].token_ids == result.token_ids


@pytest.mark.parametrize(
    "ids,budget", [([2, 1026], 6), ([True, 1026], 6), ([1025], 6), ([1025, 1026], 8)]
)
def test_cpu_forced_input_and_budget_rejection(cpu_fixture, ids, budget):
    Model, vocab, prompts, guide, make_result = cpu_fixture
    with pytest.raises(ValueError):
        runner._generate_forced_first(
            Model(),
            vocab,
            prompts,
            ids,
            max_new_tokens=budget,
            device="cpu",
            guide=guide,
            make_result=make_result,
        )


def test_helper_identity_refuses_drift_before_import(tmp_path):
    helper = tmp_path / "runs/learning/wide-first-choice-v1/summary.script.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("raise RuntimeError('should not execute')")
    with pytest.raises(ValueError, match="helper identity"):
        runner._load_helper(tmp_path)


def test_metric_aggregation_uses_new_slot_boundary_and_no_padding_credit():
    metric = {
        "exact": True,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "set_size": 2,
        "success": True,
    }
    availability = {
        "exact_available": True,
        "available_exact_miss": False,
        "source_union_contains_truth": True,
    }
    row = {
        "anchor": {"active": False},
        "anchor_correct": None,
        "newly_covered_true_ids": [],
        "newly_covered_false_ids": [],
        "false_positive_ids": [],
        "false_negative_ids": [],
        "gained_exact": False,
        "lost_exact": False,
        "metrics": {"baseline": metric, "new": metric},
        "availability": {"baseline": availability, "new": availability},
        "baseline": {"selected_slot": 14},
        "composition": {"selected_slot": 14},
    }
    total = runner._totals([row])
    assert total["inactive_anchors"] == 1 and total["false_anchors"] == 0
    assert total["new"]["added_slot_selections"] == total["baseline"]["added_slot_selections"] == 0
    assert total["new"]["f1"] == 1.0 and total["new"]["exact_count"] == 1
    row["composition"]["selected_slot"] = 37
    assert runner._totals([row])["new"]["added_slot_selections"] == 1


def test_early_execution_failure_retains_failed_summary(tmp_path):
    def failed():
        raise RuntimeError("injected import-boundary failure")

    def write(path, value):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream)

    helper = SimpleNamespace(_modules=failed, _unchanged=lambda inputs: None, _write=write)
    summary = {
        "complete": False,
        "acceptance": False,
        "input_sha256": {},
        "snapshot_sha256": {},
        "reports": [],
    }
    output = tmp_path / "summary.json"
    with pytest.raises(RuntimeError, match="import-boundary"):
        runner._execute(tmp_path, output, {}, helper, summary)
    saved = json.loads(output.read_text())
    assert not saved["complete"] and not saved["acceptance"]
    assert "injected import-boundary failure" in saved["error"]
