"""Recheck an existing coverage ceiling; do not evaluate a new selection policy."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PINS = {
    "set-candidate-coverage-v1/summary.json": "67974a44184a48ecb67b62bbb5dab4c9f3ac05a2b30e4858fee771a767ea0dc4",
    "pair-unions-v1/summary.json": "4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992",
    "pair-unions-v1/independent-audit.json": "13d9b88135eb52c768932e213cc475ac5994636a7fc228b1a7b2ad8ab538a057",
}
INPUTS: dict[str, str] = {}
ORDER = [tuple(c) for size in range(1, 5) for c in itertools.combinations(range(1, 5), size)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path, digest):
    require(sha(path) == digest, f"input hash mismatch: {path}")
    INPUTS[path.relative_to(ROOT).as_posix()] = digest
    return path


def read(path, digest):
    return json.loads(bind(path, digest).read_text(encoding="utf-8"))


def main():
    output = HERE / "summary.json"
    require(not output.exists(), "immutable output already exists")
    script_hash = sha(Path(__file__))
    loaded = {name: read(ROOT / "runs/learning" / name, digest) for name, digest in PINS.items()}
    coverage = loaded["set-candidate-coverage-v1/summary.json"]
    pair = loaded["pair-unions-v1/summary.json"]
    audit = loaded["pair-unions-v1/independent-audit.json"]
    require(audit["all_arithmetic_and_provenance_equal"] is True, "pair audit did not pass")
    require(audit["summary_sha256"] == PINS["pair-unions-v1/summary.json"], "audit binding")
    bind(ROOT / "runs/learning/set-candidate-coverage-v1/diagnose.py", coverage["script_sha256"])
    bind(ROOT / "runs/learning/pair-unions-v1/independent-audit.py", audit["script_sha256"])
    bind(ROOT / "runs/learning/pair-unions-v1/summary.script.py", pair["script_sha256"])
    require(pair["pool_order"] == [list(c) for c in ORDER[:10]], "original ten-slot order")
    vocab_paths = [p for p in pair["input_sha256"] if p.endswith("/vocabulary.json")]
    require(len(vocab_paths) == 1, "ambiguous vocabulary")
    vocab = read(ROOT / vocab_paths[0], pair["input_sha256"][vocab_paths[0]])
    tokens = vocab["tokens"]
    require(len(tokens) == len(set(tokens)) == 2049, "vocabulary size/uniqueness")
    token_ids = {name: index for index, name in enumerate(tokens)}
    cov_rows = {(r["seed"], r["subject"], r["dimension"]): r for r in coverage["observations"]}
    require(len(cov_rows) == len(coverage["observations"]) == 666, "coverage cardinality")
    rows, totals = [], []
    for seed in (1729, 1730, 1731):
        name = f"seed-{seed}.json"
        report = read(ROOT / "runs/learning/pair-unions-v1" / name, pair["report_sha256"][name])
        require(report["seed"] == seed and report["query_count"] == len(report["responses"]) == 222,
                "pair seed cardinality")
        selected_exact = available10 = available15 = new_only = covered_without_exact = absent = 0
        seen = set()
        for record in report["responses"]:
            key = (seed, record["subject"], record["dimension"])
            require(key not in seen and key in cov_rows, "query identity")
            seen.add(key)
            old = cov_rows[key]
            require(record["expected"] == old["expected"] and record["group"] == old["group"], "truth/group identity")
            truth = {token_ids[t] for t in record["expected"]}
            require(len(truth) == len(record["expected"]), "duplicate truth members")
            paths = record["source_paths"]
            require(len(paths) == 4, "source count")
            sets = []
            for rank, path in enumerate(paths, 1):
                require(path["first_rank"] == rank and path["eligible"] is True
                        and path["terminated"] is True and path["protocol_valid"] is True
                        and path["error"] is None, "source eligibility")
                ids = path["token_ids"]
                prompt = [token_ids[t] for t in ("BOS", record["subject"], record["dimension"], "SAME", "ANSWER")]
                require(ids[:5] == prompt and ids[-1] == token_ids["EOS"]
                        and all(type(i) is int and 1024 <= i < 2049 for i in ids[5:-1]), "raw grammar")
                raw = [tokens[i] for i in ids[5:-1]]
                require(raw == path["predicted"] and len(raw) == len(set(raw)), "raw targets")
                processed = [t for t in raw if t != record["subject"]]
                require(processed == path["processed_targets"] == old["candidate_targets"][rank - 1], "same source sets")
                sets.append({token_ids[t] for t in processed})
            subsets = [sorted(set().union(*(sets[r - 1] for r in ranks))) for ranks in ORDER]
            exact_ranks = [list(ranks) for ranks, ids in zip(ORDER, subsets, strict=True) if set(ids) == truth]
            require(exact_ranks == old["oracle_exact_subset_ranks_analysis_only"], "saved all15 exact inventory")
            require(bool(exact_ranks) == old["oracle_subset_union_exact"], "saved availability")
            exact_old = [i for i in range(10) if set(subsets[i]) == truth]
            exact_new = [i for i in range(10, 15) if set(subsets[i]) == truth]
            require(not exact_ranks or exact_old, "new exact availability would invalidate bound")
            duplicate_witnesses = []
            for index in exact_new:
                duplicate = next(i for i in exact_old if subsets[i] == subsets[index])
                duplicate_witnesses.append({"added_slot": index + 1, "existing_slot": duplicate + 1,
                                            "canonical_set_ids": subsets[index]})
            logits = record["symmetric_relation_logits"]
            require(len(logits) == 1025 and all(type(v) in (int, float) and math.isfinite(v)
                    and struct.unpack("f", struct.pack("f", v))[0] == v for v in logits), "FP32 logits")
            slots = record["selection"]["slots"]
            require(len(slots) == 10, "ten-slot count")
            # Replay only the accepted ten-slot baseline. Never score/select the new five slots.
            for index, slot in enumerate(slots):
                require(slot["slot"] == index + 1 and slot["source_ranks"] == list(ORDER[index])
                        and slot["set_ids"] == subsets[index] and slot["source_eligible"] is True
                        and slot["score"] == math.fsum(logits[i - 1024] for i in subsets[index]), "baseline slot/score")
            winner = min(range(10), key=lambda i: (-slots[i]["score"], i))
            selection = record["selection"]
            require(selection["selected_slot"] == winner + 1
                    and selection["selected_set_ids"] == subsets[winner]
                    and selection["selected_source_eligible"] is True
                    and selection["fallback_no_valid_source"] is False, "baseline winner")
            is_exact = set(subsets[winner]) == truth
            require(record["selected_set_metrics"]["exact"] == is_exact, "baseline exact")
            has10, has15 = bool(exact_old), bool(exact_ranks)
            union_covers = truth <= set(subsets[-1])
            require(union_covers == old["union_covers_all_truth"], "union coverage")
            selected_exact += is_exact
            available10 += has10
            available15 += has15
            new_only += has15 and not has10
            covered_without_exact += union_covers and not has15
            absent += not union_covers
            rows.append({"seed": seed, "subject": record["subject"], "dimension": record["dimension"],
                         "group": record["group"], "baseline_exact": is_exact,
                         "existing_exact_slots": [i + 1 for i in exact_old],
                         "all15_exact_source_ranks": exact_ranks,
                         "added_exact_duplicate_witnesses": duplicate_witnesses,
                         "all15_canonical_subsets": [{"source_ranks": list(ranks), "set_ids": ids}
                            for ranks, ids in zip(ORDER, subsets, strict=True)],
                         "truth_ids": sorted(truth), "union_covers_all_truth": union_covers})
        totals.append({"seed": seed, "query_count": len(seen), "baseline_exact": selected_exact,
                       "ten_slot_exact_availability": available10, "fifteen_slot_exact_availability": available15,
                       "added_exact_availability": new_only, "covered_without_exact_subset": covered_without_exact,
                       "truth_absent_from_union": absent})
    require([r["baseline_exact"] for r in totals] == [191, 192, 186], "baseline totals")
    require([r["fifteen_slot_exact_availability"] for r in totals] == [191, 192, 187], "availability totals")
    require(sum(r["covered_without_exact_subset"] for r in totals) == 43
            and sum(r["truth_absent_from_union"] for r in totals) == 53, "failure categories")
    require(pair["comparison"]["pair_selector"]["exact_count"] == 569, "accepted baseline")
    require(all(sha(ROOT / path) == digest for path, digest in INPUTS.items()), "input changed")
    require(sha(Path(__file__)) == script_hash, "executing script changed")
    result = {"scope": "Proof from existing saved validation sets; not a new policy experiment",
              "all_checks_passed": True, "query_count": 666, "subset_inventories_checked": 9990,
              "baseline_exact_count": 569, "exact_availability_both_pools": 570,
              "added_exact_availability": 0, "exact_gains_upper_bound": 0,
              "fifteen_slot_selected_exact_upper_bound": 569,
              "covered_without_exact_subset": 43, "truth_absent_from_union": 53,
              "premises": {"unchanged_saved_FP32_logits": True, "canonical_ascending_ID_fsum": True,
                           "original_ten_slots_precede_added_five_on_ties": True,
                           "all_four_frozen_sources_eligible": True, "pool_order": [list(c) for c in ORDER]},
              "deduction": ["Every added exact set duplicates an exact set already in the ten-slot pool.",
                            "Identical canonical members and fixed logits imply identical scores.",
                            "An old inexact winner already beats or precedes every old exact slot.",
                            "An added duplicate exact slot cannot displace that winner under old-first ties.",
                            "Existing exact answers may be lost; no previously inexact answer can become exact."],
              "per_seed": totals, "observations": rows, "input_sha256": INPUTS,
              "script_sha256": script_hash, "python": platform.python_version(),
              "limitations": ["No new fifteen-slot scoring/selection, F1 sweep, model forward or training",
                              "No checkpoint payload, live source, corpus or protected-test validation",
                              "Authenticates pinned reports/audit/method files and vocabulary, not a full recursive archive audit",
                              "Bound applies only to these saved candidates, logits, eligibility and specified tie order",
                              "Does not establish the new policy's actual exact count or F1; no application promotion"]}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_checks_passed": True, "query_count": 666, "exact_upper_bound": 569,
                      "script_sha256": script_hash, "summary_sha256": sha(output)}))


if __name__ == "__main__":
    main()
