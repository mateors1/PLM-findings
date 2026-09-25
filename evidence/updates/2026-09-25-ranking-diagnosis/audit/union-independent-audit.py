"""Separate bitmask/set-arithmetic audit; does not import the producing proof."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROOF_SHA = "2a45afb5c218d03cc72ae6fb5bfc6db0ab9aaa35aa49cde3d78b8bd4ca2c0c6a"
SUMMARY_SHA = "57cc984b16583bc610997ee052fc2f9d8f77a8af9061372578fed79e867f3a70"
PAIR_SHA = "4e4c46d4ebc366e02aa04cc3abda1a65cc08cec5463453ba97e25b4bc3f53992"
INPUTS = {}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, expected):
    require(digest(path) == expected, f"hash mismatch: {path}")
    INPUTS[path.relative_to(ROOT).as_posix()] = expected
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    output = HERE / "independent-audit.json"
    require(not output.exists(), "refuse immutable output overwrite")
    own_sha = digest(Path(__file__))
    summary = read(HERE / "summary.json", SUMMARY_SHA)
    require(summary["script_sha256"] == PROOF_SHA and digest(HERE / "proof.py") == PROOF_SHA,
            "producer binding")
    INPUTS[(HERE / "proof.py").relative_to(ROOT).as_posix()] = PROOF_SHA
    for name, expected in summary["input_sha256"].items():
        require(digest(ROOT / name) == expected, f"proof input drift: {name}")
        INPUTS[name] = expected
    pair = read(ROOT / "runs/learning/pair-unions-v1/summary.json", PAIR_SHA)
    vocabulary_names = [n for n in INPUTS if n.endswith("/vocabulary.json")]
    require(len(vocabulary_names) == 1, "vocabulary identity")
    tokens = read(ROOT / vocabulary_names[0], INPUTS[vocabulary_names[0]])["tokens"]
    mapping = {name: index for index, name in enumerate(tokens)}
    # Enumerate 0001..1111, independently of itertools.combinations in proof.py.
    masks = list(range(1, 16))
    ranks = {mask: tuple(bit + 1 for bit in range(4) if mask & (1 << bit)) for mask in masks}
    ordered_masks = sorted(masks, key=lambda mask: (len(ranks[mask]), ranks[mask]))
    require([list(ranks[m]) for m in ordered_masks[:10]] == pair["pool_order"], "baseline order")
    require(summary["premises"] == {
        "unchanged_saved_FP32_logits": True, "canonical_ascending_ID_fsum": True,
        "original_ten_slots_precede_added_five_on_ties": True,
        "all_four_frozen_sources_eligible": True,
        "pool_order": [list(ranks[m]) for m in ordered_masks]}, "declared premises")
    observed = {(r["seed"], r["subject"], r["dimension"]): r for r in summary["observations"]}
    require(len(observed) == len(summary["observations"]) == 666, "summary unique observations")
    totals, witness_count = [], 0
    visited = set()
    for seed in (1729, 1730, 1731):
        filename = f"seed-{seed}.json"
        report = read(ROOT / "runs/learning/pair-unions-v1" / filename, pair["report_sha256"][filename])
        require(report["seed"] == seed and len(report["responses"]) == 222, "seed coverage")
        counts = dict(seed=seed, query_count=0, baseline_exact=0, ten_slot_exact_availability=0,
                      fifteen_slot_exact_availability=0, added_exact_availability=0,
                      covered_without_exact_subset=0, truth_absent_from_union=0)
        for row in report["responses"]:
            key = (seed, row["subject"], row["dimension"])
            require(key not in visited and key in observed, "aligned unique row")
            visited.add(key)
            saved = observed[key]
            truth = {mapping[name] for name in row["expected"]}
            require(saved["truth_ids"] == sorted(truth) and saved["group"] == row["group"], "truth/group")
            require(len(row["source_paths"]) == 4, "four raw paths")
            source_sets = []
            for index, path in enumerate(row["source_paths"]):
                ids = path["token_ids"]
                require(path["first_rank"] == index + 1 and path["eligible"] is True
                        and path["terminated"] is True and path["protocol_valid"] is True
                        and path["error"] is None, "raw source eligibility")
                expected_prompt = [mapping[name] for name in
                                   ("BOS", row["subject"], row["dimension"], "SAME", "ANSWER")]
                require(ids[:5] == expected_prompt and ids[-1] == mapping["EOS"]
                        and all(1024 <= i < 2049 for i in ids[5:-1])
                        and len(ids[5:-1]) == len(set(ids[5:-1])), "raw grammar/uniqueness")
                require([tokens[i] for i in ids[5:-1]] == path["predicted"], "raw target receipt")
                members = set(ids[5:-1]) - {mapping[row["subject"]]}
                require(members == {mapping[name] for name in path["processed_targets"]}, "processed source")
                source_sets.append(members)
            by_mask = {}
            for mask in masks:
                members = set()
                for bit in range(4):
                    if mask & (1 << bit):
                        members.update(source_sets[bit])
                by_mask[mask] = members
            expected_subsets = [{"source_ranks": list(ranks[m]), "set_ids": sorted(by_mask[m])}
                                for m in ordered_masks]
            require(saved["all15_canonical_subsets"] == expected_subsets, "all15 canonical inventories")
            exact_masks = [m for m in ordered_masks if by_mask[m] == truth]
            exact_old = [i + 1 for i, m in enumerate(ordered_masks[:10]) if by_mask[m] == truth]
            require(saved["all15_exact_source_ranks"] == [list(ranks[m]) for m in exact_masks]
                    and saved["existing_exact_slots"] == exact_old, "exact subset inventory")
            witnesses = []
            for index in range(10, 15):
                mask = ordered_masks[index]
                if mask not in exact_masks:
                    continue
                require(bool(exact_old), "new exact availability violates bound")
                first_duplicate = next(old for old in exact_old
                                       if by_mask[ordered_masks[old - 1]] == by_mask[mask])
                witnesses.append({"added_slot": index + 1, "existing_slot": first_duplicate,
                                  "canonical_set_ids": sorted(by_mask[mask])})
            require(saved["added_exact_duplicate_witnesses"] == witnesses, "duplicate witnesses")
            witness_count += len(witnesses)
            # Replay only the frozen ten-slot decision, never score the added five slots.
            slots = row["selection"]["slots"]
            require(len(slots) == 10, "baseline slot count")
            logits = row["symmetric_relation_logits"]
            require(len(logits) == 1025 and all(math.isfinite(v) for v in logits), "saved logits")
            best_score, best_index = -math.inf, None
            for index in range(10):
                members = sorted(by_mask[ordered_masks[index]])
                score = math.fsum(logits[i - 1024] for i in members)
                slot = slots[index]
                require(slot["slot"] == index + 1 and slot["set_ids"] == members
                        and slot["source_ranks"] == list(ranks[ordered_masks[index]])
                        and slot["source_eligible"] is True and slot["score"] == score, "baseline score/slot")
                if score > best_score:
                    best_score, best_index = score, index
            require(best_index is not None, "baseline winner absent")
            selection = row["selection"]
            chosen = by_mask[ordered_masks[best_index]]
            require(selection["selected_slot"] == best_index + 1
                    and selection["selected_set_ids"] == sorted(chosen)
                    and selection["selected_source_eligible"] is True
                    and selection["fallback_no_valid_source"] is False, "baseline decision")
            exact = chosen == truth
            require(saved["baseline_exact"] == exact == row["selected_set_metrics"]["exact"], "baseline exact")
            covers = truth <= by_mask[15]
            require(saved["union_covers_all_truth"] == covers, "coverage classification")
            counts["query_count"] += 1
            counts["baseline_exact"] += exact
            counts["ten_slot_exact_availability"] += bool(exact_old)
            counts["fifteen_slot_exact_availability"] += bool(exact_masks)
            counts["added_exact_availability"] += bool(exact_masks) and not exact_old
            counts["covered_without_exact_subset"] += covers and not exact_masks
            counts["truth_absent_from_union"] += not covers
        totals.append(counts)
    require(totals == summary["per_seed"] and len(visited) == 666, "all per-seed totals")
    require([r["baseline_exact"] for r in totals] == [191, 192, 186]
            and [r["fifteen_slot_exact_availability"] for r in totals] == [191, 192, 187]
            and not any(r["added_exact_availability"] for r in totals), "fixed bound premises")
    for field, expected in {"query_count": 666, "subset_inventories_checked": 9990,
                            "baseline_exact_count": 569, "exact_availability_both_pools": 570,
                            "added_exact_availability": 0, "exact_gains_upper_bound": 0,
                            "fifteen_slot_selected_exact_upper_bound": 569,
                            "covered_without_exact_subset": 43, "truth_absent_from_union": 53}.items():
        require(summary[field] == expected, f"summary count: {field}")
    require(all(digest(ROOT / name) == value for name, value in INPUTS.items()), "final input drift")
    require(digest(Path(__file__)) == own_sha, "audit script drift")
    result = {"complete": True, "all_arithmetic_and_bindings_equal": True,
              "summary_sha256": SUMMARY_SHA, "proof_script_sha256": PROOF_SHA,
              "audit_script_sha256": own_sha, "input_sha256": INPUTS,
              "query_count": 666, "subset_inventory_count": 9990,
              "duplicate_exact_witnesses_checked": witness_count, "per_seed": totals,
              "exact_gains_upper_bound": 0, "selected_exact_upper_bound": 569,
              "method": "Bitmask enumeration from pinned raw paths; separate baseline winner loop; no producer imports",
              "limitations": ["No new15-slot score selection, metric sweep, model execution or protected-test access",
                              "Separate implementation, same working agent; not an independently staffed model replay",
                              "Authenticates proof-bound files; does not redo full recursive historical provenance"]}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"PASS": True, "summary_sha256": digest(output), "script_sha256": own_sha}))


if __name__ == "__main__":
    main()
