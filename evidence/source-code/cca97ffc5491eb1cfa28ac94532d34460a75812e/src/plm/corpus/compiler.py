"""Deterministic compiler from a graph snapshot to v1 protocol records."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from plm.graph.schema import connect
from plm.protocol.config import ProtocolSpec
from plm.protocol.tokenizer import Vocabulary

__all__ = ["CorpusManifest", "CorpusRecord", "canonical_graph_hash", "compile_corpus"]


class _TargetCandidate(TypedDict):
    key: str
    shared_count: int
    confidence: float


@dataclass(frozen=True)
class CorpusRecord:
    """One causal-LM training example, with loss masked through ``ANSWER``."""

    subject: str
    dimension: str
    targets: tuple[str, ...]
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class CorpusManifest:
    """Identity and counts for a byte-for-byte compiled corpus."""

    schema_version: int
    protocol_version: str
    tokenizer_hash: str
    graph_hash: str
    records_hash: str
    record_count: int
    vocabulary_size: int


def canonical_graph_hash(conn: sqlite3.Connection) -> str:
    """Hash graph structure and semantic edge attributes, excluding row IDs."""

    nodes = conn.execute("SELECT key, namespace, kind, status FROM nodes ORDER BY key").fetchall()
    relations = conn.execute("SELECT name, category FROM relation_types ORDER BY name").fetchall()
    edges = conn.execute(
        "SELECT src_node.key, dst_node.key, relation_types.name, edges.weight, "
        "edges.confidence, edges.is_canonical "
        "FROM edges "
        "JOIN nodes AS src_node ON src_node.id = edges.src "
        "JOIN nodes AS dst_node ON dst_node.id = edges.dst "
        "JOIN relation_types ON relation_types.id = edges.rel "
        "ORDER BY src_node.key, dst_node.key, relation_types.name"
    ).fetchall()
    payload = {
        "nodes": [list(row) for row in nodes],
        "relations": [list(row) for row in relations],
        "edges": [list(row) for row in edges],
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _targets(
    conn: sqlite3.Connection,
    subject: str,
    relation: str,
    target_order: tuple[str, ...],
) -> list[str]:
    rows = conn.execute(
        "WITH subject_attributes AS ("
        "  SELECT edges.dst FROM edges "
        "  JOIN nodes ON nodes.id = edges.src "
        "  JOIN relation_types ON relation_types.id = edges.rel "
        "  WHERE nodes.key = ? AND relation_types.name = ?"
        "), candidates AS ("
        "  SELECT candidate.key AS key, COUNT(DISTINCT edges.dst) AS shared_count, "
        "         MIN(edges.confidence) AS confidence "
        "  FROM edges "
        "  JOIN nodes AS candidate ON candidate.id = edges.src "
        "  JOIN relation_types ON relation_types.id = edges.rel "
        "  WHERE relation_types.name = ? AND edges.dst IN subject_attributes "
        "    AND candidate.kind = 'product' AND candidate.key != ? "
        "  GROUP BY candidate.key"
        ") SELECT key, shared_count, confidence FROM candidates",
        (subject, relation, relation, subject),
    ).fetchall()
    candidates: list[_TargetCandidate] = [
        {"key": str(row[0]), "shared_count": int(row[1]), "confidence": float(row[2])}
        for row in rows
    ]
    for criterion in reversed(target_order):
        reverse = criterion.endswith("_desc")
        if criterion.startswith("shared_attribute_count_"):
            candidates.sort(key=lambda item: item["shared_count"], reverse=reverse)
        elif criterion.startswith("confidence_"):
            candidates.sort(key=lambda item: item["confidence"], reverse=reverse)
        elif criterion.startswith("product_key_"):
            candidates.sort(key=lambda item: item["key"], reverse=reverse)
        else:  # validated by ProtocolSpec; defensive for direct private callers
            raise ValueError(f"unsupported target order criterion: {criterion}")
    return [item["key"] for item in candidates]


def compile_corpus(
    graph_db: str | Path,
    *,
    protocol: ProtocolSpec,
    output_dir: str | Path,
) -> CorpusManifest:
    """Compile all non-empty ``SAME`` buckets into JSONL and a manifest.

    The vocabulary is derived only from ``kind='product'`` nodes, so hidden
    attribute pivots can neither appear in input nor be generated as answers.
    """
    protocol.validate_for_compilation()
    mode = protocol.effective_modes[0]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conn = connect(graph_db)
    try:
        product_rows = conn.execute(
            "SELECT key FROM nodes WHERE kind = 'product' ORDER BY key"
        ).fetchall()
        products = [str(row[0]) for row in product_rows]
        vocab = Vocabulary.build_fresh(
            products,
            dimensions=tuple(protocol.dimensions),
            modes=protocol.effective_modes,
            version=protocol.version,
        )
        vocab_hash = vocab.save(output / "vocabulary.json")
        records: list[CorpusRecord] = []
        for subject in products:
            for dimension, relation in protocol.dimensions.items():
                targets = _targets(conn, subject, relation, protocol.effective_target_order)
                if not targets:
                    continue
                tokens = ["BOS", subject, dimension, mode, "ANSWER", *targets, "EOS"]
                input_ids = tuple(vocab.encode(tokens))
                answer_index = tokens.index("ANSWER")
                labels = tuple([-100] * (answer_index + 1) + list(input_ids[answer_index + 1 :]))
                records.append(
                    CorpusRecord(
                        subject=subject,
                        dimension=dimension,
                        targets=tuple(targets),
                        input_ids=input_ids,
                        labels=labels,
                    )
                )
        record_payload = "".join(
            json.dumps(asdict(record), ensure_ascii=True, separators=(",", ":")) + "\n"
            for record in records
        )
        records_hash = hashlib.sha256(record_payload.encode("utf-8")).hexdigest()
        (output / "records.jsonl").write_text(record_payload, encoding="utf-8", newline="\n")
        manifest = CorpusManifest(
            schema_version=1,
            protocol_version=protocol.version,
            tokenizer_hash=vocab_hash,
            graph_hash=canonical_graph_hash(conn),
            records_hash=records_hash,
            record_count=len(records),
            vocabulary_size=len(vocab),
        )
        (output / "manifest.json").write_text(
            json.dumps(asdict(manifest), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return manifest
    finally:
        conn.close()
