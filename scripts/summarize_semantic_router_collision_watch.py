"""Record a lightweight latest-collision watch for the semantic-router claim.

This script is intentionally offline/static: the live web search happens before
running it, and this artifact records the searched terms and readout so the
claim boundary has dated evidence. It does not train, infer, download, or call
the network.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "semantic_router_collision_watch"


DIRECT_QUERIES = [
    '"target metadata semantics" "transfer routing" cough',
    '"metadata field semantics" "transfer router"',
    '"slice-gated transfer" "calibration stacking"',
    '"safe slice gating" "metadata" audio',
]

BROAD_QUERIES = [
    '"metadata-gated" "respiratory" "audio" 2026',
    '"clinical metadata" "cough audio" "uncertainty" 2026',
    '"demographic metadata" "cough audio" "stacking"',
    '"metadata" "slice" "cough audio" "transfer"',
]

NEIGHBORS = [
    {
        "source": "Tuberculosis Screening from Cough Audio: Baseline Models, Clinical Variables, and Uncertainty Quantification",
        "url": "https://arxiv.org/abs/2601.07969",
        "decision": "adjacent_not_collision",
        "reason": "Cough-audio clinical-metadata fusion and protocol standardization, not field-semantics routing between slice-gated transfer and calibration stacking.",
    },
    {
        "source": "GLoRIA",
        "url": "https://arxiv.org/abs/2603.02464",
        "decision": "adjacent_not_collision",
        "reason": "Metadata-gated audio adaptation, not a deployment-time strategy router by metadata field safety.",
    },
    {
        "source": "PulmoVec",
        "url": "https://arxiv.org/abs/2603.15688",
        "decision": "adjacent_not_collision",
        "reason": "Demographic metadata stacking is crowded and adjacent to the calibration side, but not a safe/unsafe field-semantics rule.",
    },
]


def _table(rows: list[dict[str, str]]) -> str:
    cols = list(rows[0].keys()) if rows else []
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    payload = {
        "verdict": "NO_DIRECT_COLLISION_FOUND_IN_LATEST_WATCH",
        "checked_date": "2026-06-16",
        "direct_queries": DIRECT_QUERIES,
        "broad_queries": BROAD_QUERIES,
        "direct_query_readout": "No direct phrase collision was found for target metadata semantics, metadata field semantics, slice-gated transfer plus calibration stacking, or safe slice gating.",
        "broad_query_readout": "The broad clinical-metadata/cough-audio query surfaced the TB clinical-metadata baseline already added to the source ledger.",
        "neighbors": NEIGHBORS,
        "claim_boundary": "Keep the claim limited to metadata field semantics as a safety prior for choosing between slice-gated transfer and calibration stacking under cough dataset shift.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_collision_watch.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    query_rows = [{"type": "direct", "query": q} for q in DIRECT_QUERIES] + [
        {"type": "broad", "query": q} for q in BROAD_QUERIES
    ]
    lines = [
        "# Semantic-Router Collision Watch",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Checked date: `{payload['checked_date']}`",
        "",
        "## Queries",
        "",
        _table(query_rows),
        "",
        "## Readout",
        "",
        payload["direct_query_readout"],
        "",
        payload["broad_query_readout"],
        "",
        "## Adjacent Neighbors",
        "",
        _table(NEIGHBORS),
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
    ]
    (OUT / "SEMANTIC_ROUTER_COLLISION_WATCH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "neighbors": len(NEIGHBORS)}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_COLLISION_WATCH.md")


if __name__ == "__main__":
    main()
