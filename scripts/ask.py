"""Ask a single clinical question and inspect the full pipeline.

    python -m scripts.ask "65yo M John Smith MRN 12345 with chest pain, what workup?"

Shows the pseudonymised query, the retrieved+reranked context, the privacy gate
result, and the (re-identified) answer.
"""
from __future__ import annotations

import sys

import scripts  # noqa: F401  (path bootstrap)

from clinical_rag.pipeline import build_pipeline


def main(argv: list[str]) -> None:
    if not argv or not argv[0].strip():
        print('Usage: python -m scripts.ask "your clinical question"')
        raise SystemExit(2)
    query = argv[0]

    pipe = build_pipeline()
    print(f"Pseudonymisation backend: {pipe.pseudonymizer.backend}\n")

    result = pipe.answer(query)

    print("=" * 78)
    print("ORIGINAL QUERY (contains PHI):")
    print(" ", result.query)
    print("\nPSEUDONYMISED QUERY (what the system actually processes):")
    print(" ", result.deid.text)
    if result.deid.mapping:
        print("\n  PHI mapping (kept local only):")
        for token, value in result.deid.mapping.items():
            print(f"    {token} -> {value}")

    print("\n" + "-" * 78)
    print(f"RETRIEVED CONTEXT (top {len(result.retrieved)}, reranked="
          f"{any(r.reranked for r in result.retrieved)}):")
    for item in result.prompt.context:
        print(f"  [{item.label}] ({item.doc_id})  {item.text[:110]}...")

    print("\n" + "-" * 78)
    leaked = "NONE ✅" if not result.phi_leaks else f"LEAK ❌ {result.phi_leaks}"
    print(f"PRIVACY GATE — raw PHI in outbound prompt: {leaked}")

    print("\n" + "-" * 78)
    print("ANSWER (re-identified for clinician):")
    print(" ", result.answer)
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1:])
