from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "results" / "expansion" / "claims.json"
PAPER = ROOT / "paper" / "energy_tail_audit.tex"
FINAL_PDF = ROOT / "paper" / "final" / "best-of-n-ebm-transformer-v3.pdf"
DESKTOP_PDF = Path.home() / "OneDrive" / "Desktop" / "best-of-n-ebm-transformer-v3.pdf"

STALE_PATTERNS = [
    "best-of-n-ebm-transformer-" + "v" + "2",
    "inference " + "value " + "theorem",
]


def _pdf_page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


def main() -> int:
    failures: list[str] = []
    if not CLAIMS.exists():
        failures.append(f"missing claims file: {CLAIMS}")
    else:
        claims = json.loads(CLAIMS.read_text(encoding="utf-8"))
        if claims.get("claim_pass") is not True:
            failures.append("claim_pass is false")
        for name, value in claims.get("checks", {}).items():
            if value is not True:
                failures.append(f"claim check failed: {name}={value}")

    paper_text = PAPER.read_text(encoding="utf-8").lower()
    for pattern in STALE_PATTERNS:
        if pattern.lower() in paper_text:
            failures.append(f"stale text found in paper: {pattern}")

    for pdf in [FINAL_PDF, DESKTOP_PDF]:
        if not pdf.exists():
            failures.append(f"missing PDF: {pdf}")
            continue
        pages = _pdf_page_count(pdf)
        if pages < 25:
            failures.append(f"PDF has only {pages} pages: {pdf}")

    if failures:
        print("claim audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("claim audit complete: submission-ready v3")
    print(f"claims: {CLAIMS}")
    print(f"final: {FINAL_PDF}")
    print(f"desktop: {DESKTOP_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
