from __future__ import annotations

import csv
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "related_work_matrix.csv"
ARXIV = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

QUERIES = [
    ("best_of_n", 'all:"best-of-n" OR all:"best of n" OR all:"Best-of-N"', 35),
    ("reward_hacking", 'all:"reward hacking" OR all:"reward overoptimization"', 35),
    ("reward_model_proxy", 'all:"reward model" AND all:"overoptimization"', 35),
    ("test_time_compute", 'all:"test-time compute" AND all:"language model"', 35),
    ("verifier_search", 'all:"verifier" AND all:"language model"', 35),
    ("self_consistency", 'all:"self-consistency" AND all:"chain of thought"', 30),
    ("energy_text", 'all:"energy-based model" AND all:"text generation"', 35),
    ("energy_transformer", 'all:"energy-based" AND all:"transformer"', 35),
    ("energy_diffusion_language", 'all:"energy-based" AND all:"diffusion" AND all:"language"', 30),
    ("energy_sampling", 'all:"Langevin dynamics" AND all:"energy-based model"', 30),
    ("preference_optimization", 'all:"direct preference optimization" OR all:"preference optimization"', 35),
    ("rejection_sampling_lm", 'all:"rejection sampling" AND all:"language model"', 30),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch(topic: str, query: str, max_results: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV}?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    rows: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", NS):
        arxiv_url = entry.findtext("atom:id", default="", namespaces=NS)
        arxiv_id = arxiv_url.rsplit("/", 1)[-1]
        published = entry.findtext("atom:published", default="", namespaces=NS)
        year = published[:4] if published else ""
        title = clean(entry.findtext("atom:title", default="", namespaces=NS))
        authors = "; ".join(
            clean(author.findtext("atom:name", default="", namespaces=NS))
            for author in entry.findall("atom:author", NS)
        )
        summary = clean(entry.findtext("atom:summary", default="", namespaces=NS))
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "year": year,
                "title": title,
                "authors": authors,
                "url": arxiv_url,
                "source": "arXiv",
                "theme": topic,
                "abstract_snippet": summary[:240],
            }
        )
    return rows


def threat_level(theme: str, title: str) -> str:
    text = f"{theme} {title}".lower()
    if any(term in text for term in ["best", "reward hacking", "overoptimization", "energy-based transformers"]):
        return "high"
    if any(term in text for term in ["verifier", "test-time", "energy", "preference"]):
        return "medium"
    return "low"


def relationship(theme: str, title: str) -> tuple[str, str]:
    text = f"{theme} {title}".lower()
    if "best" in text or "reward hacking" in text or "overoptimization" in text:
        return (
            "BoN/proxy-optimization prior",
            "Threatens novelty of any generic reward-hacking claim; leaves room for EBM-specific mechanism.",
        )
    if "energy" in text or "langevin" in text:
        return (
            "EBM/energy prior",
            "Threatens novelty of broad EBM sampling claims; supports architecture-specific diagnostic framing.",
        )
    if "verifier" in text or "test-time" in text or "self-consistency" in text:
        return (
            "test-time compute prior",
            "Shows sample-and-select gains are well known; motivates studying failure modes under proxy scoring.",
        )
    if "preference" in text:
        return (
            "alignment prior",
            "Connects proxy misspecification and KL/proximity repairs to inference-time selection.",
        )
    return ("background", "Contextual related work.")


def main() -> None:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for idx, (topic, query, max_results) in enumerate(QUERIES):
        try:
            batch = fetch(topic, query, max_results)
        except Exception as exc:  # pragma: no cover - only used for live collection failures
            print(f"warning: {topic} failed: {exc}")
            batch = []
        for row in batch:
            key = row["arxiv_id"].split("v", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            relation, note = relationship(row["theme"], row["title"])
            row["id"] = f"RW{len(rows) + 1:03d}"
            row["relevance_to_this_paper"] = relation
            row["threat_level"] = threat_level(row["theme"], row["title"])
            row["novelty_relationship"] = note
            rows.append(row)
        if idx != len(QUERIES) - 1:
            time.sleep(3.1)

    if len(rows) < 100:
        raise RuntimeError(f"literature sweep returned only {len(rows)} unique rows")

    fieldnames = [
        "id",
        "year",
        "title",
        "authors",
        "source",
        "arxiv_id",
        "url",
        "theme",
        "relevance_to_this_paper",
        "threat_level",
        "novelty_relationship",
        "abstract_snippet",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT} with {len(rows)} rows")


if __name__ == "__main__":
    main()
