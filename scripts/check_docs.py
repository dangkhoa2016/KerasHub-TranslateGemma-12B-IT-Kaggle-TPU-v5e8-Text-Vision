#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb"

REQUIRED_EN_DOCS = [
    Path("README.md"),
    Path("CHANGELOG.md"),
    Path("CONTRIBUTING.md"),
    Path("SECURITY.md"),
    Path("RELEASE_NOTES_v1.0.0.md"),
    Path("NOTICE.md"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("CODE_OF_CONDUCT.md"),
    Path("clients/README.md"),
    Path("docs/API.md"),
    Path("docs/ARCHITECTURE.md"),
    Path("docs/KAGGLE.md"),
    Path("docs/BENCHMARKS.md"),
    Path(".github/SUPPORT.md"),
    Path(".github/PULL_REQUEST_TEMPLATE.md"),
    Path(".github/ISSUE_TEMPLATE/bug_report.md"),
    Path(".github/ISSUE_TEMPLATE/feature_request.md"),
]

BANNED_PATTERNS = [
    re.compile(r"private[- ]v[1-5](?:\.\d+)*", re.IGNORECASE),
    re.compile(r"pre[- ]release", re.IGNORECASE),
    re.compile(r"jkdaouvsafkukofa", re.IGNORECASE),
    re.compile(r"x11011x", re.IGNORECASE),
]


def vietnamese_pair(path: Path) -> Path:
    return path.with_name(path.stem + ".vi" + path.suffix)


def english_pair(path: Path) -> Path:
    if not path.name.endswith(".vi.md"):
        raise ValueError(path)
    return path.with_name(path.name[:-6] + ".md")


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED_EN_DOCS:
        en = ROOT / rel
        vi = ROOT / vietnamese_pair(rel)
        if not en.is_file():
            errors.append(f"missing English document: {rel}")
        if not vi.is_file():
            errors.append(f"missing Vietnamese document: {vi.relative_to(ROOT)}")

    markdown = sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts)
    for path in markdown:
        pair = english_pair(path) if path.name.endswith(".vi.md") else vietnamese_pair(path)
        if not pair.is_file():
            errors.append(f"unpaired Markdown document: {path.relative_to(ROOT)}")
            continue
        if not path.name.endswith(".vi.md"):
            en_count = line_count(path)
            vi_count = line_count(pair)
            if en_count != vi_count:
                errors.append(
                    f"line-count mismatch: {path.relative_to(ROOT)}={en_count} "
                    f"vs {pair.relative_to(ROOT)}={vi_count}"
                )

        text = path.read_text(encoding="utf-8")
        for pattern in BANNED_PATTERNS:
            if pattern.search(text):
                errors.append(f"internal-history wording in {path.relative_to(ROOT)}: {pattern.pattern}")

    if NOTEBOOK.is_file():
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        markdown_cells = [
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "markdown"
        ]
        if not markdown_cells:
            errors.append("notebook has no Markdown cells")
        for index, text in enumerate(markdown_cells, 1):
            if "**English:**" not in text:
                errors.append(f"notebook Markdown cell {index} is missing English text")
            if "**Tiếng Việt:**" not in text:
                errors.append(f"notebook Markdown cell {index} is missing Vietnamese text")
            for pattern in BANNED_PATTERNS:
                if pattern.search(text):
                    errors.append(f"internal-history wording in notebook Markdown cell {index}: {pattern.pattern}")
    else:
        errors.append(f"missing notebook: {NOTEBOOK.relative_to(ROOT)}")

    if errors:
        print("[docs] FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"[docs] PASS: {len(markdown)} Markdown files, bilingual pairs and equal line counts verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
