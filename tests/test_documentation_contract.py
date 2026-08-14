import json
import re
import unittest
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

BANNED_PUBLIC_DOC_PATTERNS = [
    re.compile(r"private[- ]v[1-5](?:\.\d+)*", re.IGNORECASE),
    re.compile(r"pre[- ]release", re.IGNORECASE),
    re.compile(r"jkdaouvsafkukofa", re.IGNORECASE),
    re.compile(r"x11011x", re.IGNORECASE),
]


def vi_path(path: Path) -> Path:
    return path.with_name(path.stem + ".vi" + path.suffix)


class DocumentationContractTests(unittest.TestCase):
    def test_required_public_docs_have_vietnamese_pairs(self):
        for rel in REQUIRED_EN_DOCS:
            en = ROOT / rel
            vi = ROOT / vi_path(rel)
            self.assertTrue(en.is_file(), f"missing English document: {rel}")
            self.assertTrue(vi.is_file(), f"missing Vietnamese document: {vi.relative_to(ROOT)}")

    def test_every_markdown_document_has_bilingual_pair(self):
        markdown = sorted(
            p for p in ROOT.rglob("*.md")
            if ".git" not in p.parts
        )
        self.assertGreater(len(markdown), 0)
        for path in markdown:
            rel = path.relative_to(ROOT)
            if path.name.endswith(".vi.md"):
                en_name = path.name[:-6] + ".md"
                pair = path.with_name(en_name)
            else:
                pair = path.with_name(path.stem + ".vi.md")
            self.assertTrue(pair.is_file(), f"unpaired Markdown document: {rel}")

    def test_bilingual_markdown_pairs_have_equal_line_counts(self):
        for path in sorted(ROOT.rglob("*.md")):
            if path.name.endswith(".vi.md"):
                continue
            pair = path.with_name(path.stem + ".vi.md")
            self.assertTrue(pair.is_file(), f"missing pair for {path.relative_to(ROOT)}")
            en_lines = path.read_text(encoding="utf-8").splitlines()
            vi_lines = pair.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                len(en_lines), len(vi_lines),
                f"line-count mismatch: {path.relative_to(ROOT)}={len(en_lines)} "
                f"vs {pair.relative_to(ROOT)}={len(vi_lines)}",
            )

    def test_public_docs_do_not_expose_internal_release_history(self):
        targets = list(ROOT.rglob("*.md"))
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        notebook_markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb["cells"]
            if cell.get("cell_type") == "markdown"
        )
        texts = [(str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")) for path in targets]
        texts.append((str(NOTEBOOK.relative_to(ROOT)) + " [markdown]", notebook_markdown))
        for label, text in texts:
            for pattern in BANNED_PUBLIC_DOC_PATTERNS:
                self.assertIsNone(pattern.search(text), f"{label}: internal history matched {pattern.pattern}")

    def test_notebook_markdown_is_bilingual(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        markdown_cells = [
            "".join(cell.get("source", []))
            for cell in nb["cells"]
            if cell.get("cell_type") == "markdown"
        ]
        self.assertGreater(len(markdown_cells), 0)
        for index, text in enumerate(markdown_cells, 1):
            self.assertIn("**English:**", text, f"markdown cell {index} missing English section")
            self.assertIn("**Tiếng Việt:**", text, f"markdown cell {index} missing Vietnamese section")

    def test_github_community_files_and_manual_ci_trigger_exist(self):
        config = ROOT / ".github/ISSUE_TEMPLATE/config.yml"
        self.assertTrue(config.is_file())
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", ci)
        self.assertIn("scripts/check_docs.py", ci)


if __name__ == "__main__":
    unittest.main()
