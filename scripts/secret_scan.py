#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

SKIP_DIRS = {
    ".git", "log", "state", ".private", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".worktrees", "worktrees",
}
RUNTIME_SECRET_PATHS = {
    PurePosixPath(".env"),
    PurePosixPath("data/api_key.txt"),
    PurePosixPath("data/restart_secret.txt"),
    PurePosixPath("data/tunnel_url.txt"),
}
SECRET_FILE_NAMES = {
    "authorized_keys",
    "connection.txt",
    "ssh_host_ed25519_key",
    "ssh_host_rsa_key",
    "id_ed25519",
    "id_rsa",
}
CONTENT_PATTERNS = (
    ("private key material", re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----")),
    (
        "credential assignment",
        re.compile(
            r"(?im)\b(?:API_KEY|RESTART_SECRET|NGROK_AUTHTOKEN)[ \t]*=[ \t]*[\"']?"
            r"([A-Za-z0-9_./:+-]{24,})"
        ),
    ),
    (
        "bearer token",
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{32,}"),
    ),
)


def _relative_runtime_secret(rel: PurePosixPath) -> bool:
    if rel == PurePosixPath(".env"):
        return True
    if rel in RUNTIME_SECRET_PATHS:
        return True
    if rel.name in SECRET_FILE_NAMES:
        return True
    if rel.name.startswith("ssh_host_") and not rel.name.endswith(".pub"):
        return True
    return False


def _content_issues(label: str, text: str) -> list[str]:
    issues: list[str] = []
    for name, pattern in CONTENT_PATTERNS:
        if pattern.search(text):
            issues.append(f"{label}: possible {name}")
    return issues


def scan_tree(root: Path) -> list[str]:
    root = Path(root)
    issues: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(root).as_posix())
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if _relative_runtime_secret(rel):
            issues.append(f"{rel}: private runtime/credential file must not be published")
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        issues.extend(_content_issues(str(rel), text))
    return issues


def scan_zip(path: Path) -> list[str]:
    issues: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            full = PurePosixPath(info.filename)
            rel = PurePosixPath(*full.parts[1:]) if len(full.parts) > 1 else full
            if any(part in SKIP_DIRS for part in rel.parts):
                issues.append(f"{info.filename}: excluded runtime directory present in archive")
                continue
            if _relative_runtime_secret(rel):
                issues.append(f"{info.filename}: private runtime/credential file present in archive")
                continue
            if info.file_size > 2_000_000:
                continue
            try:
                text = zf.read(info).decode("utf-8")
            except UnicodeDecodeError:
                continue
            issues.extend(_content_issues(info.filename, text))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan a source tree or ZIP for private credential leaks")
    parser.add_argument("path", nargs="?", default=".")
    args = parser.parse_args()
    path = Path(args.path)
    issues = scan_zip(path) if path.is_file() and path.suffix.lower() == ".zip" else scan_tree(path)
    if issues:
        for issue in issues:
            print(f"[secret-scan] ERROR: {issue}")
        return 1
    print(f"[secret-scan] PASS: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
