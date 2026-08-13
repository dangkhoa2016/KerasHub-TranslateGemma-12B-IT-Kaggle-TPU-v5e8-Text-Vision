#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TOP = {".git", "log", "state", ".private", "__pycache__"}
EXCLUDED_FILES = {
    Path(".env"),
    Path("data/api_key.txt"),
    Path("data/restart_secret.txt"),
    Path("data/tunnel_url.txt"),
    Path("PRIVATE-BACKUP-WARNING.txt"),
    Path("PRIVATE-BACKUP-MANIFEST.json"),
    Path("SOURCE-MANIFEST.sha256"),
}


def skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if rel in EXCLUDED_FILES:
        return True
    if rel.parts and rel.parts[0] in EXCLUDED_TOP:
        return True
    if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache"} for part in rel.parts):
        return True
    if path.is_file() and path.suffix in {".pyc", ".pyo", ".zip"}:
        return True
    return False


def _source_files(root: Path, output: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() != output and not skip(path, root)
    ]


def _manifest(files: list[Path], root: Path) -> str:
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + ("\n" if lines else "")


def _load_secret_scanner():
    path = ROOT / "scripts/secret_scan.py"
    spec = importlib.util.spec_from_file_location("translategemma_secret_scan", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def create_source_zip(output: Path, root: Path = ROOT) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = root.name
    files = _source_files(root, output)
    manifest = _manifest(files, root)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Add directories for convenient extraction, excluding runtime trees.
        for path in sorted(root.rglob("*")):
            if not path.is_dir() or skip(path, root):
                continue
            rel = path.relative_to(root)
            info = zipfile.ZipInfo.from_file(path, f"{prefix}/{rel.as_posix()}/")
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, b"")
        for path in files:
            rel = path.relative_to(root)
            info = zipfile.ZipInfo.from_file(path, f"{prefix}/{rel.as_posix()}")
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
        zf.writestr(f"{prefix}/SOURCE-MANIFEST.sha256", manifest)

    issues = _load_secret_scanner().scan_zip(output)
    if issues:
        output.unlink(missing_ok=True)
        raise RuntimeError("Clean source secret scan failed:\n- " + "\n- ".join(issues))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default=str(ROOT.parent / (ROOT.name + ".zip")))
    args = parser.parse_args()
    print(create_source_zip(Path(args.output)))


if __name__ == "__main__":
    main()
