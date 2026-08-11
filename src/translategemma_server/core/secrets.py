import os, secrets
from pathlib import Path
def load_or_create_secret(env_name: str, file_path: Path, required: bool) -> str:
    value=os.environ.get(env_name, "").strip()
    if value: return value
    try:
        value=file_path.read_text(encoding="utf-8").strip()
        if value: return value
    except FileNotFoundError: pass
    if not required: return ""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    value=secrets.token_urlsafe(32); tmp=file_path.with_suffix(file_path.suffix+".tmp")
    tmp.write_text(value+"\n", encoding="utf-8"); os.chmod(tmp,0o600); tmp.replace(file_path); os.chmod(file_path,0o600)
    return value
