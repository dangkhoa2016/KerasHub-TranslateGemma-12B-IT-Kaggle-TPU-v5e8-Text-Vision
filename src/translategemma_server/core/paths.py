from pathlib import Path
import logging, sys
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "log"
STATE_DIR = BASE_DIR / "state"
PRIVATE_DIR = BASE_DIR / ".private"
def configure_logging(name: str, path: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("translategemma_server")
    logger.handlers.clear(); logger.setLevel(logging.INFO); logger.propagate=False
    fmt=logging.Formatter(f"%(asctime)s | %(levelname)-7s | {name} | %(message)s")
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh=logging.FileHandler(path, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
