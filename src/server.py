#!/usr/bin/env python3
"""TranslateGemma 12B IT REST server for Kaggle TPU v5e-8.

The coordinator process runs HTTP/queue/lifecycle code only. JAX, Keras and
KerasHub are imported by the spawned TPU worker after TPU environment setup.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import threading

from translategemma_server.api.app import Runtime, create_app
from translategemma_server.core.config import Config
from translategemma_server.core.paths import LOG_DIR, configure_logging
from translategemma_server.workers.manager import TranslationManager

logger = logging.getLogger("translategemma_server")


def main() -> int:
    mp.freeze_support()
    configure_logging("api", LOG_DIR / "server.log")

    try:
        config = Config.from_env()
    except Exception:
        logger.exception("Invalid configuration")
        return 2

    manager = TranslationManager(config)
    runtime = Runtime(config, manager)
    app = create_app(runtime)

    # Delayed import keeps non-serving project tools independent of Werkzeug.
    from werkzeug.serving import make_server

    runtime.server = make_server(
        config.host,
        config.port,
        app,
        threaded=True,
    )
    manager.start_async()

    def request_shutdown(_signum, _frame) -> None:
        if runtime.shutdown_started.is_set():
            return
        runtime.shutdown_started.set()

        def shutdown() -> None:
            manager.shutdown(True, config.shutdown_timeout)
            runtime.server.shutdown()

        threading.Thread(
            target=shutdown,
            name="signal-shutdown",
            daemon=False,
        ).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    try:
        runtime.server.serve_forever()
    finally:
        if not runtime.shutdown_started.is_set():
            manager.shutdown(False, 10)
        runtime.server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
