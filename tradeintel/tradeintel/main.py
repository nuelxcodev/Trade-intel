"""Process entry point: start the background worker and serve the API on $PORT.

    python -m tradeintel.main

The worker and the API run in one process because they share one SQLite file.
"""

from __future__ import annotations

import logging
import os
import threading

from .api import config_from_env, make_server
from .service import worker_loop


def main() -> None:
    logging.basicConfig(level=os.environ.get("TI_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("tradeintel")
    cfg = config_from_env()
    if not cfg.api_key:
        log.warning("TI_API_KEY is not set: read endpoints are open and write endpoints are disabled")
    threading.Thread(target=worker_loop, name="worker", daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    log.info("serving on 0.0.0.0:%d, database %s", port, cfg.db_path)
    make_server(cfg, "0.0.0.0", port).serve_forever()


if __name__ == "__main__":
    main()
