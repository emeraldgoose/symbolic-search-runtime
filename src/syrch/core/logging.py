from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import cast


@dataclass
class LogConfig:
    level: str = "WARNING"
    format: str = "text"
    file: str | None = None


_initialized = False


def setup_logging(config: LogConfig | None = None) -> None:
    global _initialized
    if _initialized:
        return

    if config is None:
        config = LogConfig()

    logger = logging.getLogger("syrch")
    logger.setLevel(config.level.upper())
    logger.handlers.clear()

    if config.format == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    if config.file:
        handler: logging.Handler = logging.FileHandler(config.file)
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(cast(logging.Formatter, formatter))
    logger.addHandler(handler)

    _initialized = True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json

        return json.dumps(
            {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            },
            ensure_ascii=False,
        )
