import logging
import logging.config
from pathlib import Path

from src.core.config import load_config


def setup_logging():
    config = load_config()
    log_level = config.logging.level.upper()
    log_file = config.logging.path

    # Ensure the parent directory for logs exists to avoid file writing errors
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Standard Dictionary Configuration Mapping using pythonjsonlogger
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                # The explicit fields to embed into your JSON output properties
                "format": "%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "level": log_level,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "level": log_level,
                "filename": str(log_file),
                "maxBytes": 10485760,  # 10MB file rotation safety threshold
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "root": {"handlers": ["console", "file"], "level": log_level},
    }

    logging.config.dictConfig(logging_config)


# Initialize logging structure immediately upon module access
setup_logging()
logger = logging.getLogger("stock_platform")
