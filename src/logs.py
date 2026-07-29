import logging
from pathlib import Path


def configure_file_logger(name: str, path: str | Path) -> logging.Logger:
    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"tspdrone_rl.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | pid=%(process)d | %(message)s")
    )
    logger.addHandler(handler)
    return logger
