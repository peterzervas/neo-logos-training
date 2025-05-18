import logging
from pathlib import Path
from typing import Optional

def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(sh)
    return logger
