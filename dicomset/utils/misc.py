import sys
from typing import Callable

from .logging import logger

def is_windows() -> bool:
    return 'win' in sys.platform

def with_makeitso(
    makeitso: bool,
    f: Callable,
    message: str | None = None,
    ) -> None:
    if makeitso:
        f()
        if message is not None:
            logger.info(message)
    else:
        if message is not None:
            logger.info(f"Would run: {message}")
