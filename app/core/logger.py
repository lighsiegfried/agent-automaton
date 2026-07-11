"""Central logging setup. Uses rich if available, plain stdlib otherwise."""

import logging

try:
    from rich.logging import RichHandler

    _handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
    _format = "%(name)s | %(message)s"
except ImportError:  # rich is optional
    _handler = logging.StreamHandler()
    _format = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

logging.basicConfig(level=logging.INFO, format=_format, datefmt="[%X]", handlers=[_handler])


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
