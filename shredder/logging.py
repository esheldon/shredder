import sys
import logging


def setup_logging(level):
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
