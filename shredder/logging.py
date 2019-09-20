import sys
import logging


def setup_logging(level):
    if level == 'info':
        level = logging.INFO
    elif level == 'debug':
        level = logging.DEBUG
    elif level == 'warning':
        level = logging.WARNING
    elif level == 'error':
        level = logging.ERROR
    else:
        level = logging.CRITICAL

    logging.basicConfig(stream=sys.stdout, level=level)
