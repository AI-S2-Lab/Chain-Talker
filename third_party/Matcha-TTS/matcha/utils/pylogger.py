import logging

from functools import wraps
import os


def rank_zero_only(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if int(os.environ.get('RANK', '0')) == 0:
            return function(*args, **kwargs)
    return wrapped


def get_pylogger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
