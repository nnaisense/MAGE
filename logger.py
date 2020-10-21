import logging
import os
import time


class DebugModuleFilter(logging.Filter):
    def filter(self, record):
        return record.module not in ['captured_function']


def configure_logger():
    os.environ['TZ'] = 'Europe/Warsaw'  # I want to log in CET timezone regardless of the machine is
    time.tzset()

    # TODO(minor): I would like to have "I/D/etc" for Info/Debug/etc but 15/25 for nonstandard levels. Non trivial if possible at all with Formatter
    logging.basicConfig(
        format='{asctime} | {levelname[0]}{levelno:<2d} | {funcName:23s} | {message}',
        datefmt='%H:%M:%S',
        style='{',
        level=logging.INFO,
    )

    for handler in logging.getLogger().handlers:
        handler.addFilter(DebugModuleFilter())
