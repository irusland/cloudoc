import logging
import sys
from enum import Enum


class LogLevel(Enum):
    CONSOLE = 0
    LOGGING = 1

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_string(s: str):
        s = s.upper()
        if s in LogLevel._member_map_:
            return LogLevel[s]
        raise ValueError(s)


class Logger:
    LEVEL = LogLevel.LOGGING
    DEBUG_LOGGER = None
    CUT = 64

    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO,
                     fmt='%(levelname)s - %(asctime)s: %(message)s',
                     datefmt='%H:%M:%S'):
        # logging.basicConfig(format=fmt, datefmt=datefmt, level=level)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        handler = None

        if Logger.LEVEL == LogLevel.LOGGING:
            if log_file:
                handler = logging.FileHandler(log_file, mode='w+')
                formater = logging.Formatter(fmt=fmt, datefmt=datefmt)
                handler.setFormatter(formater)
            else:
                return
        elif Logger.LEVEL == LogLevel.CONSOLE:
            handler = logging.StreamHandler(stream=sys.stdout)
            formater = logging.Formatter(fmt=fmt, datefmt=datefmt)
            handler.setFormatter(formater)

        logger.addHandler(handler)
        return logger

    @staticmethod
    def configure(level=LogLevel.LOGGING, debug_path=None):
        Logger.LEVEL = level

        Logger.DEBUG_LOGGER = Logger.setup_logger(
            'debug_logger', debug_path, logging.DEBUG)

    @staticmethod
    def info(*args):
        if Logger.DEBUG_LOGGER:
            Logger.DEBUG_LOGGER.info(*args)

    @staticmethod
    def error(*args):
        if Logger.DEBUG_LOGGER:
            Logger.DEBUG_LOGGER.error(*args)

    @staticmethod
    def exception(*args):
        if Logger.DEBUG_LOGGER:
            Logger.DEBUG_LOGGER.exception(*args)
