# Error Warning and Trapping System
# ewts/config.py

import logging
import sys
import os

from .constants import (
    MODULE_NAME,
    EV_EWTS_LOGGING,
    EV_MODULE_LOGLEVEL,
    LOG_MODULE_NAME_LEN,
)
from .formatter import CustomFormatter
from .paths import get_log_file_path

def translate_ngwpc_log_level(level: str) -> str:
    level = level.strip().upper()
    return {
        "SEVERE": "ERROR",
        "FATAL": "CRITICAL",
    }.get(level, level)


def force_info(handler, logger, msg, *args):
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        msg,
        args,
        None,
    )
    handler.emit(record)


def configure_logging():
    '''
    Set logging level and specify logger configuration based on environment variables set by ngen
    '''
    logger = logging.getLogger(MODULE_NAME)

    if getattr(logger, "_initialized", False):
        return logger # logger already initialized, nothing else to do

    # Default to enabled if flag not set or is set to DISABLED
    raw_value = os.getenv(EV_EWTS_LOGGING)
    enabled = raw_value != "DISABLED"
    if raw_value is None:
        print(f"{EV_EWTS_LOGGING} not set; logging ENABLED by default")

    if not enabled:
        logger.disabled = True
        logger._initialized = True
        print(f"Module {MODULE_NAME} Logging DISABLED", flush=True)
        return logger

    print(f"Module {MODULE_NAME} Logging ENABLED", flush=True)

    logFilePath, appendEntries = get_log_file_path()

    handler = (
        logging.FileHandler(logFilePath, mode="a" if appendEntries else "w")
        if logFilePath
        else logging.StreamHandler(sys.stdout)
    )

    log_level = translate_ngwpc_log_level(
        os.getenv(EV_MODULE_LOGLEVEL, "INFO")
    )

    module_fmt = MODULE_NAME.upper().ljust(LOG_MODULE_NAME_LEN)[:LOG_MODULE_NAME_LEN]

    formatter = CustomFormatter(
        fmt=f"%(asctime)s.%(msecs)03d {module_fmt} %(levelname_padded)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Setup logger
    logger.handlers.clear() # Clear any default handlers
    logger.setLevel(log_level)
    logger.addHandler(handler)

    # Write log level INFO message to log regradless of the actual log level
    force_info(handler, logger, "Log level set to %s", log_level)
    print(f"Module {MODULE_NAME} Log Level set to {log_level}", flush=True)

    logger._initialized = True
    return logger
