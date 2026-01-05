# Error Warning and Trapping System
# ewts/paths.py

import getpass
import os
from datetime import datetime, timezone

from .constants import (
    MODULE_NAME,
    EV_NGEN_LOGFILEPATH,
    EV_MODULE_LOGFILEPATH,
    DS,
    LOG_DIR_DEFAULT,
    LOG_DIR_NGENCERF,
    LOG_FILE_EXT,
)

def create_timestamp(date_only=False, iso=False, append_ms=False):
    now = datetime.now(timezone.utc)

    if date_only:
        ts = now.strftime("%Y%m%d")
    elif iso:
        ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        ts = now.strftime("%Y%m%dT%H%M%S")

    if append_ms:
        ts += f".{now.microsecond // 1000:03d}"

    return ts

def get_log_file_path():
    # Determine the log file path using the following precedence:
    # 1) Use the ngen-provided log file path if available in the NGEN_LOG_FILE_PATH environment variable
    # 3) Otherwise, create a default module-specific log file using the module name and a UTC timestamp.
    # 3.1) First create a subdirectory under the ngenCERF data directory if available, otherwise the user home directory.
    # 3.2) Next create a subdirectory name using the username, if available, otherwise use the YYYYMMDD.
    # 3.3) Attempt to open the log file and upon failure, use stdout.

    appendEntries = True
    moduleLogFileExists = False

     # Determine if a log file has laready been opened for this module (either the ngen log or default)
    moduleEnvVar = os.getenv(EV_MODULE_LOGFILEPATH, "")
    if moduleEnvVar:
        logFilePath = moduleEnvVar
        moduleLogFileExists = True
    else:
        ngenEnvVar = os.getenv(EV_NGEN_LOGFILEPATH, "")
        if ngenEnvVar:
            logFilePath = ngenEnvVar
        else:
            print(f"Module {MODULE_NAME} Env var {EV_NGEN_LOGFILEPATH} not found. Creating default log name.")
            appendEntries = False
            baseDir = (
                f"{LOG_DIR_NGENCERF}{DS}{LOG_DIR_DEFAULT}"
                if os.path.isdir(LOG_DIR_NGENCERF)
                else f"{os.path.expanduser('~')}{DS}{LOG_DIR_DEFAULT}"
            )
            try:
                os.makedirs(baseDir, exist_ok=True)

                childDir = getpass.getuser() or create_timestamp(True)
                logFileDir = f"{baseDir}{DS}{childDir}"
                os.makedirs(logFileDir, exist_ok=True)

                logFilePath = (
                    f"{logFileDir}{DS}{MODULE_NAME}_{create_timestamp()}.{LOG_FILE_EXT}"
                )
            except Exception as e:
                print(f"Module {MODULE_NAME} {e}", flush=True)
                logFilePath = ""
    
    # Ensure log file can be opened and set module env var
    try:
        if (logFilePath):
            mode = "a" if appendEntries else "w"
            with open(logFilePath, mode):
                pass
            if not moduleLogFileExists:
                os.environ[EV_MODULE_LOGFILEPATH] = logFilePath
                print(f"Module {MODULE_NAME} Log File: {logFilePath}", flush=True)
        else:
            raise IOError
    except Exception:
        print(f"Module {MODULE_NAME} Unable to open log file: {logFilePath}", flush=True)
        print(f"Module {MODULE_NAME} Log entries will be writen to stdout", flush=True)

    return logFilePath, appendEntries
