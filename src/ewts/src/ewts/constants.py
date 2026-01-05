# Error Warning and Trapping System
# ewts/constants.py

# Values unique to each module
MODULE_NAME           = "T-Route"
EV_MODULE_LOGLEVEL    = "TROUTE_LOGLEVEL"      # This modules log level
EV_MODULE_LOGFILEPATH = "TROUTE_LOGFILEPATH"   # This modules log full log filename

# Values common to all ngen modules
EV_NGEN_LOGFILEPATH   = "NGEN_LOG_FILE_PATH"   # Environment variable log file location typically set by ngen
EV_EWTS_LOGGING       = "NGEN_EWTS_LOGGING"    # Environment variable flat to enable/disable the Error Warning and Trapping System  

DS                    = "/"                    # Directory separator
LOG_DIR_DEFAULT       = "run-logs"             # Default parent log directory string if env var empty  & ngencerf dosn't exist
LOG_DIR_NGENCERF      = "/ngencerf/data"       # ngenCERF log directory string if environement var empty.
LOG_FILE_EXT          = "log"                  # Log file name extension
LOG_MODULE_NAME_LEN   = 8                      # Width of module name for log entries


