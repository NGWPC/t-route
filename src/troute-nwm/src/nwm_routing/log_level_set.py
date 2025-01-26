import logging
import sys   
from datetime import datetime, timezone
import os
import time

def create_timestamp() -> str: 
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    

def log_level_set(input_parameters):
    '''
    Set logging level and specify logger configuration.
    
    Arguments
    ---------
    input_parameters (dict): User input logging parameters
    
    Returns
    -------
    None
    
    Notes
    -----
    In the absense of user-specified logging level, level defaults to DEGUG
    See also https://docs.python.org/3/library/logging.html
    
    '''
    #log_level = input_parameters.get("log_level", 'INFO')
    log_level = 'INFO'
    if True:
        logFilePath = os.getenv('NGEN_LOG_FILE_PATH', "")
        try:
            logFile = open(logFilePath, "a")
        except IOError:
            print("Warning: Can't Open shared Log File referenced from NGEN_LOG_FILE_PATH env. variable for TROUTE module", file=sys.stderr)
            log_file_dir = f"./run-logs/ngen_{create_timestamp()}/"
            log_file_name = "troute_log.txt"
            os.makedirs(log_file_dir, exist_ok=True)
            logFilePath = os.path.join(log_file_dir, log_file_name)
            try:
                logFile = open(logFilePath, "a")
                print(f"TROUTE is logging instead into: {logFilePath}")
            except IOError:
                print(f"Can't Open local directory Log File for TROUTE module: {logFilePath}", file=sys.stderr)
        else:
            print(f"Log File Path for TROUTE module: {logFilePath}")
        
        logging.Formatter.converter = time.gmtime
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s.%(msecs)03d TROUTE   %(levelname)s   %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S',
            handlers=[
            logging.FileHandler(logFilePath, mode='a'),  # Log to a file
            #logging.StreamHandler(sys.stdout)  
        ])
    else:       
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s]: %(message)s',
            stream=sys.stderr,
        )   
    