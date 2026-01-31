import logging
import sys
import json
import os
from datetime import datetime, timezone

class SlogStyleFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": datetime.now(timezone.utc).isoformat(), 
            "level": record.levelname,                      
            "msg": record.getMessage(),                     
            "source": {                                    
                "function": record.funcName,
                "file": record.pathname,
                "line": record.lineno
            },
            "service": "passion-tree-ai",                  
            "env": "production" if os.getenv("APP_ENV") == "production" else "development"
        }

        standard_attrs = ("name", "msg", "args", "levelname", "levelno", "pathname", "filename", 
                          "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", 
                          "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process")
        
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_record[key] = value

        if record.exc_info:
            log_record["err"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def setup_logger(is_dev: bool):
    logger = logging.getLogger("passion-tree-ai")
    
    # Prevent adding duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        if is_dev:
            # TextHandler for Development
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            logger.setLevel(logging.DEBUG)
        else:
            # JSONHandler for Production
            formatter = SlogStyleFormatter()
            logger.setLevel(logging.INFO)
            
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.propagate = False
    return logger