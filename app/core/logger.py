import logging
import sys
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """ฟอร์แมตเตอร์สำหรับเปลี่ยน Log เป็น JSON เพื่อให้ Azure Monitor อ่านง่าย"""
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "service": "passion-tree-ai"
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logger(is_dev: bool):
    logger = logging.getLogger("passion-tree-ai")
    handler = logging.StreamHandler(sys.stdout)
    
    if is_dev:
        # ช่วง Dev: อ่านง่ายๆ ใน Terminal
        formatter = logging.Formatter('%(levelname)s: %(message)s')
    else:
        # ช่วง Prod (Azure): พ่นเป็น JSON
        formatter = JSONFormatter()
        
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger