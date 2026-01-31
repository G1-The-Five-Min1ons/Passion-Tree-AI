import logging
import sys
import os
from datetime import datetime, timezone

class DevStructuredFormatter(logging.Formatter):
    """จัดรูปแบบ Log ให้เป็น key=value เหมือน slog ของ Go"""
    def format(self, record):
        # เตรียมข้อมูลพื้นฐาน
        time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"
        level = record.levelname
        # ทำ path ให้สั้นลงเหมือนใน Go
        rel_path = os.path.relpath(record.pathname, os.getcwd()) if os.path.isabs(record.pathname) else record.pathname
        source = f"{rel_path}:{record.lineno}"
        msg = record.getMessage()
        
        # จัดรูปแบบ key=value
        log_line = f"time={time} level={level} source={source} msg=\"{msg}\" service=passion-tree-ai env=development"
        
        # ถ้ามี extra attributes อื่นๆ (เช่น latency_ms, method) ให้พ่นต่อท้ายไปด้วย
        standard_attrs = ("name", "msg", "args", "levelname", "levelno", "pathname", "filename", 
                          "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", 
                          "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process")
        
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_line += f" {key}={value}"
        
        return log_line

def setup_logger(is_dev: bool):
    logger = logging.getLogger() # Root Logger
    
    
    # ปิด Log ของ Library อื่นๆ ที่ไม่จำเป็น (เหมือนที่คุยกันก่อนหน้า)
    for ext in ["httpx", "httpcore", "groq"]:
        logging.getLogger(ext).setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        if is_dev:
            # ใช้ Formatter ตัวใหม่ที่เราสร้างขึ้น
            formatter = DevStructuredFormatter()
            logger.setLevel(logging.DEBUG)
        else:
            # JSON สำหรับ Production (SlogStyleFormatter ที่คุณเขียนไว้)
            formatter = SlogStyleFormatter()
            logger.setLevel(logging.INFO)
            
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger