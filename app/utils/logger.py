import sys
import logging
from loguru import logger as base_logger

base_logger.remove()

class RAGLogger:
    @staticmethod
    def setup_logging(log_level: str = "INFO"):
        base_logger.remove()

        def formatter(record):
            if "stage" in record["extra"]:
                return (
                    "<green>{time:HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{extra[stage]: <28}</cyan> | "
                    "<level>{message}</level>\n"
                )
            else:
                return (
                    "<green>{time:HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<level>{message}</level>\n"
                )

        base_logger.add(
            sys.stderr,
            level=log_level,
            format=formatter,
            colorize=True,
            enqueue=True
        )

        
        # Add File Handler (logs/rag_service.log)
        import os
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        base_logger.add(
            os.path.join(log_dir, "rag_service.log"), 
            rotation="10 MB", 
            retention="10 days", 
            level="INFO",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}"
        )

        class InterceptHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = base_logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1
                base_logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )

        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        
        for lib in ["uvicorn.access", "httpx", "httpcore", "neo4j", "multipart"]:
            logging.getLogger(lib).handlers = [InterceptHandler()]
            logging.getLogger(lib).setLevel(logging.WARNING)
            
        return base_logger

logger = base_logger

def get_logger(stage_name: str):
    if not stage_name.startswith("["):
        stage_name = f"[{stage_name}]"
    return base_logger.bind(stage=stage_name)

__all__ = ["logger", "RAGLogger", "get_logger"]