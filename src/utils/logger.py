import logging
import sys
from pathlib import Path

_logger = None
_initialized = False

def get_logger(name: str = "wm_model", log_file: Path = None) -> logging.Logger:
    global _logger, _initialized
    
    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
    
    # If log_file provided and not yet initialized with file handler, set it up
    if log_file is not None and not _initialized:
        log_path = Path(log_file)
        
        # Remove old log file
        if log_path.exists():
            log_path.unlink()
        
        _logger.handlers.clear()
        
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        _logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        _logger.addHandler(ch)
        
        _initialized = True
    
    # If no handlers yet (called before log_file set), add console only
    if not _logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        _logger.addHandler(ch)
    
    return _logger
