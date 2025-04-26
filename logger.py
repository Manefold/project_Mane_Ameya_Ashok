# logger.py
import logging
from datetime import datetime

def setup_logger(log_file: str = None) -> logging.Logger:
    """
    Sets up a logger with console and optional file output.
    
    Args:
        log_file: Optional path to a log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('anomaly_detector')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"{log_file}_{timestamp}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Create default logger
logger = setup_logger()
