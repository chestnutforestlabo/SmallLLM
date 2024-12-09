import logging
import sys

class Logger:
    def __init__(self, log_file, name):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.enable_exception_handler()
    
    def info(self, message):
        self.logger.info(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def exception(self, exc_type, exc_value, exc_traceback):
        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    def enable_exception_handler(self):
        sys.excepthook = self.exception