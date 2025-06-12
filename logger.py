import logging
from threading import Lock
import inspect


class Logger_Singleton:
    _instance = None
    _lock = Lock()
    _filename = None

    def __new__(cls, file_name = None):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Logger_Singleton, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, file_name= None):
        if getattr(self, '_initialized', False):
            return
        self._setup_logger(file_name, "info")
        if(file_name != None and self._filename == None): 
          self._filename = file_name
        else:
          raise("Either logging was not initialized ")
        self._initialized = True

    def updatelog_level(self, level):
        if(level.lower() == 'info'):
            self.logger.setLevel(logging.INFO)
        elif(level.lower() == 'debug'):
            self.logger.setLevel(logging.DEBUG)
        elif(level.lower() == 'error'): 
            self.logger.setLevel(logging.ERROR)
        else:
            self.logger.warning(f"Unknown log level: {level} - defaulting to DEBUG") 

    def _setup_logger(self, file_name, level='info'):
        self.logger = logging.getLogger("SingletonLogger")
        self.updatelog_level(level)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(file_name)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log(self, level, method_name , message):
        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'error':
            self.logger.error(message)
        else:
            self.logger.warning(f"Unknown log level: {level} - {message}")

    def info(self, message):
        try:
            method_name = inspect.stack()[1].function
        except exception as e:
            method_name = "<unknown>"   
        log_message = f"{method_name}-{message}"

        self.logger.setLevel(logging.INFO)
        self.logger.info(log_message)
            
    def debug(self, message):
        try:
            method_name = inspect.stack()[1].function
        except exception as e:
            method_name = "<unknown>"       
        log_message = f"{method_name}-{message}"

        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(log_message)
            
    def warning(self, message):
        try:
            method_name = inspect.stack()[1].function
        except exception as e:
            method_name = "<unknown>"       
        log_message = f"{method_name}-{message}"

        self.logger.setLevel(logging.WARNING)
        self.logger.warning(log_message)

    def error(self,  message):
        self.logger.setLevel(logging.WARNING)
        try:
            method_name = inspect.stack()[1].function
        except exception as e:
            method_name = "<unknown>"       
        log_message = f"{message}"

        self.logger.setLevel(logging.ERROR)
        self.logger.error(log_message)
