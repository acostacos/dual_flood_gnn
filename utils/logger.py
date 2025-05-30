import os
import logging

class Logger(object):
    def __init__(self, log_path=None):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())

        if log_path is not None:
            # Create log directory and file if it doesn't exist
            directory = os.path.dirname(log_path)
            if directory != '' and not os.path.exists(directory):
                os.makedirs(directory)

            file_handler = logging.FileHandler(log_path, mode='a')
            self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)

        if len(self.logger.handlers) > 1:
            self.logger.handlers[1].flush() # For instant update of file handler
