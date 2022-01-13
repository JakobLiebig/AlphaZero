import logger

class Logger(logger.Base):
    def log(self, type, message):
        
        print(type, message)