import logger

class Logger(logger.Base):
    def log(self, type, message):
        if type == "loss":
            print(type, message)