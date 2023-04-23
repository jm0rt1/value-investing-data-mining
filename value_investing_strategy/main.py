
import logging
import logging.handlers
from value_investing_strategy.shared.settings import GlobalSettings


def initialize_logging():

    file_handler = logging.handlers.RotatingFileHandler(
        GlobalSettings.GLOBAL_LOGS_DIR/GlobalSettings.LoggingParams.GLOBAL_FILE_NAME,
        backupCount=GlobalSettings.LoggingParams.BACKUP_COUNT)

    logging.getLogger().addHandler(file_handler)
    file_handler.doRollover()
    logging.info("Global Logging Started")


def main():
    initialize_logging()
    try:
        # Start System
        pass

    except Exception as e:
        logging.exception(e)
        raise e
