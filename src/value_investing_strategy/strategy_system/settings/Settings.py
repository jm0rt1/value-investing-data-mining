

from pathlib import Path
from src.value_investing_strategy.shared.settings import GlobalSettings


class Settings:

    class LoggingParams():
        LOG_FILE_NAME = "strategy.log"
        LOG_FILE_PATH = GlobalSettings.STRATEGY_SYSTEM_LOGS_DIR/LOG_FILE_NAME
        BACKUP_COUNT = 20

    def __init__(self, ):
        pass

    def load(self, path: Path):
        pass
