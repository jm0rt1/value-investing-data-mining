
import logging
import logging.handlers
from src.value_investing_strategy.shared.settings import GlobalSettings


def initialize_logging():

    file_handler = logging.handlers.RotatingFileHandler(
        GlobalSettings.GLOBAL_LOGS_DIR/GlobalSettings.LoggingParams.GLOBAL_FILE_NAME,
        backupCount=GlobalSettings.LoggingParams.BACKUP_COUNT)

    logging.getLogger().addHandler(file_handler)
    file_handler.doRollover()
    logging.info("Global Logging Started")


def main():
    """run a console menu that has two options"""
    initialize_logging()
    logging.info("Program Started")
    print("Welcome to the Value Investing Strategy Program")
    print("Please select an option:")
    print("1. Run the Value Investing Strategy")
    print("2. Compare the Value Investing Strategy to the S&P 500")
    print("3. Exit the Program")
    choice = input("Enter your choice: ")
    if choice == "1":
        print("Running the Value Investing Strategy")
        from src.model_comparison.data_collection import main
        main()
    elif choice == "2":
        print("Comparing the Value Investing Strategy to the S&P 500")
        from src.model_comparison.model_comparison import main
        main()
    elif choice == "3":
        print("Exiting the Program")
        logging.info("Program Ended")
        exit()
    else:
        print("Invalid Choice")