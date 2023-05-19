import logging

LOGGING_LEVEL = logging.INFO


class Logger:
    __logger = None

    @staticmethod
    def __get_logger():
        if Logger.__logger is None:
            Logger.__logger = logging.getLogger('global')
            Logger.__logger.setLevel(LOGGING_LEVEL)

        return Logger.__logger

    @staticmethod
    def info(text: str, **kwargs):
        logger = Logger.__get_logger()
        logger.info(text, **kwargs)

    @staticmethod
    def warning(text: str, **kwargs):
        logger = Logger.__get_logger()
        logger.warning(text, **kwargs)

    @staticmethod
    def error(text: str, **kwargs):
        logger = Logger.__get_logger()
        logger.error(text, **kwargs)

    @staticmethod
    def debug(text: str, **kwargs):
        logger = Logger.__get_logger()
        logger.debug(text, **kwargs)

    @staticmethod
    def exception(text: str, **kwargs):
        logger = Logger.__get_logger()
        logger.exception(text, **kwargs)

    @staticmethod
    def critical(text: str, **kwargs):
        logger = Logger.__get_logger()
        logger.critical(text, **kwargs)
