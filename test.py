from Chokkhu.logger import logger
from Chokkhu.custom_exception import InvalidURLException

logger.info("this is testing log")

try:
    raise InvalidURLException()
except Exception as e:
    logger.error(f"Cought an exception: {str(e)}")
    