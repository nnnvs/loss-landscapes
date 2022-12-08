import logging
logging.basicConfig(filename= "std.log", filemode= 'w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info("Logging 2")
logger.info ("logging 2")

