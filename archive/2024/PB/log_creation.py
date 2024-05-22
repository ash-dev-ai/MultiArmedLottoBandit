import logging
from logging.handlers import RotatingFileHandler
import os

# Define a custom logging level: AUDIT
AUDIT_LEVEL_NUM = 35  # Numerical value between WARNING (30) and ERROR (40)
logging.addLevelName(AUDIT_LEVEL_NUM, "AUDIT")

def audit(self, message, *args, **kwargs):
    if self.isEnabledFor(AUDIT_LEVEL_NUM):
        self._log(AUDIT_LEVEL_NUM, message, args, **kwargs)

# Attach the custom level to the logging.Logger class
logging.Logger.audit = audit

# Create logs folder if it doesn't exist
log_folder_path = './logs'
if not os.path.exists(log_folder_path):
    os.makedirs(log_folder_path)

# Initialize the application logger
app_logger = logging.getLogger('ApplicationLogger')
app_logger.setLevel(logging.INFO)

# Initialize the error logger
error_logger = logging.getLogger('ErrorLogger')
error_logger.setLevel(logging.ERROR)

# Add rotating handlers to both loggers
app_rotating_handler = RotatingFileHandler(os.path.join(log_folder_path, 'app_rotating_log.log'), maxBytes=1e6, backupCount=5)
error_rotating_handler = RotatingFileHandler(os.path.join(log_folder_path, 'error_rotating_log.log'), maxBytes=1e6, backupCount=5)

# Formatter
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s')

# Set formatter for handlers
app_rotating_handler.setFormatter(formatter)
error_rotating_handler.setFormatter(formatter)

# Add handlers to loggers
app_logger.addHandler(app_rotating_handler)
error_logger.addHandler(error_rotating_handler)

# Test the loggers and custom level
app_logger.info('This is an application-level log.')
app_logger.audit('This is an audit-level log.')
error_logger.error('This is an error-level log.')
