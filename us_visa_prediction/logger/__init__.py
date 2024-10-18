import logging
from pathlib import Path
from from_root import from_root
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# Directory for logs
log_dir = Path(from_root()) / 'logs'

# Ensure the log directory exists
try:
    log_dir.mkdir(parents=True, exist_ok=True)
except OSError as e:
    raise Exception(f"Failed to create log directory: {e}")

# Log file name with timestamp
log_file = log_dir / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create a timed rotating file handler to rotate logs daily and keep 7 backup files
log_handler = TimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=7
)
log_handler.suffix = "%Y%m%d"  # File rotation will create logs with this suffix

# Logging configuration
logging.basicConfig(
    handlers=[log_handler],
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

# Add a console logger to see logs in the terminal as well
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

