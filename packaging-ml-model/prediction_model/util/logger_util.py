import logging
from pathlib import Path
import os
import sys
from datetime import datetime
from prediction_model.config import config

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

logfilename= os.path.join(config.LOGPATH, config.LOGFILE)
logging.basicConfig(level=logging.INFO,
                    filename=logfilename,
                    encoding='utf-8',
                    format="%(levelname)s:%(asctime)s:%(message)s")
logging.warning("Save me")