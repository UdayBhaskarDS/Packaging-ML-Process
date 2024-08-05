import os 
from predictive_modelling.config import config
with open(os.path.join(config.PACKAGE_ROOT,'VERSION')) as f :
    __version__ = f.read().strip()
