import os
from pathlib import Path
import sys


PROJECT_PATH = Path(os.path.abspath(__file__)).parent
sys.path.append(str(Path(PROJECT_PATH, 'src')))
DATA_PATH = Path(PROJECT_PATH , 'data')
if not DATA_PATH.exists():
    DATA_PATH.mkdir(parents=True)
VERBOSE = False
RESULTS_PATH = Path(PROJECT_PATH, 'results')
if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir(parents=True)

