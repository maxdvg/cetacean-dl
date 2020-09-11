# Max Van Gelder 6/19/2020
# sys.argv[1] is a directory containing any number of .wav files for which
# spectrograms are to be generated

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import copy
import numpy as np
import sqlite3
import sys
import math
import time
from os import listdir




