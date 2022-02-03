#!/usr/bin/env python3.8

from configparser import ConfigParser, ExtendedInterpolation
import glob
import time

from autoencoders.plotAE import visual_train_history
###############################################################################
time_start = time.time()

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("visualHistory.ini")
###############################################################################

###############################################################################
time_finish = time.time()
print(f" Run time: {time_finish - time_start: 1.0f}[s]")
