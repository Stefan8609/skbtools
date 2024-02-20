import numpy as np
from Unwrap_Fix import RMSE_offset, find_offset

"""
Goal of this file is to iterate through finding the CDOG with the following steps:
    1) Align the travel time series for a CDOG guess as close as possible with the unwrapped CDOG data
    2) Apply Geiger's method to the aligned data to find an improved guess of CDOG position
    3) Use Synthetic Annealing to find the best guess for the transducer location (with a constricted range of possibilities)
    4) Iterate back to the alignment phase with the new CDOG guess and travel time series
    5) After a certain number of iterations take the given CDOG location (determine validity by looking at residual distribution)
"""

