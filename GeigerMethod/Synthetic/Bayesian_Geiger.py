import numpy as np
import matplotlib.pyplot as plt
from advancedGeigerMethod import geigersMethod, calculateTimesRayTracing, generateRealistic, findTransponder

def  Bayesian_Geiger(iterations, n, time_noise, position_noise):
    CDog, GPS_Coordinates, transponder_coordinates_Actual, gps1_to_others, gps1_to_transponder = generateRealistic(n)