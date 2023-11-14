import numpy as np
import matplotlib.pyplot as plt

def experimentPathPlot(transponder_coordinates, CDog):
    # Plot path of experiment
    plt.scatter(transponder_coordinates[:, 0], transponder_coordinates[:, 1], s=10, marker="o",
                label="Transponder")
    plt.scatter(CDog[0], CDog[1], s=50, marker="x", color="k", label="CDog")
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title(f'GPS and transducer coordinates over course of {len(transponder_coordinates)} points')
    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.show()
    return