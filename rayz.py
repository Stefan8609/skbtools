import numpy as np
import math
import matplotlib.pyplot as plt

# Replace the following functions with actual implementations if they are not available.

cz = np.loadtxt('Thalia_Stuff/data/cz_cast2_big.txt')
depth = np.loadtxt('Thalia_Stuff/data/depth_cast2_big.txt')
gradc = np.loadtxt('Thalia_Stuff/data/gradc_cast2_big.txt')



#Find nearest algorithm to find the starting and ending indices according to incident depth
#   and the beacon depth... Then can cut all parts of c(z) not within range of depths.
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def ray_tracing(c, z, zA, zB, iga):
    # Effective range for sound velocity range
    theA = 90 - iga  # Degree between the ray and the vertical axis at z = zA
    idx = find_nearest(z, zA)
    a = np.sin(theA * math.pi / 180) / c[idx]  # Snell's constant (Clay and Medwin's)

    # Ordinary refraction analysis
    if a > 0:
        dz = np.diff(z)
        b = np.diff(c) / dz  # Gradients
        theta = np.arcsin(a * c)
        zgi = np.where(np.abs(b) < 1e-10)[0]  # Zero gradient indices

        nzgi = np.where(b != 0)[0]  # Non-zero gradient indices

        # Non-zero gradient elements
        R = 1 / (a * b[nzgi])
        ct1 = np.cos(theta[nzgi])
        ct2 = np.cos(theta[nzgi + 1])
        dz[nzgi] = (ct1 - ct2) * R
        vl = c[nzgi] / b[nzgi]
        u2 = dz[nzgi] + vl
        dt = 1 / b[nzgi] * np.log((u2 * (1 + ct1)) / (vl * (1 + ct2)))

        # Zero gradient elements
        dx = dz[zgi] * np.tan(theta[zgi])
        dt[zgi] = np.sqrt(dx**2 + (dz[zgi]**2)) / c[zgi]

        tt = np.sum(dt)
        hd = np.sum(dx)

        # If ray plot is desired
        x0 = 0
        x = np.cumsum([x0] + dz.tolist())
        plt.plot(x, z)
        plt.show()
        # Plot x, -zs using a plotting library or print the coordinates

        ve = math.sqrt(hd**2 + (zA - zB)**2) / tt

        return hd, tt, ve

print(ray_tracing(cz, depth, 10, 5000, 50))