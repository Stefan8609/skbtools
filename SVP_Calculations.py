import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

CTD = sio.loadmat('GPSData/CTD_Data/AE2008_Cast2.mat')['AE2008_Cast2']

depth_t = np.ascontiguousarray(np.genfromtxt('GPSData/depth_cast2_smoothed.txt'))
cz_t = np.ascontiguousarray(np.genfromtxt('GPSData/cz_cast2_smoothed.txt'))

depth = CTD[:,0]
temperature = CTD[:,1]
salinity = CTD[:,4]

"""This probably isn't working great


from 19. Chen, C.T.; Millero, F.J. Speed of sound in seawater at high pressures. J. Acoust. Soc. Am. 1977, 62, 1129–1135. [CrossRef]

Use this http://resource.npl.co.uk/acoustics/techguides/soundseawater/underlying-phys.html

Frederik's program gives back Decibars

Del Grosso uses kg/(cm^2)
"""

def depth_to_pressure(z, lat):
    c1 = (5.92 + 5.25*np.sin(np.abs(lat)*np.pi/180)**2)*1e-3
    c2 = 2.21*1e-6
    pressure = ((1 - c1) - np.sqrt((1 - c1) ** 2 - 4 * c2 * z)) / (2 * c2)

    #convert to kg/cm^2
    return (pressure / 10) * 1.019716213


def DelGrosso_SV(S, T, P):
    """Algorithm to calculate sound velocity given Salinity, Temperature, and Pressure
    Found in Makar 2022
    """
    # Define Constants
    const = {
        'C_T1': 5.012285, 'C_T2': -0.551184 * 10 ** -1, 'C_T3': 0.221649 * 10 ** -3,
        'C_S1': 1.329530, 'C_S2': 0.1288598 * 10 ** -3,
        'C_P1': 0.1560592, 'C_P2': 0.2449993 * 10 ** -4, 'C_P3': -0.8833959 * 10 ** -8,
        'C_ST': -0.1275936 * 10 ** -1, 'C_TP': 0.6353509 * 10 ** -2,
        'C_T2P2': 0.2656174 * 10 ** -7, 'C_TP2': -0.1593895 * 10 ** -5, 'C_TP3': 0.5222483 * 10 ** -9,
        'C_T3P': -0.4383615 * 10 ** -6,
        'C_S2P2': -0.1616745 * 10 ** -8, 'C_ST2': 0.9688441 * 10 ** -4,
        'C_S2TP': 0.4857614 * 10 ** -5, 'C_STP': -0.3406824 * 10 ** -3
    }

    # Calculate Terms
    Delta_CT = const['C_T1']*T + const['C_T2']*T**2 + const['C_T3']*T**3
    Delta_CS = const['C_S1']*S + const['C_S2']*S**2
    Delta_CP = const['C_P1']*P + const['C_P2']*P**2 + const['C_P3']*P**3
    Delta_CSTP = (const['C_TP']*T*P + const['C_T3P']*P*T**3 + const['C_TP2']*T*P**2
                  + const['C_T2P2']*T**2*P**2 + const['C_TP3']*T*P**3
                  + const['C_ST']*S*T + const['C_ST2']*S*T**2 + const['C_STP']*S*T*P
                  + const['C_S2TP']*S**2*T*P + const['C_S2P2']*S**2*P**2)

    # Compute Sound Velocity
    c = 1402.392 + Delta_CT + Delta_CS + Delta_CP + Delta_CSTP
    return c

pressure = depth_to_pressure(depth, 31.5)

print(np.max(pressure))

sound_speed = DelGrosso_SV(salinity, temperature, pressure)
print(sound_speed)
print(np.max(sound_speed))
print(np.min(sound_speed))

plt.plot(sound_speed, depth, label='DelGrosso')
plt.plot(cz_t, depth_t, label='DelGrosso Thalia')
plt.gca().invert_yaxis()
plt.title('Sound Speed Profile')
plt.xlabel('Sound Speed (m/s)')
plt.ylabel('Depth (m)')
plt.legend()
plt.show()

