import numpy as np
"""
Equations of State found here: http://resource.npl.co.uk/acoustics/techguides/soundseawater/underlying-phys.html
"""

def depth_to_pressure_Leroy(z, lat):
    """
    Algorithm to convert depth to pressure taken from Leroy
        Inputs are meters and degrees
        Calculates pressure in MPa (relative to atmospheric pressure)
        Algorithm Converts and returns in bars
    """
    g_lat = 9.7803*(1+5.3e-3 * np.sin(lat)**2)
    h_z45 = 1.00818e-2 * z - 2.456e-8 * z ** 2 - 1.25e-13 * z ** 3 + 2.8e-19 * z ** 4
    k_zlat = (g_lat - 2e-5 * z)/ (9.80612 - 2e-5 * z)

    h_zlat = h_z45 * k_zlat
    h_0 = 1.0e-2 * z / (z+100) + 6.2e-6*z

    pressure = h_zlat - h_0

    #Convert to MPa (not relative to atm) and then bars
    pressure_conversion = (pressure) * 10

    return pressure_conversion

def depth_to_pressure(z, lat):
    """
    Algorithm to convert depth to pressure taken from Thalia and Frederik
        Inputs are meters and degrees, returns bars
    """
    c1 = (5.92 + 5.25*np.sin(np.abs(lat)*np.pi/180)**2)*1e-3
    c2 = 2.21*1e-6
    pressure = ((1 - c1) - np.sqrt((1 - c1) ** 2 - 4 * c2 * z)) / (2 * c2)

    return pressure / 10


def DelGrosso_SV(S, T, P):
    """
    Algorithm to calculate sound velocity given Salinity, Temperature, and Pressure
        Input are in terms of Celsius, Practical Salinity Units, and kg/cm^2
        Returns sound speed in m/s
    """
    #Convert pressure from bars to kg/cm^2
    P = P * 1.019716213

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

def UNESCO_SV(S, T, P):
    """
    Algorithm to calculate sound velocity given Salinity, Temperature, and Pressure using UNESCO Equations
    Inputs are in terms of Celsius, Practical Salinity Units, and bars
    Returns sound speed in m/s
    """
    # Define Constants
    C = {
        '00': 1402.388, '01': 5.03830, '02': -5.81090 * 10 ** -2, '03': 3.3432 * 10 ** -4,
        '04': -1.47797 * 10 ** -6, '05': 3.1419 * 10 ** -9,
        '10': 0.153563, '11': 6.8999 * 10 ** -4, '12': -8.1829 * 10 ** -6,
        '13': 1.3632 * 10 ** -7, '14': -6.1260 * 10 ** -10,
        '20': 3.1260 * 10 ** -5, '21': -1.7111 * 10 ** -6, '22': 2.5986 * 10 ** -8,
        '23': -2.5353 * 10 ** -10, '24': 1.0415 * 10 ** -12, '30': -9.7729 * 10 ** -9, '31': 3.8513 * 10 ** -10,
        '32': -2.3654 * 10 ** -12
    }

    A = {
        '00': 1.389, '01': -1.262 * 10 ** -2, '02': 7.166 * 10 ** -5, '03': 2.008 * 10 ** -6,
        '04': -3.21 * 10 ** -8, '10': 9.4742 * 10 ** -5, '11': -1.2583 * 10 ** -5, '12': -6.4928 * 10 ** -8,
        '13': 1.0515 * 10 ** -8, '14': -2.0142 * 10 ** -10, '20': -3.9064 * 10 ** -7, '21': 9.1061 * 10 ** -9,
        '22': -1.6009 * 10 ** -10, '23': 7.994 * 10 ** -12, '30': 1.100 * 10 ** -10, '31': 6.651 * 10 ** -12,
        '32': -3.391 * 10 ** -13
    }

    B = {
        '00': -1.922 * 10 ** -2, '01': -4.42 * 10 ** -5, '10': 7.3637 * 10 ** -5, '11': 1.7950 * 10 ** -7,
    }

    D = {
        '00': 1.727 * 10 ** -3, '10': -7.9836 * 10 ** -6
    }

    # Calculate Terms
    Cw_TP = (C['00'] + C['01']*T + C['02']*T**2 + C['03']*T**3 + C['04']*T**4 + C['05']*T**5
          + (C['10'] + C['11']*T + C['12']*T**2 + C['13']*T**3 + C['14']*T**4)*P
          + (C['20'] + C['21']*T + C['22']*T**2 + C['23']*T**3 + C['24']*T**4)*P**2
          + (C['30'] + C['31']*T + C['32']*T**2)*P**3)

    A_TP = (A['00'] + A['01']*T + A['02']*T**2 + A['03']*T**3 + A['04']*T**4
          + (A['10'] + A['11']*T + A['12']*T**2 + A['13']*T**3 + A['14']*T**4)*P
          + (A['20'] + A['21']*T + A['22']*T**2 + A['23']*T**3)*P**2
          + (A['30'] + A['31']*T + A['32']*T**2)*P**3)

    B_TP = B['00'] + B['01']*T + (B['10'] + B['11']*T)*P

    D_TP = D['00'] + D['10']*P

    # Compute Sound Velocity
    c = Cw_TP + A_TP*S + B_TP*S**1.5 + D_TP*S**2
    return c

def NPL_ESV(S, T, Z, lat):
    """
    Algorithm to calculate sound velocity given Salinity, Temperature, and Depth given NPL Equations
        Inputs appear to be in terms of Celsius, Practical Salinity Units, meters, and degrees
        returns sound speed in m/s
    """
    c = (1402.5 + 5*T - 5.44e-2*T**2 + 2.1e-4*T** 3
         + 1.33*S -1.23e-2*S*T + 8.7e-5*S*T**2
         + 1.56e-2*Z + 2.55e-7*Z**2 - 7.3e-12*Z**3
         + 1.2e-6*Z*(lat - 45) - 9.5e-13*T*Z**3
         + 3e-7*T**2*Z + 1.43e-5*S*Z)
    return c

def Mackenzie_ESV(S, T, Z):
    """
    Algorithm to calculate sound velocity given Salinity, Temperature, and Depth given Mackenzie Equations
        Inputs are in terms of Celsius, Practical Salinity Units, and meters
        Returns sound speed in m/s
    """
    c = (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3 + 1.340*(S - 35) + 1.630e-2*Z
    + 1.675e-7*Z**2 - 1.025e-2*T*(S - 35) - 7.139e-13*T*Z**3)
    return c

def Coppens_ESV(S, T, Z):
    """
    Algorithm to calculate sound velocity given Salinity, Temperature, and Depth given Coppens Equations
        Inputs are in terms of Celsius, Practical Salinity Units, and meters
        Returns sound speed in m/s
    """
    T = T/10
    Z = Z/1000
    c_0 = 1449.05 + 45.7*T - 5.21*T**2 + 0.23*T**3 + (1.333 - 0.126*T + 0.009*T**2)*(S-35)
    c = c_0 + (16.23 + 0.253*T)*Z + (0.213-0.1*T)*Z**2 + (0.016 + 0.0002*(S-35))*(S-35)*T*Z
    return c

if __name__ == '__main__':
    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt

    CTD = sio.loadmat('GPSData/CTD_Data/AE2008_Cast2.mat')['AE2008_Cast2']

    depth_t = np.ascontiguousarray(np.genfromtxt('GPSData/depth_cast2_smoothed.txt'))[::100]
    cz_t = np.ascontiguousarray(np.genfromtxt('GPSData/cz_cast2_smoothed.txt'))[::100]

    depth = CTD[:, 0][::100]
    temperature = CTD[:, 1][::100]
    salinity = CTD[:, 4][::100]

    # Find index of max depth
    max_depth = np.max(depth)
    max_depth_index = np.where(depth == max_depth)[0][0]

    # Restrict data to max depth
    depth = depth[:max_depth_index]
    temperature = temperature[:max_depth_index]
    salinity = salinity[:max_depth_index]

    pressure = depth_to_pressure(depth, 31.5)

    print(np.max(pressure))

    sound_speed1 = DelGrosso_SV(salinity, temperature, pressure)
    sound_speed2 = UNESCO_SV(salinity, temperature, pressure)
    sound_speed3 = NPL_ESV(salinity, temperature, depth, 31.5)
    sound_speed4 = Mackenzie_ESV(salinity, temperature, depth)
    sound_speed5 = Coppens_ESV(salinity, temperature, depth)


    print(sound_speed1)
    print(np.max(sound_speed1))
    print(np.min(sound_speed1))

    plt.figure(figsize=(6, 8))
    plt.plot(sound_speed1, depth, label='DelGrosso')
    plt.plot(sound_speed2, depth, label='UNESCO')
    plt.plot(sound_speed3, depth, label='NPL')
    plt.plot(sound_speed4, depth, label='Mackenzie')
    plt.plot(sound_speed5, depth, label='Coppens')
    plt.plot(cz_t, depth_t, label='DelGrosso Thalia')

    plt.gca().invert_yaxis()
    plt.title('Sound Speed Profile')
    plt.xlabel('Sound Speed (m/s)')
    plt.ylabel('Depth (m)')
    plt.legend()
    plt.show()


    # Calculate the difference between each value in the sound speed profile and the closest cz_t
    sound_speed_diff1 = np.zeros_like(sound_speed1)
    sound_speed_diff2 = np.zeros_like(sound_speed2)
    sound_speed_diff3 = np.zeros_like(sound_speed3)
    sound_speed_diff4 = np.zeros_like(sound_speed4)
    sound_speed_diff5 = np.zeros_like(sound_speed5)


    # Create depth difference matrix and find closest indices all at once
    depth_diff = np.abs(depth_t[:, np.newaxis] - depth)
    closest_indices = np.argmin(depth_diff, axis=0)

    # Use the indices to compute all differences at once
    sound_speed_diff1 = sound_speed1 - cz_t[closest_indices]
    sound_speed_diff2 = sound_speed2 - cz_t[closest_indices]
    sound_speed_diff3 = sound_speed3 - cz_t[closest_indices]
    sound_speed_diff4 = sound_speed4 - cz_t[closest_indices]
    sound_speed_diff5 = sound_speed5 - cz_t[closest_indices]

    # Plot the difference
    plt.figure(figsize=(6, 8))
    plt.plot(sound_speed_diff1, depth, label='Difference (DelGrosso - cz_t)')
    plt.plot(sound_speed_diff2, depth, label='Difference (UNESCO - cz_t)')
    plt.plot(sound_speed_diff3, depth, label='Difference (NPL - cz_t)')
    plt.plot(sound_speed_diff4, depth, label='Difference (Mackenzie - cz_t)')
    plt.plot(sound_speed_diff5, depth, label='Difference (Coppens - cz_t)')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlim(-2, 2)
    plt.gca().invert_yaxis()
    plt.title('Difference in Sound Speed Profile')
    plt.xlabel('Difference in Sound Speed (m/s)')
    plt.ylabel('Depth (m)')
    plt.legend()
    plt.show()
