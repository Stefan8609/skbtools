import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math

def load_and_process_data(path):
    """Charge et traite les données d'une unité."""
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    return datetimes, x, y, z

paths = [
    'GPSData/Unit1-camp_bis.mat',
    'GPSData/Unit2-camp_bis.mat',
    'GPSData/Unit3-camp_bis.mat',
    'GPSData/Unit4-camp_bis.mat'
]

# Charger les données de tous les fichiers
all_data = [load_and_process_data(path) for path in paths]
print(all_data)

# Trouver les datetimes communs
common_datetimes = set(all_data[0][0])
for data in all_data[1:]:
    common_datetimes.intersection_update(data[0])
common_datetimes = sorted(common_datetimes)

# Appliquer le masque pour conserver uniquement les datetimes communs et les valeurs correspondantes
filtered_data = []
for datetimes, x, y, z in all_data:
    mask = np.isin(datetimes, common_datetimes)
    filtered_data.append((datetimes[mask], x[mask], y[mask], z[mask]))

# Calculer les distances entre chaque paire d'antennes
distances = {}
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for i, j in pairs:
    xi, yi, zi = filtered_data[i][1], filtered_data[i][2], filtered_data[i][3]
    xj, yj, zj = filtered_data[j][1], filtered_data[j][2], filtered_data[j][3]
    d = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)
    mean_distance = np.mean(d)
    distances[f"Antenna {i+1}-{j+1}"] = mean_distance

# Afficher les distances moyennes
for key, value in distances.items():
    print(f"Distance moyenne entre {key}: {value:.2f} m")
# Distances données
d14 = 10.20
d34 = 7.11
d24 = 11.35
d13 = 12.20
d12 = 4.93
d23 = 10.24

# Fixer la position de Antenna 4
A4 = (0, 0)

# Antenna 1 à d14 mètres de A4 le long de l'axe positif x
A1 = (d14, 0)

# Antenne 3 à d34 mètres de A4 le long de l'axe négatif y (de l'arrière à tribord)
A3 = (0, -d34)

# Pour Antenna 2
y2 = -math.sqrt(d24**2 - d14**2)  # Utilisation du théorème de Pythagore, avec un signe négatif pour l'axe y
A2 = (d14, y2)


print("Position de l'Antenne 1:", A1)
print("Position de l'Antenne 2:", A2)
print("Position de l'Antenne 3:", A3)
print("Position de l'Antenne 4 (Origine):", A4)

print("Position de l'Antenne 1:", A1)
print("Position de l'Antenne 2:", A2)
print("Position de l'Antenne 3:", A3)
print("Position de l'Antenne 4 (Origine):", A4)

# Calculer le barycentre
Bx = (A1[0] + A2[0] + A3[0] + A4[0]) / 4
By = (A1[1] + A2[1] + A3[1] + A4[1]) / 4
B = (Bx, By)

# Nouvelles coordonnées par rapport au barycentre
A1_new = (A1[0] - Bx, A1[1] - By)
A2_new = (A2[0] - Bx, A2[1] - By)
A3_new = (A3[0] - Bx, A3[1] - By)
A4_new = (A4[0] - Bx, A4[1] - By)

print("Barycentre:", B)
print("Position de l'Antenne 1 par rapport au barycentre:", A1_new)
print("Position de l'Antenne 2 par rapport au barycentre:", A2_new)
print("Position de l'Antenne 3 par rapport au barycentre:", A3_new)
print("Position de l'Antenne 4 par rapport au barycentre:", A4_new)

# Initialisation du graphe
plt.figure(figsize=(10, 10))
plt.grid(True)
plt.axhline(y=0, color='k')  # Ligne horizontale
plt.axvline(x=0, color='k')  # Ligne verticale

# Plot les antennes
plt.scatter(*A1_new, color='red', label=f"Antenne 1 {A1_new}")
plt.scatter(*A2_new, color='blue', label=f"Antenne 2 {A2_new}")
plt.scatter(*A3_new, color='green', label=f"Antenne 3 {A3_new}")
plt.scatter(*A4_new, color='black', label=f"Antenne 4 (Origine) {A4_new}")
plt.scatter(0, 0, color='purple', label="Barycentre", s=100, marker='*')  # Barycentre, s=100 pour augmenter la taille

# Titres et légendes
plt.title("Disposition des Antennes sur le Bateau")
plt.xlabel("Avant <-> Arrière")
plt.ylabel("Tribord <-> Babord")
plt.legend()

# Afficher le graphe
# plt.show()
