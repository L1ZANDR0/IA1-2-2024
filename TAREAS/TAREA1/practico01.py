import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Mamani Ramos Lizandro
# Generación de datos aleatorios controlados
np.random.seed(42)  # Para reproducibilidad
num_samples = 100
heights = np.random.uniform(1.4, 2.0, num_samples)  # Estaturas entre 1.4m y 2.0m
weights = []

for h in heights:
    if h < 1.5:
        weight = np.random.uniform(45, 55)  # Peso para estatura baja
    elif h < 1.7:
        weight = np.random.uniform(55, 70)  # Peso para estatura media-baja
    elif h < 1.9:
        weight = np.random.uniform(70, 85)  # Peso para estatura media-alta
    else:
        weight = np.random.uniform(85, 100)  # Peso para estatura alta
    weights.append(weight)

weights = np.array(weights)

# Visualización de los datos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=heights, y=weights)
plt.title("Relación entre Estatura y Peso")
plt.xlabel("Estatura (m)")
plt.ylabel("Peso (kg)")
plt.show()

# Ajuste de una curva a los datos
# Ajuste de una recta (polinomio de grado 1)
coeffs = np.polyfit(heights, weights, 1)
poly_eq = np.poly1d(coeffs)

# Visualización de la curva ajustada
plt.figure(figsize=(10, 6))
sns.scatterplot(x=heights, y=weights)
plt.plot(heights, poly_eq(heights), color='red', label=f'Curva ajustada: y={coeffs[0]:.2f}x + {coeffs[1]:.2f}')
plt.title("Relación entre Estatura y Peso con Curva Ajustada")
plt.xlabel("Estatura (m)")
plt.ylabel("Peso (kg)")
plt.legend()
plt.show()
