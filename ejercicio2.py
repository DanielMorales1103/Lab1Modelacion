import numpy as np
import matplotlib.pyplot as plt

# Definir la función objetivo
def objective_function(x, y):
    return (x - 3) ** 2 + (y - 2) ** 2

# Inicializar parámetros
N = 40  # Número de partículas
w = 0.5  # Inercia
c1 = 1.5  # Coeficiente cognitivo
c2 = 1.5  # Coeficiente social
iterations = 100  # Número máximo de iteraciones
threshold = 1e-5  # Umbral de cambio en la posición global

np.random.seed(16) 

# Inicializar posiciones y velocidades
positions = np.random.uniform(-10, 10, (N, 2))
velocities = np.random.uniform(-1, 1, (N, 2))

# Inicializar las mejores posiciones personales y globales
personal_best_positions = np.copy(positions)
personal_best_scores = objective_function(positions[:, 0], positions[:, 1])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

def update_particle(p, v, p_best, g_best, w, c1, c2):
    r1 = np.random.rand()
    r2 = np.random.rand()
    new_v = w * v + c1 * r1 * (p_best - p) + c2 * r2 * (g_best - p)
    new_p = p + new_v
    return new_p, new_v

# Guardar posiciones para graficar
history_positions = [np.copy(positions)]

# Algoritmo PSO
for iteration in range(iterations):
    previous_global_best_position = np.copy(global_best_position)
    for i in range(N):
        positions[i], velocities[i] = update_particle(
            positions[i], velocities[i], personal_best_positions[i], global_best_position, w, c1, c2)
        current_score = objective_function(positions[i, 0], positions[i, 1])
        if current_score < personal_best_scores[i]:
            personal_best_positions[i] = positions[i]
            personal_best_scores[i] = current_score
        if current_score < global_best_score:
            global_best_position = positions[i]
            global_best_score = current_score
    
    history_positions.append(np.copy(positions))
    
    # Verificar si el cambio en la mejor posición global es menor al umbral
    change = np.linalg.norm(global_best_position - previous_global_best_position)
    if change < threshold:
        break

print(f"Mejor posición global encontrada: {global_best_position}")
print(f"Valor de la función objetivo en la mejor posición global: {global_best_score}")
    
# Graficar resultados
X = np.linspace(-10, 10, 400)
Y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(X, Y)
Z = objective_function(X, Y)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfica inicial
axes[0].contourf(X, Y, Z, levels=50, cmap='viridis')
axes[0].scatter(history_positions[0][:, 0], history_positions[0][:, 1], color='red')
axes[0].set_title('Iteración Inicial')

# Gráfica de un punto medio
mid_point = len(history_positions) // 2
axes[1].contourf(X, Y, Z, levels=50, cmap='viridis')
axes[1].scatter(history_positions[mid_point][:, 0], history_positions[mid_point][:, 1], color='red')
axes[1].set_title(f'Iteración {mid_point}')

# Gráfica final
axes[2].contourf(X, Y, Z, levels=50, cmap='viridis')
axes[2].scatter(history_positions[-1][:, 0], history_positions[-1][:, 1], color='red')
axes[2].set_title('Iteración Final')

plt.show()