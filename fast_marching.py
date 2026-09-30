import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# Fonction pour calculer la distance entre deux points selon Sethian[1996]
def sethian_distance(D, F, i, j):
    m, n = D.shape
    # Calcul des valeurs des éléments voisins de D à partir desquels la distance est calculée
    if i == 0:
        a = D[i+1, j]
    elif i == m-1:
        a = D[i-1, j]
    else:
        a = min(D[i-1, j], D[i+1, j])
    if j == 0:
        b = D[i, j+1]
    elif j == n-1:
        b = D[i, j-1]
    else:
        b = min(D[i, j-1], D[i, j+1])

    # Calcul de la distance approximée
    if np.sqrt(2)*F[i,j] > np.abs(a-b):
      dij = 0.5 * ( a + b + np.sqrt(2*((F[i,j]**2)) - (a-b)**2))
    else:
      dij = (F[i,j])**2 + min(a,b)
    return dij

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_neighbors(point, shape):
    x, z = point
    neighbors = []
    x_m, x_p, z_m, z_p = x-1, x+1, z-1, z+1
    if (0 <= x_m < shape[0]):
      neighbors.append((x_m, z))
    if (0 <= x_p < shape[1]):
      neighbors.append((x_p, z))
    if (0 <= z_m < shape[1]):
      neighbors.append((x, z_m))
    if (0 <= z_p < shape[1]):
      neighbors.append((x, z_p))
    return neighbors


# Algorithme de Fast Marching
def fast_marching(start, end, cost_matrix, max_distance=100000000):
    # Local variables delcarations
    shape = cost_matrix.shape
    visited = np.zeros(shape, dtype=bool)
    narrow_band = np.zeros(shape, dtype=bool)
    distance_map = np.full(shape, max_distance, dtype=float)
    far_away = np.full(shape, True)

    # Initialization
    distance_map[start] = 0
    visited[start] = True
    far_away[start] = False
    heap = [[0, start]]

    # Marching Forward -- Update
    while heap:
        dist, current = heapq.heappop(heap)
        visited[current] = True
        narrow_band[current] = False
        neighbors = get_neighbors(current, shape)
        for neighbor in neighbors:                 
            d = sethian_distance(distance_map, cost_matrix, neighbor[0], neighbor[1])
            if d < distance_map[neighbor]:
              if far_away[neighbor]:
                distance_map[neighbor] = sethian_distance(distance_map, cost_matrix, neighbor[0], neighbor[1])
                far_away[neighbor] = False
                narrow_band[neighbor] = True 
                heapq.heappush(heap, [d, neighbor])
    return distance_map

# Fonction pour calculer la probabilité de contrails pour un point donné
def contrails_likelihood(point, cost_matrix):
    return cost_matrix[point[0], point[1]]

def bilinear_interpolation(point, distance_map, grad_x, grad_z, direction):
    if direction == "x":
      x_P, z_P = point[0], point[1]
      x_m, x_p, z_m, z_p = int(x_P), int(x_P)+1, int(z_P), int(z_P)+1
      g_mm = np.gradient(distance_map, axis=0)[x_m, z_m]
      g_pm = np.gradient(distance_map, axis=0)[x_p, z_m]
      g_mp = np.gradient(distance_map, axis=0)[x_m, z_p]
      g_pp = np.gradient(distance_map, axis=0)[x_p, z_p]
      x = np.array([x_m, x_p])
      z = np.array([z_m, z_p])
      g = np.array([[g_mm, g_pm], 
                    [g_mp, g_pp]])
      return interp2d(x, z, g, kind="linear")(x_P - grad_x, z_P - grad_z)[0]
    else:
      x_P, z_P = point[0], point[1]
      x_m, x_p, z_m, z_p = int(x_P), int(x_P)+1, int(z_P), int(z_P)+1
      g_mm = np.gradient(distance_map, axis=1)[x_m, z_m]
      g_pm = np.gradient(distance_map, axis=1)[x_p, z_m]
      g_mp = np.gradient(distance_map, axis=1)[x_m, z_p]
      g_pp = np.gradient(distance_map, axis=1)[x_p, z_p]
      x = np.array([x_m, x_p])
      z = np.array([z_m, z_p])
      g = np.array([[g_mm, g_pm], 
                    [g_mp, g_pp]])
      return interp2d(x, z, g, kind="linear")(x_P - grad_x, z_P - grad_z)[0]


def gradient_descent(start, end, distance_map, max_iterations=1000, learning_rate=1):
    current_point = np.array(end)
    path = []
    path.append(end)

    g_x, g_z =  np.gradient(distance_map, axis=0)[current_point[0], current_point[1]], np.gradient(distance_map, axis=1)[current_point[0], current_point[1]]
    new_x = current_point[0] - bilinear_interpolation(current_point, distance_map, g_x, g_z, "x")
    new_z = current_point[1] - bilinear_interpolation(current_point, distance_map, g_x, g_z, "z")
    g_x, g_z =  bilinear_interpolation(current_point, distance_map, g_x, g_z, "x"), bilinear_interpolation(current_point, distance_map, g_x, g_z, "z")
    
    current_point = (new_x, new_z)
    path.append((round(new_x), round(new_z)))

    while start not in path:
        new_x = current_point[0] - bilinear_interpolation(current_point, distance_map, g_x, g_z, "x")
        new_z = current_point[1] - bilinear_interpolation(current_point, distance_map, g_x, g_z, "z")

        g_x, g_z =  bilinear_interpolation(current_point, distance_map, g_x, g_z, "x"), bilinear_interpolation(current_point, distance_map, g_x, g_z, "z")
        
        current_point = (new_x, new_z)
        path.append((round(new_x), round(new_z)))
    return path

# Exemple d'utilisation
start = (500, 1)
end = (500, 990)
cost_matrix = np.full((50, 50), 1) 

array = np.full((50, 50), 1)


center_x, center_y = 12, 12
half_size = 3 
array[center_x - half_size:center_x + half_size + 1, center_y - half_size:center_y + half_size + 1] = 2

center_x, center_y = 37, 37
half_size = 3 
array[center_x - half_size:center_x + half_size + 1, center_y - half_size:center_y + half_size + 1] = 2

center_x, center_y = 37 , 12
half_size = 3 
array[center_x - half_size:center_x + half_size + 1, center_y - half_size:center_y + half_size + 1] = 2

center_x, center_y = 12 , 37
half_size = 3  
array[center_x - half_size:center_x + half_size + 1, center_y - half_size:center_y + half_size + 1] = 2 

matrice = np.random.randint(1, 3, size=(1000, 1000))
distance_map = fast_marching(start, end, matrice)

def plot_gradient(distances):

    grad_x, grad_y = np.gradient(distances)

    x, y = np.meshgrid(np.arange(distances.shape[0]), np.arange(distances.shape[1]))

    fig, ax = plt.subplots()
    ax.imshow(distances, cmap='gray')
    ax.quiver(x, y, -grad_y, grad_x, color='r')
    ax.set_aspect('equal')
    plt.show()

# Recherche du chemin le plus court
path = gradient_descent(start, end, distance_map)
plt.imshow(distance_map ,cmap='hot', interpolation='nearest')
path_x, path_y = zip(*path)
plt.plot(path_y, path_x, color="blue")


plot_gradient(distance_map)