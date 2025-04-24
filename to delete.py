import matplotlib.pyplot as plt

# Coordonnées des sommets
points = [
    [0, 0],  # 0
    [1, 0],  # 1
    [1, 1],  # 2
    [0, 1]   # 3
]

# Pour fermer le carré
points.append(points[0])

# Séparer les coordonnées
x, y = zip(*points)

# Tracer le carré
plt.plot(x, y, marker='o')

# Annoter les sommets
for i, (xi, yi) in enumerate(points[:-1]):  # ne pas annoter le point de fermeture
    plt.text(xi + 0.03, yi + 0.03, str(i), fontsize=12, color='red')

plt.axis('equal')
plt.grid(True)
plt.title("Carré avec sommets annotés")
plt.show()
