import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

# Création d'une image blanche RGB
image = np.ones((256, 256, 3), dtype=np.float32)

# Définition d'un triangle en coordonnées UV normalisées [0, 1]
# Format : (x, y) = (u, v)
uvs = [ np.array([
    [0.2, 0.2],  # sommet A
    [0.8, 0.2],  # sommet B
    [0.5, 0.7]   # sommet C
]),  np.array([
    [0.1, 0.6],  # sommet A
    [0.4, 0.1],  # sommet B
    [0.03, 0.8]   # sommet C
])]

for uvs in uvs : 
    # Conversion des UV en coordonnées pixels : (x, y) -> (colonne, ligne)
    c = uvs[:, 0] * image.shape[1]  # abscisses (colonnes)
    r = uvs[:, 1] * image.shape[0]  # ordonnées (lignes)

    # Utilisation de skimage.draw.polygon pour obtenir les pixels à remplir
    rr, cc = polygon(r, c, image.shape[:2])  # On fournit la taille en 2D (H, W)

    # Remplissage en rouge
    image[rr, cc] = [1, 0, 0]  # RGB = rouge

# Affichage
plt.imshow(image)
plt.axis('off')
plt.title("Triangle colorié en rouge")
plt.show()
