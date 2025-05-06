__author__ = "Clemence Hamon"
__date__ = "2025-05-06"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def precise_texture_bleeding(uv_mask, texture):
    
    """
    Remplit les zones non texturées de la texture par la couleur du pixel texturé le plus proche.

    Paramètres :
    - uv_mask : np.ndarray de forme (H, W), avec 1 pour les pixels texturés, 0 sinon.
    - texture : np.ndarray de forme (H, W, 3), image RGB.

    Retour :
    - texture remplie par bleeding.
    """

    h, w = uv_mask.shape

    # On obtient la distance et le label du pixel le plus proche texturé
    dist, labels = cv2.distanceTransformWithLabels(
        (uv_mask == 0).astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL
    )

    # Coordonnées des pixels valides (texturés)
    valid_coords = np.column_stack(np.where(uv_mask > 0))

    # Création d’une image de sortie
    result = texture.copy()

    # Table : pour chaque label (1, 2, ..., N) → coordonnée du pixel texturé
    label_to_coord = {}
    for idx, (y, x) in enumerate(valid_coords):
        label = idx + 1  # les labels commencent à 1
        label_to_coord[label] = (y, x)

    # Remplissage des pixels à partir des plus proches pixels texturés
    for y in range(h):
        for x in range(w):
            if uv_mask[y, x] == 0:
                label = labels[y, x]
                if label in label_to_coord:
                    sy, sx = label_to_coord[label]
                    result[y, x] = texture[sy, sx]

    return result

if __name__ == "__main__":
    # Chargement de l'image UV et de la texture


    uv_mask = np.load("uv_mask.npy")
    final_texture = np.load("final_texture.npy")

    final_texture_bleeded = precise_texture_bleeding(uv_mask, final_texture)

    # Affichage
    plt.imshow(final_texture_bleeded)
    plt.title("Texture après bleeding")
    plt.axis('off')
    plt.show()

    # Enregistrement
    Image.fromarray(final_texture_bleeded).save("texture_bleeded.png")
