import numpy as np

#texture_map = np.load("fichiers_intermediaires/texture_map.npy")

texture_map = np.random.random((512, 512, 3)) * 255

row, col = texture_map.shape[:2]

def red_intensity(texture_map):
    return texture_map[:, :, 0]

def green_intensity(texture_map):
    return texture_map[:, :, 1]

def blue_intensity(texture_map):
    return texture_map[:, :, 2]