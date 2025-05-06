__author__ = "Jules Imbert"
__date__ = "2025-05-06"

import numpy as np

def compute_edges(Fi) :
    """
    Renvoie edges_set, le dictionnaire qui contiet les edges et leurs sommets
    {(i, j) : (v1, v2)} signifie l'arete entre la face i et la face j, dont les sommets sont v1 et v2
    (i, j) est stocke une seule fois, avec i < j
    """
    edges_set = dict()
    # On va stocker les edges dans un dictionnaire (v1, v2) ou v1 < v2
    edge_to_faces = dict() # {(v1, v2) : [i, j]} -> faces Fi et Fj reliees par l'arrete (v1, v2)
    for i, (v1, v2, v3) in enumerate(Fi):
        edges = [tuple(sorted((v1, v2))),
                tuple(sorted((v2, v3))),
                tuple(sorted((v3, v1)))]
        for edge in edges:
            if edge in edge_to_faces :
                edge_to_faces[edge].append(i)
            else : 
                edge_to_faces[edge] = [i]
    # On ne conserve que les arretes qui separent deux faces
    for edge in edge_to_faces :
        faces = edge_to_faces[edge]
        if len(faces) == 2 :
            edges_set[tuple(sorted(faces))] = edge
            # on stocke au format (i, j) avec i < j
    return edges_set