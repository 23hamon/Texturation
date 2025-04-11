import json
import numpy as np

np.set_printoptions(precision=36, suppress=False, linewidth=200)


with open("data/absolute_transforms.json") as f :
    data = json.load(f)
    data_images = data["0"]
    for d in data_images :
        r = np.array(data_images[d][0], dtype=np.float128)
        t = np.array(data_images[d][1], dtype=np.float128)
        print(d, r, t, type(r), type(t))