#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import numpy as np
from pathlib import Path
import json
from functools import partial, partialmethod
from pprint import pprint
from multiprocessing import Pool
from datetime import datetime
import cv2
cv2.setNumThreads(4)
import time
#from IPython import embed

from ba_first_guess import ba_first_guess,complete_ba_first_guess
from ImagePair import ImagePair
from ImageWrapper import Image
from utils import RigParametersRefract as Rp,load_variables_from_json,save_variables_to_json
from io_utils import * 
from auto_calibration import auto_calibrate
from pair_graph import build_graph,complete_graph,draw_graph




def load_images(pair_info, *, keypoint_channel_name):

    pair_id, left_fn, right_fn = pair_info
    left_image = Image.from_filename(left_fn, keypoint_channel_name=keypoint_channel_name)
    right_image = Image.from_filename(right_fn, keypoint_channel_name=keypoint_channel_name)

    # print(pair_id, datetime.now() - start)
    return pair_id, left_image, right_image



def load_pairs(left_filenames,right_filenames,indexes,rp,sensi_params):
    print('loading images')
    pairs = {}
    with Pool(16) as p:                                                                                                              ### MULTIPROCESSING DE LOAD_IMAGES RENVOYANT IMAGE GAUCHE, DROITE + PAIR ID (POUR LES AVOIR DANS L'ORDRE)
        for pair_id, left_image, right_image in tqdm(p.imap_unordered(
            partial(load_images, keypoint_channel_name="g"),
            zip(indexes, left_filenames, right_filenames),                                                        ###VARIABLES DANS UN ZIP ( PAS POSSIBLE AUTREMENT AVEC MULTIPROCESSING ) [(5, 'left1.jpg', 'right1.jpg'),
            chunksize=1
        ), total=len(left_filenames)):
            
            # if left_image.descriptors is None or right_image.descriptors is None:
            #     continue

            pairs[pair_id] = ImagePair(left_image, right_image, pair_id=pair_id, rig_parameters=rp, sensi_params=sensi_params )  ##PARAM - TOLERANCE SUR LA CALIBRATION      ###PAIRES D'IMAGES STOCKÉES DANS PAIRS CLASSE IMAGEPAIR 
            if pairs[pair_id].selfdestroy==True:
                print(f"not treating pair{pair_id}")
                pairs.pop(pair_id)
                continue


            pairs[pair_id].unload_images()
            if len(pairs[pair_id].points)<sensi_params[5]:
                print(f"not enough keypoints detected for pair{pair_id} ({len(pairs[pair_id].points)}), not treating pair.")
                pairs.pop(pair_id)

    return pairs



def fetch_rp(save=False,filename='rp.json'):    #@TODO move to utils
    M = 268.73775590551185 / 2
    f_l = M * 24.4936280811
    cx_l, cy_l = 1510.08637472, 986.0160617
    air_K_l = np.array([
        [f_l, 0, cx_l],
        [0, f_l, cy_l],
        [0, 0, 1]
    ])

    f_r = M * 24.57431989
    cx_r, cy_r = 1493.22488246, 996.53835531
    air_K_r = np.array([
        [f_r, 0, cx_r],
        [0, f_r, cy_r],
        [0, 0, 1]
    ])



    #CALIB MER 1004
    # x = [-0.01924443, 0.1614238, -0.04860656, 218.0576, 7.120566, -22.87093, 0.01432869, -0.004050934, 0.9633524, -0.01648749, -0.00316129, 0.9999437]

    # #CALIB MER 1004  #2
    # x=  [-0.02607184, 0.1645179, -0.04801103, -215.0423, -9.181835, 22.21749, 0.003030494, -0.03992838, 0.9797754, -0.02231426, -0.01902701, 0.9764153]


    # Rc = x[:3]
    # Rc= cv2.Rodrigues(np.array(Rc))[0]
    # Tc = x[3:6]
    # n_l = x[6:9]
    # n_r = x[9:]
    # n_l/=np.linalg.norm(n_l)
    # n_r/=np.linalg.norm(n_r)


    # XXX great on the 21_03_2025 (big rock) 
    Rc  = cv2.Rodrigues(np.deg2rad(np.array([-0.12180768,  8.41991392, -3.1929143 ])))[0]           ###MATRICES DE ROTATION ET DE TRANSLATION DES CAMÉRAS , RODRIGUES = CONVERTIT VECTEUR EN MATRICE ROTATION
    Tc  = np.array([210.29701361,   7.24916971, -19.77955875])                                      
    n_l = np.array([0.03223987, 0.0119772 , 0.99940839])                                            ###NORMALES AUX CAMÉRAS SUR LE PLEXI
    n_r = np.array([-0.04773011, -0.02050776,  0.99864972])
    d_l = 40.0                                                                                      ###DISTANCES
    d_r = 40.0
    rp = Rp(air_K_l=air_K_l, air_K_r=air_K_r, d_l=d_l, d_r=d_r, n_l=n_l, n_r=n_r, Rc=Rc, Tc=Tc)                ###CREATION DE LA CLASSE AVEC PARAMÈTRES RP=RIGPARAMETERS
    if save:
        rp.save(filename) 
    return rp



def generate_ply(filenames,pairs=None,graph=None,sensi_params=None,rp=None):
    if sensi_params is None:
        sensi_params = load_variables_from_json('params.json')
    if rp is None:
        rp=Rp.from_json()
    left_filenames=filenames["l"]
    right_filenames=filenames["r"]
    indexes=filenames["indexes"]
    graph_Exists=(graph!=None)
    if graph_Exists:
        graph_Exists=(graph.number_of_nodes()!=0)

    if not graph_Exists:
        pairs=load_pairs(left_filenames,right_filenames,indexes,rp,sensi_params)
        graph, all_matches = build_graph(pairs,sensi_params)
        absolute_transforms,connected_components = ba_first_guess(pairs, graph, all_matches, rp, sensi_params)
        # print(absolute_transforms)
        # with open("absolute_transforms.json", 'w') as f:
        #     f.write(absolute_transforms_to_json(absolute_transforms))
        
    else:

        newpairs=load_pairs(left_filenames,right_filenames,indexes,rp,sensi_params)
        new_matches=complete_graph(graph,newpairs,pairs,sensi_params)
        pairs=pairs|newpairs
        absolute_transforms,connected_components = complete_ba_first_guess(pairs, graph,new_matches,rp,sensi_params)
        # print(absolute_transforms)
        # with open("absolute_transforms.json", 'w') as f:
        #     f.write(absolute_transforms_to_json(absolute_transforms))

    return pairs,graph,absolute_transforms,connected_components




# def main():
   
#     to_remove=[]
#     rp=Rp.from_json()
    
#                                                        #POSSIBILITÉ D'UPDATE LES FOCALES, MAIS PAS RC ET TC SOUS L'EAU
#     IMG_START, IMG_END = 1, 54
#     BASE_DIR="/home/mig/test_21_03_2025/rock_pool/downsampled"
#     # to_remove = [30]
   
#     left_filenames  = [BASE_DIR + f'/scene_l_{i:04}.jpeg' for i in range(IMG_START, IMG_END) if i not in to_remove]                      ###FILTRAGE DES NOMS D'IMAGES
#     right_filenames = [BASE_DIR + f'/scene_r_{i:04}.jpeg' for i in range(IMG_START, IMG_END) if i not in to_remove]



#     IMG_START,IMG_END=10,20
#     filenames={"l":left_filenames[IMG_START-1:IMG_END-1],"r":right_filenames[IMG_START-1:IMG_END-1],"indexes":[i for i in range(IMG_START, IMG_END) if i not in to_remove]}
#     pairs,graph=generate_ply(filenames)
#     draw_graph(graph)
    




#     IMG_START,IMG_END=20,40
#     filenames={"l":left_filenames[IMG_START:IMG_END],"r":right_filenames[IMG_START:IMG_END],"indexes":range(IMG_START, IMG_END)}
#     pairs,graph=generate_ply(filenames,pairs,graph)
#     draw_graph(graph)


# save_variables_to_json('params.json',match_treshold=0.9,            
#                         intra_pair_reconstruction_error=5,
#                         min_num_matches=30,
#                         nbatches=50,
#                         accepted_trans_loss=0.3,min_intra_matches=100)


# if __name__ == '__main__':
#     main()

# rp=fetch_rp(save=True,filename="rp_rock.json")
# # 