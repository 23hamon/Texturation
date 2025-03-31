#!/usr/bin/env python3
from json import loads
from pprint import pprint
import os
#os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import PosixPath, Path
from tqdm import tqdm
import itertools
import scipy
from scipy.optimize import least_squares, differential_evolution, direct, dual_annealing, minimize
from numba import njit
import open3d as o3d
from utils import reconstruct_3d_points
from calibration_utils import find_corners, find_circles
from utils import trace_refract_ray, reconstruct_3d_point_refract
from pairs_lists import *
from refract_calibration import calibrate_stereo_refract_global2, calibrate_stereo_air_global
from time import sleep

import multiprocessing


def main():
    multiprocessing.set_start_method("spawn")
    K_l, K_r = np.eye(3), np.eye(3)
    dist_l, dist_r = np.zeros(5), np.zeros(5)
    """
# 30 m
best_error: 4.967172772074357 best_num_inliers: 39
best_inliers: [93, 87, 45, 65, 53, 80, 82, 52, 95, 89, 48, 91, 84, 79, 72, 47, 92, 51, 54, 83, 74, 73, 94, 90, 55, 44, 56, 96, 98, 60, 99, 81, 70, 71, 78, 85, 68, 64, 97]
Rc [-1.30498738  8.41854718 -2.94117157]
Tc [-214.39713479   -8.62643753   19.8896945 ]
focals_l []
focals_r []
n_l (normalized) [ 0.02013153 -0.00840727  0.99976199]
n_r (normalized) [-0.04879483  0.00265978  0.99880528]

# 3 m
best_error: 3.7896895040484035 best_num_inliers: 34
best_inliers: [21, 15, 36, 31, 25, 0, 33, 9, 35, 3, 26, 34, 19, 14, 11, 16, 13, 37, 20, 30, 27, 32, 12, 22, 1, 41, 2, 5, 28, 23, 29, 42, 17, 24]
Rc [-1.19235187  9.20888512 -2.81728439]
Tc [-218.48968764   -6.34096969   24.33630066]
focals_l []
focals_r []
n_l (normalized) [ 0.01124623 -0.02989082  0.9994899 ]
n_r (normalized) [-0.03078449 -0.01523896  0.99940987]


# 30 m
best_error: 6.603185812501531 best_num_inliers: 38
best_inliers: [89, 82, 44, 85, 73, 65, 53, 45, 46, 74, 78, 72, 91, 99, 56, 93, 52, 79, 80, 81, 92, 63, 48, 94, 97, 87, 83, 84, 64, 47, 98, 96, 71, 95, 70, 51, 90, 2]
Rc [-1.55046654  9.08701799 -2.85652265]
Tc [-215.23314019  -10.24805769   23.50853111]
focals_l []
focals_r []
n_l (normalized) [ 0.00868317 -0.02350121  0.9996861 ]
n_r (normalized) [-0.03271353  0.00369997  0.99945792]


# 30 m
best_inliers: [73, 58, 91, 90, 82, 106, 105, 92, 107, 59, 72]
Rc [-1.00793819  9.06803736 -2.88897029]
Tc [-211.44085946  -11.38312855   19.30352704]
focals_l []
focals_r []
n_l (normalized) [-0.03831513 -0.0165606   0.99912847]
n_r (normalized) [-0.07237116 -0.02814421  0.9969806 ]

# robuste_06_02_2025
best_error: 7.014264106750488 best_num_inliers: 4
best_inliers: [1, 0, 6, 7]
Rc [  1.42017027 -11.85378813   2.07110373]
Tc [205.94207531  17.70295251  40.48073192]
focals_l []
focals_r []
n_l (normalized) [-0.03147539 -0.02031246  0.99929811]
n_r (normalized) [-0.02390627 -0.05747185  0.99806086]
    """

    #pairs = pairs_robuste_eleves
    pairs = pairs_robuste_27_02_2025_checkerboard_30m
    #pairs = pairs_robuste_27_02_2025_checkerboard_3m
    #pairs = pairs_circles_3m
    #pairs = pairs_robuste_06_02_2024
    #pairs = pairs_air_21_03_2025
    pairs = pairs_pool_21_03_2025
    #pairs = pairs_24mm_11_02_2024
    #pairs = pairs[:20]
    #pairs = pairs[36:39]
    #pairs = [*pairs[3:6], *pairs[12:15], *pairs[15:18], *pairs[21:24], *pairs[24:27]]  # XXX good pairs for pairs_robuste_eleves
    #pairs = [pairs[-9], pairs[-5], *pairs[-3:]]  # XXX good pairs for pairs_robuste_06_02_2024
    # XXX -5 is good
    # XXX -6 seems fine too
    # XXX -9 seems good too

    NUM_CORNERS = (9, 7)
    #NUM_CORNERS = (10, 8)
    N_POINTS = NUM_CORNERS[0] * NUM_CORNERS[1]
    SIZE_CHESSBOARD = 30  # in mm
    #SIZE_CHESSBOARD = 23  # in mm
    HOR_DIST_CORNERS = (NUM_CORNERS[0] - 1) * SIZE_CHESSBOARD
    VER_DIST_CORNERS = (NUM_CORNERS[1] - 1) * SIZE_CHESSBOARD

    y = np.linspace(0, VER_DIST_CORNERS, NUM_CORNERS[1])
    x = np.linspace(0, HOR_DIST_CORNERS, NUM_CORNERS[0])
    [xx, yy] = np.meshgrid(x, y)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    PTS_COORDS_Z = np.zeros((NUM_CORNERS[0] * NUM_CORNERS[1], 1), np.float32)
    OBJ_PTS_COORDS = np.float32(np.hstack((xx, yy, PTS_COORDS_Z)))
    OBJ_PTS_COORDS -= OBJ_PTS_COORDS.mean(axis=0)

    corners = {}
    all_corners_l = []
    all_corners_r = []

    if True:
        for i, (left, right) in enumerate(tqdm(pairs)):
            print(left)
            print(right)
            left_img = cv2.imread(str(left))
            right_img = cv2.imread(str(right))

            gray_left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            #corners_l = find_corners(left_img[..., 1], NUM_CORNERS)
            #corners_r = find_corners(right_img[..., 1], NUM_CORNERS)
            corners_l = find_corners(gray_left_img, NUM_CORNERS)
            corners_r = find_corners(gray_right_img, NUM_CORNERS)

            found = corners_l is not None and len(corners_l) == N_POINTS and corners_r is not None and len(corners_r) == N_POINTS
            print(found)
            if found:
                corners_l, corners_r = corners_l.reshape((-1, 2)), corners_r.reshape((-1, 2))

                if (corners_l[0] < corners_l[-1]).all():
                    corners_l = corners_l[::-1]

                if (corners_r[0] < corners_r[-1]).all():
                    corners_r = corners_r[::-1]

                corners_l = corners_l[::-1]
                corners_r = corners_r[::-1]

                corners[i] = (corners_l, corners_r)

                all_corners_l.append(corners_l)
                all_corners_r.append(corners_r)

                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(left_img[..., 1], vmin=0, vmax=255, cmap="grey")
                ax[0].scatter(corners_l[..., 0], corners_l[..., 1], c="r")

                ax[1].imshow(right_img[..., 1], vmin=0, vmax=255, cmap="grey")
                ax[1].scatter(corners_r[..., 0], corners_r[..., 1], c="r")
                #plt.show()
                plt.close(fig)
            else:
                print(f"didn't find corners for {i} (left path: {left})")

        np.savez("checkerboard_robuste.npz", all_corners_l=all_corners_l, all_corners_r=all_corners_r)
    else:
        #f = np.load("sift_points.npz")
        #f = np.load("checkerboard_3m_sift_points.npz")
        #f = np.load("checkerboard_30m_sift_points.npz")
        f = np.load("checkerboard_robuste.npz")
        all_corners_l = f["all_corners_l"]
        all_corners_r = f["all_corners_r"]

    print("number of images", len(all_corners_l))

    # XXX for air
    #selected = np.array([0, 1, 2, 3, 6, 7, 9, 10, 11, 14, 16, 17, 18, 20, 23])
    #selected = np.array(range(0, 33))
    #print(len(selected), "images")
    #sleep(1)
    #all_corners_l = all_corners_l[selected]
    #all_corners_r = all_corners_r[selected]

    # XXX for checkerboard_3m
    #selected = np.array([0, 6, 8, 2, 16, 17, 18, 20])
    #selected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    #all_corners_l = all_corners_l[selected]
    #all_corners_r = all_corners_r[selected]

    objpoints = np.array([OBJ_PTS_COORDS for _ in range(len(all_corners_l))])
    print("DBG", len(all_corners_l))

    #calibrate_stereo_air_global(all_corners_l, all_corners_r, objpoints);
    #raise SystemExit

    M = 268.73775590551185 / 2
    #f = np.load("kl_air.npz")
    #air_K_l = f["K"]
    #air_K_l[:2,:]# *= 2
    #air_dist_l = f["dist"]

    #f = np.load("kr_air.npz")
    #air_K_r = f["K"]
    #air_K_r[:2,:]# *= 2
    #air_dist_r = f["dist"]

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
    air_dist_l = np.zeros(5)
    air_dist_r = np.zeros(5)

    #calibrate_refract(air_K_l, air_dist_l, all_corners_l, objpoints)
    calibrate_stereo_refract_global2(air_K_l, air_K_r, air_dist_l, air_dist_r, all_corners_l, all_corners_r, objpoints)
    raise SystemExit

    #intrinsic_flags =  cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    #intrinsic_flags =  None# cv2.CALIB_FIX_K3 #| cv2.CALIB_ZERO_TANGENT_DIST
    intrinsic_flags =  cv2.CALIB_ZERO_TANGENT_DIST
    ret, K_l, dist_l, rvecs, tvecs = cv2.calibrateCamera(objpoints, np.array(all_corners_l), (6000, 4000), None, None, flags=intrinsic_flags)
    print(ret)
    print(rvecs)
    print(tvecs)
    ret, K_r, dist_r, rvecs, tvecs = cv2.calibrateCamera(objpoints, np.array(all_corners_r), (6000, 4000), None, None, flags=intrinsic_flags)
    print(ret)
    print(K_l)
    print(dist_l)
    print(K_r)
    print(dist_r)


    best_error = 100000
    best_R = np.zeros((3, 3))
    best_T = np.zeros((3, 1))
    best_K_l = np.zeros((3, 3))
    best_K_r = np.zeros((3, 3))
    best_dist_l = np.zeros((1, 14))
    best_dist_r = np.zeros((1, 14))

    old_K_l = K_l.copy()
    old_K_r = K_r.copy()

    num_pairs = len(all_corners_l)
    num_to_take = num_pairs - 1

    for indices in tqdm(itertools.combinations(range(num_pairs), num_to_take), total=scipy.special.comb(num_pairs, num_to_take)):
        indices = list(indices)

        ret, new_K_l, new_dist_l, new_K_r, new_dist_r, R, T, e, f = cv2.stereoCalibrate(
            objpoints[indices],
            np.array(all_corners_l)[indices],
            np.array(all_corners_r)[indices],
            K_l, dist_l, K_r, dist_r,
            #(6000, 4000), flags=cv2.CALIB_FIX_INTRINSIC)
            (6000, 4000), flags=(cv2.CALIB_FIX_INTRINSIC))

        if ret < best_error:
            best_error = ret
            best_R, best_T = R.copy(), T.copy()
            #best_K_l = new_K_l.copy()
            #best_K_r = new_K_r.copy()
            #best_dist_l = new_dist_l.copy()
            #best_dist_r = new_dist_r.copy()
            best_K_l = K_l.copy()
            best_K_r = K_r.copy()
            best_dist_l = dist_l.copy()
            best_dist_r = dist_r.copy()

            print("========== new best calib: ==========")
            print(indices)
            rot_rodr, _ = cv2.Rodrigues(R)
            print(R)
            print(rot_rodr)
            print(np.rad2deg(np.linalg.norm(rot_rodr)))
            print(T)
            print(dist_l)
            print(dist_r)
            print("error:", ret)

    print("========== final: ==========")
    print(np.rad2deg(cv2.Rodrigues(best_R)[0]))
    print(best_T)
    print()
    print(best_K_l)
    print(best_K_r)
    print(best_dist_l)
    print(best_dist_r)

    #best_K_l /= 2
    #best_K_l[-1,-1] = 1
    #best_K_r /= 2
    #best_K_r[-1,-1] = 1

    print(best_K_l)

    PL = np.zeros((3, 4))
    PL[:3, :3] = np.eye(3)
    PL = best_K_l @ PL

    PR = np.zeros((3, 4))
    PR[:3, :3] = best_R
    PR[:3, 3] = best_T.flatten()
    PR = best_K_r @ PR

    for i, (img_points_l, img_points_r) in enumerate(zip(all_corners_l, all_corners_r)):
        undistorted_img_points_l = cv2.undistortImagePoints(img_points_l, best_K_l, best_dist_l).squeeze()
        undistorted_img_points_r = cv2.undistortImagePoints(img_points_r, best_K_r, best_dist_r).squeeze()

        points_3d, errors = reconstruct_3d_points(undistorted_img_points_l, undistorted_img_points_r, PL, PR, return_errors=True)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(-points_3d))
        o3d.io.write_point_cloud(f"checkerboard_opencv_{i}.ply", pcd)

    #np.savez("/home/luca/test/photogrammetry/meshroom_extrinsic.npz", Rc=best_R, Tc=best_T)
    #np.savez("/home/luca/test/photogrammetry/meshroom_kl.npz", K=best_K_l, dist=best_dist_l)
    #np.savez("/home/luca/test/photogrammetry/meshroom_kr.npz", K=best_K_r, dist=best_dist_r)


@njit(error_model="numpy", cache=True)
def calibrate_refract_loss_point(keypoint, objpoint, air_K, rot, trans, distance, normal, thickness, eta_glass, eta_water):
    water_ro, water_rd = trace_refract_ray(keypoint, air_K, distance, normal, thickness, eta_glass, eta_water)

    water_ro = rot @ water_ro + trans
    water_rd = rot @ water_rd

    world_point = -water_ro[2] / water_rd[2] * water_rd + water_ro

    return (world_point - objpoint)[:2]



@njit(error_model="numpy", cache=True)
def _calibrate_refract_loss(distance, normal, rots, tvecs, keypoints, objpoints, air_K, thickness, eta_glass, eta_water):
    loss = np.zeros((len(keypoints), 9 * 7, 2))
    for i, (img_keypoints, img_objpoints, rot, trans) in enumerate(zip(keypoints, objpoints, rots, tvecs)):
        for j, (img_keypoint, img_objpoint) in enumerate(zip(img_keypoints, img_objpoints)):
            loss[i, j] = calibrate_refract_loss_point(img_keypoint, img_objpoint, air_K, rot, trans, distance, normal, thickness, eta_glass, eta_water)

    return loss.flatten()


def calibrate_refract_loss(x, keypoints, objpoints, air_K, thickness, eta_glass, eta_water):
    distance = x[0]
    normal = x[1:4]
    normal /= np.linalg.norm(normal)

    rots = np.empty((len(keypoints), 3, 3))
    tvecs = np.empty((len(keypoints), 3))
    for i in range(len(keypoints)):
        rot = x[4 + i * 6 : 7 + i * 6]
        trans = x[7 + i * 6 : 10 + i * 6]
        rot = cv2.Rodrigues(rot)[0]

        rots[i] = rot
        tvecs[i] = trans

    return _calibrate_refract_loss(distance, normal, rots, tvecs, keypoints, objpoints, air_K, thickness, eta_glass, eta_water)


@njit(error_model="numpy", cache=True)
def _stereo_refract_loss1(d_l, d_r, Rc, Tc, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints, penalize_angle):
    loss = []

    for i, (img_corners_l, img_corners_r) in enumerate(zip(corners_l, corners_r)):
        img_points = np.zeros((objpoints[0].shape[0], 3))
        points_mean = np.zeros(3)
        for j, (corner_l, corner_r) in enumerate(zip(img_corners_l, img_corners_r)):
            p, corner_reconstruction_loss = reconstruct_3d_point_refract(corner_l, corner_r, air_K_l, air_K_r, d_l, d_r, n_l, n_r, Rc, Tc, thickness, eta_glass, eta_water)
            #loss.append(corner_reconstruction_loss * 0.1)
            #img_points.append(p)
            img_points[j] = p
            points_mean += p

        #for j, img_point in enumerate(img_points):
        #    for k in range(len(img_points)):
        #        if j != k:
        #            dist_to_other_img_point = np.linalg.norm(img_point - img_points[k])
        #            dist_to_other_obj_point = np.linalg.norm(objpoints[0, j] - objpoints[0, k])
        #            corner_size_loss = dist_to_other_img_point - dist_to_other_obj_point

        #            loss.append(corner_size_loss)

        for j in range(6):
            for k in range(8):
                img_point = img_points[j * 9 + k]
                img_point_right = img_points[j * 9 + (k + 1)]
                img_point_down = img_points[(j + 1) * 9 + k]

                diff_right = img_point_right - img_point
                diff_down = img_point_down - img_point

                loss.append(np.linalg.norm(diff_right) - 30)
                loss.append(np.linalg.norm(diff_down) - 30)
                loss.append(1 * (diff_right / np.linalg.norm(diff_right)) @ (diff_down / np.linalg.norm(diff_down)))

        #img_points = np.array(img_points)
        points_mean /= img_points.shape[0]
        img_points -= points_mean

        u, _, _ = np.linalg.svd(img_points.T)
        normal = u[:, 2]

        if penalize_angle:
            for point in img_points:
                loss.append(5 * np.dot(normal, point))

    #penalty = -np.minimum(0, np.linalg.norm(Tc) - 150) + np.maximum(0, np.linalg.norm(Tc) - 300)# + 100 * np.sum(np.array(planarity_penalties))
    #loss.append(100 * penalty)
    return np.array(loss)# + 100 * penalty


def stereo_refract_loss1(x, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints, penalize_angle=True):
    d_l = x[0]
    d_r = x[1]
    Rc = x[2:5]
    Tc = x[5:8]
    n_l = x[8:11]
    n_r = x[11:14]

    n_l /= np.linalg.norm(n_l)
    n_r /= np.linalg.norm(n_r)

    #print(d_l, d_r, Rc, Tc, n_l, n_r)

    Rc = cv2.Rodrigues(Rc)[0]
    return _stereo_refract_loss1(d_l, d_r, Rc, Tc, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints, penalize_angle=penalize_angle)


iter_cnt = 0
best_loss = np.inf

def stereo_refract_loss_global(x, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints, penalize_angle=True):
    global iter_cnt, best_loss
    iter_cnt += 1
    #print(iter_cnt, best_loss)

    # TODO remove penalize_angle from stereo_refract_loss1
    #dist_l = np.zeros(5)
    #dist_l[-2:] = x[14:16]
    #dist_r = np.zeros(5)
    #dist_r[-2:] = x[-2:]

    #undistorted_corners_l = []
    #undistorted_corners_r = []

    #for corners in corners_l:
    #    undistorted_corners_l.append(cv2.undistortImagePoints(corners, air_K_l, dist_l).squeeze())

    #for corners in corners_r:
    #    undistorted_corners_r.append(cv2.undistortImagePoints(corners, air_K_r, dist_r).squeeze())

    losses = stereo_refract_loss1(x, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints)
    loss = losses @ losses + np.maximum(0, np.abs(x[0] - x[1]) - 40) * 10
    if loss < best_loss:
        best_loss = loss
    return loss


def _stereo_refract_loss_display(d_l, d_r, Rc, Tc, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints):
    loss = []

    for i, (img_corners_l, img_corners_r) in enumerate(zip(corners_l, corners_r)):
        img_points = np.zeros((objpoints[0].shape[0], 3))
        points_mean = np.zeros(3)
        for j, (corner_l, corner_r) in enumerate(zip(img_corners_l, img_corners_r)):
            p, corner_reconstruction_loss = reconstruct_3d_point_refract(corner_l, corner_r, air_K_l, air_K_r, d_l, d_r, n_l, n_r, Rc, Tc, thickness, eta_glass, eta_water)
            #img_points.append(p)
            img_points[j] = p
            points_mean += p

        for j in range(6):
            for k in range(8):
                img_point = img_points[j * 9 + k]
                img_point_right = img_points[j * 9 + (k + 1)]
                img_point_down = img_points[(j + 1) * 9 + k]

                loss.append(np.linalg.norm(img_point - img_point_right) - 30)
                loss.append(np.linalg.norm(img_point - img_point_down) - 30)

    return np.array(loss)


def stereo_refract_loss_display(x, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints):
    d_l = x[0]
    d_r = x[1]
    Rc = x[2:5]
    Tc = x[5:8]
    n_l = x[8:11]
    n_r = x[11:14]

    #dist_l = np.zeros(5)
    #dist_l[-2:] = x[14:16]
    #dist_r = np.zeros(5)
    #dist_r[-2:] = x[-2:]

    n_l /= np.linalg.norm(n_l)
    n_r /= np.linalg.norm(n_r)

    #undistorted_corners_l = []
    #undistorted_corners_r = []

    #for corners in corners_l:
    #    undistorted_corners_l.append(cv2.undistortImagePoints(corners, air_K_l, dist_l).squeeze())

    #for corners in corners_r:
    #    undistorted_corners_r.append(cv2.undistortImagePoints(corners, air_K_r, dist_r).squeeze())


    Rc = cv2.Rodrigues(Rc)[0]
    return _stereo_refract_loss_display(d_l, d_r, Rc, Tc, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints)


def calibrate_stereo_refract_global(air_K_l, air_K_r, air_dist_l, air_dist_r, corners_l, corners_r, objpoints):
    THICKNESS = 13
    ETA_GLASS = 1.492
    ETA_WATER = 1.34

    # TODO try to flip left and right

    #d_l, n_l = calibrate_refract(air_K_l, air_dist_l, corners_l, objpoints, THICKNESS, ETA_GLASS, ETA_WATER)
    #d_r, n_r = calibrate_refract(air_K_r, air_dist_r, corners_r, objpoints, THICKNESS, ETA_GLASS, ETA_WATER)
    d_l, n_l = 50, np.array([0, 0, 1])
    d_r, n_r = 50, np.array([0, 0, 1])
    #raise SystemExit

    undistorted_corners_l = []
    undistorted_corners_r = []

    for img_corners_l, img_corners_r in zip(corners_l, corners_r):
        print(air_dist_l)
        print(air_dist_r)
        undistorted_img_corners_l = cv2.undistortImagePoints(img_corners_l, air_K_l, air_dist_l).squeeze()
        undistorted_img_corners_r = cv2.undistortImagePoints(img_corners_r, air_K_r, air_dist_r).squeeze()

        undistorted_corners_l.append(undistorted_img_corners_l)
        undistorted_corners_r.append(undistorted_img_corners_r)

    # optimize the distances, normals, Rc, Tc, rvecs and tvecs
    x_init = np.zeros(24)
    x_init[0] = 113
    x_init[1] = 113
    x_init[2:5] = np.deg2rad(np.array([-0.5, -12.3, 2.5]))
    x_init[5:8] = np.array([-229, 27, -33])
    x_init[8:11] = np.array([0, 0, 1])
    x_init[11:14] = np.array([0, 0, 1])

    """
d_l 113.81521252440587
d_r 113.67963564549075
Rc [ -0.55059364 -12.3736715    2.49142139]
Tc [-229.36171512   27.72176803  -33.74463082]
n_l [ 0.05249476 -0.01525359  0.96281513]
n_r [ 0.05402338 -0.00702801  0.98542584]
distortion l [-0.0663613   0.24676009  0.0091849   0.00519388 -0.3702578 ]
distortion r [-0.043514    0.14587053 -0.00121753  0.00561212  0.30936471]

fun: 907.1608417873678
    """

    bounds_min = -np.inf * np.ones(14)
    bounds_max = np.inf * np.ones(14)

    bounds_min[:2] = 10  # d_l, d_r
    bounds_max[:2] = 270

    bounds_min[2:5] = -1
    bounds_max[2:5] = 1
    bounds_min[5:8] = -500
    bounds_max[5:8] = 500

    bounds_min[5] = -500 # -500
    bounds_max[5] = -200 # -100

    bounds_min[8:14] = -0.05
    bounds_max[8:14] = 0.05

    bounds_min[10] = 0.95
    bounds_min[13] = 0.95
    bounds_max[10] = 1.0
    bounds_max[13] = 1.0

    #bounds_min[-4:] = -0.5  # distortion
    #bounds_max[-4:] = 0.5

    bounds = list(zip(bounds_min, bounds_max))

    # optimize checkerboard shape
    res = differential_evolution(
        stereo_refract_loss_global,
        bounds=bounds,
        args=(air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, THICKNESS, ETA_GLASS, ETA_WATER, objpoints),
        disp=True,
        workers=31,
        tol=0,
        atol=0,
        init="sobol",
        maxiter=2200,
        mutation=(0.5, 1.5),
        popsize=15,
        #x0=x_init,
        polish=True,
    )

    print(res)
    x = res.x

    d_l = x[0]
    d_r = x[1]
    Rc = x[2:5]
    Tc = x[5:8]
    n_l = x[8:11]
    n_r = x[11:14]

    n_l /= np.linalg.norm(n_l)
    n_r /= np.linalg.norm(n_r)
    #dist_l = x[14:16]
    #dist_r = x[-2:]

    print("d_l", x[0])
    print("d_r", x[1])
    print("Rc", np.rad2deg(Rc))
    print("Tc", Tc)
    print("n_l", x[8:11])
    print("n_r", x[11:14])
    #print("distortion l", dist_l)
    #print("distortion r", dist_r)

    Rc = cv2.Rodrigues(Rc)[0]

    points = []
    for i, (img_corners_l, img_corners_r) in enumerate(zip(corners_l, corners_r)):
        img_points = []
        for j, (corner_l, corner_r) in enumerate(zip(img_corners_l, img_corners_r)):
            p, _ = reconstruct_3d_point_refract(corner_l, corner_r, air_K_l, air_K_r, d_l, d_r, n_l, n_r, Rc, Tc, THICKNESS, ETA_GLASS, ETA_WATER)
            img_points.append(p)
        points.append(np.array(img_points))


    final_loss = stereo_refract_loss_display(x, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, THICKNESS, ETA_GLASS, ETA_WATER, objpoints)
    max_loss = np.max(np.abs(final_loss))
    mean_loss = np.mean(np.abs(final_loss))

    print("max loss", max_loss)
    print("mean loss", mean_loss)

    plt.hist(np.abs(final_loss), bins=50)
    plt.show()

    for i, img_points in enumerate(points):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(img_points))
        #pcd.colors = o3d.utility.Vector3dVector()

        o3d.io.write_point_cloud(f"checkerboard_{i}.ply", pcd)


def calibrate_stereo_refract(air_K_l, air_K_r, air_dist_l, air_dist_r, corners_l, corners_r, objpoints):
    THICKNESS = 13
    ETA_GLASS = 1.492
    ETA_WATER = 1.34

    d_l, n_l = calibrate_refract(air_K_l, air_dist_l, corners_l, objpoints, THICKNESS, ETA_GLASS, ETA_WATER)
    d_r, n_r = calibrate_refract(air_K_r, air_dist_r, corners_r, objpoints, THICKNESS, ETA_GLASS, ETA_WATER)
    #raise SystemExit

    undistorted_corners_l = []
    undistorted_corners_r = []

    for img_corners_l, img_corners_r in zip(corners_l, corners_r):
        undistorted_img_corners_l = cv2.undistortImagePoints(img_corners_l, air_K_l, air_dist_l).squeeze()
        undistorted_img_corners_r = cv2.undistortImagePoints(img_corners_r, air_K_r, air_dist_r).squeeze()

        undistorted_corners_l.append(undistorted_img_corners_l)
        undistorted_corners_r.append(undistorted_img_corners_r)

    # optimize the distances, normals, Rc, Tc, rvecs and tvecs
    x_init = np.zeros(14)
    x_init[0] = d_l
    x_init[1] = d_r
    x_init[2:5] = np.deg2rad(np.array([0, -10, 0]))
    x_init[5:8] = np.array([-200, 0, 0])
    x_init[8:11] = n_l
    x_init[11:14] = n_r

    bounds_min = -np.inf * np.ones_like(x_init)
    bounds_max = np.inf * np.ones_like(x_init)

    bounds_min[:2] = 5
    bounds_max[:2] = 300

    # optimize checkerboard shape
    res = least_squares(
        stereo_refract_loss1,
        x_init,
        max_nfev=2500,
        xtol=0,
        bounds=(bounds_min, bounds_max),
        x_scale="jac",
        args=(air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, THICKNESS, ETA_GLASS, ETA_WATER, objpoints),
        verbose=2
    )
    print(res)
    x = res.x

    d_l = x[0]
    d_r = x[1]
    Rc = x[2:5]
    Tc = x[5:8]
    n_l = x[8:11]
    n_r = x[11:14]

    print("d_l", x[0])
    print("d_r", x[1])
    print("Rc", np.rad2deg(Rc))
    print("Tc", Tc)
    print("n_l", x[8:11])
    print("n_r", x[11:14])

    Rc = cv2.Rodrigues(Rc)[0]

    points = []
    for i, (img_corners_l, img_corners_r) in enumerate(zip(corners_l, corners_r)):
        img_points = []
        for j, (corner_l, corner_r) in enumerate(zip(img_corners_l, img_corners_r)):
            p, _ = reconstruct_3d_point_refract(corner_l, corner_r, air_K_l, air_K_r, d_l, d_r, n_l, n_r, Rc, Tc, THICKNESS, ETA_GLASS, ETA_WATER)
            img_points.append(p)
        points.append(np.array(img_points))


    final_loss = stereo_refract_loss_display(x, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, THICKNESS, ETA_GLASS, ETA_WATER, objpoints)
    max_loss = np.max(np.abs(final_loss))
    mean_loss = np.mean(np.abs(final_loss))

    print("max loss", max_loss)
    print("mean loss", mean_loss)

    #final_loss = final_loss.reshape((-1, 2))
    print(np.histogram(final_loss, bins=50))


    for i, img_points in enumerate(points):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(img_points))
        #pcd.colors = o3d.utility.Vector3dVector()

        o3d.io.write_point_cloud(f"checkerboard_{i}.ply", pcd)



def calibrate_refract(air_K, air_dist, corners, objpoints, thickness, eta_glass, eta_water):
    print("DBG", air_K, air_dist)
    num_pairs = len(corners)

    undistorted_corners = []

    for img_corners in corners:
        undistorted_img_corners = cv2.undistortImagePoints(img_corners, air_K, air_dist).squeeze()
        undistorted_corners.append(undistorted_img_corners)

    x_init = np.zeros(4 + num_pairs * 6)
    x_init[0] = 130
    x_init[3] = 1
    for i in range(num_pairs):
        x_init[9 + 6 * i] = -800
        x_init[4 + 6 * i : 7 + 6 * i] = 0.3

    bounds_min = -np.inf * np.ones_like(x_init)
    bounds_max = np.inf * np.ones_like(x_init)
    bounds_min[0] = 5
    bounds_max[0] = 160

    res = least_squares(
        calibrate_refract_loss,
        x_init,
        max_nfev=250,
        ftol=1e-4,
        x_scale="jac",
        bounds=(bounds_min, bounds_max),
        args=(undistorted_corners, objpoints, air_K, thickness, eta_glass, eta_water),
        verbose=2
    )

    print(res)
    x = res.x
    np.set_printoptions(threshold=np.inf, suppress=True)
    print("distance:", x[0])
    print("normal", x[1:4])
    for i in range(num_pairs):
        rot = x[4 + i * 6 : 7 + i * 6]
        trans = x[7 + i * 6 : 10 + i * 6]
        print(i)
        print(np.rad2deg(rot))
        print(trans)
        print()

    final_loss = np.abs(calibrate_refract_loss(res.x, undistorted_corners, objpoints, air_K, thickness, eta_glass, eta_water))
    max_loss = np.max(np.abs(final_loss))
    mean_loss = np.mean(np.abs(final_loss))

    plt.hist(final_loss, bins=50)
    plt.show()

    print("max loss", max_loss)
    print("mean loss", mean_loss)

    # the distance and the normal
    return x[0], x[1:4]


if __name__ == '__main__':
    main()
