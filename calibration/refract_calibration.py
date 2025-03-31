from Texturation.utils import reconstruct_3d_point_refract, trace_refract_ray
import os
#os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from multiprocessing import Pool
import multiprocessing
import cv2
from scipy.optimize import differential_evolution
import open3d as o3d
from numba import njit
import numba
import matplotlib.pyplot as plt
from datetime import datetime
from time import sleep
from tqdm import tqdm
from dataclasses import dataclass

from functools import partial
import global_refract_loss


np.set_printoptions(suppress=True)

def fit_refract_model(ransac_iteration, ransac_N, ransac_inlier_threshold, minimum_accepted_inliers, bounds, corners_l, corners_r, objpoints, D_L, D_R, air_K_l, air_K_r, thickness, eta_glass, eta_water):
    NUM_IMAGES = len(corners_l)

    all_idxs = np.arange(NUM_IMAGES)
    np.random.shuffle(all_idxs)
    maybe_inliers_idx = all_idxs[:ransac_N]
    test_idx = all_idxs[ransac_N:]

    maybe_corners_l = np.array(corners_l, dtype=np.float32)[maybe_inliers_idx].copy()
    maybe_corners_r = np.array(corners_r, dtype=np.float32)[maybe_inliers_idx].copy()
    maybe_object_points = objpoints.astype(np.float32)[maybe_inliers_idx].copy()

    # try 5 times to get a fit (sometimes the differential_evolution doesn't
    # converge to anything reasonable)
    for _ in range(1):
        res = differential_evolution(
            #stereo_refract_loss_global2,
            #multiprocessing_dummy_air,
            global_refract_loss.refract_loss_global_vectorized,
            bounds=bounds,
            args=(D_L, D_R, air_K_l, air_K_r, np.array(maybe_corners_l, dtype=np.float32), np.array(maybe_corners_r, dtype=np.float32), thickness, eta_glass, eta_water, maybe_object_points.astype(np.float32)),
            #disp=True,
            #workers=31,
            updating="deferred",
            tol=0,
            atol=0,
            #init="sobol",
            maxiter=2500,
            popsize=3,
            polish=False,
            vectorized=True,
        )
        #print(f"({ransac_iteration}) loss {res.fun}")
        #if res.fun < 1000:
        #    break

    x = res.x

    also_inlier_error = 0
    also_inlier_idx = []

    ransac_inlier_threshold = res.fun / ransac_N

    other_errors = []
    for idx in test_idx:
        l = np.array([corners_l[idx]], dtype=np.float32)
        r = np.array([corners_r[idx]], dtype=np.float32)
        o = np.array([objpoints[idx]], dtype=np.float32)

        # TODO replace multiprocessing_dummy_refract by the real function call
        test_loss = global_refract_loss.refract_loss_global(
            x,
            D_L,
            D_R,
            air_K_l,
            air_K_r,
            l,
            r,
            thickness,
            eta_glass,
            eta_water,
            o
        )

        other_errors.append((idx, test_loss))
        #print(f"({ransac_iteration}) test_loss: {test_loss}")
        if test_loss < ransac_inlier_threshold:
            also_inlier_error += test_loss
            also_inlier_idx.append(idx)


    #print(f"({ransac_iteration:03}): {len(also_inlier_idx)} inliers, fit error: {res.fun}, also inliers error: {also_inlier_error}, other_errors: {other_errors}")
    if len(also_inlier_idx) < minimum_accepted_inliers:
        return ransac_iteration, None, None, None

    # TODO refit a model with maybe_inliers_idx and also_inlier_idx
    proposed_error = (res.fun + also_inlier_error) / (ransac_N + len(also_inlier_idx))
    return ransac_iteration, proposed_error, [*maybe_inliers_idx, *also_inlier_idx], x


@njit(error_model="numpy", cache=True)
def _stereo_refract_loss_global2(d_l, d_r, Rc, Tc, focals_l, focals_r, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints):
    losses = np.zeros((len(corners_l) * 2, 63, 3))
    num_images = corners_l.shape[0]
    num_corners_per_image = corners_l.shape[1]
    #losses = np.zeros((len(corners_l) * 2, 80, 3))

    #for i, (img_corners_l, img_corners_r) in enumerate(zip(corners_l, corners_r)):
    for i in range(num_images):
        reconstructed_points1 = np.zeros((objpoints[0].shape[0], 3), dtype=np.float32)
        reconstructed_points2 = np.zeros((objpoints[0].shape[0], 3), dtype=np.float32)
        points_mean1 = np.zeros(3, dtype=np.float32)
        points_mean2 = np.zeros(3, dtype=np.float32)

        # reconstruct the points for the current pair
        for j in range(num_corners_per_image):
            modified_air_K_l = air_K_l.copy()
            modified_air_K_r = air_K_r.copy()

            modified_air_K_l[:2,:2] *= focals_l[i]
            modified_air_K_r[:2,:2] *= focals_r[i]

            ro1, rd1 = trace_refract_ray(corners_l[i, j], modified_air_K_l, d_l, n_l, thickness, eta_glass, eta_water)
            ro2, rd2 = trace_refract_ray(corners_r[i, j], modified_air_K_r, d_r, n_r, thickness, eta_glass, eta_water)

            ro2 = Rc.T @ ro2 - Rc.T @ Tc
            rd2 = Rc.T @ rd2

            b = ro2 - ro1

            #d1_cross_d2 = np.cross(rd1, rd2)
            #d1_cross_d2 = np.array([
            #    rd1[1] * rd2[2] - rd1[2] * rd2[1],
            #    rd1[2] * rd2[0] - rd1[0] * rd2[2],
            #    rd1[0] * rd2[1] - rd1[1] * rd2[0]
            #])
            d1_cross_d2_x = rd1[1] * rd2[2] - rd1[2] * rd2[1]
            d1_cross_d2_y = rd1[2] * rd2[0] - rd1[0] * rd2[2]
            d1_cross_d2_z = rd1[0] * rd2[1] - rd1[1] * rd2[0]
            cross_norm2 = d1_cross_d2_x * d1_cross_d2_x + d1_cross_d2_y * d1_cross_d2_y + d1_cross_d2_z * d1_cross_d2_z
            cross_norm2 = np.maximum(0.0000001, cross_norm2)

            #t1 = np.linalg.det(np.array([
            #    [b[0], rd2[0], d1_cross_d2[0]],
            #    [b[1], rd2[1], d1_cross_d2[1]],
            #    [b[2], rd2[2], d1_cross_d2[2]]
            #])) / np.maximum(0.00001, cross_norm2)

            t1 = (b[0] * rd2[1] * d1_cross_d2_z + b[1] * rd2[2] * d1_cross_d2_x + rd2[0] * d1_cross_d2_y * b[2]
                  - b[2] * rd2[1] * d1_cross_d2_x - b[0] * d1_cross_d2_y * rd2[2] - b[1] * rd2[0] * d1_cross_d2_z) / cross_norm2

            #t2 = np.linalg.det(np.array([
            #    [b[0], rd1[0], d1_cross_d2[0]],
            #    [b[1], rd1[1], d1_cross_d2[1]],
            #    [b[2], rd1[2], d1_cross_d2[2]]
            #])) / np.maximum(0.00001, cross_norm2)
            t2 = (b[0] * rd1[1] * d1_cross_d2_z + b[1] * rd1[2] * d1_cross_d2_x + rd1[0] * d1_cross_d2_y * b[2]
                  - b[2] * rd1[1] * d1_cross_d2_x - b[0] * d1_cross_d2_y * rd1[2] - b[1] * rd1[0] * d1_cross_d2_z) / cross_norm2

            p1 = ro1 + t1 * rd1
            p2 = ro2 + t2 * rd2

            reconstructed_points1[j] = p1
            reconstructed_points2[j] = p2

            points_mean1 += p1
            points_mean2 += p2

        points_mean1 /= reconstructed_points1.shape[0]
        points_mean2 /= reconstructed_points2.shape[0]

        reconstructed_points = (reconstructed_points1 + reconstructed_points2) / 2
        points_mean = (points_mean1 + points_mean2) / 2

        # align the reconstructed points with the objpoints
        reconstructed_points_centered = reconstructed_points - points_mean
        img_objpoints_centered = objpoints[i]

        H = reconstructed_points_centered.T @ img_objpoints_centered
        V, _, W = np.linalg.svd(H)
        V /= abs(np.linalg.det(V))
        W /= abs(np.linalg.det(W))
        d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
        if d:
            V[:, -1] = -V[:, -1]

        rot = np.dot(V, W)

        losses[2 * i + 0] = (rot.T @ reconstructed_points_centered.T).T - img_objpoints_centered
        losses[2 * i + 1] = reconstructed_points2 - reconstructed_points1

    return losses.flatten()


def stereo_refract_loss_global2(x, d_l, d_r, air_K_l, air_K_r, corners_l, corners_r, THICKNESS, ETA_GLASS, ETA_WATER, objpoints):
    #d_l = x[0]
    #d_r = x[1]
    Rc = x[0:3]
    Tc = x[3:6]
    n_l = x[6:9]
    n_r = x[9:12]
    focals_l = x[12::2].copy()
    focals_r = x[13::2].copy()

    n_l /= np.linalg.norm(n_l)
    n_r /= np.linalg.norm(n_r)

    Rc = cv2.Rodrigues(Rc)[0]
    losses = _stereo_refract_loss_global2(d_l, d_r, Rc, Tc, focals_l, focals_r, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, THICKNESS, ETA_GLASS, ETA_WATER, objpoints)
    #return losses @ losses
    #return (np.sqrt((1 + losses * losses)) - 1).sum()
    return np.log(1 + losses * losses).sum()


# TODO replace multiprocessing_dummy_refract by the real function call
def multiprocessing_dummy_refract(*args):
    return global_refract_loss.refract_loss_global(*args)


# TODO replace multiprocessing_dummy_refract by the real function call
def multiprocessing_dummy_air(*args):
    return global_refract_loss.air_loss_global(*args)


def calibrate_stereo_refract_global2(air_K_l, air_K_r, air_dist_l, air_dist_r, corners_l, corners_r, objpoints):
    THICKNESS = 13
    ETA_GLASS = 1.492
    ETA_WATER = 1.34
    NUM_IMAGES = len(corners_l)
    D_L = 40.0
    D_R = D_L

    undistorted_corners_l = []
    undistorted_corners_r = []

    for img_corners_l, img_corners_r in zip(corners_l, corners_r):
        undistorted_img_corners_l = cv2.undistortImagePoints(img_corners_l, air_K_l, air_dist_l).squeeze()
        undistorted_img_corners_r = cv2.undistortImagePoints(img_corners_r, air_K_r, air_dist_r).squeeze()

        undistorted_corners_l.append(undistorted_img_corners_l)
        undistorted_corners_r.append(undistorted_img_corners_r)

    RANSAC_N = 5
    RANSAC_INLIER_THRESHOLD = 1.0
    MINIMUM_ACCEPTED_INLIERS = 0

    # optimize the distances, normals, Rc, Tc, rvecs and tvecs
    #bounds_min = -np.inf * np.ones(12 + 2 * RANSAC_N, dtype=np.float32)
    #bounds_max = np.inf * np.ones(12 + 2 * RANSAC_N, dtype=np.float32)
    bounds_min = -np.inf * np.ones(12)
    bounds_max = np.inf * np.ones(12)

    # Rc
    bounds_min[0:3] = -.35
    bounds_max[0:3] = .35

    # Tc
    bounds_min[3:6] = -500
    bounds_max[3:6] = 500

    # n_x, n_y
    bounds_min[6:12] = -0.05
    bounds_max[6:12] = 0.05

    # n_z
    bounds_min[8] = 0.95
    bounds_min[11] = 0.95
    bounds_max[8] = 1.0
    bounds_max[11] = 1.0

    # focals
    #bounds_min[12:] = 0.98
    #bounds_max[12:] = 1.02

    bounds = list(zip(bounds_min, bounds_max))

    best_error = np.inf
    best_num_inliers = 0
    best_inliers = None
    best_x = None

    fit_model = partial(fit_refract_model,
        ransac_N=RANSAC_N,
        ransac_inlier_threshold=RANSAC_INLIER_THRESHOLD,
        minimum_accepted_inliers=MINIMUM_ACCEPTED_INLIERS,
        bounds=bounds,
        corners_l=corners_l,
        corners_r=corners_r,
        objpoints=objpoints,
        D_L=D_L,
        D_R=D_R,
        air_K_l=air_K_l,
        air_K_r=air_K_r,
        thickness=THICKNESS,
        eta_glass=ETA_GLASS,
        eta_water=ETA_WATER
    )

    RANSAC_N_ITER = 1000

    with Pool(24) as p:
        for result in tqdm(p.imap_unordered(fit_model, range(RANSAC_N_ITER)), total=RANSAC_N_ITER):
            ransac_iteration, proposed_error, inliers, x = result

            if proposed_error is None:
                continue

            if proposed_error < best_error:
                best_error = proposed_error
                best_inliers = inliers
                best_num_inliers = len(best_inliers)
                best_x = x
                print("best_error:", best_error, "best_num_inliers:", best_num_inliers)
                print("best_inliers:", best_inliers)

                Rc = x[0:3]
                Tc = x[3:6]
                n_l = x[6:9]
                n_r = x[9:12]
                focals_l = x[12::2]
                focals_r = x[13::2]

                print("Rc", np.rad2deg(Rc))
                print("Tc", Tc)
                print("focals_l", focals_l)
                print("focals_r", focals_r)

                n_l /= np.linalg.norm(n_l)
                n_r /= np.linalg.norm(n_r)

                print("n_l (normalized)", n_l)
                print("n_r (normalized)", n_r)

    raise SystemExit

    # optimize checkerboard shape
    res = differential_evolution(
        #stereo_refract_loss_global2,
        #multiprocessing_dummy_refract,
        global_refract_loss.refract_loss_global_vectorized,
        bounds=bounds,
        args=(D_L, D_R, air_K_l, air_K_r, np.array(undistorted_corners_l, dtype=np.float32), np.array(undistorted_corners_r, dtype=np.float32), THICKNESS, ETA_GLASS, ETA_WATER, objpoints.astype(np.float32)),
        disp=True,
        updating="deferred",
        vectorized=True,
        #workers=31,
        tol=0,
        atol=0,
        #init="sobol",
        maxiter=5000,
        popsize=3,
        polish=False,
    )

    print(res)
    x = res.x

    # unpack the solution and print it
    Rc = x[0:3]
    Tc = x[3:6]
    n_l = x[6:9]
    n_r = x[9:12]
    focals_l = x[12::2]
    focals_r = x[13::2]

    print("Rc", np.rad2deg(Rc))
    print("Tc", Tc)
    print("focals_l", focals_l)
    print("focals_r", focals_r)

    n_l /= np.linalg.norm(n_l)
    n_r /= np.linalg.norm(n_r)

    print("n_l (normalized)", n_l)
    print("n_r (normalized)", n_r)

    Rc = cv2.Rodrigues(Rc)[0]

    # reconstruct the checkerboards and save them to `.ply` files
    #points = []
    #reconstruction_losses = []
    #for i, (img_corners_l, img_corners_r) in enumerate(zip(undistorted_corners_l, undistorted_corners_r)):
    #    img_points = []
    #    for j, (corner_l, corner_r) in enumerate(zip(img_corners_l, img_corners_r)):
    #        modified_air_K_l = air_K_l.copy()
    #        modified_air_K_r = air_K_r.copy()

    #        modified_air_K_l[:2,:2] *= focals_l[i]
    #        modified_air_K_r[:2,:2] *= focals_r[i]

    #        p, reconstruction_loss = reconstruct_3d_point_refract(corner_l, corner_r, modified_air_K_l, modified_air_K_r, D_L, D_R, n_l, n_r, Rc, Tc, THICKNESS, ETA_GLASS, ETA_WATER)
    #        reconstruction_losses.append(reconstruction_loss)
    #        # TODO print the reconstruction losses
    #        img_points.append(p)

    #    points.append(np.array(img_points))


    final_loss = _stereo_refract_loss_global2(D_L, D_R, Rc, Tc, focals_l, focals_r, air_K_l, air_K_r, n_l, n_r, np.array(undistorted_corners_l, dtype=np.float32), np.array(undistorted_corners_r, dtype=np.float32), THICKNESS, ETA_GLASS, ETA_WATER, objpoints.astype(np.float32))
    final_loss = final_loss.reshape((len(corners_l) * 2, 63, 3))

    reconstruction_losses = final_loss[1::2, ...]
    alignment_losses = final_loss[0::2, ...]

    print("max recon loss", np.max(np.abs(reconstruction_losses)))
    print("mean recon loss", np.mean(np.abs(reconstruction_losses)))
    print("median recon loss", np.median(np.abs(reconstruction_losses)))
    print("max alignment loss", np.max(np.abs(alignment_losses)))
    print("mean alignment loss", np.mean(np.abs(alignment_losses)))
    print("median alignment loss", np.median(np.abs(alignment_losses)))

    #plt.hist(np.abs(final_loss), bins=150)
    #plt.savefig("loss_histogram.svg")
    ##plt.show()

    #fig, ax = plt.subplots(1, 2)
    #ax[0].scatter(
    #    np.array(corners_l).reshape((-1, 2))[:, 0],
    #    np.array(corners_l).reshape((-1, 2))[:, 1],
    #    c = reconstruction_losses.flatten(),
    #    alpha=0.3,
    #    s=2,
    #)
    #ax[0].set_xlim([0, 6000])
    #ax[0].set_ylim([4000, 0])
    #ax[0].set_aspect("equal")

    #ax[1].scatter(
    #    np.array(corners_r).reshape((-1, 2))[:, 0],
    #    np.array(corners_r).reshape((-1, 2))[:, 1],
    #    c = reconstruction_losses.flatten(),
    #    alpha=0.3,
    #    s=2,
    #)
    #ax[1].set_xlim([0, 6000])
    #ax[1].set_ylim([4000, 0])
    #ax[1].set_aspect("equal")

    #fig.savefig("loss_scatter.svg")

    #for i, img_points in enumerate(points):
    #    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(img_points))

    #    o3d.io.write_point_cloud(f"checkerboard_{i}.ply", pcd)


# TODO
#@dataclass
#class RansacConfig:
#


def fit_air_model(ransac_iteration, ransac_N, ransac_inlier_threshold, minimum_accepted_inliers, bounds, corners_l, corners_r, objpoints):
    NUM_IMAGES = len(corners_l)

    #print(ransac_iteration)
    all_idxs = np.arange(NUM_IMAGES)
    np.random.shuffle(all_idxs)
    maybe_inliers_idx = all_idxs[:ransac_N]
    test_idx = all_idxs[ransac_N:]

    maybe_corners_l = np.array(corners_l, dtype=np.float32)[maybe_inliers_idx].copy()
    maybe_corners_r = np.array(corners_r, dtype=np.float32)[maybe_inliers_idx].copy()
    maybe_object_points = objpoints.astype(np.float32)[maybe_inliers_idx].copy()

    # try 5 times to get a fit (sometimes the differential_evolution doesn't
    # converge to anything reasonable)
    for _ in range(1):
        res = differential_evolution(
            #stereo_refract_loss_global2,
            #multiprocessing_dummy_air,
            global_refract_loss.air_loss_global_vectorized,
            bounds=bounds,
            args=(
                maybe_corners_l,
                maybe_corners_r,
                maybe_object_points
            ),
            #disp=True,
            #workers=31,
            updating="deferred",
            tol=0,
            atol=0,
            #init="sobol",
            maxiter=2500,
            popsize=3,
            polish=False,
            vectorized=True,
        )
        #print("loss: ", res.fun)
        if res.fun < 1000:
            break
        #print(f"retrying {ransac_iteration}...")

    x = res.x

    also_inlier_error = 0
    also_inlier_idx = []

    for idx in test_idx:
        l = np.array([corners_l[idx]], dtype=np.float32)
        r = np.array([corners_r[idx]], dtype=np.float32)
        o = np.array([objpoints[idx]], dtype=np.float32)

        # TODO replace multiprocessing_dummy_refract by the real function call
        test_loss = multiprocessing_dummy_air(x, l, r, o)
        if test_loss < ransac_inlier_threshold:
            also_inlier_error += test_loss
            also_inlier_idx.append(idx)

    # TODO refit a model with maybe_inliers_idx and also_inlier_idx

    #print(f"num inliers {ransac_iteration}:", len(also_inlier_idx))
    if len(also_inlier_idx) < minimum_accepted_inliers:
        return ransac_iteration, None, None, None

    proposed_error = (res.fun + also_inlier_error) / (ransac_N + len(also_inlier_idx))
    return ransac_iteration, proposed_error, [*maybe_inliers_idx, *also_inlier_idx], x


def calibrate_stereo_air_global(corners_l, corners_r, objpoints):
    NUM_IMAGES = len(corners_l)

    # optimize the distances, normals, Rc, Tc, rvecs and tvecs
    bounds_min = -np.inf * np.ones(12)
    bounds_max = np.inf * np.ones(12)

    # Rc (rodrigues)
    bounds_min[0:3] = -1
    bounds_max[0:3] = 1

    # Tc
    bounds_min[3:6] = -500
    bounds_max[3:6] = 500

    # f, cx, cy for both cameras
    bounds_min[6] = 20
    bounds_max[6] = 50
    bounds_min[7] = 2950 / 2
    bounds_max[7] = 3050 / 2
    bounds_min[8] = 1950 / 2
    bounds_max[8] = 2050 / 2
    #bounds_min[7] = 0
    #bounds_max[7] = 3000
    #bounds_min[8] = 0
    #bounds_max[8] = 2000

    bounds_min[9] = 20
    bounds_max[9] = 50
    bounds_min[10] = 2950 / 2
    bounds_max[10] = 3050 / 2
    bounds_min[11] = 1950 / 2
    bounds_max[11] = 2050 / 2
    #bounds_min[10] = 0
    #bounds_max[10] = 3000
    #bounds_min[11] = 0
    #bounds_max[11] = 2000

    bounds = list(zip(bounds_min, bounds_max))

    # optimize checkerboard shape
    RANSAC_N = 5
    RANSAC_INLIER_THRESHOLD = 0.5
    MINIMUM_ACCEPTED_INLIERS = 2

    best_error = np.inf
    best_num_inliers = 0
    best_inliers = None

    partial_fit_model = partial(
        fit_air_model,
        ransac_N=RANSAC_N,
        ransac_inlier_threshold=RANSAC_INLIER_THRESHOLD,
        minimum_accepted_inliers=MINIMUM_ACCEPTED_INLIERS,
        bounds=bounds,
        corners_l=corners_l,
        corners_r=corners_r,
        objpoints=objpoints
    )

    RANSAC_N_ITER = 1000

    with Pool(24) as p:
        for result in tqdm(p.imap_unordered(partial_fit_model, range(RANSAC_N_ITER)), total=RANSAC_N_ITER):
            ransac_iteration, proposed_error, inliers, x = result

            if proposed_error is None:
                continue

            if proposed_error < best_error:
                best_error = proposed_error
                best_inliers = inliers
                best_num_inliers = len(best_inliers)
                print("best_error:", best_error, "best_num_inliers:", best_num_inliers)
                print("best_inliers:", best_inliers)

                Rc = x[0:3]
                Tc = x[3:6]
                k_l = x[6:9]
                k_r = x[9:12]

                print("Rc", np.rad2deg(Rc))
                print("Tc", Tc)
                print("k_l", k_l)
                print("k_r", k_r)

    # TODO reconstruct the checkerboards using the best fitted model and save them
