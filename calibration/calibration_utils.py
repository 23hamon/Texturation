from numba import njit
import numpy as np
import cv2


def find_corners(mask, num_corners):
    """Flags to be adjusted"""
    # pyrdown twice, detect on that, multiply the detected coords by 4 and then refine
    down_mask = cv2.pyrDown(mask)
    #down_mask = cv2.pyrDown(down_mask)

    # TODO test checkboard detection on all three channels and averaging of points

    flags = cv2.CALIB_CB_NORMALIZE_IMAGE

    _, corners = cv2.findChessboardCornersSB(down_mask, num_corners)#, flags=flags)
    return corners
    ret, corners = cv2.findChessboardCorners(down_mask, num_corners, None)
    #ret, corners = cv2.findChessboardCorners(mask, num_corners, None)

    if ret:
        #corners *= 4
        return cv2.cornerSubPix(down_mask, corners, (51,51), (-1,-1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001))

    return None


def find_circles(mask, num_circles):
    ret, corners = cv2.findCirclesGrid(mask, num_circles, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    if ret:
        return corners

    return None
