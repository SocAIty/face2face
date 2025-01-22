# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    # Calculate transformation matrix using cv2.estimateAffinePartial2D
    M, _ = cv2.estimateAffinePartial2D(lmk, dst)

    return M

def transform(data, center, output_size, scale, rotation):
    # Create rotation matrix
    rot_mat = cv2.getRotationMatrix2D(
        (center[0] * scale, center[1] * scale),
        rotation,
        scale
    )

    # Adjust translation to center the output
    rot_mat[0, 2] += (output_size / 2) - (center[0] * scale)
    rot_mat[1, 2] += (output_size / 2) - (center[1] * scale)

    # Apply transformation
    cropped = cv2.warpAffine(
        data,
        rot_mat,
        (output_size, output_size),
        borderValue=0.0
    )

    return cropped, rot_mat

#def transform(data, center, output_size, scale, rotation):
#    scale_ratio = scale
#    rot = float(rotation) * np.pi / 180.0
#    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
#    t1 = trans.SimilarityTransform(scale=scale_ratio)
#    cx = center[0] * scale_ratio
#    cy = center[1] * scale_ratio
#    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
#    t3 = trans.SimilarityTransform(rotation=rot)
#    t4 = trans.SimilarityTransform(translation=(output_size / 2,
#                                                output_size / 2))
#    t = t1 + t2 + t3 + t4
#    M = t.params[0:2]
#    cropped = cv2.warpAffine(data,M, (output_size, output_size), borderValue=0.0)
#    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


def estimate_affine_matrix_3d23d(X, Y):
    """
    Estimate the affine transformation matrix using a least-squares solution.

    Args:
        X: ndarray of shape (n, 3). Fixed 3D points.
        Y: ndarray of shape (n, 3). Corresponding moving 3D points (Y = PX).

    Returns:
        P_Affine: ndarray of shape (3, 4). Affine camera matrix, where
                  the third row is [0, 0, 0, 1].
    """
    if X.shape[1] != 3 or Y.shape[1] != 3:
        raise ValueError("Both X and Y must have exactly 3 columns.")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")

    # Add the homogeneous coordinate to X
    X_homo = np.hstack((X, np.ones((X.shape[0], 1))))  # Shape: (n, 4)

    # Solve the least-squares problem
    P = np.linalg.lstsq(X_homo, Y, rcond=None)[0].T  # Affine matrix, Shape: (3, 4)

    return P


def P2sRt(P):
    ''' decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return rx, ry, rz

