"""
99% of the code in this file was taken from
https://blender.stackexchange.com/questions/40650/blender-camera-from-3x4-matrix
"""
import bpy
from mathutils import Matrix
import numpy as np
from typing import *


def KRT_from_P(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     Reference implementations:
      - Oxford's visual geometry group matlab toolbox
      - Scilab Image Processing toolbox
    :param P: 3x4 numpy matrix
    :return:  K, R, T such that P = K*[R | T], det(R) positive and K has positive diagonal
    """

    N = 3
    H = P[:, 0:N]  # if not numpy,  H = P.to_3x3()

    [K, R] = rf_rq(H)

    K /= K[-1, -1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = np.diag(np.sign(np.diag(K)))

    K = K * sg
    R = sg * R
    # det(R) negative, just invert; the proj equation remains same:
    if np.linalg.det(R) < 0:
        R = -R
    # C = -H\P[:,-1]
    C = np.linalg.lstsq(-H, P[:, -1], rcond=1)[0]
    T = -R * C
    return K, R, T


def rf_rq(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    RQ decomposition of a numpy matrix, using only libs that already come with
    blender by default

    Author: Ricardo Fabbri
    Reference implementations:
      Oxford's visual geometry group matlab toolbox
      Scilab Image Processing toolbox
    :param P: 3x4 numpy matrix P
    :return: numpy matrices r,q
    """

    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = np.linalg.qr(P[::-1, ::-1], 'complete')
    q = q.T
    q = q[::-1, ::-1]
    r = r.T
    r = r[::-1, ::-1]

    if np.linalg.det(q) < 0:
        r[:, 0] *= -1
        q[0, :] *= -1
    return r, q


def get_blender_camera_from_3x4_P(P, scale) -> Tuple[bpy.types.Object, bpy.types.Camera]:
    """
    Creates a blender camera consistent with a given 3x4 computer vision P matrix.
    :param P: numpy 3x4
    :param scale: resolution scale percentage as in GUI, known a priori
    :return:
    """
    # get krt
    K, R_world2cv, T_world2cv = KRT_from_P(np.matrix(P))

    scene = bpy.context.scene
    sensor_width_in_mm = K[1, 1] * K[0, 2] / (K[0, 0] * K[1, 2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0, 2] * 2  # principal point assumed at the center
    resolution_y_in_px = K[1, 2] * 2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    # s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio
    f_in_mm = K[0, 0] / s_u
    # recover original resolution
    scene.render.resolution_x = resolution_x_in_px / scale
    scene.render.resolution_y = resolution_y_in_px / scale
    scene.render.resolution_percentage = scale * 100

    # Use this if the projection matrix follows the convention listed in my answer to
    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Use this if the projection matrix follows the convention from e.g. the matlab calibration toolbox:
    # R_bcam2cv = Matrix(
    #     ((-1, 0,  0),
    #      (0, 1, 0),
    #      (0, 0, 1)))

    R_cv2world = R_world2cv.T
    rotation = Matrix(R_cv2world.tolist()) * R_bcam2cv
    location = -R_cv2world * T_world2cv

    # create a new camera
    camera_data = bpy.data.cameras.new(name='CamFrom3x4P')
    camera_object = bpy.data.objects.new('CamFrom3x4PObj', camera_data)
    camera_object.location = location

    # Lens
    camera_data.type = 'PERSP'
    camera_data.lens = f_in_mm
    camera_data.lens_unit = 'MILLIMETERS'
    camera_data.sensor_width = sensor_width_in_mm
    camera_object.matrix_world = Matrix.Translation(location) * rotation.to_4x4()

    #     cam.shift_x = -0.05
    #     cam.shift_y = 0.1
    #     cam.clip_start = 10.0
    #     cam.clip_end = 250.0
    #     empty = bpy.data.objects.new('DofEmpty', None)
    #     empty.location = origin+Vector((0,10,0))
    #     cam.dof_object = empty
    return camera_object, camera_data


def test2():
    P = Matrix([
        [1.0347, 0.    , 0.8982, 0.],
        [0.    , 1.0313, 0.5411, 0.],
        [0.    , 0.    , 0.001 ,  0.]
    ])
    # This test P was constructed as k*[r | t] where
    #     k = [2 0 10; 0 3 14; 0 0 1]
    #     r = [1 0 0; 0 -1 0; 0 0 -1]
    #     t = [231 223 -18]
    # k, r, t = KRT_from_P(numpy.matrix(P))
    get_blender_camera_from_3x4_P(P, 1)