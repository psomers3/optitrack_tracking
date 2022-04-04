import numpy as np
import scipy.spatial.transform as transform
import pandas as pd
from typing import *
from itertools import permutations

XYZW2WXYZ = [-1, -4, -3, -2]
WXYZ2XYZW = [-3, -2, -1, -4]


def get_optitrack_rotation_from_markers(cad_positions: Union[List[List[float]], np.ndarray],
                                        recorded_marker_positions: np.ndarray,
                                        recorded_body_positions: np.ndarray,
                                        recorded_body_orientations: np.ndarray,
                                        samples_to_use: int = 200,
                                        scalar_first: bool = False) -> Tuple[transform.Rotation, List[int]]:
    """

    :param cad_positions: 3d cad marker positions relative to markers' CG
    :param recorded_marker_positions: optitrack recorded positions
    :param recorded_body_positions: optitrack rigid body recorded positions
    :param recorded_body_orientations: optitrack rigid body orientations as quanternions
    :param samples_to_use: how many samples to use for the optimization
    :param scalar_first: whether the quaternions are given as scalar first or not
    :return: A tuple with the scipy Rotation object to go from CAD to optitrack and the permutation of cad points to
             match optitrack's marker order.
    """
    recorded_marker_positions = recorded_marker_positions[:samples_to_use]
    valid_entries = ~np.isnan(recorded_marker_positions).any(axis=2)
    recorded_marker_positions = recorded_marker_positions[valid_entries.all(axis=-1)]
    recorded_body_positions = recorded_body_positions[:samples_to_use][valid_entries.all(axis=-1)]
    recorded_body_orientations = recorded_body_orientations[:samples_to_use][valid_entries.all(axis=-1)]
    if scalar_first:
        recorded_body_orientations = recorded_body_orientations[:, WXYZ2XYZW]

    aligned_balls = np.zeros(recorded_marker_positions.shape)
    for i in range(len(recorded_marker_positions)):
        opti_rotation = transform.Rotation.from_quat(recorded_body_orientations[i])  # type: transform.Rotation
        b = recorded_marker_positions[i]
        shifted = b - recorded_body_positions[i]
        aligned_balls[i] = opti_rotation.inv().apply(shifted)

    # figure out permutation to use by sampling 5 measurements
    random_sample_indexes = np.random.randint(0, len(recorded_marker_positions), 20)
    perms = list(set(permutations(np.arange(start=0, stop=len(recorded_marker_positions[0])))))
    found_permutations = []
    for sample_index in random_sample_indexes:
        errors = []
        rotations = []  # type: List[transform.Rotation]
        measurement_sample = aligned_balls[sample_index]
        for p in perms:
            result = transform.Rotation.align_vectors(cad_positions[p, :], measurement_sample)
            rotations.append(result[0])
            errors.append(result[1])
        min_index = np.argmin(np.array(errors))
        found_permutations.append(perms[min_index])
    found_permutations = {found_permutations.count(x): x for x in found_permutations}
    optimal_permutation = found_permutations[max(found_permutations.keys())]

    # reorder cad positions and then repeat to have matching vector sizes
    cad_ball_positions = cad_positions[optimal_permutation, :]
    cad_ball_positions = np.expand_dims(cad_ball_positions, axis=0)
    cad_ball_positions = np.repeat(cad_ball_positions, axis=0, repeats=len(aligned_balls))
    cad_ball_positions = np.reshape(cad_ball_positions, (cad_ball_positions.shape[0] * cad_ball_positions.shape[1], cad_ball_positions.shape[-1]))
    opti_balls = np.reshape(aligned_balls, (aligned_balls.shape[0] * aligned_balls.shape[1], aligned_balls.shape[-1]))
    results = transform.Rotation.align_vectors(cad_ball_positions, opti_balls)
    return results[0].inv(), optimal_permutation


def get_optitrack_rotation_from_dataframe(data_frame: pd.DataFrame,
                                          position_name: str,
                                          orientation_name: str,
                                          ball_names: List[str],
                                          cad_ball_positions: Union[List[List[float]], np.ndarray],
                                          samples_to_use: int = 200
                                          ) -> Tuple[transform.Rotation, Iterable[float]]:
    """
    Returns a scipy rotation object from the markers' geometric center in CAD coordinates to the optitrack
    coordinates.

    :param data_frame: pandas dataframe containing the recorded optitrack data
    :param position_name: data_frame column title with object's opti-track position
    :param orientation_name: data_frame column title with object's opti-track orientation
    :param ball_names: list of the column names corresponding to the opti-track markers belonging to the object
    :param cad_ball_positions: list of positions of the balls in CAD coordinate system (origin at the geometric
                               center of balls)
    :param samples_to_use: how many of the provided data samples to use.
    :return: a scipy rotation object rotating the CAD coordinate to the local optitrack coordinate and a
             permutation of the order the cad_balls that matches the order of the opti-track balls.
    """

    positions = np.array(list(data_frame[position_name].to_numpy()))
    orientations = np.array(list(data_frame[orientation_name].to_numpy()))
    balls = np.array(list(data_frame[ball_names].to_numpy().tolist()))

    return get_optitrack_rotation_from_markers(cad_positions=cad_ball_positions,
                                               recorded_body_positions=positions,
                                               recorded_body_orientations=orientations,
                                               recorded_marker_positions=balls,
                                               samples_to_use=samples_to_use)

