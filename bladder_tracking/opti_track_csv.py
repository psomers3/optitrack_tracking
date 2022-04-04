"""
Set of functions for extracting rigid body data from optitrack csv files
"""

from collections import defaultdict
from bladder_tracking.transformations import *
import scipy.spatial.transform as transform

recursive_dd = lambda: defaultdict(recursive_dd)


def get_prepared_df(data: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, dict]:
    """

    :param data: either an already loaded dataframe or path to the csv file
    :return: a tuple containing the loaded dataframe and a dictionary of the rigid body data
    """
    if isinstance(data, str):
        data = pd.read_csv(data, quotechar='"', sep=',', header=list(range(1, 6)))
    rigid_bodies = recursive_dd()

    for i in range(len(data.columns)):
        h = data.columns[i]  # type: Tuple[str, str, str, str, str]
        if 'Rigid Body' in h[0]:
            if 'Marker' not in h[0]:
                rigid_bodies[h[1]][h[-2]][h[-1]] = i

    for i in range(len(data.columns)):
        h = data.columns[i]  # type: Tuple[str, str, str, str, str]
        if h[0] == 'Marker':
            split_name = h[1].split('_')
            marker_id = int(split_name[-1])
            marker_body_name = ''.join(split_name[:-1])
            if marker_body_name in rigid_bodies:
                rigid_bodies[marker_body_name]['Markers'][marker_id][h[-1]] = i

    return data, rigid_bodies


def get_marker_positions(data: Union[str, pd.DataFrame], rigid_body: str) -> np.ndarray:
    """
    Returns the recorded marker positions
    :param data: either an already loaded dataframe or path to the csv file
    :param rigid_body: the name of the rigid body
    :return: numpy array of positions of shape [num_samples, num_markers, 3]
    """

    data, parsing_dict = get_prepared_df(data)
    markers = []
    positions = ['X', 'Y', 'Z']
    for marker in parsing_dict[rigid_body]['Markers'].values():
        markers.append(np.asarray([data[data.columns[marker[x]]].to_numpy() for x in positions]))

    markers_as_numpy = np.asarray(markers).T
    return markers_as_numpy.swapaxes(1, 2)


def get_rigid_body_data(data: Union[str, pd.DataFrame], rigid_body: str) -> Tuple[np.array, np.array]:
    """
    Get the positions and orientations of a rigid body
    :param data: either an already loaded dataframe or path to the csv file
    :param rigid_body: the name of the rigid body
    :return: a tuple of positions and orientations
    """

    data, parsing_dict = get_prepared_df(data)
    positions = ['X', 'Y', 'Z']
    orientations = ['X', 'Y', 'Z', 'W']
    position_data = np.asarray([data[data.columns[parsing_dict[rigid_body]['Position'][x]]] for x in positions]).T
    orientation_data = np.asarray([data[data.columns[parsing_dict[rigid_body]['Rotation'][x]]] for x in orientations]).T

    return position_data, orientation_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    tracker_ball1 = [-31.94985, 39.75, -10.93251]
    tracker_ball2 = [-81.40743, -35.25, 19.95353]
    tracker_ball3 = [23.95005, 25.75, 39.07951]
    tracker_ball4 = [89.40723, -30.25, -48.10053]

    tracker_cad_balls = np.array([tracker_ball1, tracker_ball2, tracker_ball3, tracker_ball4])

    part = 'bladder'
    cad_balls = tracker_cad_balls*1e-3

    csv_file = r'C:\Users\Somers\Desktop\optitrack\Take 2022-03-17 04.31.54 PM.csv'
    data, parsing = get_prepared_df(csv_file)
    positions = get_marker_positions(data, part)
    recorded_positions, recorded_orientations = get_rigid_body_data(data, part)
    rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(cad_positions=cad_balls,
                                                                     recorded_body_positions=recorded_positions,
                                                                     recorded_body_orientations=recorded_orientations,
                                                                     recorded_marker_positions=positions,
                                                                     samples_to_use=500)
    print(rotation_to_opti_local.as_euler('xyz', degrees=True))
    index = 1000
    markers = positions[index]
    orientation = recorded_orientations[index]
    position = recorded_positions[index]
    shifted = markers - position
    aligned = transform.Rotation.from_quat(orientation).inv().apply(shifted)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    rotated_cad = rotation_to_opti_local.apply(cad_balls)
    ax.scatter(cad_balls[:, 0], cad_balls[:, 1], cad_balls[:, 2], color='y')
    ax.scatter(rotated_cad[:, 0], rotated_cad[:, 1], rotated_cad[:, 2], color='r')
    ax.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2], color='b')
    ax.set_xlim(-.11, .11)
    ax.set_ylim(-.11, .11)
    ax.set_zlim(-.11, .11)
    np.savez('test_points', cad_points=cad_balls, opti_points=shifted)
    plt.show(block=True)