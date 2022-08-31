import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from optitrack_tools import get_optitrack_rotation_from_markers
from optitrack_tools.endoscope_definitions import Endoscope
from typing import *


def invert_affine_transform(matrix: np.ndarray) -> np.ndarray:
    """
    get the inversion of a 4x4 transformation matrix

    :param matrix: a 4x4 affine transformation
    :return:
    """
    inverted_r = np.linalg.inv(matrix[:3, :3])
    m = np.eye(4)
    m[:3, :3] = inverted_r
    m[:3, 3] = -(inverted_r @ matrix[:3, 3])
    return m


class EndoscopeTrajectory:
    """
    A class to recreate the trajectory of the recorded camera.
    """
    cam_name = 'cam'
    bladder_name = 'bladder'

    def __init__(self,
                 data: Union[str, dict],
                 endoscope: Endoscope,
                 invert_cam_rotation: bool = False,
                 relative_trajectory: bool = False):
        """
        :param data: either the path to or the already loaded numpy data file that was recorded.
        :param endoscope: which endoscope was used for recording from one of ENDOSCOPES <- found in bladder_tracking
        :param invert_cam_rotation: Whether to reverse the direction the camera is rotated on the endoscope. This
                                    may need to be done because the direction is not deterministic. So... try and see.
        :param relative_trajectory: Whether to return the trajectory relative to the first recorded position
        """
        self.endoscope = endoscope
        self.cam_direction = -1 if invert_cam_rotation else 1
        self.rotation_cam_2_endo: Rotation = Rotation.from_euler(angles=[0, 90-self.endoscope.angle, 0],
                                                                 seq='XYZ',
                                                                 degrees=True)
        tip_2_light = self.endoscope.shaft
        light_2_balls = self.endoscope.light_2_balls
        self.vector_cam_to_balls = np.array((tip_2_light + light_2_balls)) * 1e-3
        endo_cad_balls = self.endoscope.cad_balls * 1e-3

        if isinstance(data, str):
            data = np.load(data)
        self.rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(endo_cad_balls,
                                                                              recorded_marker_positions=data[f'{self.endoscope.rigid_body_name}-markers'],
                                                                              recorded_body_positions=data[f'{self.endoscope.rigid_body_name}'][:, :3],
                                                                              recorded_body_orientations=data[f'{self.endoscope.rigid_body_name}'][:, 3:],
                                                                              scalar_first=False)

        self.optitrack_times = np.squeeze(data["optitrack_received_timestamps"] - data["optitrack_received_timestamps"][0])
        self.recorded_orientations = data[f'{self.endoscope.rigid_body_name}'][:, 3:]
        self.recorded_camera_orientations = data[f'{self.cam_name}'][:, 3:]
        self.recorded_bladder_orientations = data[f'{self.bladder_name}'][:, 3:]
        self.recorded_positions = data[f'{self.endoscope.rigid_body_name}'][:, :3]
        self.recorded_bladder_positions = data[f'{self.bladder_name}'][:, :3]

        # slerp = spherical linear interpolation (interpolation between orientations)
        self.slerp = Slerp(self.optitrack_times, Rotation.from_quat(self.recorded_orientations))
        self.cam_slerp = Slerp(self.optitrack_times, Rotation.from_quat(self.recorded_camera_orientations))
        self.bladder_slerp = Slerp(self.optitrack_times, Rotation.from_quat(self.recorded_bladder_orientations))

        self.initial_cam_translate = self.rotation_to_opti_local.apply(-np.array(self.vector_cam_to_balls))

        # average the first 10 measurements to get the "zero" position of the camera to endoscope angle
        self.zero_angle: float = 0
        self.zero_angle = np.average(np.array([self.get_camera_angle(index=x, zeroed=False) for x in range(10)]))
        self.__relative_trajectory = False
        self.__zero_absolute_transform = self.get_absolute_orientation(t=0)
        self.__zero_relative_transform = self.get_relative_orientation(t=0)
        self.__relative_trajectory = relative_trajectory

    @property
    def times(self) -> np.ndarray:
        """
        :return: The timestamps that correspond to opti-track measurements.
        """
        return self.optitrack_times

    def get_camera_angle(self, index: int = None, t: float = None, degrees=True, zeroed: bool = True) -> float:
        """
        Get the angle between the endoscope shaft and the attached camera.

        :param index: index of the recorded data. should be left None if t is specified.
        :param t: exact time to get angle for. This will be linearly interpolated from the data. Leave None if index is
                  specified.
        :param degrees: whether to return degrees or radians.
        :param zeroed: whether to give the value relative to the internally stored zero point.
        :return: the angle between the camera and endoscope.
        """
        assert (index is not None and t is None) or (t is not None and index is None)
        if index is not None:
            r_cam_tracker = Rotation.from_quat(self.recorded_camera_orientations[index]).inv()
            r_endo_tracker = Rotation.from_quat(self.recorded_orientations[index])
        else:
            r_cam_tracker = self.cam_slerp(t)  # from opti-track to cam sys
            r_endo_tracker = self.slerp(t).inv()  # from endo sys to opti sys
        resultant_r = r_cam_tracker * r_endo_tracker  # type: Rotation
        angle = np.arctan(np.linalg.norm(resultant_r.as_mrp()))*4
        offset = self.zero_angle if zeroed else 0
        direction = self.cam_direction if zeroed else 1
        if degrees:
            return (np.degrees(angle) - offset) * direction
        else:
            (angle - offset) * direction

    def get_absolute_orientation(self, index: int = None, t: float = None) -> np.ndarray:
        """
            Get the affine transformation that puts the endoscope camera in **absolute** (optitrack) coordinates. The
            camera is assumed to start at the origin pointing in the -Z direction.

            :param index: the desired  index position from the recorded data
            :param t: the desired time position using interpolation (slerp).
            :return: 4x4 affine transformation matrix
        """
        assert (index is not None and t is None) or (t is not None and index is None)
        cam_angle: Rotation = Rotation.from_euler(angles=[0, 0, 90 + self.get_camera_angle(index=index, t=t)],
                                                  seq='XYZ',
                                                  degrees=True)
        if index is not None:
            r: Rotation = Rotation.from_quat(self.recorded_orientations[index])
            p = self.recorded_positions[index]
        else:
            r: Rotation = self.slerp(t)
            x = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 0])
            y = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 1])
            z = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 2])
            p = np.array([x, y, z])

        # cam toolpoint position & orientation in opti-track. This matches blender
        cam_orientation = r * self.rotation_to_opti_local * self.rotation_cam_2_endo * cam_angle
        cam_position = r.apply(self.initial_cam_translate) + p
        full_transform = np.eye(4)
        full_transform[:3, :3] = cam_orientation.as_matrix()
        full_transform[:3, 3] = cam_position
        if self.__relative_trajectory:
            full_transform[:3, 3] = full_transform[:3, 3] - self.__zero_absolute_transform[:3, 3]
            full_transform[:3, :3] = np.linalg.inv(self.__zero_absolute_transform[:3, :3]) @ full_transform[:3, :3]
            full_transform[:3, 3] = np.linalg.inv(self.__zero_absolute_transform[:3, :3]) @ full_transform[:3, 3]

        return full_transform

    def get_relative_orientation(self, index: int = None, t: float = None) -> np.ndarray:
        """
            Get the affine transformation that puts the endoscope camera in **bladder** coordinates. This is useful because
            it essentially "cancels out" the movement of the bladder during measurements.

            :param index: the desired  index position from the recorded data.
            :param t: the desired time position using interpolation (slerp).
            :return: 4x4 affine transformation matrix.
        """
        if index is not None:
            bladder_r = Rotation.from_quat(self.recorded_bladder_orientations[index])
            bladder_p = self.recorded_bladder_positions[index]
        else:
            bladder_r = self.bladder_slerp(t)
            bx = np.interp(t, xp=self.optitrack_times, fp=self.recorded_bladder_positions[:, 0])
            by = np.interp(t, xp=self.optitrack_times, fp=self.recorded_bladder_positions[:, 1])
            bz = np.interp(t, xp=self.optitrack_times, fp=self.recorded_bladder_positions[:, 2])
            bladder_p = np.array([bx, by, bz])

        if self.__relative_trajectory:
            self.__relative_trajectory = False
            cam_transform = self.get_absolute_orientation(index, t)
            self.__relative_trajectory = True
        else:
            cam_transform = self.get_absolute_orientation(index, t)

        # cam tool-point position to bladder coordinate system
        cam_transform[:3, :3] = bladder_r.as_matrix() @ cam_transform[:3, :3]
        cam_transform[:3, 3] = bladder_r.apply(cam_transform[:3, 3] - bladder_p)

        if self.__relative_trajectory:
            cam_transform[:3, 3] = cam_transform[:3, 3] - self.__zero_relative_transform[:3, 3]
            cam_transform[:3, :3] = np.linalg.inv(self.__zero_relative_transform[:3, :3]) @ cam_transform[:3, :3]
            cam_transform[:3, 3] = np.linalg.inv(self.__zero_relative_transform[:3, :3]) @ cam_transform[:3, 3]

        return cam_transform
