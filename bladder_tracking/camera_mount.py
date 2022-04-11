import bpy
from mathutils import Matrix, Vector, Quaternion
from bladder_tracking.opti_track_csv import *
from bladder_tracking.transformations import get_optitrack_rotation_from_markers, XYZW2WXYZ, WXYZ2XYZW
import os
from scipy.spatial.transform import Rotation, Slerp


class CameraMount:
    name = 'cam'
    """ This name should match the rigid body name from optitrack """

    def __init__(self, data: Union[str, pd.DataFrame, dict],
                 files: Union[str, List[str]] = "C:/Users/Somers/Desktop/optitrack/camera_mount.stl",
                 opti_track_csv: bool = True,
                 collection: bpy.types.Collection = None):
        if collection is not None:
            self.collection = collection
        else:
            self.collection = bpy.context.scene.collection
        self.collection.hide_render = False

        files = [files] if isinstance(files, str) else files
        self.stl_objects = []
        self.opti_track_csv = opti_track_csv
        for f in files:
            bpy.ops.import_mesh.stl(filepath=f)
            obj = bpy.data.objects[os.path.splitext(os.path.basename(f))[0]]
            bpy.context.scene.collection.objects.unlink(obj)
            self.stl_objects.append(obj)
            self.collection.objects.link(obj)

        self.cad_balls = np.array([[-44.3395, 79.50937, -43.17249],
                                   [-38.61143, -9.2119, 62.74736],
                                   [-34.43467, -75.99019, 0.63252],
                                   [117.38562, 5.69273, -21.56939]]) * 1e-3

        self.geometric_center = Vector([-31.51551, -11.10362, -9.18387])
        if not opti_track_csv:
            if isinstance(data, str):
                data = np.load(data)
            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(self.cad_balls,
                                                                                  recorded_marker_positions=data[f'{self.name}-markers'],
                                                                                  recorded_body_positions=data[f'{self.name}'][:, :3],
                                                                                  recorded_body_orientations=data[f'{self.name}'][:, 3:],
                                                                                  scalar_first=False)
            self.optitrack_times = np.squeeze(data["optitrack_timestamps"] - data["optitrack_timestamps"][0])
            self.recorded_orientations = data[f'{self.name}'][:, XYZW2WXYZ]  # save as w, x, y, z
            rotations = Rotation.from_quat(self.recorded_orientations[:, WXYZ2XYZW])  # switch scalar position
            self.recorded_positions = data[f'{self.name}'][:, :3]
            self.slerp = Slerp(self.optitrack_times, rotations)
        else:
            data, parsing = get_prepared_df(data)
            positions = get_marker_positions(data, f'{self.name}')
            self.recorded_positions, self.recorded_orientations = get_rigid_body_data(data, f'{self.name}')
            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(cad_positions=self.cad_balls,
                                                                                  recorded_body_positions=self.recorded_positions,
                                                                                  recorded_body_orientations=self.recorded_orientations,
                                                                                  recorded_marker_positions=positions,
                                                                                  samples_to_use=500,
                                                                                  scalar_first=True)

        for s in self.stl_objects:
            s.rotation_mode = 'QUATERNION'
            # !!! Blender quaternions are [w, x, y, z] and scipy is [x, y, z, w]
            mw = s.matrix_world
            imw = mw.inverted()
            me = s.data
            origin = self.geometric_center
            local_origin = imw @ origin
            me.transform(Matrix.Translation(-local_origin))
            opti_quat = Quaternion(self.rotation_to_opti_local.as_quat()[XYZW2WXYZ])
            me.transform(Matrix.Rotation(opti_quat.angle, 4, opti_quat.axis))
            s.scale = Vector([0.001, 0.001, 0.001])

    def put_to_location(self, index: int = None, t: float = None):
        """

        :param index: puts the model to the index position from the recorded data
        :param t: puts the model to the time position using interpolation (slerp). currently not supported for optitrack
                  csv files.
        :return: None
        """
        assert (index is not None and t is None) or (t is not None and index is None)

        if index is not None:
            q = Quaternion(self.recorded_orientations[index])
            p = Vector(self.recorded_positions[index])
        else:
            q = Quaternion(self.slerp(t).as_quat()[XYZW2WXYZ])
            x = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 0])
            y = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 1])
            z = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 2])
            p = [x, y, z]

        for s in self.stl_objects:
            s.rotation_quaternion = q
            s.location = p

    def keyframe_insert(self, frame):
        for s in self.stl_objects:
            s.keyframe_insert(data_path="location", frame=frame)
            s.keyframe_insert(data_path="rotation_quaternion", frame=frame)