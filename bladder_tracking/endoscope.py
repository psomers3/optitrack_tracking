import bpy
from mathutils import Matrix, Vector, Euler, Quaternion
from bladder_tracking.opti_track_csv import *
from bladder_tracking.transformations import get_optitrack_rotation_from_markers, XYZW2WXYZ, WXYZ2XYZW
from scipy.spatial.transform import Rotation, Slerp
import os


class Endoscope:
    name = 'endo-front'

    def __init__(self, data: Union[str, pd.DataFrame, dict],
                 opti_track_csv: bool = True,
                 stl_model: str = None):

        R_cam_2_endo = transform.Rotation.from_euler(angles=[0, 60, 90], seq='XYZ', degrees=True)  # type: transform.Rotation
        R_cam_2_endo_euler = R_cam_2_endo.as_euler('xyz', False)
        T_endo_2_light_in_endo = np.array([330, 0, 0])  # mm
        T_light_2_balls_in_light = np.array([-24.9378, 8.3659, 13.40425])  # mm

        total_offset = (T_endo_2_light_in_endo + T_light_2_balls_in_light) * 1e-3  # meters
        if stl_model is not None:
            self.tracker_geometric_center = Vector([-9.43783, 8.36593, 13.40425])
            bpy.ops.import_mesh.stl(filepath=stl_model)
            self.stl_object = bpy.data.objects[os.path.splitext(os.path.basename(stl_model))[0]]

        # Endo_mount ball positions from balls cg using CAD balls cg coordinate system
        endo_cad_balls = np.array([[29.66957, 47.6897, 83.68694],
                                   [29.66957, -64.42155, 83.68694],
                                   [-70.6104, -48.7631, -69.62806],
                                   [11.27126, 65.49496, -97.74583]])

        if not opti_track_csv:
            if isinstance(data, str):
                data = np.load(data)
            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(endo_cad_balls,
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
            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(cad_positions=endo_cad_balls,
                                                                                  recorded_body_positions=self.recorded_positions,
                                                                                  recorded_body_orientations=self.recorded_orientations,
                                                                                  recorded_marker_positions=positions,
                                                                                  samples_to_use=500,
                                                                                  scalar_first=True)
        self.create_endoscope()
        s = self.camera_object
        s.rotation_euler = Euler(Vector(R_cam_2_endo_euler))  # endoscope angle
        s.location = Vector(-total_offset)
        self.tracker = bpy.data.objects.new(name=self.name, object_data=None)
        self.camera_object.parent = self.tracker
        bpy.data.scenes["Scene"].camera = self.camera_object
        bpy.context.scene.collection.objects.link(self.tracker)
        self.cad_2_opti = Quaternion(self.rotation_to_opti_local.as_quat()[XYZW2WXYZ])

        self.tracker.rotation_mode = 'QUATERNION'

        if stl_model is not None:
            s = self.stl_object
            mw = s.matrix_world
            imw = mw.inverted()
            me = s.data
            origin = self.tracker_geometric_center
            local_origin = imw @ origin
            me.transform(Matrix.Translation(-local_origin))
            s.scale = Vector([0.001, 0.001, 0.001])
            self.stl_object.parent = self.tracker

    def create_endoscope(self):
        collection = bpy.data.collections.new("Endoscope")
        bpy.context.scene.collection.children.link(collection)
        camera_data = bpy.data.cameras.new(name='Camera_Data')
        camera_data.clip_start = 0.001  # 10mm
        camera_data.clip_end = 100  # 1m
        camera_data.sensor_width = 24  # 1mm
        camera_data.lens_unit = "FOV"
        camera_data.angle = np.radians(90)  # 110 degrees field of view
        camera_object = bpy.data.objects.new('Camera', camera_data)
        camera_object.location = (0, 0, 0)
        camera_object.rotation_euler = (0, 0, 0)
        collection.objects.link(camera_object)

        # create light datablock, set attributes
        light_data = bpy.data.lights.new(name="Light_Data", type='SPOT')
        light_data.energy = 0.001  # 1mW
        light_data.shadow_soft_size = 0.001  # set radius of Light Source (5mm)
        light_data.spot_blend = 0.5  # smoothness of spotlight edges
        light_data.spot_size = np.radians(120)  #

        # create new object with our light datablock
        light_left = bpy.data.objects.new(name="light_left", object_data=light_data)

        # create new object with our light datablock
        light_right = bpy.data.objects.new(name="light_right", object_data=light_data)
        light_offset = 0.001
        light_left.location = (-light_offset, 0, 0)
        light_right.location = (light_offset, 0, 0)
        light_left.parent = camera_object
        light_right.parent = camera_object

        # link light object
        collection.objects.link(light_right)
        collection.objects.link(light_left)
        self.camera_object = camera_object

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
        self.tracker.matrix_world = q.to_matrix().to_4x4() @ self.cad_2_opti.to_matrix().to_4x4()
        self.tracker.location = p

    def keyframe_insert(self, frame):
        self.tracker.keyframe_insert(data_path="location", frame=frame)
        self.tracker.keyframe_insert(data_path="rotation_quaternion", frame=frame)