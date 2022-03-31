import bpy
from mathutils import Matrix, Vector, Euler, Quaternion
from bladder_tracking.opti_track_csv import *
from bladder_tracking.transformations import get_optitrack_rotation_from_markers
from ast import literal_eval
import os


class Endoscope:
    front_tracker_name = 'endo-front'

    def __init__(self, data_frame: Union[str, pd.DataFrame],
                 opti_track_csv: bool = True,
                 stl_model: str = None):

        R_cam_2_endo = transform.Rotation.from_euler(angles=[0, 60, 0], seq='XYZ', degrees=True)  # type: transform.Rotation
        R_cam_2_endo_euler = R_cam_2_endo.as_euler('xyz', False)
        T_endo_2_light_in_endo = np.array([330, 0, 0])  # mm
        T_light_2_balls_in_light = np.array([-24.9378, 8.3659, 13.40425])  # mm

        total_offset = (T_endo_2_light_in_endo + T_light_2_balls_in_light) * 1e-3  # meters
        if stl_model is not None:
            self.tracker_geometric_center = Vector([-9.43783, 8.36593, 13.40425])
            bpy.ops.import_mesh.stl(filepath=stl_model)
            self.stl_object = bpy.data.objects[os.path.splitext(os.path.basename(stl_model))[0]]

        # Endo_mount ball positions from balls cg using CAD balls cg coordinate system
        endo_cad_ball1 = [29.66957, 47.6897, 83.68694]
        endo_cad_ball2 = [29.66957, -64.42155, 83.68694]
        endo_cad_ball3 = [-70.6104, -48.7631, -69.62806]
        endo_cad_ball4 = [11.27126, 65.49496, -97.74583]
        endo_cad_balls = np.array([endo_cad_ball1, endo_cad_ball2, endo_cad_ball3, endo_cad_ball4])

        if not opti_track_csv:
            if isinstance(data_frame, str):
                data_frame = pd.read_csv(data_frame, quotechar='"', sep=',', converters={idx: literal_eval for idx in range(1, 100)})

            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_dataframe(data_frame,
                                                                                    f'{self.front_tracker_name}-position',
                                                                                    f'{self.front_tracker_name}-rotation',
                                                                                    [f'{self.front_tracker_name}-1', f'{self.front_tracker_name}-2', f'{self.front_tracker_name}-3', f'{self.front_tracker_name}-4'],
                                                                                    endo_cad_balls * 1e-3)
            # print(self.rotation_to_opti_local.as_euler('xyz', degrees=True))
            positions = np.array(list(data_frame[f'{self.front_tracker_name}-position'].to_numpy()))  # meters
            orientations = np.array(list(data_frame[f'{self.front_tracker_name}-rotation'].to_numpy()))  # quaternions (x, y, z, w)
            self.recorded_positions, self.recorded_orientations = positions, orientations
        else:
            data, parsing = get_prepared_df(data_frame)
            positions = get_marker_positions(data, f'{self.front_tracker_name}')
            self.recorded_positions, self.recorded_orientations = get_rigid_body_data(data, f'{self.front_tracker_name}')
            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(cad_positions=endo_cad_balls,
                                                                                  recorded_body_positions=self.recorded_positions,
                                                                                  recorded_body_orientations=self.recorded_orientations,
                                                                                  recorded_marker_positions=positions,
                                                                                  samples_to_use=500)
        self.create_endoscope()
        s = self.camera_object
        s.rotation_euler = Euler(Vector(R_cam_2_endo_euler))  # endoscope angle
        s.location = Vector(-total_offset)
        self.tracker = bpy.data.objects.new(name="endotracker_front", object_data=None)
        self.camera_object.parent = self.tracker
        bpy.data.scenes["Scene"].camera = self.camera_object
        bpy.context.scene.collection.objects.link(self.tracker)
        self.cad_2_opti = Quaternion(self.rotation_to_opti_local.as_quat())

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
        camera_data.sensor_width = 1  # 1mm
        camera_data.lens_unit = "FOV"
        camera_data.angle = np.radians(110)  # 110 degrees field of view
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

    def reset(self):
        self.tracker.location = Vector([0, 0, 0])
        self.tracker.rotation_euler = Euler(Vector([0, 0, 0]))

    def put_to_location(self, index: int):
        quat = Quaternion(self.recorded_orientations[index][[3, 0, 1, 2]])
        position = Vector(self.recorded_positions[index])
        # self.reset()
        self.tracker.matrix_world = self.cad_2_opti.to_matrix().to_4x4() @ quat.to_matrix().to_4x4()
        self.tracker.location = position

    def keyframe_insert(self, frame):
        self.tracker.keyframe_insert(data_path="location", frame=frame)
        self.tracker.keyframe_insert(data_path="rotation_quaternion", frame=frame)