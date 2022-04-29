import bpy
import numpy as np
from mathutils import Matrix, Vector, Euler, Quaternion
from bladder_tracking.opti_track_csv import *
from bladder_tracking.camera_mount import CameraMount
from bladder_tracking.transformations import get_optitrack_rotation_from_markers, XYZW2WXYZ, WXYZ2XYZW
from bladder_tracking.blender_cam import get_blender_camera_from_3x4_P
from scipy.spatial.transform import Rotation, Slerp
import os
import json


def new_material(name: str):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()
    return mat


class Endoscope:
    name = 'endo-front'

    def __init__(self,
                 data: Union[str, pd.DataFrame, dict],
                 opti_track_csv: bool = True,
                 stl_model: str = None,
                 light_surfaces: str = None,
                 camera_mount_stl: str = None,
                 invert_cam_rotation: bool = False,
                 camera_params: str = None):

        self.cam_direction = -1 if invert_cam_rotation else 1
        self.collection = bpy.data.collections.new("Endoscope")
        bpy.context.scene.collection.children.link(self.collection)
        self.collection.hide_render = False
        camera_angle = 30
        rotation_cam_2_endo = transform.Rotation.from_euler(angles=[0, 90-camera_angle, 90],
                                                            seq='XYZ',
                                                            degrees=True)  # type: transform.Rotation

        endo_2_light = np.array([321.5, 0, 0])  # mm  length of endoscope starting at where the light comes in.
        light_2_balls = np.array([-24.9378, 8.3659, 13.40425])
        """ in mm.  Distance in CAD from light/fiber intersection to markers CG. """

        total_offset = np.array((endo_2_light + light_2_balls) * 1e-3)  # meters
        self.cad_geometric_center = Vector([-9.43783, 8.36593, 13.40425])
        """ The vector from the CAD origin to the opti markers CG """

        endo_cad_balls = np.array([[29.66957, 47.6897, 83.68694],
                                   [29.66957, -64.42155, 83.68694],
                                   [-70.6104, -48.7631, -69.62806],
                                   [11.27126, 65.49496, -97.74583]])
        """ Distances to each marker in CAD from the markers CG. Uses the CAD coordinate sys. """

        if camera_params is None:
            self.projection_matrix = Matrix([[1.0347, 0., 0.8982, 0.],
                                             [0., 1.0313, 0.5411, 0.],
                                             [0.,     0.,  0.001, 0.]])
            """ The transposed calibration matrix from Matlab's calibration tool """
        else:
            temp = np.zeros((3, 4))
            temp[:, :3] = np.array(json.load(open(camera_params, 'r'))['IntrinsicMatrix']).T
            self.projection_matrix = temp

        if stl_model is not None:
            bpy.ops.import_mesh.stl(filepath=stl_model)
            self.stl_object = bpy.data.objects[os.path.splitext(os.path.basename(stl_model))[0]]
            bpy.context.scene.collection.objects.unlink(self.stl_object)
            self.collection.objects.link(self.stl_object)

        self.camera_object, self.camera_data = self.create_camera()
        self.set_lighting(light_surfaces)

        if not opti_track_csv:
            if isinstance(data, str):
                data = np.load(data)
            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(endo_cad_balls,
                                                                                  recorded_marker_positions=data[f'{self.name}-markers'],
                                                                                  recorded_body_positions=data[f'{self.name}'][:, :3],
                                                                                  recorded_body_orientations=data[f'{self.name}'][:, 3:],
                                                                                  scalar_first=False)
            self.optitrack_times = np.squeeze(data["optitrack_received_timestamps"] - data["optitrack_received_timestamps"][0])
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

        self.endo_endpoint = bpy.data.objects.new(name="endo_endpoint", object_data=None)
        self.collection.objects.link(self.endo_endpoint)
        self.camera_object.parent = self.endo_endpoint
        self.endo_endpoint.rotation_euler = Euler(Vector(rotation_cam_2_endo.as_euler('xyz', False)))  # endoscope angle
        self.endo_endpoint.location = Vector(-total_offset)

        self.tracker = bpy.data.objects.new(name=self.name, object_data=None)
        self.endo_endpoint.parent = self.tracker
        bpy.data.scenes["Scene"].camera = self.camera_object
        self.collection.objects.link(self.tracker)
        self.cad_2_opti = Quaternion(self.rotation_to_opti_local.as_quat()[XYZW2WXYZ])
        self.tracker.rotation_mode = 'QUATERNION'

        if stl_model is not None:
            s = self.stl_object
            mw = s.matrix_world
            imw = mw.inverted()
            me = s.data
            origin = self.cad_geometric_center
            local_origin = imw @ origin
            me.transform(Matrix.Translation(-local_origin))
            s.scale = Vector([0.001, 0.001, 0.001])
            self.stl_object.parent = self.tracker

        self.camera_mount = CameraMount(data=data, opti_track_csv=opti_track_csv, collection=self.collection, files=camera_mount_stl)
        self.zero_angle = np.average(np.array([self.get_camera_angle(x) for x in range(10)]))

        # This line because I don't know how to get my collections to render otherwise...
        [bpy.context.scene.collection.objects.link(x) for x in self.collection.objects]

    def get_camera_angle(self, index: int = None, t: float = None):
        assert (index is not None and t is None) or (t is not None and index is None)
        if index is not None:
            r_cam_tracker = Rotation.from_quat(self.camera_mount.recorded_orientations[index][WXYZ2XYZW]).inv()
            r_endo_tracker = Rotation.from_quat(self.recorded_orientations[index][WXYZ2XYZW])
        else:
            r_cam_tracker = self.camera_mount.slerp(t)  # from opti-track to cam sys
            r_endo_tracker = self.slerp(t).inv()  # from cam sys to opti sys
        resultant_r = r_cam_tracker * r_endo_tracker  # type: Rotation
        return np.arctan(np.linalg.norm(resultant_r.as_mrp()))*4

    def set_lighting(self, stl_file: str):
        if stl_file is not None:
            bpy.ops.import_mesh.stl(filepath=stl_file)
            stl_object = bpy.data.objects[os.path.splitext(os.path.basename(stl_file))[0]]
            bpy.context.scene.collection.objects.unlink(stl_object)
            stl_object.scale = Vector([0.001, 0.001, 0.001])
            stl_object.rotation_euler = Vector(np.radians([90, 0, 90]))
            self.collection.objects.link(stl_object)
            stl_object.parent = self.camera_object
            mat = new_material('light_emission')
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            output = nodes.new(type='ShaderNodeOutputMaterial')
            shader = nodes.new(type='ShaderNodeEmission')
            mixer = nodes.new(type='ShaderNodeMixShader')
            backfacing = nodes.new(type='ShaderNodeNewGeometry')
            nodes["Emission"].inputs[0].default_value = (1, 1, 1, 1)
            nodes["Emission"].inputs[1].default_value = 100
            links.new(backfacing.outputs[6], mixer.inputs[0])
            links.new(shader.outputs[0], mixer.inputs[1])
            links.new(mixer.outputs[0], output.inputs[0])
            stl_object.data.materials.append(mat)

        else:
            # create light datablock, set attributes
            light_data = bpy.data.lights.new(name="Light_Data", type='SPOT')
            light_data.specular_factor = 0.004
            light_data.volume_factor = .85
            light_data.diffuse_factor = 0.003
            light_data.energy = 5  # Watts
            light_data.shadow_soft_size = 0.0005  # set radius of Light Source (mm)
            light_data.spot_blend = 1.0  # smoothness of spotlight edges
            light_data.spot_size = np.radians(65)  #

            # create new object with our light datablock
            light_left = bpy.data.objects.new(name="light_left", object_data=light_data)

            # create new object with our light datablock
            light_right = bpy.data.objects.new(name="light_right", object_data=light_data)
            light_offset = 0.004
            light_left.location = (-light_offset, 0, 0)
            light_right.location = (light_offset, 0, 0)
            light_left.parent = self.camera_object
            light_right.parent = self.camera_object

            # link light object
            self.collection.objects.link(light_right)
            self.collection.objects.link(light_left)

    def create_camera(self) -> Tuple[bpy.types.Object, bpy.types.Camera]:
        """
        Helper function to create the camera object
        :return:
        """
        camera_object, camera_data = get_blender_camera_from_3x4_P(self.projection_matrix, scale=1)
        camera_data.clip_start = 0.001
        camera_data.clip_end = 100
        camera_object.location = [0, 0, 0]
        camera_object.rotation_euler = [0, 0, 0]
        # self.camera_data.lens_unit = "FOV"
        # self.camera_data.angle = np.radians(90)  # 110 degrees field of view
        self.collection.objects.link(camera_object)
        return camera_object, camera_data

    def put_to_location(self, index: int = None, t: float = None) -> None:
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
            cam_angle = self.get_camera_angle(index=index)
        else:
            q = Quaternion(self.slerp(t).as_quat()[XYZW2WXYZ])
            x = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 0])
            y = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 1])
            z = np.interp(t, xp=self.optitrack_times, fp=self.recorded_positions[:, 2])
            p = [x, y, z]
            cam_angle = self.get_camera_angle(t=t)
        self.tracker.matrix_world = q.to_matrix().to_4x4() @ self.cad_2_opti.to_matrix().to_4x4()
        self.tracker.location = p
        self.camera_mount.put_to_location(index, t)
        self.camera_object.rotation_euler = [0, 0, self.cam_direction*(cam_angle - self.zero_angle)]

    def keyframe_insert(self, frame):
        self.tracker.keyframe_insert(data_path="location", frame=frame)
        self.tracker.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        self.camera_object.keyframe_insert(data_path="rotation_euler", frame=frame)
        self.camera_mount.keyframe_insert(frame)