import bpy
from mathutils import Matrix, Vector, Euler, Quaternion
from bladder_tracking.opti_track_csv import *
from bladder_tracking.transformations import get_optitrack_rotation_from_markers
import os
from ast import literal_eval


class Bladder:
    bladder_name = 'bladder'

    def __init__(self, data_frame: Union[str, pd.DataFrame],
                 files: List[str] = "C:/Users/Somers/Desktop/optitrack/test.stl",
                 opti_track_csv: bool = True):

        files = [files] if isinstance(files, str) else files
        self.stl_objects = []
        for f in files:
            bpy.ops.import_mesh.stl(filepath=f)
            self.stl_objects.append(bpy.data.objects[os.path.splitext(os.path.basename(f))[0]])

        self.tracker_ball1 = [-31.94985, 39.75, -10.93251]
        self.tracker_ball2 = [-81.40743, -35.25, 19.95353]
        self.tracker_ball3 = [23.95005, 25.75, 39.07951]
        self.tracker_ball4 = [89.40723, -30.25, -48.10053]

        self.tracker_geometric_center = Vector([-7.16498, 9.75, 76.21033])
        self.tracker_cad_balls = np.array([self.tracker_ball1, self.tracker_ball2, self.tracker_ball3, self.tracker_ball4])
        if not opti_track_csv:
            if isinstance(data_frame, str):
                data_frame = pd.read_csv(data_frame, quotechar='"', sep=',',
                                         converters={idx: literal_eval for idx in range(1, 100)})

            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_dataframe(data_frame,
                                                                                    f'{self.bladder_name}-position',
                                                                                    f'{self.bladder_name}-rotation',
                                                                                    [f'{self.bladder_name}-1',
                                                                                     f'{self.bladder_name}-2',
                                                                                     f'{self.bladder_name}-3',
                                                                                     f'{self.bladder_name}-4'],
                                                                                    self.tracker_cad_balls * 1e-3)

            positions = np.array(list(data_frame[f'{self.bladder_name}-position'].to_numpy()))  # meters
            orientations = np.array(
                list(data_frame[f'{self.bladder_name}-rotation'].to_numpy()))  # quaternions (x, y, z, w)
            self.recorded_positions, self.recorded_orientations = positions, orientations
        else:
            data, parsing = get_prepared_df(data_frame)
            positions = get_marker_positions(data, f'{self.bladder_name}')
            self.recorded_positions, self.recorded_orientations = get_rigid_body_data(data, f'{self.bladder_name}')
            self.rotation_to_opti_local, _p = get_optitrack_rotation_from_markers(cad_positions=self.tracker_cad_balls,
                                                                                  recorded_body_positions=self.recorded_positions,
                                                                                  recorded_body_orientations=self.recorded_orientations,
                                                                                  recorded_marker_positions=positions,
                                                                                  samples_to_use=500)

        for s in self.stl_objects:
            s.rotation_mode = 'QUATERNION'
            # !!! Blender quaternions are [w, x, y, z] and scipy is [x, y, z, w]
            mw = s.matrix_world
            imw = mw.inverted()
            me = s.data
            origin = self.tracker_geometric_center
            local_origin = imw @ origin
            me.transform(Matrix.Translation(-local_origin))
            opti_quat = Quaternion(self.rotation_to_opti_local.as_quat()[[3, 0, 1, 2]])
            me.transform(Matrix.Rotation(opti_quat.angle, 4, opti_quat.axis))
            s.scale = Vector([0.001, 0.001, 0.001])

    def put_stl_to_location(self, index: int):
        quat = Quaternion(self.recorded_orientations[index][[3, 0, 1, 2]])
        position = Vector(self.recorded_positions[index])
        for s in self.stl_objects:
            s.rotation_quaternion = quat
            s.location = position

    def keyframe_insert(self, frame):
        for s in self.stl_objects:
            s.keyframe_insert(data_path="location", frame=frame)
            s.keyframe_insert(data_path="rotation_quaternion", frame=frame)