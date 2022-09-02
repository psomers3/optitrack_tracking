import bpy
import numpy as np
from bladder_tracking import Bladder, BlenderEndoscope
from optitrack_tools.endoscope_definitions import ENDOSCOPES
import os
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import cv2

argv = sys.argv
argv = argv[argv.index("--") + 1:]

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA' if (sys.platform != 'darwin') else 'METAL'
for dev in prefs.devices:
    dev.use = True
    # dev.use = (dev.type != 'CPU')

bpy.ops.wm.save_userpref()


def init():
    scene = init_scene()
    depth_node, img_node = init_render_settings(scene)
    make_animation()
    return depth_node, img_node


def init_render_settings(scene):
    # Set render resolution
    scene.render.engine = render_engine
    if render_engine == 'CYCLES' and do_rendering:
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
        scene.cycles.adaptive_threshold = noise_threshold
        scene.cycles.adaptive_min_samples = 64
        scene.cycles.samples = 128
        scene.cycles.use_auto_tile = use_tiling
        scene.cycles.tile_size = tile_size
        scene.cycles.denoiser = denoise_option
        scene.cycles.device = 'GPU'
    scene.render.resolution_x = video_resolution[0]
    scene.render.resolution_y = video_resolution[1]
    scene.render.resolution_percentage = resolution_percent

    scene.render.image_settings.color_mode = 'RGB'

    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    clear_current_render_graph(tree)
    return create_render_graph(tree)


def make_animation():
    camera_matrices = np.zeros((len(video_times), 4, 4))
    for i, t in enumerate(video_times):
        if t >= 0:
            bladder.put_to_location(t=t)
            endoscope.put_to_location(t=t)

        endoscope.keyframe_insert(frame=i)
        bladder.keyframe_insert(frame=i)
        bpy.context.view_layer.update()
        a = endoscope.camera_object.matrix_world.copy()
        camera_matrices[i] = a
        if i == 3000:
            print(a)
    np.savez(file=os.path.join(recording_path, 'camera_trajectory'), trajectory=camera_matrices)
    bpy.context.scene.frame_end = i


def clear_current_render_graph(tree):
    # clear default/old nodes
    for n in tree.nodes:
        tree.nodes.remove(n)


def create_render_graph(tree):
    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True  # TODO: make this non-context based

    # create depth output node
    depth_node = tree.nodes.new('CompositorNodeOutputFile')
    depth_node.format.file_format = "OPEN_EXR"

    # create image output node
    img_node = tree.nodes.new('CompositorNodeOutputFile')
    img_node.format.file_format = "PNG"

    # Links
    links = tree.links
    links.new(rl.outputs[2], depth_node.inputs['Image'])  # link Z to output
    links.new(rl.outputs['Image'], img_node.inputs['Image'])  # link image to output
    return depth_node, img_node


def init_scene():
    scene = bpy.data.scenes["Scene"]

    master_collection = bpy.context.scene.collection
    # disable all collections that are still active
    for collection in master_collection.children:
        collection.hide_render = True
    return scene


def render_frames():
    bpy.context.scene.camera = endoscope.camera_object

    if render_skip == 1:
        scene.frame_start = render_starting_at_frame
        bpy.ops.render.render(animation=True, scene=scene.name)
    else:
        for i, t in tqdm(enumerate(video_times)):
            if i < render_starting_at_frame:
                continue
            if i % render_skip == 0:
                scene.frame_set(i)
                bpy.ops.render.render(write_still=True)  # render still image


def renderAnimation(img_node, depth_node, screenshot_folder):
    depth_node.base_path = screenshot_folder
    img_node.base_path = screenshot_folder
    if do_rendering:
        render_frames()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('recording_directory', type=str,
                        help='directory where the video and numpy recording files are.')
    parser.add_argument('--render_skip', type=int,  help='render every i-th frame, where i is the value provided.'
                                                         ' [default=1]', default=1)
    parser.add_argument('--render', action='store_true',
                        help='activates rendering of the animation.', default=False)
    parser.add_argument('--start_frame', type=int, help='which frame to start rendering at. [default=1] ', default=1)
    parser.add_argument('--video_delay', type=float,
                        help='a delay to help synchronize the video and movement. [default=0.1]', default=0.1)
    parser.add_argument('--resolution_percent', type=float,
                        help='fraction (0-1) to reduce the rendering resolution by (faster) [default=0.5]', default=0.5)
    parser.add_argument('--denoise', type=str, choices=['OPTIX', 'OPENIMAGEDENOISE'],
                        help='denoising option depending on computer. default = \'OPTIX\'', default='OPTIX')
    parser.add_argument('--tile_size', type=int,
                        help='tile size for GPU computing. tune for graphics card. [default=1024]', default=256)
    parser.add_argument('--no_tiling', action='store_false',
                        help='disables the auto_tile_size variable for GPU rendering. [default=True]', default=True)
    parser.add_argument('--render_engine', type=str, help='which renderer to use. [default=\'CYCLES\']',
                        default='CYCLES', choices=['CYCLES', 'EEVEE'])
    parser.add_argument('--reverse_cam', action='store_true', help='switches the relative direction that the camera is '
                                                                   'rotated against the endoscope', default=False)
    parser.add_argument('--noise_threshold', type=float, help='rendering noise threshold. [default=0.01]', default=0.01)
    args = parser.parse_args(args=argv)

    obj_map = {}
    context = bpy.context
    scene = context.scene

    for c in scene.collection.children:
        scene.collection.children.unlink(c)

    bpy.data.scenes["Scene"].unit_settings.length_unit = 'MILLIMETERS'

    recording_path = os.path.abspath(args.recording_directory)
    do_rendering = args.render
    render_skip = args.render_skip
    render_starting_at_frame = args.start_frame
    video_time_delay = args.video_delay
    denoise_option = args.denoise
    tile_size = args.tile_size
    resolution_percent = int(100*args.resolution_percent)
    use_tiling = args.no_tiling
    render_engine = args.render_engine if args.render_engine == "CYCLES" else "BLENDER_EEVEE"
    reverse_cam = args.reverse_cam
    noise_threshold = args.noise_threshold

    vid_file = os.path.join(recording_path, 'video.mp4')
    cap = cv2.VideoCapture(vid_file)
    ret, frame = cap.read()
    video_resolution = (frame.shape[1], frame.shape[0])
    data_file = os.path.join(recording_path, 'data.npz')
    save_path = os.path.join(recording_path, 'depth_rendering')
    data = np.load(data_file)
    video_times = np.squeeze(data['video_timestamps'] - data['optitrack_received_timestamps'][0] - video_time_delay)

    bladder = Bladder(data_file, ['./models/bladder_tracker.stl',
                                  './models/bladder_1.stl',
                                  './models/bladder_2.stl'],
                      opti_track_csv=False)
    endoscope = BlenderEndoscope(data_file,
                                 endoscope=ENDOSCOPES.ITO,
                                 stl_model_path='./models',
                                 opti_track_csv=False,
                                 light_surfaces='./models/ITO_light.stl',
                                 camera_mount_stl='./models/camera_mount.stl',
                                 invert_cam_rotation=reverse_cam,
                                 camera_params=os.path.join(recording_path, 'cam_params.json'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    depth_node, img_node = init()
    renderAnimation(img_node, depth_node, save_path)
    for a in bpy.context.screen.areas:
        if a.type == 'VIEW_3D':
            a.spaces.active.clip_end = 1e8
            break