import bpy
import numpy as np
from bladder_tracking import *
import os

obj_map = {}
context = bpy.context
scene = context.scene

for c in scene.collection.children:
    scene.collection.children.unlink(c)

bpy.data.scenes["Scene"].unit_settings.length_unit = 'MILLIMETERS'
scene.render.image_settings.file_format = 'PNG'  # set output format to .png

data_file = r'C:\Users\Somers\Desktop\test_recording\data.npz'
save_path = r'C:\Users\Somers\Desktop\test_recording\depth_rendering'
video_times = np.squeeze(np.load(data_file)['video_timestamps'] - np.load(data_file)['video_timestamps'][0])
bladder = Bladder(data_file, ['C:/Users/Somers/Desktop/optitrack/1.STL',
                              'C:/Users/Somers/Desktop/optitrack/2.stl',
                              'C:/Users/Somers/Desktop/optitrack/3.stl'],
                  opti_track_csv=False)
camera = Endoscope(data_file, stl_model='C:/Users/Somers/Desktop/optitrack/endoscope.stl', opti_track_csv=False)


def init():
    scene = init_scene()
    depth_node, img_node = init_render_settings(scene)
    init_animation()
    return depth_node, img_node


def init_render_settings(scene):
    # Set render resolution
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 920
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.engine = 'BLENDER_EEVEE'

    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    clear_current_render_graph(tree)
    return create_render_graph(tree)


def init_animation():
    i = 0
    for t in video_times:
        bladder.put_to_location(t=t)
        bladder.keyframe_insert(frame=i)
        camera.put_to_location(t=t)
        camera.keyframe_insert(frame=i)
        i += 1
    bpy.context.scene.frame_end = i


def clear_current_render_graph(tree):
    # clear default/old nodes
    for n in tree.nodes:
        tree.nodes.remove(n)


def create_render_graph(tree):
    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')

    # create depth output node
    depth_node = tree.nodes.new('CompositorNodeOutputFile')
    depth_node.format.file_format = "OPEN_EXR"

    # create image output node
    img_node = tree.nodes.new('CompositorNodeOutputFile')
    img_node.format.file_format = "PNG"

    # Links
    links = tree.links
    links.new(rl.outputs['Depth'], depth_node.inputs['Image'])  # link Z to output
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
    for i, t in enumerate(video_times):
        scene.frame_set(i)
        scene.render.filepath = os.path.join(save_path, f'{t}')
        bpy.ops.render.render(write_still=True)  # render still image


def renderAnimation(img_node, depth_node, screenshot_folder):
    depth_node.base_path = screenshot_folder
    img_node.base_path = screenshot_folder
    render_frames()


if __name__ == '__main__':
    depth_node, img_node = init()
    depthmap_folder = 'C:/Users/Somers/Desktop/optitrack/'
    renderAnimation(img_node, depth_node, depthmap_folder)
    for a in bpy.context.screen.areas:
        if a.type == 'VIEW_3D':
            a.spaces.active.clip_end = 1e8
            break