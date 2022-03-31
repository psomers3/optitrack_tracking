import bpy
from bladder_tracking import *

obj_map = {}
context = bpy.context
scene = context.scene

for c in scene.collection.children:
    scene.collection.children.unlink(c)

bpy.data.scenes["Scene"].unit_settings.length_unit = 'MILLIMETERS'

# csv_file = r'C:\Users\Somers\Desktop\optitrack\Take 2022-03-17 02.47.34 PM.csv'  # moving endo
# csv_file = r'C:\Users\Somers\Desktop\optitrack\Take 2022-03-17 02.01.37 PM.csv'  # moving endo
csv_file = r'C:\Users\Somers\Desktop\optitrack\Take 2022-03-17 02.16.20 PM.csv'  # pointing at bladder?
# csv_file = r'C:\Users\Somers\Desktop\optitrack\Take 2022-03-17 04.31.54 PM.csv'
# csv_file = r'C:\Users\Somers\Desktop\optitrack\Take 2022-03-17 04.13.55 PM.csv'
# csv_file = r'C:\Users\Somers\Desktop\optitrack\recording-101\Take 2022-03-16 02.01.35 PM.csv'

bladder = Bladder(csv_file, ['C:/Users/Somers/Desktop/optitrack/1.STL',
                             'C:/Users/Somers/Desktop/optitrack/2.stl',
                             'C:/Users/Somers/Desktop/optitrack/3.stl'])
camera = Endoscope(csv_file, stl_model='C:/Users/Somers/Desktop/optitrack/endoscope.stl')


def init(camera_poses, focal):
    scene = init_scene()
    # bpy.ops.object.light_add(type='POINT')
    depth_node, img_node = init_render_settings(scene)
    init_animation(camera_poses)
    return depth_node, img_node


def init_render_settings(scene):
    # Set render resolution
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.engine = 'BLENDER_EEVEE'

    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    clear_current_render_graph(tree)
    return create_render_graph(tree)


def init_animation(camera_poses):
    for i in range(1, len(bladder.recorded_positions)):
        bladder.put_stl_to_location(i)
        bladder.keyframe_insert(frame=i)
        camera.put_to_location(i)
        camera.keyframe_insert(i)
    bpy.context.scene.frame_end = len(camera_poses)


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


def render():
    # render
    bpy.ops.render.render(animation=True)


def renderAnimation(img_node, depth_node, screenshot_folder):
    depth_node.base_path = screenshot_folder
    img_node.base_path = screenshot_folder
    # render()


if __name__ == '__main__':
    camera_poses = []
    focal_length = 0.003
    depth_node, img_node = init(camera_poses, focal_length)
    depthmap_folder = 'C:/Users/Somers/Desktop/optitrack/'
    renderAnimation(img_node, depth_node, depthmap_folder)
    for a in bpy.context.screen.areas:
        if a.type == 'VIEW_3D':
            a.spaces.active.clip_end = 1e8
            break