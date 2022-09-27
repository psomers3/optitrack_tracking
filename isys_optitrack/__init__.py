from isys_optitrack.optitrack_tools import ENDOSCOPES, Endoscope, EndoscopeTrajectory, invert_affine_transform
try:
    from isys_optitrack.bladder_tracking import BlenderBladder, BlenderEndoscope, BlenderCameraMount
except ImportError as e:
    print('blender\'s mathutils not found! This may be installable with \'pip install mathutils\'')
