import os
import numpy as np
import cv2 as cv
from tqdm import trange
import json
from typing import *
import shutil
from matplotlib import pyplot as plt
plt.ion()

from isys_optitrack.optitrack_tools.endoscope_trajectory import EndoscopeTrajectory
from isys_optitrack.optitrack_tools.endoscope_definitions import Endoscope, ENDOSCOPES
from isys_optitrack.image_tools.masking import get_circular_mask_4_img, ImageCroppingException


def _path_create(path: str):
    """
    create path if it doesn't exist
    :param path: directory to create
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def generate_transform_matrix(original: np.ndarray,
                              offset: Union[Iterable[float], np.ndarray] = None) \
        -> np.ndarray:
    """
    https://github.com/NVlabs/instant-ngp/discussions/153?converting=1#discussioncomment-2187648
    changes from blender info to instant-ngp transform format
    :param original: original 4x4 transorm matrix
    :param offset: optional [x, y, z] distance to offset the position before transforming.
    :return: 4x4 transformation matrix
    """
    xf_rot = np.eye(4)
    xf_rot[:3, :3] = original[:3, :3]

    xf_pos = np.eye(4)
    average_position = 0 if not offset else offset
    xf_pos[:3, 3] = original[:3, 3] - average_position

    M = np.asarray([[0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    xf = M @ xf_pos
    assert np.abs(np.linalg.det(xf) - 1.0) < 1e-4
    xf = xf @ xf_rot
    return xf


def save_images_and_traj_for_ngp(input_directory: str,
                                 output_dir: str,
                                 endoscope: Endoscope = ENDOSCOPES.TUEBINGEN,
                                 video_time_delay: float = 0.1,
                                 invert_cam_rotation: bool = False,
                                 frame_samples: int = None,
                                 runtime_stops: Tuple[float] = None,
                                 aabb_scaling: int = 1,
                                 model_scale: float = 0.33,
                                 bladder_offset: bool = False) -> None:
    """
    create a dataset from a recorded endoscope video (with optitrack data) for use with NVIDIA's instant-NGP

    :param input_directory: directory where the recorded data is located
    :param output_dir: directory to output the converted trajectory and images
    :param endoscope: what endoscope was used for the recording.
    :param video_time_delay: a delay in seconds to help synchronize the video and movement.
    :param invert_cam_rotation: whether to invert the relative direction of endoscope rotation to camera. This needs to
                                be figured out with trial and error.
    :param frame_samples: optionally only take these number of samples. The video is divided evenly to sample across the
                          entire runtime.
    :param runtime_stops: start and end times to use as the runtime. [start_in_seconds, end_in_seconds]
    :param aabb_scaling: For natural scenes where there is a background visible outside the unit cube, it is necessary
                         to set the parameter aabb_scaling to a power of 2 integer up to 128  (1, 2, 4, 8, ..., 128)
    :param model_scale: scaling factor for the entire model.
    :param bladder_offset: whether to return the trajectory relative to the bladder mold
    """
    if not os.path.exists(input_directory):
        raise FileNotFoundError(input_directory)
    _path_create(output_dir)
    image_directory = os.path.join(output_dir, 'images')
    _path_create(image_directory)
    data = np.load(os.path.join(input_directory, 'data.npz'))

    trajectory = EndoscopeTrajectory(data, invert_cam_rotation=invert_cam_rotation, relative_trajectory=True,
                                     endoscope=endoscope)
    video_times = np.squeeze(data['video_timestamps'] - data['optitrack_received_timestamps'][0] - video_time_delay)
    vid_capture = cv.VideoCapture(os.path.join(input_directory, 'video.mp4'))
    num_frames = int(vid_capture.get(cv.CAP_PROP_FRAME_COUNT))
    positions = []
    with open(os.path.join(input_directory, 'cam_params.json')) as f:
        camera_params = json.load(f)
    output_json = {"fl_x": camera_params['FocalLength'][0],
                   "fl_y": camera_params['FocalLength'][1],
                   "k1": camera_params['RadialDistortion'][0],
                   "k2": camera_params['RadialDistortion'][1],
                   "p1": camera_params['TangentialDistortion'][0],
                   "p2": camera_params['TangentialDistortion'][1],
                   "cx": camera_params['PrincipalPoint'][0],
                   "cy": camera_params['PrincipalPoint'][1],
                   "w": camera_params['ImageSize'][1],
                   "h": camera_params['ImageSize'][0],
                   "aabb_scale": aabb_scaling,
                   "scale": model_scale,
                   "offset": [0.5, 0.5, 0.5],
                   "frames": []}

    frames_to_include = []
    times_to_include = []
    if frame_samples:
        skip_percentage = 1 / frame_samples
        percentages = [i*skip_percentage for i in range(frame_samples+1)]
        frames_to_include = [int(p*num_frames) for p in percentages]
        if runtime_stops:
            runtime_stops = runtime_stops[0]+video_times[0], runtime_stops[1]+video_times[0]
            runtime_length = runtime_stops[1] - runtime_stops[0]
            times_to_include = [runtime_stops[0]+p*runtime_length for p in percentages]
            times_to_include.reverse()

    for vid_frame_index in trange(num_frames):
        good, frame = vid_capture.read()
        if not good:
            continue

        if frame_samples and not runtime_stops:
            if vid_frame_index not in frames_to_include:
                continue

        t = video_times[vid_frame_index]
        if t < 0:
            continue
        if runtime_stops:
            if t < times_to_include[-1]:
                continue
            else:
                times_to_include.pop()
                if len(times_to_include) == 0:
                    break

        cam_extrinsic = trajectory.get_relative_orientation(t=t) if bladder_offset else \
            trajectory.get_absolute_orientation(t=t)
        transformed_extrinsic = generate_transform_matrix(cam_extrinsic)
        # transformed_extrinsic = cam_extrinsic
        image_name = f'{vid_frame_index:04d}.jpg'

        try:
            mask = np.logical_not(get_circular_mask_4_img(frame, scale_radius=0.75)).astype(np.uint8)
        except ImageCroppingException as e:
            continue
        mask_name = f'dynamic_mask_{vid_frame_index:04d}.png'

        cv.imwrite(os.path.join(image_directory, mask_name), mask)
        cv.imwrite(os.path.join(image_directory, image_name), frame)
        output_json['frames'].append({'file_path': os.path.join('images', image_name),
                                      # 'sharpness':  31.752987436300323,
                                      'transform_matrix': transformed_extrinsic.tolist()})
        positions.append(transformed_extrinsic[:3, 3])

    positions = np.asarray(positions)
    cg = np.mean(positions, axis=0)
    adjusted_positions = positions - cg
    fig = plt.figure()
    axes: List[plt.Axes] = fig.subplots(3, 1)
    [axes[i].plot(adjusted_positions[:, i]) for i in range(3)]
    fig.show()
    plt.show(block=True)
    output_json['offset'] = (-cg*model_scale + np.asarray(output_json['offset'])).tolist()
    print('Writing trajectory file...')
    with open(os.path.join(output_dir, 'transforms.json'), 'w') as fp:
        json.dump(output_json, fp)
