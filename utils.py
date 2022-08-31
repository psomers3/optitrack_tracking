import os
import numpy as np
import cv2 as cv
from tqdm import trange
import json
from typing import *
import shutil

from endoscope_trajectory import EndoscopeTrajectory
from optitrack_tools.endoscope_definitions import Endoscope, ENDOSCOPES


def _path_create(path: str):
    """
    create path if it doesn't exist
    :param path: directory to create
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def save_images_and_traj_for_ngp(input_directory: str,
                                 output_dir: str,
                                 endoscope: Endoscope = ENDOSCOPES.TUEBINGEN,
                                 video_time_delay: float = 0.1,
                                 invert_cam_rotation: bool = False,
                                 frame_samples: int = None,
                                 runtime_stops: Tuple[float, float] = None,
                                 aabb_scaling: int = 1) -> None:
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
    """
    if not os.path.exists(input_directory):
        raise FileNotFoundError(input_directory)
    _path_create(output_dir)
    image_directory = os.path.join(output_dir, 'images')
    _path_create(image_directory)
    data = np.load(os.path.join(input_directory, 'data.npz'))

    trajectory = EndoscopeTrajectory(data, invert_cam_rotation=invert_cam_rotation, relative_trajectory=True, endoscope=endoscope)
    video_times = np.squeeze(data['video_timestamps'] - data['optitrack_received_timestamps'][0] - video_time_delay)
    vid_capture = cv.VideoCapture(os.path.join(input_directory, 'video.mp4'))
    num_frames = int(vid_capture.get(cv.CAP_PROP_FRAME_COUNT))
    positions = []

    with open(os.path.join(input_directory, 'cam_params.json')) as f:
        camera_params = json.load(f)
    output_json = {"camera_angle_x": 0.7481849417937728,
                   "camera_angle_y": 1.2193576119562444,
                   "fl_x": camera_params['FocalLength'][0],
                   "fl_y": camera_params['FocalLength'][1],
                   "k1": camera_params['RadialDistortion'][0],
                   "k2": camera_params['RadialDistortion'][1],
                   "p1": camera_params['TangentialDistortion'][0],
                   "p2": camera_params['TangentialDistortion'][1],
                   "cx": camera_params['PrincipalPoint'][0],
                   "cy": camera_params['PrincipalPoint'][1],
                   "w": camera_params['ImageSize'][0],
                   "h": camera_params['ImageSize'][1],
                   "aabb_scale": aabb_scaling,
                   "scale": .50,
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

        cam_extrinsic = trajectory.get_absolute_orientation(t=t)
        image_name = f'{vid_frame_index:04d}.jpg'
        cv.imwrite(os.path.join(image_directory, image_name), frame)
        output_json['frames'].append({'file_path': os.path.join('images', image_name),
                                      'sharpness':  31.752987436300323,
                                      'transform_matrix': cam_extrinsic.tolist()})
        positions.append(cam_extrinsic[:3, 3])

    positions = np.asarray(positions)
    print('Writing trajectory file...')
    with open(os.path.join(output_dir, 'transforms.json'), 'w') as fp:
        json.dump(output_json, fp)
