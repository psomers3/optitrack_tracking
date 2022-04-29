import open3d as o3d
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import re
from tqdm.contrib import tzip
from argparse import ArgumentParser
import masking
from masking import get_circular_mask_4_img
import json
from trajectory import EndoscopeTrajectory, invert_affine_transform

vis = o3d.visualization.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('recording_directory', type=str,
                        help='directory where the video and numpy recording files are. Rendered images should be in '
                             'a subfolder called \'depth_rendering\'.')
    parser.add_argument('--render_skip', type=int,
                        help='render every i-th frame, where i is the value provided. [default=1]', default=1)
    parser.add_argument('--start_frame', type=int, help='which frame to start rendering at. If None, will infer from '
                                                        'file names [default=None] ', default=None)
    parser.add_argument('--video_delay', type=float,
                        help='a delay to help synchronize the video and movement. [default=0.1]', default=0.1)
    args = parser.parse_args()

    video_time_delay = args.video_delay
    directory = args.recording_directory
    depth_directory = os.path.join(directory, 'depth_rendering')
    data = np.load(os.path.join(directory, 'data.npz'))
    trajectory = EndoscopeTrajectory(data, invert_cam_rotation=True)
    video_times = np.squeeze(data['video_timestamps'] - data['optitrack_received_timestamps'][0] - video_time_delay)

    depth_imgs = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'exr']
    images = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'png']
    skip = args.render_skip
    if args.start_frame is not None:
        frame_start = args.start_frame
    else:
        frame_start = int(re.findall('.(\d+).', os.path.basename(depth_imgs[0]))[0])

    cap = cv.VideoCapture(os.path.join(directory, 'video.mp4'))

    print(f'number of frames: {len(depth_imgs)}')
    good = False
    i = frame_start - 1
    mask = None
    frame = None

    while not good and mask is None:
        good, frame = cap.read()
        if good:
            good = False
            try:
                mask = get_circular_mask_4_img(frame)
            except masking.ImageCroppingException:
                pass
            continue
    cap.release()
    cap = cv.VideoCapture(os.path.join(directory, 'video.mp4'))
    cap.set(1, frame_start-1)

    with open(os.path.join(directory, 'cam_params.json')) as f:
        camera_params = json.load(f)
    blender_traj = np.load(os.path.join(directory, 'camera_trajectory.npz'))['trajectory']

    intrinsic_matrix = np.asarray(camera_params['IntrinsicMatrix']).T
    print(intrinsic_matrix)
    intrinsic_cam = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_cam.intrinsic_matrix = intrinsic_matrix
    intrinsic_cam.width = frame.shape[1]
    intrinsic_cam.height = frame.shape[0]
    op3_cam = o3d.camera.PinholeCameraParameters()
    op3_cam.intrinsic = intrinsic_cam

    point_cloud = o3d.geometry.PointCloud()
    flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.001,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for depth_file, image in tzip(depth_imgs, images):
        while not good:
            good, frame = cap.read()
            if good and i % skip != 0:
                good = False
                i += 1
                continue
        t = video_times[i]
        if t < 0:
            continue

        if i % 5 == 0:
            extrinsic_matrix = trajectory.get_extrinsic_matrix(t=t)

            d_img = cv.imread(depth_file, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            d_img = cv.cvtColor(d_img, cv.COLOR_RGB2GRAY)
            d_img = cv.resize(d_img, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_LINEAR)
            d_img = np.where(mask, d_img, 1e5).astype(np.float32)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(frame),
                                                                      depth=o3d.geometry.Image(d_img),
                                                                      depth_trunc=1e4,
                                                                      depth_scale=1,
                                                                      convert_rgb_to_intensity=False)
            volume.integrate(image=rgbd,
                             intrinsic=op3_cam.intrinsic,
                             extrinsic=invert_affine_transform(extrinsic_matrix@flip))
        i += 1
        good = False
    vis.add_geometry(volume.extract_point_cloud())
    vis.run()
    vis.destroy_window()
    cap.release()
    cv.destroyAllWindows()





