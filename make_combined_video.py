import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import re
from tqdm.contrib import tzip
from argparse import ArgumentParser
import masking
from masking import get_circular_mask_4_img


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('recording_directory', type=str,
                        help='directory where the video and numpy recording files are. Rendered images should be in '
                             'a subfolder called \'depth_rendering\'.')
    parser.add_argument('--render_skip', type=int,
                        help='render every i-th frame, where i is the value provided. [default=1]', default=1)
    parser.add_argument('--start_frame', type=int, help='which frame to start rendering at. If None, will infer from '
                                                        'file names [default=None] ', default=None)
    parser.add_argument('--mask', type=bool,
                        help='adds a circular mask based on the first frame of the video [default=True]', default=True)
    parser.add_argument('--max_depth', type=float, help='the depth value in meters to clip values to [default=0.1]', default=0.1)

    args = parser.parse_args()

    directory = args.recording_directory
    depth_directory = os.path.join(directory, 'depth_rendering')
    depth_imgs = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'exr']
    images = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'png']
    depth_max = args.max_depth
    skip = args.render_skip
    if args.start_frame is not None:
        frame_start = args.start_frame
    else:
        frame_start = int(re.findall('.(\d+).', os.path.basename(depth_imgs[0]))[0])
    with_mask = args.mask

    cap = cv.VideoCapture(os.path.join(directory, 'video.mp4'))
    fps = int(20 / skip)
    video_codec = cv.VideoWriter_fourcc(*'mp4v')
    depth_writer = cv.VideoWriter(os.path.join(directory, 'video_with_depth.mp4'),
                                  video_codec,
                                  fps, (1920*2, 1080))
    img_writer = cv.VideoWriter(os.path.join(directory, 'video_with_img.mp4'),
                                video_codec,
                                fps, (1920*2, 1080))
    print(f'number of frames: {len(depth_imgs)}')
    good = False
    i = 0
    last_mask = None
    if with_mask:
        while not good and last_mask is None:
            good, frame = cap.read()
            if good:
                good = False
                try:
                    last_mask = np.expand_dims(get_circular_mask_4_img(frame), -1)
                except masking.ImageCroppingException:
                    pass
                continue
        cap.release()
        cap = cv.VideoCapture(os.path.join(directory, 'video.mp4'))
    cap.set(1, frame_start-1)

    for depth_img, image in tzip(depth_imgs, images):
        while not good and i < len(depth_imgs):
            good, frame = cap.read()
            if good and i % skip != 0:
                good = False
                i += 1
                continue
        i += 1
        good = False

        d_img = cv.imread(depth_img, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        d_img = cv.resize(d_img, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_LINEAR)
        d_img = np.where(d_img > depth_max, 255, 255*(d_img/depth_max)).astype(np.uint8)
        if with_mask:
            d_img = np.where(last_mask, d_img, 0)
        d_img_combined = np.concatenate((frame, d_img), axis=1)
        depth_writer.write(d_img_combined)

        img = cv.resize(cv.imread(image, -1), (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_LINEAR)

        if with_mask:
            img = np.where(last_mask, img, 0)
        img_combined = np.concatenate((frame, img), axis=1)
        img_writer.write(img_combined)


    cap.release()
    img_writer.release()
    depth_writer.release()
    cv.destroyAllWindows()





