import numpy as np
import cv2 as cv
import os
import re
from tqdm.contrib import tzip

import masking
from masking import get_circular_mask_4_img

directory = r'C:\Users\Somers\Desktop\test_recording3'
depth_directory = os.path.join(directory, 'depth_rendering')
depth_imgs = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'exr']
images = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'png']
depth_max = .1  # meters
skip = 1
frame_start = 475
with_mask = True

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

for depth_img, image in tzip(depth_imgs, images):
    while not good and i < (len(depth_imgs) + frame_start - 1):
        good, frame = cap.read()
        if (good and i % skip != 0) or (good and i < frame_start):
            good = False
            i += 1
            continue
    i += 1
    good = False
    d_img = cv.imread(depth_img, -1)
    d_img = np.where(d_img > depth_max, 255, 255*(d_img/depth_max)).astype(np.uint8)
    if with_mask:
        d_img = np.where(last_mask, d_img, 0)
    d_img_combined = np.concatenate((frame, d_img), axis=1)
    depth_writer.write(d_img_combined)

    img = cv.imread(image)
    if with_mask:
        img = np.where(last_mask, img, 0)
    img_combined = np.concatenate((frame, img), axis=1)
    img_writer.write(img_combined)


cap.release()
img_writer.release()
depth_writer.release()
cv.destroyAllWindows()





