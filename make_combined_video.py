import numpy as np
import cv2 as cv
import os
import re

directory = r'C:\Users\Somers\Desktop\test_recording'
depth_directory = os.path.join(directory, 'depth_rendering')
depth_imgs = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'exr']
images = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda k: re.findall('.(\d+).', k)) if x[-3:] == 'png']
depth_max = .200  # meters

cap = cv.VideoCapture(os.path.join(directory, 'video.mp4'))
fps = 30
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
for depth_img, image in zip(depth_imgs, images):
    while not good:
        good, frame = cap.read()
    good = False
    d_img = cv.imread(depth_img, -1)
    d_img = np.where(d_img > depth_max, 255, 255*(d_img/depth_max)).astype(np.uint8)
    d_img_combined = np.concatenate((frame, d_img), axis=1)
    depth_writer.write(d_img_combined)

    img = cv.imread(image)
    img_combined = np.concatenate((frame, img), axis=1)
    img_writer.write(img_combined)


cap.release()
img_writer.release()
depth_writer.release()
cv.destroyAllWindows()





