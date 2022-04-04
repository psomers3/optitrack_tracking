import numpy as np
import cv2 as cv
import os

directory = r'C:\Users\Somers\Desktop\test_recording'
depth_directory = os.path.join(directory, 'depth_rendering')
depth_imgs = [os.path.join(depth_directory, x) for x in sorted(os.listdir(depth_directory), key=lambda x: float(x[:-5]))]

cap = cv.VideoCapture(os.path.join(directory, 'video.mp4'))
fps = 30
video_codec = cv.VideoWriter_fourcc(*'mp4v')
writer = cv.VideoWriter(os.path.join(directory, 'video_with_depth.mp4'),
                        video_codec,
                        fps, (1920*2, 1080))
print(f'number of frames: {len(depth_imgs)}')
good = False
i = 0
for depth_img in depth_imgs:
    while not good:
        good, frame = cap.read()
    good = False
    depth = cv.imread(depth_img)
    combined = np.concatenate((frame, depth), axis=1)
    writer.write(combined)

cap.release()
writer.release()
cv.destroyAllWindows()





