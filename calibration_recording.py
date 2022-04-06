from threading import Thread, Event
from queue import Queue
import cv2 as cv
import time
import os

cam_port = 2
record_as_images = True
record_frequency = 5  # Hz
save_location = 'C:\\Users\\Somers\\Desktop\\test_recording3'

stop_requested = Event()
pause_length = 1/record_frequency
video_codec = cv.VideoWriter_fourcc(*'mp4v')


def record_cam(cam_instance: cv.VideoCapture) -> None:
    """
    :param cam_instance: an already opened cv.VideoCapture object

    """
    if not os.path.exists(os.path.join(save_location, 'calibration')):
        os.makedirs(os.path.join(save_location, 'calibration'))
    q = Queue()
    stop_writing = Event()
    good, frame = cam_instance.read()
    while not good:
        good, frame = cam_instance.read()
    if not record_as_images:
        writer = cv.VideoWriter(os.path.join(save_location, 'calibration', 'video.mp4'),
                                video_codec,
                                20, frame.shape[:2][::-1])

    def write_video():
        last_t = 0
        while not stop_writing.is_set():
            if not q.empty():
                t, f = q.get()
                if t - last_t > pause_length:
                    last_t = t
                    if record_as_images:
                        cv.imwrite(os.path.join(save_location, 'calibration', f'{t:0.4f}.jpg'), f)
                    else:
                        writer.write(f)
                q.task_done()
            else:
                time.sleep(0.001)

    writing_thread = Thread(target=write_video)
    writing_thread.start()

    while not stop_requested.is_set():
        good, frame = cam_instance.read()
        if good:
            q.put((time.time(), frame))

    stop_writing.set()
    writing_thread.join()
    while not stop_requested.is_set():
        time.sleep(0.01)
    if not record_as_images:
        writer.release()


if __name__ == '__main__':
    cap = cv.VideoCapture(cam_port, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    cam_thread = Thread(target=record_cam, args=[cap])
    input("press enter to start recording.")
    cam_thread.start()
    print('recording... have fun!\n')
    input('press enter stop recording.')
    stop_requested.set()
    cam_thread.join()
    cap.release()
    cv.destroyAllWindows()

