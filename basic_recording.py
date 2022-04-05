from threading import Thread, Event
from queue import Queue
import cv2 as cv
from typing import *
import natnetclient.natnet
import numpy as np
import time
import os

cam_port = 2
optitrack_record_frequency = 120  # Hz
rigid_body_names = ['bladder', 'endo-front', 'cam']
save_location = 'C:\\Users\\Somers\\Desktop\\test_recording'


stop_requested = Event()
video_ready = Event()
opti_ready = Event()
opti_pause = 1/optitrack_record_frequency
video_codec = cv.VideoWriter_fourcc(*'mp4v')


def record_cam(cam_instance: cv.VideoCapture, time_buffer: np.ndarray) ->None:
    """
    :param time_buffer: a numpy array that will contain the timestamps of each video frame when this function returns
    :param cam_instance: an already opened cv.VideoCapture object

    """
    q = Queue()
    stop_writing = Event()
    good, frame = cam_instance.read()
    while not good:
        good, frame = cam_instance.read()
    writer = cv.VideoWriter(os.path.join(save_location, 'video.mp4'),
                            video_codec,
                            20, frame.shape[:2][::-1])
    time_buffer.resize((int(1e3), 1), refcheck=False)

    def write_video():
        i = 0
        while not stop_writing.is_set():
            if not q.empty():
                t, frame = q.get()
                time_buffer[i] = t
                writer.write(frame)
                i += 1
                if i == len(time_buffer):
                    time_buffer.resize((len(time_buffer) + int(1e2), 1), refcheck=False)
                q.task_done()
            else:
                time.sleep(0.001)

    writing_thread = Thread(target=write_video)
    writing_thread.start()
    video_ready.set()
    opti_ready.wait()

    while not stop_requested.is_set():
        good, frame = cam_instance.read()
        if good and frame.any():
            q.put((time.time(), frame))

    stop_writing.set()
    writing_thread.join()
    time_buffer.resize((i, *time_buffer.shape[1:]), refcheck=False)
    while not stop_requested.is_set():
        time.sleep(0.01)
    writer.release()


def record_optitrack(rigid_bodies: List[natnetclient.natnet.RigidBody],
                     bodies_buffer: np.array,
                     marker_buffers: List[np.ndarray],
                     time_buffer: np.ndarray) -> None:
    """
    :param rigid_bodies: a list of the rigid body objects that need to be recorded
    :param bodies_buffer: a numpy array that will contain the position and orientation of each body
    :param marker_buffers: a list of numpy arrays that will contain the corresponding rigid body's marker positions
    :param time_buffer: a numpy array that will contain the time stamps of when the opti-track data was read.
                        WARNING: This is not a thread safe library, so as values for each object are being read, they
                                 may be updated so theoretically may not be 100% synchronized.
    """
    bodies_buffer.resize((int(1e3), len(rigid_bodies), 7), refcheck=False)  # x, y, z, qw, qx, qy, qz
    time_buffer.resize((int(1e3), 1), refcheck=False)
    for i in range(len(marker_buffers)):
        marker_buffers[i].resize((int(1e3), len(rigid_bodies[i].markers), 3), refcheck=False)  # x, y, z

    i = 0
    opti_ready.set()
    video_ready.wait()

    while not stop_requested.is_set():
        time_buffer[i] = time.time()
        for j in range(len(rigid_bodies)):
            bodies_buffer[i, j] = *rigid_bodies[j].position, *rigid_bodies[j].quaternion
            marker_buffers[j][i] = [m.position for m in rigid_bodies[j].markers]

        i += 1
        if i == len(bodies_buffer):
            bodies_buffer.resize((i + int(1e3), *bodies_buffer.shape[1:]), refcheck=False)
            time_buffer.resize((i + int(1e3), 1), refcheck=False)
            [m.resize((i + int(1e3), len(rigid_bodies[k].markers), 3), refcheck=False) for k, m in enumerate(marker_buffers)]
        while time.time() - time_buffer[i-1] < opti_pause:
            time.sleep(opti_pause/10)
    bodies_buffer.resize((i, *bodies_buffer.shape[1:]), refcheck=False)
    time_buffer.resize((i, 1), refcheck=False)
    [m.resize((i, len(rigid_bodies[k].markers), 3), refcheck=False) for k, m in enumerate(marker_buffers)]
    while not stop_requested.is_set():
        time.sleep(0.01)


if __name__ == '__main__':
    try:
        opti_client = natnetclient.natnet.NatClient(client_ip='127.0.0.1', data_port=1511, comm_port=1510)
        rigid_bodies = [opti_client.rigid_bodies[r] for r in rigid_body_names]
    except Exception as e:
        print('Error when connecting to optitrack')
        raise e

    cap = cv.VideoCapture(cam_port, cv.CAP_DSHOW)  # make sure this number is correct for the computer
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    vid_t_buffer = np.zeros(0)
    opti_t_buffer = np.zeros(0)
    opti_buffer = np.zeros(0)
    marker_buffers = [np.zeros(0) for _ in rigid_bodies]
    cam_thread = Thread(target=record_cam, args=[cap, vid_t_buffer])
    opti_thread = Thread(target=record_optitrack, args=[rigid_bodies, opti_buffer, marker_buffers, opti_t_buffer])

    input("press enter to start recording. Don't forget to start recording on the optitrack system first.")
    cam_thread.start()
    opti_thread.start()
    print('recording... have fun!\n')
    input('press enter stop recording.')
    stop_requested.set()
    cam_thread.join()
    opti_thread.join()
    cap.release()
    cv.destroyAllWindows()

    print(f'number of video frames: {len(vid_t_buffer)}')
    print(f'number of opti frames: {len(opti_buffer)}')
    optitrack_data = {'optitrack_timestamps': opti_t_buffer}
    i = 0
    for rigid_body_name in rigid_body_names:
        optitrack_data[rigid_body_name] = opti_buffer[:, i, :]
        optitrack_data[f'{rigid_body_name}-markers'] = marker_buffers[i]
        i += 1
    np.savez(file=os.path.join(save_location, 'data'),
             video_timestamps=vid_t_buffer,
             **optitrack_data)

