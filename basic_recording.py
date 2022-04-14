from threading import Thread, Event
from queue import Queue
import cv2 as cv
from typing import *
import natnetclient.natnet
import numpy as np
import time
import os
from datetime import datetime
from distutils.util import strtobool
from argparse import ArgumentParser
from witmotion import IMU, protocol


def record_cam(cam_instance: cv.VideoCapture, time_buffer: np.ndarray) -> None:
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
                            video_fps, frame.shape[:2][::-1])
    time_buffer.resize((int(1e3), 1), refcheck=False)

    def write_video():
        i = 0
        while not stop_writing.is_set() or not q.empty():
            if not q.empty():
                t, f = q.get()
                time_buffer[i] = t
                writer.write(f)
                i += 1
                if i == len(time_buffer):
                    time_buffer.resize((len(time_buffer) + int(1e2), 1), refcheck=False)
                q.task_done()
            else:
                time.sleep(0.001)
        time_buffer.resize((i, *time_buffer.shape[1:]), refcheck=False)

    writing_thread = Thread(target=write_video)
    writing_thread.start()
    video_ready.set()
    [ready_event.wait() for ready_event in ready_events]

    while not stop_requested.is_set():
        _t = time.time()
        good, frame = cam_instance.read()
        if good:
            q.put((_t, frame))

    stop_writing.set()
    writing_thread.join()
    while not stop_requested.is_set():
        time.sleep(0.01)
    writer.release()


def record_imu(imu: IMU,
               data_buffer: np.ndarray) -> None:

    data_buffer.resize((int(1e4), 13), refcheck=False)
    _i = [0]

    def callback_record(msg):
        # if isinstance(msg, protocol.TimeMessage):
        #     self.last_timestamp = msg.timestamp  TODO: Make this work
        if isinstance(msg, protocol.AccelerationMessage):
            data_buffer[_i[0], 0] = time.time()
            data_buffer[_i[0], 1:4] = msg.a
        elif isinstance(msg, protocol.AngularVelocityMessage):
            data_buffer[_i[0], 4:7] = msg.w
        elif isinstance(msg, protocol.AngleMessage):
            data_buffer[_i[0]][7:10] = (msg.roll, msg.pitch, msg.yaw)
        elif isinstance(msg, protocol.MagneticMessage):
            data_buffer[_i[0]][10:13] = msg.mag
            _i[0] = 1 + _i[0]
            if _i[0] >= len(data_buffer):
                data_buffer.resize((_i[0] + int(1e3), *data_buffer.shape[1:]), refcheck=False)
        # elif isinstance(msg, protocol.QuaternionMessage):
        #     self.last_q = msg.q

    imu_ready.set()
    [ready_event.wait() for ready_event in ready_events]
    imu.subscribe(callback_record)
    stop_requested.wait()
    imu.close()
    data_buffer.resize((_i[0], *data_buffer.shape[1:]), refcheck=False)


def record_optitrack(rigid_bodies: List[natnetclient.natnet.RigidBody],
                     bodies_buffer: np.ndarray,
                     marker_buffers: List[np.ndarray],
                     time_buffer: np.ndarray,
                     receive_time_buffer) -> None:
    """
    :param rigid_bodies: a list of the rigid body objects that need to be recorded
    :param bodies_buffer: a numpy array that will contain the position and orientation of each body
    :param marker_buffers: a list of numpy arrays that will contain the corresponding rigid body's marker positions
    :param time_buffer: a numpy array that will contain the time stamps of when the opti-track data was read.
                        WARNING: This is not a thread safe library, so as values for each object are being read, they
                                 may be updated so theoretically may not be 100% synchronized.
    """
    bodies_buffer.resize((int(1e4), len(rigid_bodies), 7), refcheck=False)  # x, y, z, qw, qx, qy, qz
    receive_time_buffer.resize((int(1e4), 1), refcheck=False)
    time_buffer.resize((int(1e4), 1), refcheck=False)

    for i in range(len(marker_buffers)):
        marker_buffers[i].resize((int(1e4), len(rigid_bodies[i].markers), 3), refcheck=False)  # x, y, z

    i = 0
    opti_ready.set()
    [ready_event.wait() for ready_event in ready_events]

    while not stop_requested.is_set():
        receive_time_buffer[i] = time.time()
        time_buffer[i] = opti_client.timestamp

        for j in range(len(rigid_bodies)):
            bodies_buffer[i, j] = *rigid_bodies[j].position, *rigid_bodies[j].quaternion
            marker_buffers[j][i] = [m.position for m in rigid_bodies[j].markers]

        i += 1
        if i == len(bodies_buffer):
            bodies_buffer.resize((i + int(1e3), *bodies_buffer.shape[1:]), refcheck=False)
            receive_time_buffer.resize((i + int(1e3), 1), refcheck=False)
            time_buffer.resize((i + int(1e3), 1), refcheck=False)
            [m.resize((i + int(1e3), len(rigid_bodies[k].markers), 3), refcheck=False) for k, m in enumerate(marker_buffers)]
        while time.time() - receive_time_buffer[i-1] < opti_pause:
            time.sleep(opti_pause/10)
    bodies_buffer.resize((i, *bodies_buffer.shape[1:]), refcheck=False)
    receive_time_buffer.resize((i, 1), refcheck=False)
    time_buffer.resize((i, 1), refcheck=False)
    [m.resize((i, len(rigid_bodies[k].markers), 3), refcheck=False) for k, m in enumerate(marker_buffers)]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('save_directory', type=str,
                        help='The folder location the data will be saved in. The data is saved '
                                                         'in a folder with the timestamp as a name')
    parser.add_argument('--imu', type=lambda x: bool(strtobool(x)),
                        help='Whether or not a WIT Motion IMU is to be used [default=True]',
                        default="True")
    parser.add_argument('--imu_port', type=str, help='The serial port name for the imu [default=\'COM8\']',
                        default='COM8')
    parser.add_argument('--imu_rate', type=int, help='The recording rate in Hz for the imu (max=200) [default=100]',
                        default=100)
    parser.add_argument('--cam', type=lambda x: bool(strtobool(x)),
                        help='Whether or not to record a USB camera signal [default=True]',
                        default="True")
    parser.add_argument('--cam_port', type=int, help='What camera number (windows) to use [default=2]', default=2)
    parser.add_argument('--optitrack', type=lambda x: bool(strtobool(x)),
                        help='whether or not to record an opti-track signal [default=True]', default="True")
    parser.add_argument('--optitrack_rate', type=int, help='the recording rate in Hz for optitrack [default=120]',
                        default=120)
    parser.add_argument('--rigid_bodies', type=List[str],
                        help='list of rigid bodies from optitrack [default=\'bladder endo-front cam\']',
                        default=['bladder', 'endo-front', 'cam'])
    parser.add_argument('--video_fps', type=int, help='assumed fps for video streaming [default=30]', default=30)

    args = parser.parse_args()

    imu = args.imu
    imu_port = args.imu_port
    imu_rate = args.imu_rate
    cam = args.cam
    cam_port = args.cam_port
    optitrack = args.optitrack
    optitrack_rate = args.optitrack_rate
    rigid_body_names = args.rigid_bodies
    save_location = args.save_directory
    video_fps = args.video_fps

    stop_requested = Event()
    video_ready = Event()
    opti_ready = Event()
    imu_ready = Event()

    opti_pause = 1 / optitrack_rate
    video_codec = cv.VideoWriter_fourcc(*'mp4v')
    threads = []
    ready_events = []

    if optitrack:
        try:
            opti_client = natnetclient.natnet.NatClient(client_ip='127.0.0.1', data_port=1511, comm_port=1510)
            rigid_bodies = [opti_client.rigid_bodies[r] for r in rigid_body_names]
        except Exception as e:
            print('Error when connecting to optitrack')
            raise e
        opti_t_buffer = np.zeros(0)
        opti_t_received_buffer = np.zeros(0)
        opti_buffer = np.zeros(0)
        marker_buffers = [np.zeros(0) for _ in rigid_bodies]
        threads.append(Thread(target=record_optitrack, args=[rigid_bodies, opti_buffer, marker_buffers, opti_t_buffer, opti_t_received_buffer]))
        ready_events.append(opti_ready)

    if cam:
        cap = cv.VideoCapture(cam_port, cv.CAP_DSHOW)  # make sure this number is correct for the computer
        if not cap.isOpened():
            raise ValueError(f'Could not open camera on port {cam_port}.')
        vid_t_buffer = np.zeros(0)
        threads.append(Thread(target=record_cam, args=[cap, vid_t_buffer]))
        ready_events.append(video_ready)

    if imu:
        wit_imu = IMU(path='COM6', baudrate=115200)
        wit_imu.set_update_rate(imu_rate)
        imu_data_buffer = np.zeros(0)
        threads.append(Thread(target=record_imu, args=[wit_imu, imu_data_buffer]))
        ready_events.append(imu_ready)

    input("Press enter to start recording. Don't forget to start recording on the optitrack system first!")
    save_location = os.path.join(save_location, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_location)
    save_file = os.path.join(save_location, 'data')

    [t.start() for t in threads]
    [ready_event.wait() for ready_event in ready_events]

    print('recording... have fun!\n')
    input('press enter stop recording.')
    stop_requested.set()
    [t.join() for t in threads]
    data_dictionary = {}

    if cam:
        cap.release()
        cv.destroyAllWindows()
        print(f'number of video frames: {len(vid_t_buffer)}')
        data_dictionary['video_timestamps'] = vid_t_buffer

    if optitrack:
        print(f'number of opti frames: {len(opti_buffer)}')
        data_dictionary['optitrack_timestamps'] = opti_t_buffer
        data_dictionary['optitrack_received_timestamps'] = opti_t_received_buffer
        i = 0
        for rigid_body_name in rigid_body_names:
            data_dictionary[rigid_body_name] = opti_buffer[:, i, :]
            data_dictionary[f'{rigid_body_name}-markers'] = marker_buffers[i]
            i += 1

    if imu:
        print(f'number of imu samples: {len(imu_data_buffer)}')
        data_dictionary['imu'] = imu_data_buffer

    np.savez(file=save_file,
             **data_dictionary)
    print('Data saved!')

