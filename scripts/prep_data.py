from isys_optitrack.optitrack_tools.utils import save_images_and_traj_for_ngp
from typing import *
from isys_optitrack import ENDOSCOPES, Endoscope
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory', type=str, help='Directory with optitrack recorded data')
    parser.add_argument('--out', type=str, help='Directory to output the images and trajectory file')
    parser.add_argument('--endo', type=str, choices=[k for k in ENDOSCOPES.__dict__.keys() if '__' not in k],
                        help='Which endoscope was used to make the recording.')
    parser.add_argument("--delay", type=float, default=0.1, help='Time delay between video and optitrack recordings.'
                                                                 'default = 0.1 sec')
    parser.add_argument('--invert_cam', action='store_true', help='Invert the relative direction between camera and '
                                                                  'endoscope.')
    parser.add_argument('--samples', type=int, help='Only use this number of samples. Runtime is split evenly')
    parser.add_argument('--stops', type=float, nargs=2, help='Video range in seconds to use. default uses whole video')
    parser.add_argument('--aabb', type=int, help='aabb scaling for instant-ngp. This should be a power of 2. default=1')
    parser.add_argument('--scale', type=float, help="scaling factor for entire model. default=0.33", default=0.33)
    parser.add_argument('--relative', action='store_true', help='subtract the bladder mold trajectory from'
                                                                ' the measured one.')
    args = parser.parse_args()

    directory: str = args.directory
    output_directory: str = args.out
    endoscope_used: Endoscope = ENDOSCOPES.__dict__[args.endo]
    video_time_delay: float = args.delay
    invert_cam: bool = bool(args.invert_cam)
    frame_samples: int = args.samples
    runtime_stops: Tuple[float] = tuple(args.stops) if args.stops else None
    aabb_scaling: int = args.aabb
    model_scale: float = args.scale
    bladder_offset: bool = bool(args.relative)
    save_images_and_traj_for_ngp(input_directory=directory,
                                 output_dir=output_directory,
                                 endoscope=endoscope_used,
                                 video_time_delay=video_time_delay,
                                 invert_cam_rotation=invert_cam,
                                 frame_samples=frame_samples,
                                 runtime_stops=runtime_stops,
                                 aabb_scaling=aabb_scaling,
                                 model_scale=model_scale,
                                 bladder_offset=bladder_offset)
