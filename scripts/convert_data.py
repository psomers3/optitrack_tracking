import numpy as np
import json
import argparse
import cv2 as cv
import os
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert data from nvidia\'s instant-ngp format to unisurf format')
    parser.add_argument('out', type=str, help='directory to put the converted data')
    parser.add_argument('input_dir', type=str, help='source directory')
    args = parser.parse_args()

    output_dir: str = args.out
    input_dir: str = args.input_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'image'))
        os.makedirs(os.path.join(output_dir, 'mask'))

    with open(os.path.join(input_dir, 'transforms.json'), 'r') as f:
        transforms = json.load(f)

    inv_intrinsic_matrix = np.zeros((4, 4))
    inv_intrinsic_matrix[0, 0] = transforms['fl_x']
    inv_intrinsic_matrix[1, 1] = transforms['fl_y']
    inv_intrinsic_matrix[0, 2] = transforms['cx']
    inv_intrinsic_matrix[1, 2] = transforms['cy']
    inv_intrinsic_matrix[-1, -1] = 1
    inv_intrinsic_matrix[-2, -2] = 1

    intrinsic_matrix = np.divide(1,
                                 inv_intrinsic_matrix,
                                 out=np.zeros_like(inv_intrinsic_matrix),
                                 where=inv_intrinsic_matrix != 0)
    scaling = np.zeros((4, 4))
    scaling[:3, :3] = np.eye(3)*transforms['scale']
    scaling[:3, -1] = np.asarray(transforms['offset'])
    scaling[-1, -1] = 1

    unisurf_arrays = {}

    frames = transforms['frames']
    print('converting matrices and moving images...')
    for i, frame in enumerate(tqdm(frames)):
        unisurf_arrays[f'world_mat_{i}'] = np.asarray(frame['transform_matrix'])
        transform = unisurf_arrays[f'world_mat_{i}']
        unisurf_arrays[f'world_mat_inv_{i}'] = np.divide(1,
                                                         transform,
                                                         out=np.zeros_like(transform),
                                                         where=transform != 0)
        unisurf_arrays[f'scale_mat_{i}'] = scaling
        unisurf_arrays[f'scale_mat_inv_{i}'] = np.divide(1,
                                                         scaling,
                                                         out=np.zeros_like(scaling),
                                                         where=scaling != 0)
        unisurf_arrays[f'camera_mat_{i}'] = intrinsic_matrix
        unisurf_arrays[f'camera_mat_inv_{i}'] = inv_intrinsic_matrix
        img_path = os.path.join(input_dir, frame['file_path'])
        img = cv.imread(img_path)
        cv.imwrite(os.path.join(output_dir, 'image', f'{i:06d}.png'), img)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(os.path.dirname(img_path), f'dynamic_mask_{img_name}.png')
        img = cv.imread(mask_path)

        cv.imwrite(os.path.join(output_dir, 'mask', f'{i:03d}.png'), img)

    np.savez(os.path.join(output_dir, 'cameras'), **unisurf_arrays)
    print('Done!')
