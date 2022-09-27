import numpy as np
import os
from PIL import Image
import cv2
import random


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def draw_circles(img, circles):
    res = np.copy(img)
    for (x, y, r) in circles[:5]:
        cv2.circle(res, (x, y), r, (0, 255, 0), 4)
    return res


def get_circular_mask_4_img(img: np.ndarray, scale_radius: float = 1.0) -> np.ndarray:
    """
    Returns a mask of the same size as img with a circular mask of 1 where the endoscopic image is.

    :param img: endoscopic image
    :param scale_radius: optional resize of the found circular mask.
    :raise ImageCroppingException:  when a proper circle can't be found.
    :return: a boolean array with True values within the circular mask
    """
    gray_img = rgb2gray(img)
    ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    img_height = thresh_img.shape[0]
    img_width = thresh_img.shape[1]
    points = []
    # go through lines with small steps to find points that may be the contour of the circle
    for y in range(0, img_height, 5):
        # get index pairs for consecutive runs of True values
        idx_pairs = np.where(np.diff(np.hstack(([False], thresh_img[y] == 255, [False]))))[0].reshape(-1, 2)
        # assert that runs have been found
        if len(idx_pairs) == 0:
            continue
        run_lengths = np.diff(idx_pairs, axis=1)
        # assert that there is only one "long" run
        if len(idx_pairs) > 1:
            if np.sort(run_lengths)[1] > 20:
                continue
        x1, x2 = idx_pairs[run_lengths.argmax()]  # Longest island
        run_length = x2 - x1
        # filter out short runs like for text etc.
        if run_length < 0.2 * img_width:
            continue
        points = points + [(x1, y), (x2, y)]
    n_samples = 15
    if len(points) < n_samples * 3:
        raise ImageCroppingException(img, "Not enough samples to process frame")
    max_circle = get_biggest_circle(points, n_samples)
    return create_circular_mask(h=img_height, w=img_width, center=max_circle[0:2], radius=max_circle[-1] * scale_radius)


def squarify(M, pad_constant):
    (a, b) = M.shape[0:2]
    if a > b:
        pad_val = a - b
        pad_left = pad_val // 2
        pad_right = pad_val - pad_left
        padding = (0, 0), (pad_left, pad_right), (0, 0)
    else:
        pad_val = b - a
        pad_top = pad_val // 2
        pad_bot = pad_val - pad_top
        padding = (pad_top, pad_bot), (0, 0), (0, 0)

    if len(M.shape) < 3:
        padding = padding[0:2]
    return np.pad(M, tuple(padding), mode='constant', constant_values=pad_constant)


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return None

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    return [cx, cy, radius]


def create_circular_mask(h, w, center=None, radius=None):
    """ https://stackoverflow.com/a/44874588 """
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def reject_outliers(data, m=2):
    d = np.abs(data - np.median(data, axis=0))
    mdev = np.median(d, axis=0)
    s = np.divide(d, mdev, out=np.zeros_like(data), where=mdev != 0)
    return np.all(s < m, axis=1)


def get_biggest_circle(points, n_samples=20):
    n_points = 3 * n_samples
    samples = random.sample(points, n_points)
    samples = np.reshape(samples, newshape=(n_samples, 3, 2))
    circles = []
    for point_set in samples:
        circle = define_circle(*point_set)
        if circle is not None:
            circles.append(circle)
    circles = np.array(circles)
    circles = circles[reject_outliers(circles)]
    max_circle_idx = np.argmax(circles[:, 2])
    max_circle = circles[max_circle_idx, :].astype(int)
    return max_circle


def save_frame(frame, directory, file_name, size=None, uint8=False):
    pil_img = Image.fromarray(frame)
    if uint8:
        pil_img = pil_img.convert('L')
    if size is not None:
        pil_img = pil_img.resize((size, size))
    filename = f"{os.path.splitext(file_name)[0]}.png"
    pil_img.save(os.path.join(directory, filename))


class ImageCroppingException(Exception):
    """Raised when an edoscop image can not be cropped"""

    def __init__(self, img, message="Image cropping failed"):
        og_height = img.shape[0]
        og_width = img.shape[1]
        width = 200
        height = int(og_height / og_width * width)
        self.img = cv2.resize(img, (width, height), cv2.INTER_AREA)
        self.message = message
        super().__init__(self.message)