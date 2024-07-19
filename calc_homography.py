import numpy as np
import cv2
import json

def calc_homography():
    pts_tv = np.array([
        [1584, 698, 1],
        [1804, 704, 1],
        [1498, 981, 1],
        [1752, 971, 1]
    ], dtype=np.float32)
    pts_pv = np.array([
        [864, 332, 1],
        [1359, 572, 1],
        [138, 589, 1],
        [534, 1030, 1]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pts_pv, pts_tv)
    return H

def save_json(fname, d):
    with open(fname, "w") as f:
        json.dump(d, f)


if __name__ == '__main__':
    H = calc_homography()
    save_json('homography.json', H.tolist())

