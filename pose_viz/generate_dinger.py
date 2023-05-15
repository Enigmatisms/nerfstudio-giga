"""from MVS dataset to nerfstudio json file."""
import json
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import natsort
import numpy as np


def load_from_single_file(path: str):
    """Load camera from single file"""
    with open(path, "r") as f:
        line = f.readline()
        extr = None
        intr = None
        while line:
            if line.startswith("extrinsic"):
                result = []
                for _ in range(4):
                    line = f.readline()
                    digits = line[:-1].strip().split(" ")
                    result.append([float(digit) for digit in digits])
                w2c = np.float32(result)
                c2w = np.linalg.inv(w2c)
                # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
                c2w[0:3, 1:3] *= -1
                c2w = c2w[np.array([1, 0, 2, 3]), :]
                c2w[2, :] *= -1
                extr = c2w
            elif line.startswith("intrinsic"):
                result = []
                for _ in range(3):
                    line = f.readline()
                    digits = line[:-1].strip().split(" ")
                    result.append([float(digit) for digit in digits])
                intr = np.float32(result)
            line = f.readline()
        return extr, intr


def input_extrinsics(input_path: str):
    """Read all extrinsics"""
    img_names = filter(lambda x: x.endswith("txt"), os.listdir(input_path))
    all_cams = list(img_names)
    total_imgs = natsort.natsorted(all_cams)
    all_exts = []
    all_ints = []
    for name in total_imgs:
        extrinsic, intrinsic = load_from_single_file(os.path.join(input_path, name))
        all_exts.append(extrinsic)
        all_ints.append(intrinsic)
    print(f"{len(all_exts)} camera poses in total.")
    return np.float32(all_exts), np.float32(all_ints), total_imgs


def extract_focal(K: np.ndarray, img: np.ndarray):
    H, W, _ = img.shape
    fx, fy = K[0, 0], K[1, 1]
    return 2.0 * np.arctan2(W, 2.0 * fx), 2.0 * np.arctan2(W, 2.0 * fy)


def export_json(all_exts, all_ints, names, input_scene, img_ext="jpg"):
    image_folder = os.path.join(input_scene, "images")
    img_names = filter(lambda x: x.endswith(img_ext), os.listdir(image_folder))
    all_imgs = list(img_names)

    example_int = all_ints[0]
    example_img = plt.imread(os.path.join(image_folder, all_imgs[0]))
    camera_height, camera_width, _ = example_img.shape
    # fov_x, fov_y = extract_focal(example_int, example_img)
    # print(f"FOV x: {fov_x:.4f}, FOV y: {fov_y:.4f}")

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    applied_transform[2, :] *= -1

    train_file = {
        "w": camera_width,
        "h": camera_height,
        "fl_x": float(example_int[0, 0]),
        "fl_y": float(example_int[1, 1]),
        "cx": float(example_int[0, 2]),
        "cy": float(example_int[1, 2]),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_model": "OPENCV",
        "applied_transform": applied_transform.tolist(),
        "frames": [],
    }

    test_file = deepcopy(train_file)

    check_set = set(int(name[:-4]) for name in all_imgs)
    for name in names:
        index = int(name[: name.rfind("_")])
        if index not in check_set:
            frame = {"transform_matrix": all_exts[index].tolist(), "colmap_im_id": index}
            test_file["frames"].append(frame)
        else:
            frame = {
                "file_path": f"./images/{index:08d}.jpg",
                "transform_matrix": all_exts[index].tolist(),
                "colmap_im_id": index,
            }
            train_file["frames"].append(frame)

    print(f"Training set: { len(train_file['frames']) } images. Test set: { len(test_file['frames']) } images.")
    with open(os.path.join(input_scene, "transforms_dinger.json"), "w") as output:
        json.dump(train_file, output, indent=4)
    with open(os.path.join(input_scene, "test_dinger.json"), "w") as output:
        json.dump(test_file, output, indent=4)


if __name__ == "__main__":
    input_scene = sys.argv[1]
    all_exts, all_ints, names = input_extrinsics(os.path.join(input_scene, "cams"))
    export_json(all_exts, all_ints, names, input_scene)
