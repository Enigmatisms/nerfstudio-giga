"""
    Visualize scene camera extrinsics
"""

import json
import os
import sys
from copy import deepcopy

import configargparse
import cv2 as cv
import matplotlib.pyplot as plt
import natsort
import numpy as np
import tqdm

def folder_path(path: str, comment: str = ""):
    """Make valid folder"""
    if not os.path.exists(path):
        if comment: print(comment)
        os.makedirs(path)
    return path

def load_from_single_file(path: str):
    """Load GIGAMVS camera param"""
    with open(path, 'r') as f:
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
    """Load GIGAMVS camera intrinsics"""
    img_names = filter(lambda x: x.endswith("txt"), os.listdir(input_path))
    all_cams = [name for name in img_names]
    total_imgs = natsort.natsorted(all_cams)
    all_exts = []
    all_ints = []
    for name in total_imgs:
        extrinsic, intrinsic = load_from_single_file(os.path.join(input_path, name))
        all_exts.append(extrinsic)
        all_ints.append(intrinsic)
    print(f"{len(all_exts)} camera poses in total.")
    return np.float32(all_exts), np.float32(all_ints), total_imgs

def extract_focal(shape: tuple, K: np.ndarray):
    """Extract focal from camera intrinsics"""
    H, W = shape
    fx, fy = K[0, 0], K[1, 1]
    return 2. * np.arctan2(W, 2. * fx), 2. * np.arctan2(H, 2. * fy)

def export_json(all_exts, all_ints, names, input_scene, scale, img_ext = 'jpg', merge_train_test = False, scene_name = None):
    """Output NeRF json scene file
        - merge_train_test: if True, train.json will have test poses (with file_path = test_view)
    """
    image_folder = os.path.join(input_scene, "images")
    img_names = filter(lambda x: x.endswith(img_ext), os.listdir(image_folder))
    all_imgs = [name for name in img_names]

    example_img = plt.imread(os.path.join(image_folder, all_imgs[0]))
    example_int = all_ints[0]

    check_set = set(int(name[:-4]) for name in all_imgs)
    H, W, _ = example_img.shape
    scaled_h, scaled_w = H, W
    if scale < 9.9e-1:
        scaled_h, scaled_w = int(H * scale), int(W * scale)
        num_images = len(all_imgs)
        output_folder = folder_path(os.path.join(input_scene, "image_scaled"))
        print(f"Scale image from {(H, W)} to {(scaled_h, scaled_w)}")
        print(f"Estimated GPU memory after scaling: {scaled_w * scaled_h * 6 * num_images / (1024 ** 3):.3f} GB for {num_images} images.")
        for img_name in tqdm.tqdm(all_imgs):
            img = cv.imread(os.path.join(image_folder, img_name))
            img = cv.resize(img, (scaled_w, scaled_h))
            cv.imwrite(os.path.join(output_folder, img_name), img)
        print("Scaled images output folder 'image_scaled/'")
    else:
        print(f"No scaling, image shape {(H, W)}")

    fov_x, fov_y = extract_focal((H, W), example_int)
    print(f"FOV x: {fov_x:.4f}, FOV y: {fov_y:.4f}")
    
    train_file = {"camera_angle_x": fov_x, "camera_angle_y": fov_y, 
        "aabb_scale": 2.0, "scale": 0.05, "fl_x": example_int[0, 0].item() * scale,
        "fl_y": example_int[1, 1].item() * scale,
        "cx": scaled_w / 2. ,
        "cy": scaled_h / 2.,
        "w": scaled_w,
        "h": scaled_h,
        "frames": [], 
    }
    test_file = deepcopy(train_file)

    image_pos = 'images' if scale > 9.9e-1 else 'image_scaled'
    train_cnt = 0
    merged_test_view = 0
    for name in names:
        index = int(name[:name.rfind("_")])
        if scene_name == "Library" and index == 58:
            continue
        transform = all_exts[index]
        # transform[:3, :] = Z_ROT @ transform[:3, :]
        if index not in check_set:
            frame = {"transform_matrix": transform.tolist(), "original_name": name}
            test_file["frames"].append(frame)
            if merge_train_test:        # for test view occlusion loss
                frame = {"file_path": "test_view", "transform_matrix": transform.tolist()}
                train_file["frames"].append(frame)
                merged_test_view += 1
        else:
            train_cnt += 1
            frame = {"file_path": f"{image_pos}/frame_{train_cnt:05d}.jpg", "transform_matrix": transform.tolist(), "original_name": name}
            train_file["frames"].append(frame)
            
    print(f"Training set: { train_cnt } images. Merged test view: {merged_test_view}. Test set: { len(test_file['frames']) } images.")
    output_train_file = "train_merged.json" if merge_train_test else "train.json"
    with open(os.path.join(input_scene, output_train_file), "w", encoding = 'utf-8') as output:
        json.dump(train_file, output, indent=4)
    with open(os.path.join(input_scene, "test.json"), "w", encoding = 'utf-8') as output:
        json.dump(test_file, output, indent=4)

def parser_opts():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("-i", "--input_scene",  required = True, help = "Input scene file", type = str)
    parser.add_argument("-n", "--name",         required = True, help = "Name of the scene", type = str)
    parser.add_argument("-s", "--scale",        default = 1.0, help = "Scaling of the scene", type = float)
    parser.add_argument("-m", "--merge",        default = False, action = "store_true", help = "whether to merge train / test dataset")
    return parser.parse_args()

if __name__ == '__main__':
    opts = parser_opts()
    all_exts, all_ints, names = input_extrinsics(os.path.join(opts.input_scene, "cams"))
    export_json(all_exts, all_ints, names, opts.input_scene, opts.scale, merge_train_test = opts.merge, scene_name = opts.name)
