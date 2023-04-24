"""
    Visualize scene camera extrinsics
"""

import json
import os
import sys
from copy import deepcopy

import cv2 as cv
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3
import natsort
import numpy as np
import tqdm
from cam_viz import Z_ROT, CameraPoseVisualizer


def folder_path(path: str, comment: str = ""):
    if not os.path.exists(path):
        if comment: print(comment)
        os.makedirs(path)
    return path

def load_from_single_file(path: str):
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
                # c2w[0:3, 1:3] *= -1
                # c2w = c2w[np.array([1, 0, 2, 3]), :]
                # c2w[2, :] *= -1
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
    H, W = shape
    fx, fy = K[0, 0], K[1, 1]
    return 2. * np.arctan2(W, 2. * fx), 2. * np.arctan2(H, 2. * fy)

def export_json(all_exts, all_ints, names, input_scene, scale, img_ext = 'jpg'):
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
    for name in names:
        index = int(name[:name.rfind("_")])
        transform = all_exts[index]
        # transform[:3, :] = Z_ROT @ transform[:3, :]
        if index not in check_set:
            frame = {"transform_matrix": transform.tolist()}
            test_file["frames"].append(frame)
        else:
            train_cnt += 1
            print(f"Saving {name} to {train_cnt:05d}")
            frame = {"file_path": f"{image_pos}/frame_{train_cnt:05d}.jpg", "transform_matrix": transform.tolist(), "original_name": name}
            train_file["frames"].append(frame)
            
    print(f"Training set: { len(train_file['frames']) } images. Test set: { len(test_file['frames']) } images.")
    with open(os.path.join(input_scene, "train.json"), "w") as output:
        json.dump(train_file, output, indent=4)
    with open(os.path.join(input_scene, "test.json"), "w") as output:
        json.dump(test_file, output, indent=4)
    

if __name__ == '__main__':
    input_scene = sys.argv[1]
    scale = 1.0
    if len(sys.argv) > 2:
        scale = float(sys.argv[2])
    all_exts, all_ints, names = input_extrinsics(os.path.join(input_scene, "cams"))
    export_json(all_exts, all_ints, names, input_scene, scale)
    mins = all_exts[:, :-1, -1].min(axis = 0) - 0.5
    maxs = all_exts[:, :-1, -1].max(axis = 0) + 0.5

    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]])
    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    visualizer.patch_transform(all_exts, 0.6, 0.3)
    visualizer.customize_legend([f'pose {i}' for i in range(all_exts.shape[0])])
    visualizer.show()