"""
    Camera optimization dataset generation
    @author: Qianyue He
    @date: 2023-5-1
"""

import json
import os

import configargparse
import cv2 as cv
import natsort
import numpy as np
import tqdm


def parser_opts():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("-i", "--input_scene",      required = True, help = "Input scene", type = str)
    parser.add_argument("-m", "--mode",             required = True, default = "new", choices = ["none", "new", "no_skew"], 
                        help = "Input modes to choose from.", type = str)
    parser.add_argument("-p", "--input_pose_path",  default = "../../dataset/images_and_cams/full/pose_align/", 
                        help = "Input pose and camera intrinsics file.", type = str)
    parser.add_argument("-o", "--output_name",      default = "transforms_opt.json", help = "Output file name", type = str)
    parser.add_argument("--input_img_path",         default = "../renders/", help = "Input image file path", type = str)
    parser.add_argument("--output_path",            default = "../../dataset/images_and_cams/full/", 
                        help = "Where the images and transform.json are stored", type = str)

    parser.add_argument("-s", "--scale",            default = 1.0, help = "Scale the image (originally, 1/8 resolution)", type = float)
    parser.add_argument("-f", "--filter",           default = False, action = "store_true", help = "Do bilateral filtering")
    return parser.parse_args()

def get_images(img_path: str, opts: configargparse.Namespace):
    all_names = os.listdir(img_path)
    all_names = natsort.natsorted(all_names)
    image_infos = {}
    print("Loading images:")
    for i, name in enumerate(tqdm.tqdm(all_names)):
        image_id = int(name[:-4])
        dict_key = f"{image_id:08d}_cam.txt"
        new_name = f"frame_{i + 1:05d}.jpg"
        file_path = os.path.join(img_path, name)
        img = cv.imread(file_path)
        if abs(opts.scale - 1) > 1e-5:
            h, w, _ = img.shape
            scaled_h, scaled_w = int(h * opts.scale), int(w * opts.scale)
            img = cv.resize(img, (scaled_w, scaled_h))
        if opts.filter:
            original = img.copy().astype(np.float32)
            img = cv.bilateralFilter(original, 7, 10, 50)
        image_infos[dict_key] = (new_name, img)
    return image_infos

if __name__ == "__main__":
    opts = parser_opts()
    img_path = os.path.join(opts.input_img_path, opts.input_scene)
    image_infos = get_images(img_path, opts)

    mode_mapping = {"none": "", "new": "_new", "no_skew": "_no_skew"}
    test_json = os.path.join(opts.input_pose_path, opts.input_scene, f"output{mode_mapping[opts.mode]}.json")
    with open(test_json, "r") as file:
        test_data = json.load(file)
    
    output_folder = os.path.join(opts.output_path, f"{opts.input_scene}_opt")
    image_folder = os.path.join(output_folder, "images")
    if not os.path.exists(output_folder):
        # recursive folder making
        os.makedirs(image_folder)

    posed_frames = {frame["original_name"]:frame["camera_to_world"] for frame in test_data["camera_path"]}
    example_frame = test_data["camera_path"][0]
    output_json = {
        "w": test_data["render_width"],
        "h": test_data["render_height"],
        "fl_x": example_frame["fx"],
        "fl_y": example_frame["fy"],
        "cx": test_data["render_width"] / 2,
        "cy": test_data["render_height"] / 2,
        "k1": example_frame["k1"],
        "k2": example_frame["k2"],
        "p1": example_frame["p1"],
        "p2": example_frame["p2"],
        "camera_model": "OPENCV",
        "frames": []
    }
    
    if len(image_infos) != len(posed_frames):
        raise ValueError(f"Pose and images can not form bi-jection. Pose num: {len(posed_frames)}, image num: {len(image_infos)}")
    print("Organizing json file and exporting images:")
    for name, info_pack in tqdm.tqdm(image_infos.items()):
        if name not in posed_frames:
            raise ValueError(f"There is an image without pose, image id: {name}")
        pose = np.float32(posed_frames[name]).reshape(4, 4).tolist()
        image_name, image = info_pack
        # no depth and colmap_im_id
        output_json["frames"].append({
            "file_path": f"images/{image_name}",
            "transform_matrix": pose,
            "original_name": name
        })
        cv.imwrite(os.path.join(image_folder, image_name), image)
        
    output_json["applied_transform"] = test_data["applied_transform"]

    output_file_path = os.path.join(output_folder, opts.output_name)
    with open(output_file_path, 'w', encoding = 'utf-8') as file:
        print(f"Output processed pose optimzation dataset to {output_file_path}")
        json.dump(output_json, file, indent = 4)