import json
import os

import configargparse
import numpy as np
from copy import deepcopy
from pathlib import Path

def get_idx(path: str):
    return int(path[path.find("_")+1:path.find(".")])

def parser_opts():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("-i", "--input_scene",  required = True, help = "Input scene file", type = str)
    parser.add_argument("--outpath",            required = True, help = "Output file path", type = str)
    parser.add_argument("-o", "--output_scene", default = "transforms_new.json", help = "Output scene file", type = str)
    parser.add_argument("-m", "--merge",        default = False, action = "store_true", help = "whether to merge train / test dataset")
    parser.add_argument("-t", "--transform",    default = False, action = "store_true", help = "Whether to cancel out the applied transform")
    parser.add_argument("-n", "--no_skew",      default = False, action = "store_true", help = "Whether to use skews in the transform.json")
    parser.add_argument("--skip",               default = False, action = "store_true", help = "Whether to skip generating train embedded json (for DayaTemple)")
    parser.add_argument("--overwrite",          default = False, action = "store_true", help = "Whether to enable overwriting")
    parser.add_argument("--scale",              default = 1.0, help = "Scale intrinsic and image wh (if HLOC uses half resolution images)", type=float)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parser_opts()
    file_path = opts.input_scene
    origin_name = "train_merged.json" if opts.merge else "train.json"
    colmap_name = "transforms.json"
    output_name = opts.output_scene

    origin_path = os.path.join(file_path, origin_name)
    colmap_path = os.path.join(file_path, colmap_name)
    output_path = os.path.join(opts.outpath, output_name)

    with open(origin_path, "r") as file:
        train_data = json.load(file)
    with open(colmap_path, "r") as file:
        after_data = json.load(file)
    if opts.scale > 1.1:
        to_scale = ("w", "h", "fl_x", "fl_y", "cx", "cy")
        for item in to_scale:
            after_data[item] *= opts.scale
    # preprocess

    should_overwrite = False
    for frame in after_data["frames"]:
        depth_file_path = frame["depth_file_path"]
        if not depth_file_path.endswith(".pfm"):
            main_part = Path(depth_file_path).stem
            frame["depth_file_path"] = f"depths/{main_part}.pfm"
            should_overwrite = True

    if should_overwrite and opts.overwrite:
        # DayaTemple transform 需要放大一倍
        if "overwritten" not in after_data:     # 防止二次 scaling 与处理
            after_data["overwritten"] = True
            with open(colmap_path, "w") as file:
                json.dump(after_data, file, indent = 4)

    if opts.skip:
        exit(0)

    frames = {get_idx(frame["file_path"]):frame for frame in after_data["frames"]}
    test_views = []
    for frame in train_data["frames"]:
        # "original_name": "00000001_cam.txt"
        # file_path": "images/frame_00002.jpg",
        ori_name = frame["file_path"]
        if ori_name == "test_view":
            test_views.append(frame)
            continue
        idx = get_idx(ori_name)
        matrix = frame["transform_matrix"]
        if opts.transform:
            np_mat = np.float32(matrix)
            np_mat[:3, :3] = np_mat[:3, :3] @ np.float32([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            matrix = np_mat.tolist()
        # Note that in the lastest version, where we skipped 00058(frame_49) in the first place
        # We might not need to skip anything later
        # Note that for library, there is no 49, therefore the images after 49 will have id mismatch
        # which can be a big problem, therefore we should double check the frame in frames of index [idx - 1]
        # if idx matches, we replace transform matrix, otherwise we might need to insert a frame here
        # check carefully
        if idx in frames:
            frames[idx]["transform_matrix"] = matrix 
        else:
            print(f"Warning: {idx} does not exist. Skipping for now.")
    after_data["frames"] = list(frames.values()) + test_views
    if opts.no_skew:
        if "k1" not in train_data:
            after_data["k1"] = 0
            after_data["k2"] = 0
            after_data["p1"] = 0
            after_data["p2"] = 0
            after_data["fl_x"] = train_data["fl_x"]
            after_data["fl_y"] = train_data["fl_y"]

    with open(output_path, "w") as file:
        json.dump(after_data, file, indent = 4)
        