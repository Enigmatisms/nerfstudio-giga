"""
    Replace the poses in output.json by the optimized poses
"""

import json
import os

import configargparse
import numpy as np


def parser_opts():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("-i", "--input_path",      required = True, help = "Input scene", type = str)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parser_opts()
    mode_mapping = {"none": "", "new": "_new", "no_skew": "_no_skew", "colmap": "_colmap"}
    input_json_path = os.path.join(opts.input_path, f"output{mode_mapping[opts.mode]}_opt.json")
    output_json_path = os.path.join(opts.input_path, f"output{mode_mapping[opts.mode]}_replace.json")

    with open(input_json_path, 'r', encoding = 'utf-8') as file:
        opt_poses = json.load(file)

    pose_num = len(opt_poses["camera_path"])
    new_frames = []
    for i in range(pose_num):
        pose_frame = opt_poses["camera_path"][i]
        if pose_frame["original_name"] = "00000006_cam.txt":
            new_frames.append(pose_frame)
    opt_poses["camera_path"] = new_frames
    with open(output_json_path, 'w', encoding = 'utf-8') as file:
        json.dump(opt_poses, file, indent = 4)