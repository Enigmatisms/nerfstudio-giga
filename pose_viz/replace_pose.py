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
    parser.add_argument("-s", "--input_scene",      required = True, help = "Input scene", type = str)
    parser.add_argument("-n", "--input_name",       required = True, help = "Input name", type = str)
    parser.add_argument("-m", "--mode",             required = True, choices = ["none", "new", "no_skew"], 
                        help = "Input modes to choose from.", type = str)
    parser.add_argument("-i", "--input_path",       default = "../outputs/", help = "Input name", type = str)
    parser.add_argument("-o", "--output_json_path", default = "../../dataset/images_and_cams/full/pose_align/", 
                        help = "Input pose and camera intrinsics file.", type = str)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parser_opts()
    input_json_path = os.path.join(opts.input_path, opts.input_scene, "depth-nerfacto", opts.input_name, "optimized_poses.json")
    truncate_scene = opts.input_scene.split("_")[0]
    mode_mapping = {"none": "", "new": "_new", "no_skew": "_no_skew"}
    overwrite_path  = os.path.join(opts.output_json_path, truncate_scene) 
    origin_json_path = os.path.join(overwrite_path, f"output{mode_mapping[opts.mode]}.json")
    output_json_path = os.path.join(overwrite_path, f"output{mode_mapping[opts.mode]}_opt.json")

    with open(input_json_path, 'r', encoding = 'utf-8') as file:
        opt_poses = json.load(file)

    with open(origin_json_path, 'r', encoding = 'utf-8') as file:
        template = json.load(file)


    template_pose_num = len(template["camera_path"])
    assert(template_pose_num == len(opt_poses["frames"]))

    for i in range(template_pose_num):
        template["camera_path"][i]["camera_to_world"] = \
            opt_poses["frames"][i]["camera_to_world"]

    with open(output_json_path, 'w', encoding = 'utf-8') as file:
        json.dump(template, file, indent = 4)
    print(f"Output to '{output_json_path}'")