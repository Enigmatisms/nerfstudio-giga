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
    parser.add_argument("-m", "--mode",             required = True, choices = ["none", "colmap", "new", "no_skew"], 
                        help = "Input modes to choose from.", type = str)
    parser.add_argument("-o", "--output_json_path", required = True,
                        help = "Input pose and camera intrinsics file.", type = str)
    
    parser.add_argument("-i", "--input_path",       default = "./outputs/", help = "Input name", type = str)
    parser.add_argument("--scale",                  default = 1.0, help = "Intrinsic scaling factor.", type = float)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parser_opts()
    input_json_path = os.path.join(opts.input_path, opts.input_scene, "depth-nerfacto", opts.input_name, "optimized_poses.json")
    truncate_scene = opts.input_scene.split("_")[0]
    mode_mapping = {"none": "", "new": "_new", "no_skew": "_no_skew", "colmap": "_colmap"}
    overwrite_path  = os.path.join(opts.output_json_path, truncate_scene) 
    origin_json_path = os.path.join(overwrite_path, f"output{mode_mapping[opts.mode]}.json")
    output_json_path = os.path.join(overwrite_path, f"output{mode_mapping[opts.mode]}_opt.json")

    with open(input_json_path, 'r', encoding = 'utf-8') as file:
        opt_poses = json.load(file)

    with open(origin_json_path, 'r', encoding = 'utf-8') as file:
        template = json.load(file)


    template_pose_num = len(template["camera_path"])
    assert(template_pose_num == len(opt_poses["frames"]))

    intrinsic_exist = False
    for i in range(template_pose_num):
        pose_frame = opt_poses["frames"][i]
        if (truncate_scene == "DayaTemple" and template["camera_path"][i]["original_name"] == "00000024_cam.txt") or \
           (truncate_scene == "PeonyGarden" and template["camera_path"][i]["original_name"] == "00000006_cam.txt"):
            # manually skip some of the cases (since the images are blurred)
            pass
        else:
            template["camera_path"][i]["camera_to_world"] = \
                pose_frame["camera_to_world"]
        if "fx" in pose_frame:
            intrinsic_exist = True
            template["camera_path"][i]["fx"] = pose_frame["fx"] * opts.scale
            template["camera_path"][i]["fy"] = pose_frame["fy"] * opts.scale
        if "k1" in pose_frame:
            all_names = ("k1", "k2", "k3", "k4", "p1", "p2")
            for name in all_names:
                template["camera_path"][i][name] = pose_frame[name]

    with open(output_json_path, 'w', encoding = 'utf-8') as file:
        json.dump(template, file, indent = 4)
    print(f"Output to '{output_json_path}'. Use intrinsics: {intrinsic_exist}, scaling: {opts.scale}")