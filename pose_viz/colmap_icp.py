import json
import os

import configargparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from colorama import Fore, Style
from colorama import init as colorama_init
from scipy.spatial.transform import Rotation, Slerp

correct_pose = [-1,1,-1]

def matrix_nerf2ngp(matrix):
    matrix[:, 0] *= correct_pose[0]
    matrix[:, 1] *= correct_pose[1]
    matrix[:, 2] *= correct_pose[2]
    return matrix

def icpIter(data, gt, rot, trans, threshold:float):
    query_c, pair_c = np.zeros((3, 1), dtype = float), np.zeros((3, 1), dtype = float)
    qpt = np.zeros((3, 3), dtype = float)
    query_start_p, base_start_p = rot @ data[0] + trans, gt[0]
    valid_cnt = 0
    max_dist = 0
    max_dist_valid = 0
    avg_scale1, avg_scale2 = 0, 0
    for query_pt, value_pt in zip(data, gt):
        query_pt = rot @ query_pt + trans
        dist = np.linalg.norm(query_pt - value_pt)
        max_dist = max_dist if dist < max_dist else dist
        if dist > threshold:
            continue
        max_dist_valid = max_dist_valid if dist < max_dist_valid else dist
        valid_cnt += 1
        avg_scale1 += np.linalg.norm(query_pt - query_start_p)
        avg_scale2 += np.linalg.norm(value_pt - base_start_p)
        query_c += query_pt
        pair_c += value_pt
        qpt += (query_pt - query_start_p) @ (value_pt - base_start_p).T
    print(f"valid_cnt: {valid_cnt}, max_dist: {max_dist}, max_dist_valid: {max_dist_valid}", end = '')
    if valid_cnt:
        avg_scale = avg_scale2 / avg_scale1
        print(f", avg_scale: {avg_scale}")
        query_c /= valid_cnt
        pair_c /= valid_cnt
        qpt -= valid_cnt * (query_c - query_start_p) @ (pair_c - base_start_p).T
        u, _, vh = np.linalg.svd(qpt)
        r = vh.T @ u.T
        rr = vh.T @ np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., np.linalg.det(r)]]) @ u.T
        return avg_scale, rr, pair_c - rr @ query_c, max_dist
    else:
        print("")
        print("Invalid result with valid cnt = 0, please check your transformation and threshold setting.")
        print("Set a bigger threshold / make the threshold decay slower")
        raise ValueError("Invalid result with valid cnt = 0")

def icp(data, gt):
    # for DayaTemple: threshold 2 and max_dist decay should be 0.2 (small), HaiyanHall decay should be big
    threshold = 100
    scale = 1.0
    rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    trans = np.zeros((3, 1))
    cnt = 0
    while threshold > 0.00001 and cnt < 100:
        scale1, rot1, trans1, max_dist = icpIter(data, gt, rot * scale, trans, threshold)
        scale = scale * scale1
        rot = rot1 @ rot
        trans = rot1 @ trans + trans1
        threshold = max_dist * 0.9
        cnt += 1
    result = np.zeros((4, 4), dtype = float)
    result[0:3, 0:3] = rot
    result[0:3, 3:4] = trans
    result[3, 3] = 1
    colors1, colors2 = [], []
    for query_pt, _ in zip(data, gt):
        query_pt = rot @ query_pt + trans
        colors1.append([0., 1., 0.])
        colors2.append([0., 0., 1.])
    return scale, result, colors1, colors2

def transform_scale(scale, trans, pos):
    result = np.zeros((4, 4), dtype = float)
    result[0:3, 0:3] = trans[0:3, 0:3] @ pos[0:3, 0:3]
    result[0:3, 3:4] = scale * trans[0:3, 0:3] @ pos[0:3, 3:4] + trans[0:3, 3:4]
    result[3, 3] = 1
    return result

def transform_poses(scale, trans, pos, applied_tf = None):
    result = np.zeros((4, 4), dtype = float)
    if applied_tf is not None:
        applied_tf = np.float32(applied_tf)
        if applied_tf.shape[0] != applied_tf.shape[1]:
            applied_tf = np.concatenate([applied_tf, np.float32([[0, 0, 0, 1]])], axis = 0)
        inv_applied_transform = np.linalg.inv(applied_tf)
    else:
        inv_applied_transform = np.ones((4, 4), dtype = pos.dtype)
    print(inv_applied_transform)
    c2w = trans @ inv_applied_transform
    result[0:3, 0:3] = c2w[:3, :3] @ pos[0:3, 0:3]
    result[0:3, 3:4] = scale * c2w[:3, :3] @ (pos[0:3, 3:4] - c2w[0:3, 3:4])
    result[3, 3] = 1
    return result

def customized_trajectory(
    json_file: dict, t_mat: np.ndarray, scene_scale = 1.0,
    scale_factor: int = 1, applied_tf: np.ndarray = None
):
    fx = json_file["fl_x"] * scale_factor
    fy = json_file["fl_y"] * scale_factor
    print(json_file["w"], json_file["h"])
    traj_file = {"camera_type": "perspective", "use_intrinsics": True,
        "render_height": int(json_file["h"] * scale_factor), "render_width": int(json_file["w"] * scale_factor), 
        "camera_path": []
    }
    for frame in json_file["frames"]:
        if t_mat is not None:
            tf = transform_poses(scene_scale, t_mat, frame["transform_matrix"], applied_tf)
        else:
            tf = np.float32(frame["transform_matrix"])
        tf = tf.ravel()
        camera = {"camera_to_world": tf.tolist(), "aspect": 1, "fx": fx, "fy": fy}
        traj_file["camera_path"].append(camera)
    return traj_file

def visualize_paths(opts):
    render_path   = os.path.join(opts.input_path, opts.scene_name, opts.output_name)
    studio_path   = os.path.join(opts.input_path, opts.scene_name, "path.json")
    tr_path       = os.path.join(opts.input_path, opts.scene_name, "dataparser_transforms.json")

    with open(render_path,'r')as f:     # original json file
        render_data = json.load(f)
    with open(studio_path,'r')as f:     # COLMAP output: re-numbered
        studio_data = json.load(f)
    with open(tr_path,'r')as f:     # COLMAP output: re-numbered
        transforms = json.load(f)
    T, scale = np.float32(transforms["transform"]), transforms["scale"]

    render_pos = []
    studio_pos = []


    colors1 = []
    colors2 = []
    for frame1 in render_data['camera_path']:
        input_t = np.float32(frame1['camera_to_world']).reshape(4, 4)
        # This applied tf is not read but I knew it beforehand
        applied_tf = np.float32([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        tr_matrix = transform_poses(scale, T, input_t, applied_tf)

        render_pos.append(tr_matrix[0:3, 3:4])
        colors1.append([1, 0, 0])

    for frame1 in studio_data['camera_path']:
        tr_matrix = np.float32(frame1['camera_to_world']).reshape(4, 4)
        studio_pos.append(tr_matrix[0:3, 3:4])
        colors2.append([0, 0, 1])

    render_pos = np.asarray(render_pos).squeeze()
    studio_pos = np.asarray(studio_pos).squeeze()
    
    colorama_init()

    print(f"Rendering pose in [Test] is shown in {Fore.RED}red (1, 0, 0){Style.RESET_ALL}.")
    print(f"Path pose in [Test] is shown in {Fore.BLUE}blue (0, 0, 1){Style.RESET_ALL}.")

    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.show_skybox(False)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(render_pos)
    pcd.colors = o3d.utility.Vector3dVector(colors1)
    vis.add_geometry('render pos', pcd)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(studio_pos)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    vis.add_geometry('path pos', pcd2)
    
    # for idx in range(0, cmp_pts.shape[0]):
    #     vis.add_3d_label(cmp_pts[idx], "{}".format(idx+1))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()


def parser_opts():
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("--input_path",      required = True, help = "Input scene file folder", type = str)
    parser.add_argument("--scene_name",      required = True, help = "Input scene name", type = str)
    parser.add_argument("--origin_name",     default = "train.json", help = "Input original pose json file name", type = str)
    parser.add_argument("--parser_name",     default = "dataparser_transforms.json", help = "Scale and transform filename", type = str)
    parser.add_argument("--colmap_name",     default = "transforms.json", help = "Input colmap pose json file name", type = str)
    parser.add_argument("--eval_name",       default = "test.json", help = "Eval pose json file name", type = str)
    parser.add_argument("--output_name",     default = "output.json", help = "Output pose json file name", type = str)
    parser.add_argument("--scale",           default = 1, help = "Scale camera intrinsics", type = float)
    parser.add_argument("--invalid_th",      default = 0.2, help = "Threshold for invalid pose alignment", type = float)
    parser.add_argument("-v", "--visualize", default = False, action = "store_true", help = "Whether to visualize result")
    parser.add_argument("-g", "--generate_original", default = False, action = "store_true", help = "Whether to generate non-scaled path")
    return parser.parse_args()


def main_output(opts):
    origin_path = os.path.join(opts.input_path, opts.scene_name, opts.origin_name)
    colmap_path = os.path.join(opts.input_path, opts.scene_name, opts.colmap_name)
    eval_path   = os.path.join(opts.input_path, opts.scene_name, opts.eval_name)
    parse_path  = os.path.join(opts.input_path, opts.scene_name, opts.parser_name)
    output_path = os.path.join(opts.input_path, opts.scene_name, opts.output_name)

    with open(origin_path,'r')as f:     # original json file
        data1 = json.load(f)
    with open(colmap_path,'r')as f:     # COLMAP output: re-numbered
        data2 = json.load(f)
    with open(eval_path,'r')as f:
        data3 = json.load(f)

    data = []
    gt_m = []
    gt = []
    frame2_map: dict = {}
    applied_tf = data2.get('applied_transform', None)
    for frame2 in data2['frames']:
        matrix2 = np.array(frame2['transform_matrix'], np.float32)
        frame2_map[frame2['file_path']] = matrix2
    for frame1 in data1['frames']:
        matrix1: np.ndarray = np.float32(frame1['transform_matrix'])
        
        if frame1['file_path'] in frame2_map:
            matrix2 = frame2_map[frame1['file_path']]
            data.append(matrix1[0:3, 3:4])
            gt.append(matrix2[0:3, 3:4])
            gt_m.append(matrix2)
        else:
            print(f"{Fore.YELLOW}Warning, '{frame1['file_path']}' not exist in the COLMAP pose.{Style.RESET_ALL}")
    print(len(gt), len(data))
    for i in range(2):
        print(f"Iteration {i}")
        scale, avg_trans, colors1, colors2 = icp(data, gt)
        print(f"Scale: {scale}")
        print(f"Average translation: {avg_trans}")

        data_trans_m = []
        data_trans = []
        for frame1 in data1['frames']:
            tr_matrix = np.float32(frame1['transform_matrix'])
            if opts.scene_name == 'Library' and ('49' in frame1['file_path']):
                print(frame1['file_path'], frame1['original_name'])
                continue
            matrix1 = transform_scale(scale, avg_trans, tr_matrix)
            data_trans.append(matrix1[0:3, 3:4])
            data_trans_m.append(matrix1)
        
        invalid_exist = False
        for i in range(len(data)):
            if np.linalg.norm(gt[i] - data_trans[i]) > opts.invalid_th:
                gt[i] = data_trans[i]
                gt_m[i] = data_trans_m[i]
                invalid_exist = True
        if invalid_exist == False:
            break
    data_test = []
    data_test_fix = []
    color_test = []
    color_test_fix = []
    for frame1 in data3['frames']:
        tr_matrix = np.float32(frame1['transform_matrix'])
        matrix1 = transform_scale(scale, avg_trans, tr_matrix)[0:3, 3:4]
        data_test.append(matrix1)
        color_test.append([1., 1., 0.])
        color_test_fix.append([1., 0., 1.])
        min_dist = np.linalg.norm(matrix1 - data_trans[0])
        delta2_t = delta_t = gt[0] - data_trans[0]
        min_dist2 = min_dist * 2
        for dt, g in zip(data_trans, gt):
            d = np.linalg.norm(matrix1 - dt)
            if d < min_dist:
                min_dist = d
                delta_t = g - dt
            elif d < min_dist2:
                min_dist2 = d
                delta2_t = g - dt
        matrix1_fix = matrix1 + (delta_t + delta2_t) / 2
        data_test_fix.append(matrix1_fix)

    if opts.visualize:
        data_trans = np.asarray(data_trans).squeeze()
        data_test = np.asarray(data_test).squeeze()
        data_test_fix = np.asarray(data_test_fix).squeeze()
        gt = np.asarray(gt).squeeze()
        colorama_init()

        print(f"Original pose in [Test] is shown in {Fore.YELLOW}yellow (1, 1, 0){Style.RESET_ALL}.")
        print(f"Transformed pose in [Test] is shown in {Fore.MAGENTA}magenta (1, 0, 1){Style.RESET_ALL}.")

        print(f"Pose in [Origin] is shown in {Fore.GREEN}green (0, 1, 0){Style.RESET_ALL}.")
        print(f"Pose in [COLMAP] is shown in {Fore.BLUE}blue (0, 0, 1){Style.RESET_ALL}.")
        print(f"There are {data_test.shape[0]} test views in total.")
        print(f"There are {gt.shape[0]}, {data_trans.shape[0]} original matching views in total.")

        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        vis.show_settings = True
        vis.show_skybox(False)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data_trans)
        pcd.colors = o3d.utility.Vector3dVector(colors1)
        vis.add_geometry('data_trans', pcd)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(gt)
        pcd2.colors = o3d.utility.Vector3dVector(colors2)
        vis.add_geometry('cmp_pts', pcd2)
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(data_test)
        pcd3.colors = o3d.utility.Vector3dVector(color_test)
        vis.add_geometry('data_test', pcd3)
        pcd4 = o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(data_test_fix)
        pcd4.colors = o3d.utility.Vector3dVector(color_test_fix)
        vis.add_geometry('data_test_fix', pcd4)
        lines = o3d.geometry.LineSet()
        lines_index = []
        for i in range(gt.shape[0]):
            lines_index.append([i, i + gt.shape[0]])
        lines.lines = o3d.utility.Vector2iVector(lines_index)
        lines.points = o3d.utility.Vector3dVector(np.concatenate((data_trans, gt), axis=0))
        vis.add_geometry('lines', lines)
        for idx in range(0, gt.shape[0]):
            vis.add_3d_label(gt[idx], f"{idx+1}")
        vis.reset_camera_to_default()

        app.add_window(vis)
        app.run()
    else:
        with open(colmap_path, 'r')as f:
            data_val = json.load(f)
        with open(parse_path,'r')as f:
            parser_data = json.load(f)
        t_mat, scale = np.float32(parser_data["transform"]), parser_data["scale"]
        if opts.generate_original:
            for frame in data_val['frames']:
                frame['transform_matrix'] = np.float32(frame['transform_matrix'])
        else:
            for frame in data_val['frames']:
                tr_matrix = np.float32(frame['transform_matrix'])
                matrix = transform_scale(scale, avg_trans, tr_matrix)
                m_trans = matrix[0:3, 3:4]

                delta2_t = delta_t = gt[0] - data_trans[0]
                delta2_r = delta_r = gt_m[0][0:3, 0:3] @ np.linalg.inv(data_trans_m[0][0:3, 0:3])
                min_dist = np.linalg.norm(m_trans - data_trans[0])
                min_dist2 = min_dist * 2
                for dt, g in zip(data_trans_m, gt_m):
                    d = np.linalg.norm(m_trans - dt[0:3, 3:4])
                    if d < min_dist:
                        min_dist = d
                        delta_t = g[0:3, 3:4] - dt[0:3, 3:4]
                        delta_r = g[0:3, 0:3] @ np.linalg.inv(dt[0:3, 0:3])
                    elif d < min_dist2:
                        min_dist2 = d
                        delta2_t = g[0:3, 3:4] - dt[0:3, 3:4]
                        delta2_r = g[0:3, 0:3] @ np.linalg.inv(dt[0:3, 0:3])
                matrix[0:3, 3:4] = m_trans + (delta_t + delta2_t) / 2
                q1 = Rotation.from_matrix(delta_r)
                q2 = Rotation.from_matrix(delta2_r)
                slerp = Slerp([0, 1], Rotation.concatenate([q1, q2]))
                q_fix = slerp(0.5)
                u, s, v =  np.linalg.svd(q_fix.as_matrix() @ matrix[0:3, 0:3])
                matrix[0:3, 0:3] = u @ v
                frame['transform_matrix'] = matrix
        result = customized_trajectory(data_val, t_mat, scale, opts.scale, applied_tf)
        with open(output_path, "w") as outfile:
            json.dump(result, outfile, indent=4)

if __name__ == "__main__":
    opts = parser_opts()
    main_output(opts)