import json
import os

import configargparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from colorama import Fore, Style
from colorama import init as colorama_init

colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

def transform_poses(scale, trans, pos, applied_tf = None):
    result = np.zeros((4, 4), dtype = float)
    if applied_tf is not None:
        applied_tf = np.float32(applied_tf)
        if applied_tf.shape[0] != applied_tf.shape[1]:
            applied_tf = np.concatenate([applied_tf, np.float32([[0, 0, 0, 1]])], axis = 0)
        inv_applied_transform = np.linalg.inv(applied_tf)
    else:
        inv_applied_transform = np.ones((4, 4), dtype = pos.dtype)
    c2w = trans @ inv_applied_transform
    result[0:3, 0:3] = c2w[:3, :3] @ pos[0:3, 0:3]
    result[0:3, 3:4] = scale * c2w[:3, :3] @ (pos[0:3, 3:4] - c2w[0:3, 3:4])
    result[3, 3] = 1
    return result

# Draw arrow from https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(origin, end, length, scale=1, color = [1, 0, 0]):
    assert(not np.all(end == origin))
    vec = (end - origin) * length
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return (mesh)

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

    vec_s1 = []
    vec_s2 = []

    colors1 = []
    colors2 = []
    for frame1 in render_data['camera_path']:
        input_t = np.float32(frame1['camera_to_world']).reshape(4, 4)
        # This applied tf is not read but I knew it beforehand
        applied_tf = np.float32([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        tr_matrix = transform_poses(scale, T, input_t, applied_tf)
        vec_s1.append([
            tr_matrix[0:3, 0:3] @ np.float32([0, 0, 1]),
            tr_matrix[0:3, 0:3] @ np.float32([0, 1, 0]),
            tr_matrix[0:3, 0:3] @ np.float32([1, 0, 0])
        ])
        render_pos.append(tr_matrix[0:3, 3:4])
        colors1.append([1, 0, 0])

    # poses from nerfstudio generated path needs no transformation
    # the output should be transformed by transform_pose, then the result can be used
    for frame1 in studio_data['camera_path']:
        tr_matrix = np.float32(frame1['camera_to_world']).reshape(4, 4)
        vec_s2.append([
            tr_matrix[0:3, 0:3] @ np.float32([0, 0, 1]),
            tr_matrix[0:3, 0:3] @ np.float32([0, 1, 0]),
            tr_matrix[0:3, 0:3] @ np.float32([1, 0, 0])
        ])
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

    render_len = render_pos.shape[0]
    studio_len = studio_pos.shape[0]
    for i in range(0, render_len, 1):
        start_p = render_pos[i]
        arrows = vec_s1[i]
        for j in range(3):
            end_p = start_p + arrows[j]
            arrow = get_arrow(start_p, end_p, length = 0.03, scale=0.5, color = colors[j])
            vis.add_geometry(f"render_arrow{3 * i + j}", arrow)
    for i in range(0, studio_len, 1):
        start_p = studio_pos[i]
        arrows = vec_s2[i]
        for j in range(3):
            end_p = start_p + arrows[j]
            arrow = get_arrow(start_p, end_p, length = 0.03, scale=0.5, color = colors[j])
            vis.add_geometry(f"studio_arrow{3 * i + j}", arrow)
    
    # for idx in range(0, cmp_pts.shape[0]):
    #     vis.add_3d_label(cmp_pts[idx], "{}".format(idx+1))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()
    app.quit()
    

def parser_opts():
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("--input_path",      required = True, help = "Input scene file folder", type = str)
    parser.add_argument("--scene_name",      required = True, help = "Input scene name", type = str)
    parser.add_argument("--output_name",     default = "output.json", help = "Output pose json file name", type = str)
    return parser.parse_args()

if __name__ == "__main__":
    opts = parser_opts()
    visualize_paths(opts)