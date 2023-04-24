"""
    Modified from github: https://github.com/demul/extrinsic2pyramid/blob/main/util/camera_pose_visualizer.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Z_ROT = np.float32([
    [ 0, 0, 1],
    [ 0, 1, 0],
    [-1, 0, 0]
])

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.colors = None
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        
        vertex_transformed = (vertex_std @ extrinsic.T)[..., :-1]
        meshes = [[vertex_transformed[0], vertex_transformed[1], vertex_transformed[2]],
                  [vertex_transformed[0], vertex_transformed[2], vertex_transformed[3]],
                  [vertex_transformed[0], vertex_transformed[3], vertex_transformed[4]],
                  [vertex_transformed[0], vertex_transformed[4], vertex_transformed[1]],
                  [vertex_transformed[1], vertex_transformed[2], vertex_transformed[3], vertex_transformed[4]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))
        
    def generate_color_map(self, num = 30):
        half = num >> 1
        other_half = num - half
        interval = 1. / half
        other_interval = 1. / other_half
        self.colors = [np.float32([interval * i, 0, 0]) for i in range(half, 0, -1)]
        self.colors.extend([np.float32([0, 0, other_interval * i]) for i in range(other_half)])

    @staticmethod
    def local_pyramid(cam_axis = 'z', cam_backward = False, focal_len_scaled=5, aspect_ratio=0.3):
        forward_len = focal_len_scaled
        if cam_backward:
            forward_len *= -1
        if cam_axis == 'x':
            vertex_std = np.array([[0, 0, 0, 1],
                                   [forward_len, focal_len_scaled * aspect_ratio,  -focal_len_scaled * aspect_ratio, 1],
                                   [forward_len, focal_len_scaled * aspect_ratio,  focal_len_scaled * aspect_ratio,  1],
                                   [forward_len, -focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio,  1],
                                   [forward_len, -focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, 1]])
        elif cam_axis == 'y':
            vertex_std = np.array([[0, 0, 0, 1],
                                   [focal_len_scaled * aspect_ratio,  forward_len, -focal_len_scaled * aspect_ratio, 1],
                                   [focal_len_scaled * aspect_ratio,  forward_len, focal_len_scaled * aspect_ratio,  1],
                                   [-focal_len_scaled * aspect_ratio, forward_len, focal_len_scaled * aspect_ratio,  1],
                                   [-focal_len_scaled * aspect_ratio, forward_len, -focal_len_scaled * aspect_ratio, 1]])
        else:
            vertex_std = np.array([[0, 0, 0, 1],
                                   [focal_len_scaled * aspect_ratio,  -focal_len_scaled * aspect_ratio, forward_len, 1],
                                   [focal_len_scaled * aspect_ratio,  focal_len_scaled * aspect_ratio,  forward_len, 1],
                                   [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio,  forward_len, 1],
                                   [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, forward_len, 1]])
        return vertex_std
        
    def patch_transform(self, extrinsics, focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = CameraPoseVisualizer.local_pyramid(cam_axis = 'z', cam_backward = False, focal_len_scaled=focal_len_scaled, aspect_ratio=aspect_ratio)
        self.generate_color_map(extrinsics.shape[0])
        for i, extrinsic in enumerate(extrinsics):
            # T = extrinsic
            T = extrinsic
            vertex_transformed = (vertex_std @ T.T)[..., :-1]
            meshes = [[vertex_transformed[0], vertex_transformed[1], vertex_transformed[2]],
                    [vertex_transformed[0], vertex_transformed[2], vertex_transformed[3]],
                    [vertex_transformed[0], vertex_transformed[3], vertex_transformed[4]],
                    [vertex_transformed[0], vertex_transformed[4], vertex_transformed[1]],
                    [vertex_transformed[1], vertex_transformed[2], vertex_transformed[3], vertex_transformed[4]]]
            self.ax.add_collection3d(
                Poly3DCollection(meshes, facecolors=self.colors[i], linewidths=0.3, edgecolors=self.colors[i], alpha=0.6))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            if self.colors is None:
                color = plt.cm.rainbow(idx / len(list_label))
            else:
                color = self.colors[idx]
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        self.ax.set_aspect('equal', adjustable='box')
        plt.title('Extrinsic Parameters')
        plt.show()