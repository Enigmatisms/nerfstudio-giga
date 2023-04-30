# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ray generator.
"""
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle


class RayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        cameras: Camera objects containing camera info.
        pose_optimizer: pose optimization module, for optimizing noisy camera intrinsics/extrinsics.
    """

    def __init__(self, cameras: Cameras, pose_optimizer: CameraOptimizer) -> None:
        super().__init__()
        self.cameras = cameras
        self.pose_optimizer = pose_optimizer
        self.register_buffer("image_coords", cameras.get_image_coords(), persistent=False)

    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]

        # This is ray generator, for rays sampled from unseen poses, we need to skip pose refining

        camera_opt_to_camera = self.pose_optimizer(c)

        ray_bundle = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
            camera_opt_to_camera=camera_opt_to_camera,
        )
        return ray_bundle

class PerturbRayGenerator(RayGenerator):
    """This ray generator aims to generate some rays from an unseen view
        The idea comes from Info-NeRF, using KL divergence between seen and 
        unseen views to regularize density in the space.
        For rays generated from unseen poses, there will be no RGB supervision, 
        nor is pose refinement here meaningful, therefore we manually disable 
        the gradient flow during forward-pass
    """

    def __init__(self,
            cameras: Cameras, pose_optimizer: CameraOptimizer,
            perturb_sigma: float = 4.5, random_ratio: float = 0.2,
            max_perturb_iter: float = 150000, random_seed: int = 114514
        ) -> None:
        super().__init__(cameras, pose_optimizer)
        self.perturb_sigma = perturb_sigma
        self.random_ratio = random_ratio            # probability of sampling a ray bundle with unseen views
        self.max_perturb_iter = max_perturb_iter
        self.iter_cnt = 0
        random.seed(random_seed)

    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        self.iter_cnt += 1
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]

        # This is ray generator, for rays sampled from unseen poses, we need to skip pose refining

        camera_opt_to_camera = self.pose_optimizer(c)

        # c shape (4096), coords shape (4096, 2), ray_indices (4096, 3), camera_opt_to_camera (4096, 3, 4)
        if random.random() > self.random_ratio or self.iter_cnt > self.max_perturb_iter:    # Fallback to RayGenerator
            ray_bundle: RayBundle = self.cameras.generate_rays(
                camera_indices=c.unsqueeze(-1),
                coords=coords,
                camera_opt_to_camera=camera_opt_to_camera,
            )
            ray_bundle.has_unseen = False
        else:
            camera_opt_to_camera = camera_opt_to_camera.detach()    # disable pose refinement when sampling unseen views
            half_num    = ray_indices.shape[0] >> 1
            half_c      = c[:half_num]
            half_coords = coords[:half_num]
            half_exts   = camera_opt_to_camera[:half_num]       # may be the gradient should be masked

            half_exts = half_exts[None, ...].expand(2, -1, -1, -1).reshape(-1, 3, 4)       # duplicate the extrinsics
            trunc_degs = np.random.uniform(-self.perturb_sigma, self.perturb_sigma, (half_num, 3))
            rots = Rot.from_euler('xyz', trunc_degs, degrees = True)
            perturb_rotms = rots.as_matrix()
            perturb_rots  = torch.from_numpy(perturb_rotms).to(half_exts.device).to(half_exts.dtype)
            half_exts[half_num:, :, :-1] = perturb_rots @ half_exts[half_num:, :, :-1]                  # only rotate

            half_c = half_c[None, ...].expand(2, -1).reshape(-1, 1)                   # duplicate the camera indices
            half_coords = half_coords[None, ...].expand(2, -1, -1).reshape(-1, 2)     # duplicate the image coordinates

            ray_bundle: RayBundle = self.cameras.generate_rays(
                camera_indices=half_c,
                coords=half_coords,
                camera_opt_to_camera=half_exts,
            )
            ray_bundle.has_unseen = True

        return ray_bundle
    
class AllViewsRayGenerator(RayGenerator):
    """ This ray generator will also generate rays in the test views
        to help us regularize the density of views not seen in the training set,
        preventing floaters from blocking test views. The rays under test views:
        - should have all the losses but occlusion loss disabled
    """
    def __init__(self, cameras: Cameras, pose_optimizer: CameraOptimizer, train_view_num: int):
        super().__init__(cameras, pose_optimizer)
        self.train_view_num = train_view_num

    def forward(self, ray_indices: TensorType) -> RayBundle:
        ray_bundle = super().forward(ray_indices)
        ray_bundle.has_test_view = (ray_indices[:, 0] >= self.train_view_num).any().item()
        return ray_bundle
    