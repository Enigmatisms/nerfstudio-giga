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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Tuple, Type, Union

import torch
import tyro
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (ExponentialDecaySchedulerConfig,
                                          SchedulerConfig)
from nerfstudio.utils import poses as pose_utils


@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    distortion_opt: Literal["off", "fixed", "full"] = "off"
    """Distortion optimization strategy to use."""

    intrinsic_opt: Literal["off", "fixed", "full"] = "off"
    """Camera intrinsics optimization strategy to use."""

    position_noise_std: float = 0.0
    """Noise to add to initial positions. Useful for debugging."""

    orientation_noise_std: float = 0.0
    """Noise to add to initial orientations. Useful for debugging."""

    optimizer: AdamOptimizerConfig = AdamOptimizerConfig(lr=6e-4, eps=1e-15)
    """ADAM parameters for camera optimization."""

    scheduler: SchedulerConfig = ExponentialDecaySchedulerConfig(max_steps=10000)
    """Learning rate scheduler for camera optimizer.."""

    param_group: tyro.conf.Suppress[str] = "camera_opt"
    """Name of the parameter group used for pose optimization. Can be any string that doesn't conflict with other
    groups."""


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        else:
            assert_never(self.config.mode)

        if self.config.intrinsic_opt != "off" or self.config.distortion_opt != "off":
            # Keep the state dict the same
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))

        if self.config.distortion_opt == "full":
            self.distortion_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        elif self.config.distortion_opt == "fixed":
            self.distortion_adjustment = torch.nn.Parameter(torch.zeros((1, 6), device=device))
        elif self.config.distortion_opt == "off":
            self.distortion_adjustment = None
        else:
            assert_never(self.config.distortion_opt)

        if self.config.intrinsic_opt == "full":
            self.intrinsic_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 2), device=device))
        elif self.config.intrinsic_opt == "fixed":
            self.intrinsic_adjustment = torch.nn.Parameter(torch.zeros((1, 2), device=device))
        elif self.config.intrinsic_opt == "off":
            self.intrinsic_adjustment = None
        else:
            assert_never(self.config.intrinsic_opt)

        

        # Initialize pose noise; useful for debugging.
        if config.position_noise_std != 0.0 or config.orientation_noise_std != 0.0:
            assert config.position_noise_std >= 0.0 and config.orientation_noise_std >= 0.0
            std_vector = torch.tensor(
                [config.position_noise_std] * 3 + [config.orientation_noise_std] * 3, device=device
            )
            self.pose_noise = exp_map_SE3(torch.normal(torch.zeros((num_cameras, 6), device=device), std_vector))
        else:
            self.pose_noise = None

    def forward(
        self,
        indices: TensorType["num_cameras"],
    ) -> Tuple[TensorType["num_cameras", 3, 4], TensorType["num_cameras", 6]]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        pose_outputs = []
        distortion = torch.zeros(6, device=self.device)[None, :].tile(indices.shape[0], 1)
        delta_ints = torch.zeros(2, device=self.device)[None, :].tile(indices.shape[0], 1)

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            pose_outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            pose_outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        else:
            assert_never(self.config.mode)

        if self.config.distortion_opt == "full":
            distortion = self.distortion_adjustment[indices, :]
        elif self.config.distortion_opt == "fixed":
            # fixed distortion clones its parameter
            distortion = self.distortion_adjustment.broadcast_to((indices.shape[0], 6))

        if self.config.intrinsic_opt == "full":
            delta_ints = self.intrinsic_adjustment[indices, :]
        elif self.config.intrinsic_opt == "fixed":
            # fixed distortion clones its parameter
            delta_ints = self.intrinsic_adjustment.broadcast_to((indices.shape[0], 2))

        # Apply initial pose noise.
        if self.pose_noise is not None:
            pose_outputs.append(self.pose_noise[indices, :, :])

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(pose_outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            pose = torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        else:
            pose = functools.reduce(pose_utils.multiply, pose_outputs)
        return (pose, distortion, delta_ints)
