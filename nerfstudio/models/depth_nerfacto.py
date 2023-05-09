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
Nerfacto augmented with depth supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.losses import DepthLossType, depth_loss, ds_occlusion_loss
from nerfstudio.model_components.scheduler import SimpleScheduler
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps


@dataclass
class DepthNerfactoModelConfig(NerfactoModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: DepthNerfactoModel)
    min_depth_loss_mult: float = 1e-4
    """Multiplier (minimum) of the depth loss."""
    max_depth_loss_mult: float = 5e-3
    """Multiplier (maximum) of the depth loss."""
    depth_loss_iter: int = 50000
    """Apply depth loss for this number of iteration"""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    occlusion_sigma: float = 1.0
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    """Depth loss type."""


class DepthNerfactoModel(NerfactoModel):
    """Depth loss augmented nerfacto model.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: DepthNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])
        self.depth_sch = SimpleScheduler(self.config.max_depth_loss_mult, 
                    self.config.min_depth_loss_mult, self.config.depth_loss_iter, mode = 'linear', hold_expired = True)

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        
        # if the model is freezed, there will be no ["depth_image"] in a batch
        if self.training and not self.config.freeze_field:
            metrics_dict["depth_loss"] = 0.0
            sigma = self._get_sigma().to(self.device)
            termination_depth = batch["depth_image"].to(self.device)
            weight_list_len = len(outputs["weights_list"])
            for i in range(weight_list_len):
                ray_samples = outputs["ray_samples_list"][i]
                metrics_dict["depth_loss"] += depth_loss(
                    weights=outputs["weights_list"][i],
                    ray_samples=ray_samples,
                    termination_depth=termination_depth,
                    predicted_depth=outputs["depth"],
                    sigma=sigma,
                    directions_norm=outputs["directions_norm"],
                    is_euclidean=self.config.is_euclidean_depth,
                    depth_loss_type=self.config.depth_loss_type,
                ) / weight_list_len

            # Qianyue He's note: depth supervised occlusion loss can be enabled here
            # metrics_dict["ds_occ_loss"] = ds_occlusion_loss(
            #     outputs["density"], outputs["ray_samples"], termination_depth, self.config.occlusion_sigma
            # )
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and not self.config.freeze_field:
            assert metrics_dict is not None and "depth_loss" in metrics_dict
            depth_loss_mult = self.depth_sch.update()
            loss_dict["depth_loss"] = depth_loss_mult * metrics_dict["depth_loss"]
            # occ_loss_mult = self.occ_mult_sch.get()
            # loss_dict["ds_occ_loss"] = occ_loss_mult * metrics_dict["ds_occ_loss"]
            # loss_dict["ds_occ_loss_nw"] = loss_dict["ds_occ_loss"].detach()
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Appends ground truth depth to the depth image."""
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        ground_truth_depth = batch["depth_image"]
        if not self.config.is_euclidean_depth:
            ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

        ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
        predicted_depth_colormap = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=torch.min(ground_truth_depth),
            far_plane=torch.max(ground_truth_depth),
        )
        images["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
        depth_mask = ground_truth_depth > 0
        metrics["depth_mse"] = torch.nn.functional.mse_loss(
            outputs["depth"][depth_mask], ground_truth_depth[depth_mask]
        )
        return metrics, images

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(  # pylint: disable=attribute-defined-outside-init
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma
