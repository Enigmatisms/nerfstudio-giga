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
    Simple weight (multiplier) scheduler by Qianyue He 2023.4.21
    Offering linear / arctan (faster than exponential) decay
"""

import numpy as np


class SimpleScheduler:
    """Scheduler, offering linear and arctan weight adjustment
       This might not be useful, if depth is dense
    """
    def __init__(self, start_v, end_v, iter_num, mode = 'arctan', hold_expired = False):
        self.start_v = start_v
        self.end_v   = end_v
        self.iter    = iter_num
        self.it      = 0
        self.mode    = mode
        self.hold_expired = hold_expired
        self.now     = 0
        if mode == 'arctan':
            ratio = self.end_v / self.start_v - 1
            self.alpha = np.tan(np.pi / 2. * (ratio)) / self.iter
        else:       # linear
            self.alpha = (self.end_v - self.start_v) / self.iter

    def update(self):
        """Update current state and return weight"""
        mult = self.alpha * self.it
        self.it += 1
        if self.mode == 'arctan':
            self.now = max(2. / np.pi * np.arctan(self.alpha * self.it) + 1, self.end_v / self.start_v) * self.start_v
            return self.now
        else:
            self.now = max(self.start_v + mult, self.end_v)
            return self.now
        
    def get(self):
        return self.now

    def valid_update(self):
        """Whether the current state is valid"""
        return (self.it < self.iter) or self.hold_expired
