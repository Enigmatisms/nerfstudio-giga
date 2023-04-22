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
            return max(2. / np.pi * np.arctan(self.alpha * self.it) + 1, self.end_v / self.start_v) * self.start_v
        else:
            return max(self.start_v + mult, self.end_v)

    def valid_update(self):
        """Whether the current state is valid"""
        return (self.it < self.iter) or self.hold_expired
