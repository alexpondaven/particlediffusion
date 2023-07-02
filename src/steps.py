"""Framework for description of diffusion steps"""

import numpy as np
from collections import defaultdict

class Steps():
    def __init__(self, nlevels=20, init_method="score", weights=[]):
        """
        init_method: score or repulsive_no_noise
        """
        # Set up step object
        # Default steps is a score step with auto step size at each noise level
        # dictionary of lists of tuples (step_method, num_steps, weights)
        # weights is the weight assigned to different prompts
        super().__init__()
        self.nlevels = nlevels
        self.steps = defaultdict(list)

        if init_method:
            for level in range(nlevels):
                self.steps[level].append((init_method,1, weights))
    
    def add(self, level, method, steps, weights=[]):
        self.steps[level] = [(method, steps, weights)] + self.steps[level]
    
    def add_list(self, levels, method, steps, weights=[]):
        for i, level in enumerate(levels):
            self.steps[level] = [(method, steps[i], weights)] + self.steps[level]

    def add_all(self, method, steps, weights=[]):
        for level in range(self.nlevels):
            self.add(level, method, steps, weights)
