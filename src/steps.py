"""Framework for description of diffusion steps"""

import numpy as np
from collections import defaultdict

class Steps():
    def __init__(self, nlevels=20):
        # Set up step object
        # Default steps is a score step with auto step size at each noise level
        # dictionary of lists of tuples (step_method, num_steps, )
        super().__init__()
        self.nlevels = nlevels
        self.steps = defaultdict(list)

        for level in range(nlevels):
            self.steps[level].append(("score",1))
    
    def add(self, level, method, steps):
        self.steps[level] = [(method, steps)] + self.steps[level]
    
    def add_list(self, levels, method, steps):
        for i, level in enumerate(levels):
            self.steps[level] = [(method, steps[i])] + self.steps[level]

    def add_all(self, method, steps):
        for level in range(self.nlevels):
            self.add(level, method, steps)
