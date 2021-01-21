import torch
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trace, Y = sample['trace'], sample['maskorder'],

        return {'trace': torch.from_numpy(trace),
                 'maskorder': torch.from_numpy(np.array([Y]))}


class Horizontal_Scaling_0_1(object):

    def __call__(self, sample):
        trace, Y = sample['trace'], sample['maskorder']

        scale = 1.0 / (torch.max(trace).item() - torch.min(trace).item())
        trace = trace.sub(torch.min(trace).item()).mul(scale)

        return {'trace': trace,
                 'maskorder': Y}

class Horizontal_Scaling_m1_1(object):

    def __call__(self, sample):
        trace, Y = sample['trace'], sample['maskorder']

        scale = 1.0 / (torch.max(trace).item() - torch.min(trace).item())
        trace = trace.sub(torch.min(trace).item()).mul(scale)
        trace = trace.mul(1- (-1)).add(-1)
        return {'trace': trace,
                'maskorder': Y}
