import itertools

import torch
from torch.utils.data.sampler import Sampler

from mmgpt.train.distributed import world_info_from_env


class InfiniteSampler(Sampler):
    def __init__(self, dataset: int, shuffle: bool = True, seed: int = 0):
        self._size = len(dataset)
        self._shuffle = shuffle
        self._seed = int(seed)
        _, rank, world_size = world_info_from_env()

        self._rank = rank
        self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()
