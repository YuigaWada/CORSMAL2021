# MLExp ReproducibleDataLoader
# https://github.com/OtsuKotsu/MLExp/blob/main/mlexp/utils/data/dataloader.py

# Copyright (c) 2021 OtsuKotsu

# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

"""define dataloader that has additional settings"""
import os
import random
from typing import Any, Callable, List, Optional, Sequence, TypeVar

import numpy
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
_collate_fn_t = Callable[[List[T]], Any]
_sampler_t = Sampler[Sequence[int]]


class ReproducibleDataLoader(DataLoader):
    """
    extended dataloader that has
        - reproducible seed setting
        - preferred setting for num_workers, pin_memory
    """

    def __init__(
        self,
        dataset: Dataset[T_co],
        specified_seed: int,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[_sampler_t] = None,
        num_workers: int = -1,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        timeout: float = 0,
        multiprocessing_context=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False
    ):
        """constructor
        Args:
            dataset (Dataset[T_co]): same as torch.utils.data.dataloader
            specified_seed (int): fix seed for reproducibility
            batch_size (Optional[int], optional): same as torch.utils.data.dataloader. Defaults to 1.
            shuffle (bool, optional): same as torch.utils.data.dataloader. Defaults to False.
            sampler (Optional[Sampler[int]], optional): same as torch.utils.data.dataloader. Defaults to None.
            batch_sampler (Optional[_sampler_t], optional): same as torch.utils.data.dataloader. Defaults to None.
            num_workers (int, optional): if no positive integer specified, os.cpu_count will be used. Defaults to -1.
            collate_fn (Optional[_collate_fn_t], optional): same as torch.utils.data.dataloader. Defaults to None.
            pin_memory (bool, optional): same as torch.utils.data.dataloader. Defaults to True.
            drop_last (bool, optional): same as torch.utils.data.dataloader. Defaults to False.
            timeout (float, optional): same as torch.utils.data.dataloader. Defaults to 0.
            multiprocessing_context ([type], optional): same as torch.utils.data.dataloader. Defaults to None.
            prefetch_factor (int, optional): same as torch.utils.data.dataloader. Defaults to 2.
            persistent_workers (bool, optional): same as torch.utils.data.dataloader. Defaults to False.
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self._decide_workers_num(specified=num_workers),
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=self._worker_init_fn,
            generator=torch.Generator().manual_seed(specified_seed),
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def _worker_init_fn(self, worker_id):
        """
        see discussion at https://github.com/pytorch/pytorch/issues/5059
        and the article at https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        """
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    def _decide_workers_num(self, specified: int = -1) -> int:
        if specified > 0:
            return specified
        workers = os.cpu_count()
        if workers is None:
            return 2
        return workers
