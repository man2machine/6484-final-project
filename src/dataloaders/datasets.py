""" Datasets for core experimental results """

from functools import partial
import os
import io
from pathlib import Path

import logging
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from PIL import Image  # Only used for Pathfinder
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
import torchtext
from datasets import load_dataset, DatasetDict, Value
from models.utils import D4RLTrajectoryDataset, S4D4RLTrajectoryDataset

# from pytorch_lightning import LightningDataModule

from src.utils import permutations, is_list
import pickle

# Default data path is environment variable or hippo/data
if (default_data_path := os.getenv("DATA_PATH")) is None:
    default_data_path = Path(__file__).parent.parent.parent.absolute()
    default_data_path = default_data_path / "data"
else:
    default_data_path = Path(default_data_path).absolute()


class TBPTTDataLoader(torch.utils.data.DataLoader):
    """
    Adapted from https://github.com/deepsound-project/samplernn-pytorch
    """

    def __init__(
        self, 
        dataset, 
        batch_size, 
        chunk_len,
        overlap_len,
        *args, 
        **kwargs
    ):
        super().__init__(dataset, batch_size, *args, **kwargs)
        
        # Zero padding value, given by the dataset
        self.zero = dataset.zero if hasattr(dataset, "zero") else 0

        # Size of the chunks to be fed into the model
        self.chunk_len = chunk_len

        # Keep `overlap_len` from the previous chunk (e.g. SampleRNN requires this)
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            x, y, *z = batch

            # Pad with self.overlap_len - 1 zeros
            x = torch.cat(
                [
                    torch.zeros((x.shape[0], self.overlap_len - 1, *x.shape[2:])).to(x.device).to(x.dtype) + self.zero,
                    x
                ],
                dim=1,
            )
            y = torch.cat(
                [
                    torch.zeros((y.shape[0], self.overlap_len - 1, *y.shape[2:])).to(y.device).to(y.dtype) + self.zero,
                    y
                ],
                dim=1,
            )
            z = [
                torch.cat(
                    [
                        torch.zeros((z[i].shape[0], self.overlap_len - 1, *z[i].shape[2:])).to(z[i].device).to(z[i].dtype),
                        z[i]
                    ],
                    dim=1,
                )
                for i in range(len(z)) if len(z[i].shape) > 1
            ]

            _, seq_len, *_ = x.shape

            reset = True

            for seq_begin in list(range(self.overlap_len - 1, seq_len, self.chunk_len))[:-1]:
                from_index = seq_begin - self.overlap_len + 1
                to_index = seq_begin + self.chunk_len
                # TODO: check this
                # Ensure divisible by overlap_len
                if self.overlap_len > 0:
                    to_index = min(to_index, seq_len - ((seq_len - self.overlap_len + 1) % self.overlap_len))
            
                x_chunk = x[:, from_index:to_index]
                if len(y.shape) == 3:
                    y_chunk = y[:, seq_begin:to_index]
                else:
                    y_chunk = y
                z_chunk = [z_[:, from_index:to_index] for z_ in z if len(z_.shape) > 1]

                yield (x_chunk, y_chunk, *z_chunk, reset)

                reset = False

    def __len__(self):
        raise NotImplementedError()


# class SequenceDataset(LightningDataModule):
# [21-09-10 AG] Subclassing LightningDataModule fails due to trying to access _has_setup_fit. No idea why
class SequenceDataset:
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class
    # Subclasses can provide a list of default arguments which are automatically registered as attributes
    # TODO apparently there is a python 3.8 decorator that basically does this
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, tbptt=False, chunk_len=None, overlap_len=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        # Arguments for TBPTT: only used if tbptt is True and are passed to TBPTTDataLoader 
        self.tbptt = tbptt
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len

        # Add all arguments to self
        init_args = self.init_defaults
        init_args.update(
            dataset_cfg
        )  # TODO this overrides the default dict which is bad
        for k, v in init_args.items():
            setattr(self, k, v)

        self.init()  # Extra init stuff if desired # TODO get rid of this

        # train, val, test datasets must be set by class instantiation
        self.dataset_train = None
        self.dataset_val = None

    def init(self):
        pass

    def setup(self):
        """This method should set self.dataset_train and self.dataset_val"""
        raise NotImplementedError

    def split_train_val(self, val_split):
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    @staticmethod
    def collate_fn(batch, resolution=1):
        """batch: list of (x, y) pairs"""
        def _collate(batch, resolution=1):
            # From https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum(x.numel() for x in batch)
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                x = torch.stack(batch, dim=0, out=out)
                if resolution is not None:
                    x = x[:, ::resolution] # assume length is first axis after batch
                return x
            else:
                return torch.tensor(batch)

        x, y = zip(*batch)
        # Drop every nth sample
        # x = torch.stack(x, dim=0)[:, ::resolution]
        # y = torch.LongTensor(y)
        # y = torch.tensor(y)
        # y = torch.stack(y, dim=0)
        x = _collate(x, resolution=resolution)
        y = _collate(y, resolution=None)
        return x, y

    def train_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        if train_resolution is None:
            train_resolution = [1]
        if not is_list(train_resolution):
            train_resolution = [train_resolution]
        assert len(train_resolution) == 1, "Only one train resolution supported for now"

        return self._dataloader(
            self.dataset_train,
            resolutions=train_resolution,
            shuffle=True,
            **kwargs,
        )[0]

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def _eval_dataloader(self, dataset, train_resolution, eval_resolutions, **kwargs):
        if eval_resolutions is None:
            eval_resolutions = [1]
        if not is_list(eval_resolutions):
            eval_resolutions = [eval_resolutions]

        kwargs["shuffle"] = False if "shuffle" not in kwargs else kwargs["shuffle"]
        dataloaders = self._dataloader(
            dataset,
            resolutions=eval_resolutions,
            # shuffle=False,
            **kwargs,
        )

        return (
            {
                str(res) if res > 1 else None: dl
                for res, dl in zip(eval_resolutions, dataloaders)
            }
            if dataloaders is not None
            else None
        )

    def _dataloader(self, dataset, resolutions, **loader_args):
        if dataset is None:
            return None

        if self.tbptt:
            DataLoader = partial(TBPTTDataLoader, chunk_len=self.chunk_len, overlap_len=self.overlap_len)
        else:
            DataLoader = torch.utils.data.DataLoader

        return [
            DataLoader(
                dataset=dataset,
                collate_fn=partial(self.collate_fn, resolution=resolution)
                if self.collate_fn is not None
                else None,
                **loader_args,
            )
            for resolution in resolutions
        ]

    def __str__(self):
        return self._name_


class MNIST(SequenceDataset):
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    l_output = 0
    L = 784

    @property
    def init_defaults(self):
        return {
            "permute": True,
            "val_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(self.d_input, self.L).t()),
        ]  # (L, d_input)
        if self.permute:
            # below is another permutation that other works have used
            # permute = np.random.RandomState(92916)
            # permutation = torch.LongTensor(permute.permutation(784))
            permutation = permutations.bitreversal_permutation(self.L)
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: x[permutation])
            )
        # TODO does MNIST need normalization?
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
        transform = torchvision.transforms.Compose(transform_list)
        self.dataset_train = torchvision.datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class GymHopper(SequenceDataset):
    _name_ = "hopper"
    d_input = 15 # 1 for reward, 11 for state dim, 3 for action dim
    d_output = 3 # 3 for action dim
    l_output = None
    rtg_scale = 1000
    rtg_sparse_flag = False
    L = 20

    @property
    def init_defaults(self):
        return {
            "val_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = default_data_path / "hopper-medium-v2.pkl"
        self.dataset_train = S4D4RLTrajectoryDataset(
            self.data_dir,
            context_len=self.L,
            rtg_scale=self.rtg_scale,
            rtg_sparse_flag=self.rtg_sparse_flag,
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"

