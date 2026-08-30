# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     displuser_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import os
from argparse import Namespace
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import Compose

# -

from myutils.datasets.audio import (
    ASV2021_AudioDs,
    ASV2021LA_AudioDs,
    ASVSpoof5_AudioDs,
    MLAAD_AudioDs,
)
from myutils.tools import color_print
from myutils.torch.transforms.audio import AudioRawBoost, SpecAugmentTransform_Wave
from myutils.torchaudio.transforms import (
    LFCC,
    RandomNoise,
    RawBoost,
    RandomBackgroundNoise,
)

# + editable=true slideshow={"slide_type": ""}
try:
    # from .datasets import ADD2023, LAV_DF_Audio, LibriSeVoc, WaveFake, DECRO
    from .tools import WaveDataset
except ImportError:
    # from datasets import ADD2023, LAV_DF_Audio, LibriSeVoc, WaveFake, DECRO
    from tools import WaveDataset



def make_ASVSpoof5(cfg):
    ds = ASVSpoof5_AudioDs(root_path=cfg.root_path)
    data_splits = ds.get_splits(only_test_vocoder=True)
    # data_splits.test = []
    # data_splits.test += ds.get_vocoder_test_splits()
    return data_splits
# -
# 其他数据集需要ASV5测试集做测试的时候使用
def get_ASV5_test_splits(root_path="/home/user/data/ASVSpoof5/ASVSpoof5"):
    ds = ASVSpoof5_AudioDs(root_path=root_path)
    res = ds.get_splits()
    return res.test
# ### ASV 2021


# +
def get_ASV2021_test_splits(root_path="/home/user/data/LA_2021/ASVspoof2021_LA_eval"):
    dataset = ASV2021_AudioDs(root_path=root_path)
    data_splits = dataset.get_test_splits()
    return data_splits


def get_ASV2021_whole_test_split(root_path="/home/user/data/LA_2021/ASVspoof2021_LA_eval"):
    dataset = ASV2021_AudioDs(root_path=root_path)
    test = dataset.get_whole_test_split()
    return [test]

def get_ASV2021_DF_test_split(root_path="/home/user/data/2021DF/ASVspoof2021_DF_eval"):
    dataset = ASV2021_AudioDs(root_path=root_path)
    test = [dataset.get_whole_test_split()]
    test_vocoder = test + dataset.get_test_splits()
    return test_vocoder

# -


def make_ASV2021(cfg):
    dataset = ASV2021_AudioDs(root_path=cfg.root_path)
    if cfg.task == "inner_eval":
        color_print("ASVspoof 2021 DF task: inner evaluation")
        data_splits = dataset.get_splits()

    data_splits.test += get_ASV2021_whole_test_split(root_path=cfg.root_path)
    return data_splits


def make_ASV2021_LA(cfg):
    dataset = ASV2021LA_AudioDs(root_path=cfg.root_path)
    if cfg.task == "inner_eval":
        color_print("ASVspoof 2021 LA task: inner evaluation")
        data_splits = dataset.get_splits()

    data_splits.test = [data_splits.test]
    # data_splits.test.append(get_ASV2019_test_split())
    # data_splits.test += get_ASV2021_whole_test_split()
    data_splits.test += get_ASV2021_DF_test_split()
    return data_splits



# ### MLAAD


def make_MLAAD(cfg):

    dataset = MLAAD_AudioDs(root_path=cfg.root_path)
    color_print("MLAAD: load splits")
    if cfg.task == "cross_lang":
        # data_splits = dataset.get_splits(language_list=["fr","it","pl","ru","uk"])
        data_splits = dataset.get_splits(language_list=["en"])
        data_splits.test=[]
    elif cfg.task == "same_lang":
        data_splits = dataset.get_splits_languagelist(language_list=["en", "de", "es","fr","it","pl","ru","uk"])
    elif cfg.task == "de_es_ru":
        print("MLDDA, load DE, ES, RU subsets for training!!!!!!!")
        data_splits = dataset.get_splits(language_list=["de", "es", "ru"])
        # append_test(data_splits)
    
    return data_splits


# ## Dict

MAKE_DATASETS = {
    "ASV2021": make_ASV2021,
    "ASV2021_LA": make_ASV2021_LA,
    "ASVSpoof5": make_ASVSpoof5,
    "MLAAD": make_MLAAD,
}


# # Build DataLoaders


def build_feature(cfg):
    if cfg.audio_feature == "LFCC":
        return LFCC()
    return None


# ## Transform

from myutils.torchaudio.transforms import RandomAudioCompression
from myutils.torchaudio.transforms.self_operation import (
    AudioToTensor,
    CentralAudioClip,
    RandomAudioClip,
    RandomPitchShift,
    RandomSpeed,
)

from audio_augmentations import *


def build_transforms(cfg=None, args=None):
    # t1 = RandomNoise(snr_min_db=10.0, snr_max_db=120.0, p=1.0)
    # # t = RawBoost(algo=[5], p=0.5)
    # t2 = RandomSpeed(min_speed=0.5, max_speed=2.0, p=1.0)
    # t3 = RandomPitchShift(p=1.0)

    # sr = 16000
    # num_samples=48000
    # transforms = [
    #     RandomResizedCrop(n_samples=num_samples),
    #     RandomApply([PolarityInversion()], p=0.8),
    #     RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
    #     RandomApply([Gain()], p=0.2),
    #     HighLowPass(sample_rate=sr), # this augmentation will alwusers be applied in this aumgentation chain!
    #     RandomApply([Deluser(sample_rate=sr)], p=0.5),
    #     RandomApply([PitchShift(
    #         n_samples=num_samples,
    #         sample_rate=sr
    #     )], p=0.4),
    #     RandomApply([Reverb(sample_rate=sr)], p=0.3)
    # ]

    # return {
    #     "train": transforms,
    #     "val": [
    #         CentralAudioClip(length=48000),
    #         AudioToTensor(),
    #     ],
    # }

    res = {
        "train": [
            RandomSpeed(min_speed=0.5, max_speed=2.0, p=0.5),
            # RandomAudioCompression(p=0.9),
            # RandomSpeed(min_speed=0.5, max_speed=2.0, p=1.0),
            RandomAudioClip(length=48000),
            RandomNoise(snr_min_db=10.0, snr_max_db=120.0, p=1.0),
            AudioToTensor(),
            # RandomApply([PitchShift(n_samples=48000, sample_rate=16000)], p=0.5),
            # RandomPitchShift(p=0.5),
        ],
        "val": [
            CentralAudioClip(length=48000),
            AudioToTensor(),
        ],
    }

    if args is not None and args.test_noise:
        res["test_noise"] = [
            CentralAudioClip(length=48000),
            RandomBackgroundNoise(
                16000,
                noise_dir="/home/user/data/0-原始数据集/musan/noise",
                p=1.0,
                min_snr_db=args.test_noise_level,
                max_snr_db=args.test_noise_level,
                noise_type=args.test_noise_type,
            ),
            AudioToTensor(),
        ]

    # if args.cfg.startswith('MPE_LCNN'):
    #     from myutilstorchaudio.transforms import MPE_LFCC
    #     for key in res:
    #         res[key].append(MPE_LFCC())

    return res


# ## Common Opeations


def build_dataloader(data: pd.DataFrame, cfg, is_training: bool = True, args=None):
    transforms = build_transforms(cfg.transforms, args=args)
    transform = transforms["train"] if is_training else transforms["val"]

    _ds = WaveDataset(
        data,
        sample_rate=cfg.sample_rate,
        normalize=True,
        transform=transform,
        dtype="tensor",
    )

    if not is_training and cfg.test_batch_size > 0:
        batch_size = cfg.test_batch_size
    else:
        batch_size = cfg.batch_size

    _dl = DataLoader(
        _ds,
        batch_size=batch_size,
        # num_workers=cfg.num_workers,
        num_workers=20,
        pin_memory=True,
        shuffle=True if is_training else False,
        # shuffle=True,
        prefetch_factor=2,
        drop_last=True if is_training else False,
    )
    return _ds, _dl


# ## Door


def over_sample_dataset(data, column="label"):
    n_fake = len(data[data[column] == 0])
    n_real = len(data[data[column] == 1])
    if n_fake == n_real:
        return data
    if n_fake > n_real:
        sampled = data[data[column] == 1].sample(n=n_fake - n_real, replace=True)
        balanced_data = pd.concat([data, sampled])
    else:
        sampled = data[data[column] == 0].sample(n=n_real - n_fake, replace=True)
        balanced_data = pd.concat([data, sampled])

    balanced_data = balanced_data.copy().reset_index(drop=True)
    return balanced_data


def print_audio_splits_label_distribution(audio_splits):
    res = {}
    for _split in ["train", "val", "test"]:
        _data = getattr(audio_splits, _split)
        res[_split] = ""
        if isinstance(_data, list):
            for _data2 in _data:
                tmp = _data2.groupby("label").count()
                num_0 = tmp.loc[0][0] if 0 in tmp.index else 0
                num_1 = tmp.loc[1][0] if 1 in tmp.index else 0
                res[_split] += f" {num_0}/{num_1}"
        else:
            tmp = _data.groupby("label").count()
            num_0 = tmp.loc[0][0] if 0 in tmp.index else 0
            num_1 = tmp.loc[1][0] if 1 in tmp.index else 0
            res[_split] += f" {num_0}/{num_1}"

    color_print(
        f"Fake/Real label distribution in train/val/test: {res['train']}, {res['val']}, {res['test']}"
    )


def make_data(cfg, args=None):
    # make audio splits (pd.DataFrame)
    audio_splits = MAKE_DATASETS[cfg.name](cfg.dataset_cfg)
    audio_splits.train = over_sample_dataset(audio_splits.train, column="label")

    print_audio_splits_label_distribution(audio_splits)

    # make dataset and dataloaders
    train_ds, train_dl = build_dataloader(
        audio_splits.train, cfg, is_training=True, args=args
    )
    train_ds2, train_dl2 = build_dataloader(
        audio_splits.train, cfg, is_training=False, args=args
    )
    val_ds, val_dl = build_dataloader(
        audio_splits.val, cfg, is_training=False, args=args
    )
    if isinstance(audio_splits.test, list):
        test_ds, test_dl = [], []
        for _test in audio_splits.test:
            _ds, _dl = build_dataloader(_test, cfg, is_training=False, args=args)
            test_ds.append(_ds)
            test_dl.append(_dl)
    else:
        test_ds, test_dl = build_dataloader(
            audio_splits.test, cfg, is_training=False, args=args
        )

    # collect all dataloaders
    ds = Namespace(
        train=train_ds, val=val_ds, test=test_ds, train_wo_transform=train_ds2
    )
    dl = Namespace(
        train=train_dl, val=val_dl, test=test_dl, train_wo_transform=train_dl2
    )

    print(args)
    if args is not None and args.test_noise:
        color_print("!!!!Test robustness: Load audio with background noise")
        test_noise = build_transforms(args=args)["test_noise"]
        if isinstance(dl.test, list):
            for _dl in dl.test:
                _dl.dataset.transform = test_noise
            # dl.test = dl.test[1]
        else:
            dl.test.transform = test_noise

    return ds, dl
