"""
inference_dataset.py

Dataset related to the continious waveforms. 
"""
from typing import Dict

import numpy as np
import pandas as pd
import torch
from obspy import UTCDateTime
from obspy.clients.filesystem.tsindex import Client
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from phasenet.conf import DataConfig, InferenceConfig


class SeedSqliteDataset(Dataset):
    def __init__(self, inference_conf: InferenceConfig, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.client = Client(database=str(inference_conf.sqlite_path))
        # * index all the requirement
        requirement = pd.read_csv(inference_conf.continious_requirement_path)
        requirement.sort_values(
            by=["network", "station", "start_time", "end_time"], ascending=[True, True, True, True])

        self.all_splits = []
        for irow in range(len(requirement)):
            row = requirement.iloc[irow]
            time_diff = (UTCDateTime(row.end_time)-UTCDateTime(row.start_time))
            steps = np.ceil(
                time_diff/inference_conf.continious_handle_time)
            for istep in range(steps):
                start = UTCDateTime(
                    row.start_time)+istep*inference_conf.continious_handle_time
                end = UTCDateTime(row.start_time)+(istep+1) * \
                    inference_conf.continious_handle_time
                self.all_splits.append((row.network, row.station, start, end))

    def __len__(self) -> int:
        return len(self.all_splits)

    def __getitem__(self, idx: int) -> Dict:
        net, sta, start, end = self.all_splits[idx]
        st = self.client.get_waveforms(net, sta, "*", "*", start, end)
        # * the transform will expect st as the input, after processing, return tensor dict
        res = {
            "stream": st
        }
        if self.transform:
            res = self.transform(res)
        return res


class ProcessSeedTransform:
    def __init__(self, data_conf: DataConfig) -> None:
        # * current we only do filtering
        self.data_conf = data_conf

    def __call__(self, sample: Dict) -> Dict:
        stream = sample["stream"]
        stream.filter('bandpass', freqmin=self.data_conf.filter_freqmin, freqmax=self.data_conf.filter_freqmax,
                      corners=self.data_conf.filter_corners, zerophase=self.data_conf.filter_zerophase)
        sample["stream"] = stream
        return sample


class StreamToTensorTransform:
    def __init__(self, inference_conf: InferenceConfig) -> None:
        self.inference_conf = inference_conf

    def __call__(self, sample: Dict) -> Dict:
        # * to avoid large memory cost, pop stream after this step
        stream = sample.pop("stream")
        components = ["R", "T", "Z"]
        components_replace = ["E", "N", "Z"]
        traces = []
        for i in range(3):
            trace = stream.select(component=components[i])
            if len(trace) == 0:
                trace = stream.select(component=components_replace[i])
            trace = trace[0]
            traces.append(trace)
        min_length = min(len(item) for item in traces)
        # we have to pad 0, so min_length can be divied by sliding_step
        # and it's at least width
        div = np.ceil(min_length/self.inference_conf.sliding_step)
        min_length = self.inference_conf.sliding_step*div
        if min_length < self.inference_conf.width:
            min_length = self.inference_conf.width
        # in the extreme case, min_length might be 0
        data = torch.zeros(3, min_length)
        for i in range(3):
            data[i, :min_length] = traces.data[:min_length]
        sample["data"] = data
        return sample


class StreamNormalizeTransform:
    def __init__(self, inference_conf: InferenceConfig) -> None:
        # * do sliding window normalization for the waveform data
        self.inference_conf = inference_conf

    def __call__(self, sample: Dict) -> Dict:
        data = sample["data"]
        length = data.shape[1]
        steps = (length-self.inference_conf.width)//self.inference_conf.sliding_step+1
        # * calculate each sliding windows' mean and std
        means = np.zeros((3, steps))
        stds = np.zeros((3, steps))
        times = np.zeros(steps)
        for istep in range(steps):
            cur = data[:, istep*self.inference_conf.sliding_step:istep *
                       self.inference_conf.sliding_step+self.inference_conf.width]
            means[:, istep] = torch.mean(cur, axis=1).numpy()
            stds[:, istep] = torch.std(cur, axis=1).numpy()
            times[istep] = istep * self.inference_conf.sliding_step + \
                self.inference_conf.width//2
        interp_func_means = interp1d(times, means, kind="cubic", bounds_error=False,
                                     fill_value="extrapolate", assume_sorted=True)
        interp_func_stds = interp1d(times, stds, kind="cubic", bounds_error=False,
                                    fill_value="extrapolate", assume_sorted=True)
        # * now we normalize the raw dataset
        data_mean = torch.tensor(interp_func_means(np.arange(length)))
        data_std = torch.tensor(interp_func_stds(np.arange(length)))
        data_std[data_std < 1e-8] = 1.
        data = (data-data_mean)/data_std
        sample["data"] = data
        return sample
