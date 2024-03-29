"""
dataset.py

convert the asdf file with the data information to pytorch's dataset
expected asdf file:
1. data streams are stored in waveforms, tagged with raw_recording
2. the phase arrival time and the reference time are stored in auxiliary
"""
from os.path import isfile, join
from typing import Dict, List

import numpy as np
import torch
from obspy import UTCDateTime
from pyasdf import ASDFDataSet
from torch.utils.data import Dataset

from phasenet.conf import DataConfig


class WaveFormDataset(Dataset):
    def __init__(self, data_conf: DataConfig, data_type: str = "train", transform=None, stack_transform=None, replace_noise_transform=None, scale_at_end_transform=None, prepare: bool = False) -> None:
        super().__init__()
        self.data_conf = data_conf
        self.data_type = data_type
        self.transform = transform
        self.stack_transform = stack_transform
        self.replace_noise_transform = replace_noise_transform
        self.scale_at_end_transform = scale_at_end_transform

        # path related
        asdf_file_path = ""
        if self.data_type == "train":
            asdf_file_path = join(data_conf.data_dir, data_conf.train)
        elif self.data_type == "val":
            asdf_file_path = join(data_conf.data_dir, data_conf.val)
        elif self.data_type == "test":
            asdf_file_path = join(data_conf.data_dir, data_conf.test)
        else:
            raise Exception("data type must be train, val, or test!")

        cache_path = asdf_file_path+".pt"
        # to prepare data
        self.data: Dict[str, torch.Tensor] = {}
        # left and right, comparing with data, nearby
        self.left_data: Dict[str, torch.Tensor] = {}
        self.right_data: Dict[str, torch.Tensor] = {}
        self.noise_data: Dict[str, torch.Tensor] = {}
        self.label: Dict[str, torch.Tensor] = {}
        self.wave_keys: List[str] = []
        if self.data_conf.load_ps_freq:
            self.ps_freqs: Dict[str, float] = {}

        if prepare:
            # load files and save to torch cache
            # if has already been prepared
            if isfile(cache_path):
                return
            # prefetch data
            with ASDFDataSet(asdf_file_path, mode="r") as ds:
                self.wave_keys = ds.waveforms.list()
                aux_keys = [item.replace('.', "_") for item in self.wave_keys]
                for wk, ak in zip(self.wave_keys, aux_keys):
                    self.add_data(ds, wk, ak)
            # save cache
            self.save(cache_path)

        else:
            # already saved, now the fetch stage
            if not isfile(cache_path):
                raise Exception(f"cache path {cache_path} has no file.")
            self.load(cache_path)

    def add_data(self, ds: ASDFDataSet, wk: str, ak: str) -> None:
        # * handle times
        # add label
        arrival_times: List[float] = []
        if self.data_conf.load_ps_freq:
            ps_freq = ds.auxiliary_data["FPS"][ak].data[:][0]
        for phase in self.data_conf.phases:
            arrival_times.append(ds.auxiliary_data[phase][ak].data[:][0])
        # here we use nan, as some arrivals might be nan <=> not existing in our dataset
        start = np.nanmin(arrival_times)-self.data_conf.left_extend
        end = np.nanmin(arrival_times)+self.data_conf.right_extend
        if start < 0:
            # smaller than start time, reset it to 0
            # as tetsed, we always have start<=tp<=ts<=end
            end += -start
            start = 0
        left_signal_start = start-self.data_conf.win_length
        # cut window 10 seconds before to do taper
        left_signal_start -= 10
        left_signal_end = start
        right_signal_start = end
        right_signal_end = end+self.data_conf.win_length
        # update arrival_times based on start
        arrival_times = [item-start for item in arrival_times]
        # cut the dataset based on ref time
        ref_time = UTCDateTime(
            ds.auxiliary_data["REFTIME"][ak].data[:][0])
        start, end = ref_time+start, ref_time+end
        left_signal_start, left_signal_end = ref_time + \
            left_signal_start, ref_time+left_signal_end
        right_signal_start, right_signal_end = ref_time + \
            right_signal_start, ref_time+right_signal_end
        # * handle slices
        stream = ds.waveforms[wk].raw_recording
        # here we assume sampling_rate should be the same
        sampling_rate: float = stream[0].stats.sampling_rate

        res = torch.zeros(
            3, int(sampling_rate*self.data_conf.win_length))
        left_res = torch.zeros(
            3, int(sampling_rate*self.data_conf.win_length))
        right_res = torch.zeros(
            3, int(sampling_rate*self.data_conf.win_length))
        noise_res = torch.zeros(
            3, int(sampling_rate*self.data_conf.win_length))
        components = ["R", "T", "Z"]
        components_replace = ["E", "N", "Z"]
        for i in range(3):
            trace = stream.select(component=components[i])
            if len(trace) == 0:
                trace = stream.select(component=components_replace[i])[0]
            else:
                trace = trace[0]
            if start < trace.stats.starttime or end > trace.stats.endtime or trace.stats.endtime-self.data_conf.win_length-10 < end:
                # both signal and noise should be able to cut
                raise Exception(
                    f"{wk} has incorrect time or its length is too small")
            # signal processing
            # trace.detrend()
            # trace.taper(max_percentage=self.data_conf.taper_percentage)
            # taper and detrend will make it difficult to handle signal_left
            trace.filter('bandpass', freqmin=self.data_conf.filter_freqmin, freqmax=self.data_conf.filter_freqmax,
                         corners=self.data_conf.filter_corners, zerophase=self.data_conf.filter_zerophase)
            # cut
            wave = trace.slice(starttime=start, endtime=end)
            left_wave = trace.slice(
                starttime=left_signal_start, endtime=left_signal_end)
            right_wave = trace.slice(
                starttime=right_signal_start, endtime=right_signal_end)
            # cut the noise window 10 seconds before the end point to avoid the taper effect
            noise_wave = trace.slice(
                starttime=trace.stats.endtime-self.data_conf.win_length-10, endtime=trace.stats.endtime-10)
            # to torch
            wave_data = torch.from_numpy(
                wave.data)
            res[i, :] = wave_data[:res.shape[1]]
            noise_data = torch.from_numpy(
                noise_wave.data)
            noise_res[i, :] = noise_data[:noise_res.shape[1]]
            # left wave might not be that long, we need to fill the noise to avoid abrupt jump
            left_wave_data = torch.from_numpy(left_wave.data)
            # cut window 10 seconds before to do taper
            left_wave_data = left_wave_data[int(10*sampling_rate):]
            if len(left_wave_data) > 0:
                left_wave_data = left_wave_data[:left_res.shape[1]]
                left_res[i, -len(left_wave_data):] = left_wave_data[:]
                left_res[i, :-len(left_wave_data)
                         ] = noise_res[i, :-len(left_wave_data)]
            else:
                left_res[i, :] = noise_res[i, :]
            # right wave is reliable
            right_wave_data = torch.from_numpy(
                right_wave.data)
            right_res[i, :] = right_wave_data[:right_res.shape[1]]

        # update arrivals to idx of points
        arrival_times_new = []
        for item in arrival_times:
            if np.isnan(item):
                # we use -900000000 to indicate nan, as here arrival_times has the meaning of indices
                arrival_times_new.append(-900000000)
            else:
                arrival_times_new.append(round(item*sampling_rate))
        arrival_times = arrival_times_new

        self.data[wk] = res
        self.left_data[wk] = left_res
        self.right_data[wk] = right_res
        self.noise_data[wk] = noise_res
        self.label[wk] = torch.tensor(arrival_times, dtype=torch.int)
        if self.data_conf.load_ps_freq:
            self.ps_freqs[wk] = ps_freq

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int, stack_main: bool = True) -> Dict:
        # dict
        key = self.wave_keys[idx]
        sample = {
            "data": self.data[key],
            "left_data": self.left_data[key],
            "right_data": self.right_data[key],
            "noise_data": self.noise_data[key],
            "arrivals": self.label[key],
            "key": key
        }
        if self.data_conf.load_ps_freq:
            sample["ps_freqs"] = self.ps_freqs[key]
        if self.transform:
            sample = self.transform(sample)
        if stack_main and self.stack_transform:
            random_idx = torch.randint(len(self.data), (1,)).item()
            while random_idx == idx:
                random_idx = torch.randint(len(self.data), (1,)).item()
            random_sample = self.__getitem__(random_idx, stack_main=False)
            if torch.rand(1).item() <= self.data_conf.stack_ratio:
                # stack based on the ratio
                sample = self.stack_transform(sample, random_sample)
        if self.replace_noise_transform:
            sample = self.replace_noise_transform(sample)
        if self.scale_at_end_transform:
            sample = self.scale_at_end_transform(sample)
        return sample

    def save(self, file_name: str) -> None:
        # save the dataset to pt files
        tosave = {
            "wave_keys": self.wave_keys,
            "data": self.data,
            "left_data": self.left_data,
            "right_data": self.right_data,
            "noise_data": self.noise_data,
            "label": self.label
        }
        if self.data_conf.load_ps_freq:
            tosave["ps_freqs"] = self.ps_freqs
        torch.save(tosave, file_name)

    def load(self, file_name: str) -> None:
        # load the data from pt files
        toload = torch.load(file_name)
        self.wave_keys: List[str] = toload['wave_keys']
        self.data: Dict[str, torch.Tensor] = toload['data']
        self.left_data: Dict[str, torch.Tensor] = toload['left_data']
        self.right_data: Dict[str, torch.Tensor] = toload['right_data']
        self.noise_data: Dict[str, torch.Tensor] = toload['noise_data']
        self.label: Dict[str, torch.Tensor] = toload['label']
        if self.data_conf.load_ps_freq:
            self.ps_freqs: Dict[str, float] = toload["ps_freqs"]
