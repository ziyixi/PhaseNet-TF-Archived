"""
load_data.py

convert the asdf file with the data information to pytorch's dataset
expected asdf file:
1. data streams are stored in waveforms, tagged with raw_recording
2. the phase arrival time and the reference time are stored in auxiliary
"""
from collections import defaultdict
from typing import List, Optional, Type

import torch
from obspy import Trace, UTCDateTime
from pyasdf import ASDFDataSet
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


class WaveFormDataset(Dataset):
    """
    Dataset representing the waveform and phase tag
    """

    def __init__(self, asdf_file: str, transform: Optional[Type[Module]] = None):
        """
        asdf_file represents the asdf file path with the expected content
        transform is the transformation class
        """
        self.ds = ASDFDataSet(asdf_file, mode="r")
        self.wave_keys: List[str] = self.ds.waveforms.list()
        self.aux_keys = [item.replace('.', "_") for item in self.wave_keys]
        self.transform = transform
        # some settings
        self.phases = ["TP", "TS", "TPS"]  # the phases name in aux
        self.window_length = 120  # in seconds, data and noise
        # we have to valid all the data is valid, also set the delta of the
        self.sampling_rate = 0.
        self.valid()

    def valid(self):
        # all the sampling_rate is the same
        ref_sampling_rate: float = self.ds.waveforms[self.wave_keys[0]
                                                     ].raw_recording[0].stats.sampling_rate
        for key in self.wave_keys[1:]:
            for each in self.ds.waveforms[key].raw_recording:
                test_sampling_rate: float = each.stats.sampling_rate
                if test_sampling_rate != ref_sampling_rate:
                    raise Exception(
                        "sampling rate inside the asdf dataset is not consistent")
        # set sampling rate
        self.sampling_rate = ref_sampling_rate

    def __len__(self):
        return len(self.wave_keys)

    def __getitem__(self, idx: int):
        wave_key = self.wave_keys[idx]
        aux_key = self.aux_keys[idx]
        # load phases travel time
        phase_travel_times = defaultdict(float)
        for phase in self.phases:
            phase_travel_times[phase] = self.ds.auxiliary_data[phase][aux_key].data[:][0]
        # extend half before and after the mid
        mid = (max(phase_travel_times.values()) +
               min(phase_travel_times.values()))/2
        start, end = mid-self.window_length/2, mid+self.window_length/2
        if start < 0:
            # smaller than start time, reset it to 0
            # as tetsed, we always have start<=tp<=ts<=end
            end += -start
            start = 0
        # cut the dataset based on ref time
        ref_time = UTCDateTime(
            self.ds.auxiliary_data["REFTIME"][aux_key].data[:][0])
        start, end = ref_time+start, ref_time+end
        # for the data, we cut it between start and end
        stream = self.ds.waveforms[wave_key].raw_recording
        # start should be after starttime, end should be before endtime
        # endtime-window_length should be after end
        res = torch.zeros(int(self.sampling_rate*self.window_length), 3)
        # the order here is always R,T,Z (assume we don't have 123)
        trace: Trace
        for i, trace in enumerate(stream):
            if start < trace.stats.starttime or end > trace.stats.endtime or trace.stats.endtime-self.window_length < end:
                raise Exception(
                    f"{wave_key} has incorrect time or its length is too small")
            wave_data = torch.from_numpy(
                trace.slice(starttime=start, endtime=end).data)
            res[:len(wave_data), i] = wave_data
