"""
dataset.py

convert the asdf file with the data information to pytorch's dataset
expected asdf file:
1. data streams are stored in waveforms, tagged with raw_recording
2. the phase arrival time and the reference time are stored in auxiliary
"""
from trace import Trace
from typing import Dict, List

import torch
from obspy import Trace, UTCDateTime
from phasenet.conf.load_conf import Config
from pyasdf import ASDFDataSet
from torch.utils.data import Dataset
from tqdm import tqdm


class WaveFormDataset(Dataset):
    """
    Waveform dataset and phase arrival time tag.
    """

    def __init__(self, conf: Config, data_type: str = "train", transform=None, progress=False, debug=False, debug_dict={}) -> None:
        super().__init__()
        self.conf = conf
        self.data_type = data_type
        self.transform = transform

        # * asdf path
        asdf_file_path = ""
        if self.data_type == "train":
            asdf_file_path = conf.data.train
        elif self.data_type == "test":
            asdf_file_path = conf.data.test
        else:
            raise Exception("data type must be train or test!")

        # * prefetch data
        # data dict, flag dict
        self.data: Dict[str, torch.Tensor] = {}
        self.label: Dict[str, torch.Tensor] = {}
        with ASDFDataSet(asdf_file_path, mode="r") as ds:
            wave_keys: List[str] = ds.waveforms.list()
            aux_keys = [item.replace('.', "_") for item in wave_keys]
            if debug:
                wave_keys = wave_keys[:debug_dict['size']]
                aux_keys = aux_keys[:debug_dict['size']]
            if progress:
                iters = tqdm(zip(wave_keys, aux_keys), total=len(wave_keys))
            else:
                iters = zip(wave_keys, aux_keys)
            for wk, ak in iters:
                self.add_data(ds, wk, ak)
        # for quick get idx
        self.wave_keys = wave_keys

    def add_data(self, ds: ASDFDataSet, wk: str, ak: str) -> None:
        # add label
        arrival_times: List[float] = []
        for phase in self.conf.data.phases:
            arrival_times.append(ds.auxiliary_data[phase][ak].data[:][0])
        # mid = (max(arrival_times) + min(arrival_times))/2
        # start, end = mid-self.conf.preprocess.left_extend_mid, mid + \
        #     self.conf.preprocess.right_extend_mid
        start = min(arrival_times)-self.conf.preprocess.left_extend
        end = min(arrival_times)+self.conf.preprocess.right_extend
        if start < 0:
            # smaller than start time, reset it to 0
            # as tetsed, we always have start<=tp<=ts<=end
            end += -start
            start = 0
        # update arrival_times based on start
        arrival_times = [item-start for item in arrival_times]
        # cut the dataset based on ref time
        ref_time = UTCDateTime(
            ds.auxiliary_data["REFTIME"][ak].data[:][0])
        start, end = ref_time+start, ref_time+end
        stream = ds.waveforms[wk].raw_recording
        # here we assume sampling_rate should be the same
        sampling_rate: float = stream[0].stats.sampling_rate

        res = torch.zeros(
            3, int(sampling_rate*self.conf.preprocess.win_length))
        components = ["R", "T", "Z"]
        for i in range(3):
            trace = stream.select(component=components[i])[0]
            if start < trace.stats.starttime or end > trace.stats.endtime or trace.stats.endtime-self.conf.preprocess.win_length < end:
                # both signal and noise should be able to cut
                raise Exception(
                    f"{wk} has incorrect time or its length is too small")
            wave = trace.slice(starttime=start, endtime=end)
            # signal processing
            wave.detrend()
            wave.taper(max_percentage=self.conf.preprocess.taper_percentage)
            wave.filter('bandpass', freqmin=self.conf.preprocess.filter_freqmin, freqmax=self.conf.preprocess.filter_freqmax,
                        corners=self.conf.preprocess.filter_corners, zerophase=self.conf.preprocess.filter_zerophase)
            # to torch
            wave_data = torch.from_numpy(
                wave.data)
            res[i, :] = wave_data[:res.shape[1]]

        # update arrivals to idx of points
        arrival_times = [round(item*sampling_rate) for item in arrival_times]

        self.data[wk] = res
        self.label[wk] = torch.tensor(arrival_times, dtype=torch.int)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        # dict
        key = self.wave_keys[idx]
        # sample = (self.data[key], self.label[key])
        sample = {
            "data": self.data[key],
            "arrivals": self.label[key],
            "key": key
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
