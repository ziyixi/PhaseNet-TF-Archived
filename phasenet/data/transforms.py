"""
transforms.py

custome transforms applied to waveform dataset.
"""
from typing import Dict

import torch
from phasenet.conf import DataConfig, SpectrogramConfig
from phasenet.data.sgram import GenSgram


class RandomShift:
    def __init__(self, data_conf: DataConfig) -> None:
        self.width = data_conf.width

    def __call__(self, sample: Dict) -> Dict:
        sample_updated = sample.copy()
        data, arrivals = sample_updated["data"], sample_updated["arrivals"]
        left_data, right_data = sample_updated["left_data"], sample_updated["right_data"]
        # determine the shift range
        # we filter pos vals to avoid considering negative indices, i.e., nan
        left_bound = -torch.max(arrivals[arrivals > 0])
        right_bound = self.width-torch.min(arrivals[arrivals > 0])
        # update arrivals
        shift = torch.randint(left_bound, right_bound, (1,)).item()
        arrivals_shifted = arrivals.clone()
        for i in range(len(arrivals)):
            arrivals_shifted[i] += shift
        # update data
        data_shifted = data.roll(shift, dims=1)
        if shift == 0:
            pass
        elif shift > 0:
            data_shifted[:, :shift] = left_data[:, -shift:]
        else:
            # note shift<0 when modifying the code
            data_shifted[:, shift:] = right_data[:, :-shift]
        sample_updated.update({
            "data": data_shifted,
            "arrivals": arrivals_shifted
        })
        return sample_updated


class GenLabel:
    def __init__(self, data_conf: DataConfig) -> None:
        self.label_shape = data_conf.label_shape
        self.label_width = data_conf.label_width

    def __call__(self, sample: Dict) -> Dict:
        sample_updated = sample.copy()
        data, arrivals = sample_updated["data"], sample_updated["arrivals"]
        res = torch.zeros(len(arrivals)+1, data.shape[1])
        if self.label_shape == "gaussian":
            label_window = torch.exp(-(torch.arange(-self.label_width //
                                     2, self.label_width//2+1))**2/(2*(self.label_width/6)**2))
        elif self.label_shape == "triangle":
            label_window = 1 - \
                torch.abs(
                    2/self.label_width * (torch.arange(-self.label_width//2, self.label_width//2+1)))
        else:
            raise Exception(
                f"label shape {self.label_shape} is not supported!")

        # the first class set as noise
        for i, idx in enumerate(arrivals):
            # the index for arrival times
            start = idx-self.label_width//2
            end = idx+self.label_width//2+1
            if start >= 0 and end <= res.shape[1]:
                res[i+1, start:end] = label_window
        # can sum as the first row is 0
        res[0, :] = 1-torch.sum(res, 0)
        sample_updated.update({
            "label": res
        })
        return sample_updated


class ReplaceNoise:
    def __init__(self, data_conf: DataConfig) -> None:
        self.noise_replace_ratio = data_conf.noise_replace_ratio

    def __call__(self, sample: Dict) -> Dict:
        sample_updated = sample.copy()
        if torch.rand(1).item() > self.noise_replace_ratio:
            return sample_updated
        else:
            data: torch.Tensor = sample_updated['data'].clone()
            label: torch.Tensor = sample_updated['label'].clone()
            noise_data: torch.Tensor = sample_updated['noise_data']
            # replace by noise
            data[:, :] = noise_data[:, :]
            label[0, :] = 1.
            label[1:, :] = 0.
            sample_updated.update({
                'data': data,
                'label': label
            })
            return sample_updated


class StackRand:
    def __init__(self, data_conf: DataConfig) -> None:
        self.min_stack_gap = data_conf.min_stack_gap

    def __call__(self, sample: Dict, random_sample: Dict) -> Dict:
        # * stack data / label
        sample_updated = sample.copy()
        arrivals, data, label = sample_updated['arrivals'],  sample_updated['data'],  sample_updated['label']
        random_arrivals, random_data, random_label = random_sample[
            'arrivals'],  random_sample['data'],  random_sample['label']
        # if arrivals overlap, skip
        for i in range(len(arrivals)):
            for j in range(len(random_arrivals)):
                # avoid nan
                if arrivals[i] < -800000000 or random_arrivals[i] < -800000000:
                    continue
                if torch.abs(arrivals[i]-random_arrivals[j]) <= self.min_stack_gap:
                    return sample_updated
        # handle stacking
        stack_data = data+random_data
        stack_label = label+random_label
        # * here we have to scale the poss to 1 for signals and recalculate the noise
        stack_label = torch.clamp_max(stack_label, 1.0)
        stack_label[0, :] = 1-torch.sum(stack_label[1:], 0)

        sample_updated.update({
            'data': stack_data,
            'label': stack_label
        })
        return sample_updated


class ScaleAmp:
    def __init__(self, data_conf: DataConfig) -> None:
        self.max_amp = data_conf.scale_max_amp
        self.global_max = data_conf.scale_global_max
        self.scale_norm = data_conf.scale_norm

    def __call__(self, sample: Dict) -> Dict:
        if not self.scale_norm:
            return self.scale_max_call(sample)
        else:
            return self.scale_norm_call(sample)

    def scale_max_call(self, sample: Dict) -> Dict:
        sample_updated = sample.copy()
        data: torch.Tensor = sample_updated["data"].clone()
        left_data: torch.Tensor = sample_updated["left_data"].clone()
        right_data: torch.Tensor = sample_updated["right_data"].clone()
        noise_data: torch.Tensor = sample_updated["noise_data"].clone()
        if self.global_max:
            raw_max = torch.max(torch.abs(data))
            data = data/raw_max
            left_data = left_data/raw_max
            right_data = right_data/raw_max
            noise_data = noise_data/raw_max
        else:
            for ich in range(data.shape[0]):
                raw_max = torch.max(torch.abs(data[ich]))
                data[ich, :] = data[ich, :]/raw_max
                left_data[ich, :] = left_data[ich, :]/raw_max
                right_data[ich, :] = right_data[ich, :]/raw_max
                noise_data[ich, :] = noise_data[ich, :]/raw_max
        sample_updated.update({
            "data": data,
            "left_data": left_data,
            "right_data": right_data,
            "noise_data": noise_data
        })
        return sample_updated

    def scale_norm_call(self, sample: Dict) -> Dict:
        sample_updated = sample.copy()
        data: torch.Tensor = sample_updated["data"].clone()
        if self.global_max:
            # data ch, nt
            mean_vals = torch.mean(data, axis=1, keepdim=True)
            data = data-mean_vals
            max_std_val = torch.max(torch.std(data, axis=1))
            if max_std_val == 0:
                max_std_val = torch.ones(1)
            data = data/max_std_val
        else:
            mean_vals = torch.mean(data, axis=1, keepdim=True)
            data = data-mean_vals
            std_vals = torch.std(data, axis=1, keepdim=True)
            std_vals[std_vals == 0] = 1
            data = data/std_vals
        sample_updated.update({
            "data": data
        })
        return sample_updated


class TransformSgram:
    def __init__(self, spec_conf: SpectrogramConfig) -> None:
        self.sgram_trans = GenSgram(spec_conf)

    def __call__(self, sample: Dict) -> Dict:
        sample_updated = sample.copy()
        wave = sample_updated["data"]
        sample_updated["sgram"] = self.sgram_trans(wave)
        return sample_updated
