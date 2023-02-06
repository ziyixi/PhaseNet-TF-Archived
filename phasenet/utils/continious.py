from pathlib import Path
from typing import Dict, List

import obspy
import pytorch_lightning as pl
import torch
from obspy import UTCDateTime
from pytorch_lightning.callbacks import BasePredictionWriter

from phasenet.conf import InferenceConfig


def convert_continious_to_batch(input_array: torch.Tensor, width: int, sliding_step: int) -> torch.Tensor:
    """
    Assume the input array has shape 1,cha,time, convert to div,cha,time
    here cha=time//sliding_step, and time%sliding_step==0
    """
    _, cha, time = input_array.shape
    steps = (time-width)//sliding_step+1
    res = torch.zeros(steps, cha, width, device=input_array.device)
    for istep in range(steps):
        res[istep, :, :] = input_array[0, :, istep *
                                       sliding_step:width+istep*sliding_step]
    return res


def convert_batch_to_continious(input_array: torch.Tensor, width: int, sliding_step: int) -> torch.Tensor:
    """
    Assume the input array has shape div,cha,time, convert to 1,cha,time
    here cha=time//sliding_step, and time%sliding_step==0

    This step also involves combing the result from several predictions for a single point
    """
    steps, cha, time = input_array.shape
    time = (steps-1)*sliding_step+width
    res = torch.zeros(1, cha, time, device=input_array.device)
    counts = torch.zeros(1, cha, time, device=input_array.device)
    for istep in range(steps):
        res[0, :, istep * sliding_step:width+istep *
            sliding_step] += input_array[istep, :, :]
        counts[0, :, istep * sliding_step:width+istep *
               sliding_step] += 1
    res = res/counts
    return res


class InferenceWriter(BasePredictionWriter):
    def __init__(self,  phases: List[str], inference_conf: InferenceConfig) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = inference_conf.inference_output_dir
        self.phases = phases
        self.sampling_rate = inference_conf.sampling_rate

        self.save_prediction_stream = inference_conf.save_prediction_stream
        self.save_waveform_stream = inference_conf.save_waveform_stream
        self.save_phase_arrivals = inference_conf.save_phase_arrivals

    def write_on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prediction, batch_indices, batch: Dict, batch_idx: int, dataloader_idx: int) -> None:
        if "data" not in batch:
            return

        start = UTCDateTime(batch["start"][0])
        end = UTCDateTime(batch["end"][0])
        net = batch["net"][0]
        sta = batch["sta"][0]

        # * save to phase_arrivals.csv
        if self.save_phase_arrivals:
            phase_save_path = self.output_dir/"phase_arrivals.csv"
            if trainer.global_rank == 0 and (not phase_save_path.is_file()):
                with phase_save_path.open("w") as f:
                    f.write("net,sta,start,end,phase,point,time,amp\n")

            with phase_save_path.open("a") as f:
                for iphase, phase in enumerate(self.phases):
                    for arrival, amp in zip(prediction["arrivals"][0][iphase], prediction["amps"][0][iphase]):
                        phase_offset = float(
                            f"{arrival/self.sampling_rate:.2f}")
                        f.write(
                            f"{net},{sta},{str(start)},{str(end)},{phase},{arrival},{str(start+phase_offset)},{amp:.2f}\n")

        # * save to net.sta.start.end.waveform.sac
        if self.save_waveform_stream:
            fname = self.output_dir / \
                f"{net}.{sta}.{str(start)}.{str(end)}.waveform.mseed"
            stream = obspy.Stream()
            for icomponent in range(len(batch["ids"])):
                d = batch["raw_data"][0][iphase].detach().cpu().numpy()
                trace = obspy.Trace(data=d)
                trace.stats.starttime = start
                trace.stats.sampling_rate = self.sampling_rate
                network, station, locaton, channel = batch["ids"][icomponent][0].split(
                    ".")
                trace.stats.network = network
                trace.stats.station = station
                trace.stats.locaton = locaton
                trace.stats.channel = channel
                stream += trace
            stream.write(str(fname), format="MSEED")

        # * write the prediction result stream
        if self.save_prediction_stream:
            fname = self.output_dir / \
                f"{net}.{sta}.{str(start)}.{str(end)}.prediction.mseed"
            stream = obspy.Stream()
            for iphase, phase in enumerate(self.phases):
                d = prediction["predict"][0][iphase+1].detach().cpu().numpy()
                trace = obspy.Trace(data=d)
                trace.stats.starttime = start
                trace.stats.sampling_rate = self.sampling_rate
                trace.stats.network = net
                trace.stats.station = sta
                trace.stats.locaton = ""
                trace.stats.channel = phase
                stream += trace
            stream.write(str(fname), format="MSEED")
