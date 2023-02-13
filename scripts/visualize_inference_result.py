"""
This program is designed to massively generate pdf files showing the waveforms predictions.
"""
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from matplotlib import pyplot as plt
from mpi4py import MPI
from obspy import UTCDateTime

comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
size = comm.Get_size()
rank = comm.Get_rank()

# * ======= configurations =======
# basedir = Path("/mnt/scratch/xiziyi/inference/train_bench_debug_max")
basedir = Path(
    "/mnt/home/xiziyi/Packages_Research/PhaseNet-PyTorch/continious_inference_result")
reference_path = Path(
    "/mnt/home/xiziyi/Packages_Research/PhaseNet-PyTorch/ai4eps/phase_picks.csv")
# output_path = Path("/mnt/scratch/xiziyi/inference/train_bench_debug_max_figs")
output_path = Path(
    "/mnt/home/xiziyi/Packages_Research/PhaseNet-PyTorch/continious_inference_result_figs")


def get_keys_this_rank():
    all_seeds = sorted(basedir.glob("*waveform.mseed"))
    keys = [".".join(item.name.split(".")[:-2]) for item in all_seeds]
    return np.array_split(keys, size)[rank]


def prepare_reference_pd():
    df = pd.read_csv(reference_path)
    res = defaultdict(lambda: defaultdict(list))
    for i in range(len(df)):
        row = df.iloc[i]
        res[row.station_id][row.phase_type].append(UTCDateTime(row.phase_time))
    for sta in res:
        for phase in res[sta]:
            res[sta][phase].sort()
    return res


def plot_wave_predict(key, phase_ref):
    wave = obspy.read(basedir/(key+".waveform.mseed"))
    predict = obspy.read(basedir/(key+".prediction.mseed"))
    e, n, z = wave[0].data, wave[1].data, wave[2].data
    p, s, ps = predict[0].data, predict[1].data, predict[2].data

    net, sta, start, _, end, _ = key.split(".")
    start_utctime = UTCDateTime(start)
    end_utctime = UTCDateTime(end)
    phases = ["P", "S", "PS"]
    refs = defaultdict(list)
    for phase in phases:
        left = bisect_left(phase_ref[sta][phase], start_utctime)
        right = bisect_left(phase_ref[sta][phase], end_utctime)
        refs[phase].extend(phase_ref[sta][phase][left:right])
        refs[phase] = [item-start_utctime for item in refs[phase]]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(
        100, 20), gridspec_kw={'wspace': 0, 'hspace': 0})
    start = start.split(":")[0]
    end = end.split(":")[0]
    axes[0].set_title(f"{net}.{sta}  {start}->{end}", fontsize=24)

    x = np.arange(len(n))/wave[0].stats.sampling_rate
    axes[0].plot(x, e, c="black", lw=0.1, label="E")
    axes[0].legend(fontsize=30)
    axes[1].plot(x, n, c="black", lw=0.1, label="N")
    axes[1].legend(fontsize=30)
    axes[2].plot(x, z, c="black", lw=0.1, label="Z")
    axes[2].legend(fontsize=30)

    axes[3].plot(x, p, c="red", lw=0.5, label="P")
    axes[3].plot(x, s, c="blue", lw=0.5, label="S")
    axes[3].plot(x, ps, c="purple", lw=0.5, label="PS")
    axes[3].legend(fontsize=30)
    axes[3].set_xticks(np.arange(0, 3601, 50))
    axes[3].set_xlabel("time (s)")

    for i in range(3):
        for each in refs["P"]:
            axes[i].axvline(x=each, ls='--', lw=1, c="red")
        for each in refs["S"]:
            axes[i].axvline(x=each, ls='--', lw=1, c="blue")
        for each in refs["PS"]:
            axes[i].axvline(x=each, ls='--', lw=1, c="purple")

    fig.savefig(output_path/f"{key}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    keys_this_rank = get_keys_this_rank()
    phase_ref = prepare_reference_pd()
    for key in keys_this_rank:
        plot_wave_predict(key, phase_ref)
