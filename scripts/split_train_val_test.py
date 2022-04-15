"""
split_train_val_test.py

given the asdf file, split it into the trainning dataset and the test dataset.
train+valid 80% val 10% test 10%, so here we split to 8:2
"""
from os.path import basename, dirname, join
from typing import List

import click
import numpy as np
from pyasdf import ASDFDataSet
from tqdm import tqdm


def generate_new(fname: str, data_lists: List[str], raw: ASDFDataSet, desc: str):
    """
    generate new asdf file based on data_lists from raw
    """
    with ASDFDataSet(fname, mode="w") as ds:
        ds.add_quakeml(raw.events)
        for key in tqdm(data_lists, desc=desc):
            aux_key = key.replace(".", "_")
            stream = raw.waveforms[key]['raw_recording']
            ds.add_waveforms(stream, tag="raw_recording")
            for tag in raw.auxiliary_data.list():
                val = raw.auxiliary_data[tag][aux_key].data[:]
                ds.add_auxiliary_data(
                    data=val, data_type=tag, path=aux_key, parameters={})
            inv = raw.waveforms[key]["StationXML"]
            ds.add_stationxml(inv)


@click.command()
@click.option("--asdf_data_path", help="input ASDF file path")
@click.option("--seed", help="input ASDF file path", type=int, default=-1)
@click.option("--split_ratio", help="input ASDF file path", type=str, default="0.8,0.1,0.1")
def main(asdf_data_path: str, seed: int, split_ratio: str):
    """
    given the input asdf data path, the random seed, generate the trainning and test dataset
    the random seed is default to -1, which means the random seed
    """
    split_ratio_list = [float(item) for item in split_ratio.split(",")]

    asdf_base = basename(asdf_data_path)
    asdf_head = ".".join(asdf_base.split(".")[:-1])
    asdf_dir = dirname(asdf_data_path)
    train_base = f"{asdf_head}_train.h5"
    val_base = f"{asdf_head}_val.h5"
    test_base = f"{asdf_head}_test.h5"
    train_path = join(asdf_dir, train_base)
    val_path = join(asdf_dir, val_base)
    test_path = join(asdf_dir, test_base)

    if seed >= 0:
        np.random.seed(seed)

    # handle asdf files
    with ASDFDataSet(asdf_data_path, mode="r") as raw:
        all_lists: List[str] = raw.waveforms.list()
        np.random.shuffle(all_lists)
        split_num0 = int(len(all_lists)*split_ratio_list[0])
        split_num2 = int(len(all_lists)*split_ratio_list[2])
        train_lists, val_lists, test_lists = all_lists[:split_num0], all_lists[
            split_num0:-split_num2], all_lists[-split_num2:]
        # generate new files
        generate_new(train_path, train_lists, raw, "train")
        generate_new(val_path, val_lists, raw, "val")
        generate_new(test_path, test_lists, raw, "test")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
