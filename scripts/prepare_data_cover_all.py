"""
prepare_data.py
convert the TongML data into asdf file.
"""
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from os.path import join
from typing import DefaultDict, List, Tuple

import click
import numpy as np
import obspy
import pandas as pd
from loguru import logger
from obspy.core.event import Origin, Catalog
from obspy.core.event.event import Event
from obspy.core.inventory.inventory import Inventory
from obspy.core.inventory.network import Network
from obspy.core.inventory.station import Station
from pyasdf import ASDFDataSet
from tqdm import tqdm


@dataclass
class TravelTimes:
    """
    Store the travel time information
    """
    p: float = 0
    s: float = 0
    ps: float = 0
    ps_freq: float = 0
    origin_time: str = ""


def load_csv(file_name: str) -> Tuple[DefaultDict[str, Event], DefaultDict[str, Inventory], DefaultDict[str, TravelTimes]]:
    """
    load the csv file, return three dicts with event/station/(event,station) as the key
    the first dict's value is the event object
    the second dict's value is the station object
    the third dict's value is TravelTimes
    """
    csv_data = pd.read_csv(file_name)
    # generate events and inventories
    event_dict = defaultdict(lambda: Event())
    inventory_dict = defaultdict(lambda: Inventory())
    times_dict = defaultdict(lambda: TravelTimes())

    for i in range(len(csv_data)):
        pd_row = csv_data.iloc[i]
        event_key = pd_row['EVENT_ID']
        station_key = pd_row['STATION']
        event_station_key = f"{event_key}_{station_key}"
        # events
        if event_key not in event_dict:
            origin = Origin(time=obspy.UTCDateTime(
                pd_row['ORIGIN_TIME']), longitude=pd_row['ELON'], latitude=pd_row['ELAT'], depth=pd_row['EDEP'])
            event = Event(origins=[origin])
            event_dict[event_key] = event
        # stations
        if event_station_key in inventory_dict:
            raise Exception(
                f"Multiple rows have same {event_key}/{station_key}")
        station = Station(code=pd_row['STATION'],
                          longitude=0, latitude=0, depth=0)
        # here we use event id as the network
        network = Network(code=event_key, stations=[station])
        inventory = Inventory(networks=[network])
        inventory_dict[event_station_key] = inventory
        # times
        if event_station_key in times_dict:
            raise Exception(
                f"Multiple rows have same {event_key}/{station_key}")
        times_dict[event_station_key] = TravelTimes(
            p=pd_row['PTIME'], s=pd_row['STIME'], ps=pd_row['PSTIME'], ps_freq=pd_row['PS_FREQ'], origin_time=pd_row['ORIGIN_TIME'])

    return event_dict, inventory_dict, times_dict


def get_sac_paths(sac_dir_ps: str, sac_dir_s: str, sac_dir_p: str) -> DefaultDict[str, List[str]]:
    """
    given the root path, get all sac files path, with the key as the connected EVENT_ID and STATION
    """
    res = defaultdict(list)

    sacs1 = glob(join(sac_dir_ps, "**/*sac"), recursive=True)
    sacs2 = glob(join(sac_dir_s, "**/*sac"), recursive=True)
    sacs3 = glob(join(sac_dir_p, "**/*sac"), recursive=True)

    def check(current_check: str):
        for key in res:
            if len(res[key]) != 3:
                raise Exception(
                    f"{key} in the {current_check} directory contains {len(res[key])} files.")

    for sac in sacs1:
        # here I do this as the key is a connected event and station
        key = sac.split("/")[-1].split(".")[0]
        res[key].append(sac)
    check("PS")

    for sac in sacs2:
        key = sac.split("/")[-1].split(".")[0]
        if len(res[key]) < 3:
            res[key].append(sac)
    check("S")

    for sac in sacs3:
        key = sac.split("/")[-1].split(".")[0]
        if len(res[key]) < 3:
            res[key].append(sac)
    check("P")

    return res


@click.command()
@click.option("--asdf_path", help="output ASDF file path")
@click.option("--csv_path", help="input csv file path")
@click.option("--sac_dir_ps", help="input sac directory in PS dir")
@click.option("--sac_dir_s", help="input sac directory in S dir")
@click.option("--sac_dir_p", help="input sac directory in P dir")
def main(asdf_path: str, csv_path: str, sac_dir_ps: str, sac_dir_s: str, sac_dir_p: str):
    """
    given the output asdf path, the data info csv file, and the sac files root (TongaML/PickedPS)
    generate the asdf file storing all the information for later use
    """
    # init
    event_dict, inventory_dict, times_dict = load_csv(csv_path)
    sac_dict = get_sac_paths(sac_dir_ps, sac_dir_s, sac_dir_p)

    # for all combinations of event_station, load the data
    with ASDFDataSet(asdf_path, mode="w") as ds:
        # add events
        ds.add_quakeml(Catalog(events=list(event_dict.values())))
        # add possible waveforms
        for event_station_key in tqdm(times_dict, desc="waveform"):
            event_key = '_'.join(event_station_key.split('_')[:-1])
            station_key = event_station_key.split('_')[-1]
            if event_station_key not in sac_dict:
                logger.info(f"{event_station_key} is not in sac dataset")
            else:
                if len(sac_dict[event_station_key]) != 3:
                    logger.info(
                        f"{event_station_key} has {len(sac_dict[event_station_key])} traces")
                else:
                    stream = obspy.Stream()
                    for each in sac_dict[event_station_key]:
                        trace = obspy.read(each)[0]
                        trace.stats.network = event_key
                        trace.stats.station = station_key
                        stream += trace
                    ds.add_waveforms(stream, tag="raw_recording",
                                     event_id=event_dict[event_key])
        # add stationxml
        for event_station_key in tqdm(inventory_dict, desc="station"):
            ds.add_stationxml(inventory_dict[event_station_key])
        # add auxiliary data
        for event_station_key in tqdm(times_dict, desc="auxiliary"):
            ds.add_auxiliary_data(data=np.array(
                [times_dict[event_station_key].p]), data_type="TP", path=event_station_key, parameters={})
            ds.add_auxiliary_data(data=np.array(
                [times_dict[event_station_key].s]), data_type="TS", path=event_station_key, parameters={})
            ds.add_auxiliary_data(data=np.array(
                [times_dict[event_station_key].ps]), data_type="TPS", path=event_station_key, parameters={})
            ds.add_auxiliary_data(data=np.array(
                [times_dict[event_station_key].ps_freq]), data_type="FPS", path=event_station_key, parameters={})
            # also the origin time, note here we need encode to put the string into hdf5
            ds.add_auxiliary_data(data=np.array([times_dict[event_station_key].origin_time.encode(
            )]), data_type="REFTIME", path=event_station_key, parameters={})


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
