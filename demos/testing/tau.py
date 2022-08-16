import itertools
import os
import shutil
from pathlib import Path
from scipy import signal
import pandas as pd
import uuid
import h5py as h5
from time import time
import numpy as np

from dask.distributed import Client, LocalCluster, progress
import dask.array as da

from multiprocessing import shared_memory

import traceback
import logging
import sys

def dummy(n):
    print(n)

def find_peaks(trace, prominence, meta=None):

    _, d = signal.find_peaks(trace, prominence=prominence)
    df = pd.DataFrame([d["left_bases"], d["right_bases"]]).transpose()
    df.drop_duplicates(inplace=True)
    df.columns = ["f0", "f1"]

    traces = []
    for _, row in df.iterrows():
        l, r = row.values
        tr = trace[l:r]
        traces.append(tr)
    # df["trace"] = traces

    if meta is not None:
        for key in meta.keys():
            df[key] = meta[key]

    return df

def find_peaks_block(data_package, indices, prominence, output):

    if type(data_package) == dict:

        x0, x1, y0, y1 = indices
        if output is not None:
            output_path = output.joinpath(f"{x0}x{y0}.pickle")
            if output_path.is_file():
                return 1

        shm_data = shared_memory.SharedMemory(name=data_package["name"])
        block = np.ndarray(data_package["shape"], dtype=data_package["dtype"], buffer=shm_data.buf)[:, x0:x1, y0:y1]
        _, bX, bY = block.shape

        results = []
        for x, y in itertools.product(range(bX), range(bY)):
            results.append(find_peaks(block[:, x, y], prominence=prominence,
                                      meta={"X":x0+x, "Y":y0+y}))

        shm_data.close()

    else:

        x0, y0 = indices
        if output is not None:
            output_path = output.joinpath(f"{x0}x{y0}.pickle")
            if output_path.is_file():
                return 1

        block = data_package
        _, bX, bY = block.shape

        results = []
        for x, y in itertools.product(range(bX), range(bY)):
            results.append(find_peaks(block[:, x, y], prominence=prominence,
                                      meta={"X":x0+x, "Y":y0+y}))


    results = pd.concat(results)

    if output is None:
        return results
    else:
        results.to_pickle(output_path)
        return 1

class Tau:

    def __init__(self, path, indices=None, loc=None, use_shared_memory=True):

        self.path = Path(path)
        self.use_shared_memory = use_shared_memory

        if self.path.suffix == ".h5":
            arr = h5.File(self.path.as_posix(), "r")[loc]
            data = da.from_array(arr)

        else:
            data = da.from_tiledb(self.path.as_posix())

        Z, X, Y = data.shape

        print("Loading data ...")
        if use_shared_memory:

            if indices is None:

                self.shm_data = shared_memory.SharedMemory(create=True, size=data.dtype.itemsize * data.size)
                self.snp_data = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm_data.buf)

                self.snp_data[:] = data.compute()
            else:
                cz, cx, cy = indices
                z0, z1 = cz if cz is not None else (0, Z)
                x0, x1 = cx if cx is not None else (0, X)
                y0, y1 = cy if cy is not None else (0, Y)

                dummy = np.zeros((z1-z0, x1-x0, y1-y0), dtype=data.dtype)
                self.shm_data = shared_memory.SharedMemory(create=True, size=dummy.dtype.itemsize * dummy.size)
                self.snp_data = np.ndarray(dummy.shape, dtype=dummy.dtype, buffer=self.shm_data.buf)

                self.snp_data[:] = data[z0:z1, x0:x1, y0:y1].compute()

            print("load finished!")

            self.data_package = {"name":self.shm_data.name, "shape":self.snp_data.shape, "dtype":self.snp_data.dtype}
            self.shape = self.snp_data.shape

        else:

            if indices is None:
                self.data = data.compute()
            else:
                cz, cx, cy = indices
                z0, z1 = cz if cz is not None else (0, Z)
                x0, x1 = cx if cx is not None else (0, X)
                y0, y1 = cy if cy is not None else (0, Y)

                self.data = data[z0:z1, x0:x1, y0:y1].compute()

            self.shape = self.data.shape
            print("load finished!")

    def run(self, prominence, steps=16, output=None):

        if output is not None:
            if not output.is_dir():
                output.mkdir()

        Z, X, Y = self.shape

        try:
            with LocalCluster() as lc:
                with Client(lc) as client:
                    futures = []
                    for (x0, y0) in itertools.product(range(0, X, steps), range(0, Y, steps)):

                        if self.use_shared_memory:
                            futures.append(client.submit(find_peaks_block,
                                                         self.data_package, (x0, min(X, x0+steps), y0, min(Y, y0+steps)),
                                                         prominence, output))

                        else:
                            futures.append(client.submit(find_peaks_block,
                                                         self.data[x0:min(X, x0+steps), y0:min(Y, y0+steps)], (x0, y0),
                                                         prominence, output))

                    res = client.gather(futures)
        finally:

            self.shm_data.close()
            self.shm_data.unlink()

        if output is not None:
            res = pd.concat([pd.read_pickle(output.joinpath(file)) for file in os.listdir(output.as_posix())])
            output_path = output.parent.joinpath("z_events.p")
            res.to_pickle(output_path)
            shutil.rmtree(output.as_posix())
        else:
            res = pd.concat(res)
            output_path = self.path.parent.joinpath("z_events.p")
            res.to_pickle(output_path)

        print("Saved results to: ", output_path)

if __name__ == "__main__":

    tau = Tau("/media/STORAGE/data/22A7x6-3.delta",
              # indices=(None, (0, 10), (0, 10))
              )

    print("run analysis ...")
    t0 = time()
    tau.run(prominence=5, steps=4,
            output=Path("/home/janrei1@ad.cmm.se/Desktop/del/del/events")
            )

    t1 = time()
    print("Runtime: {:.2f}min".format((t1-t0)/60))
