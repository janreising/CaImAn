import itertools
import os
import shutil
from pathlib import Path
from scipy import signal
import pandas as pd
import h5py as h5
from time import time
import numpy as np

from dask.distributed import Client, LocalCluster, progress
import dask.array as da

def find_peaks(trace, prominence, meta=None):

    peaks = {}

    _, d = signal.find_peaks(trace, prominence=prominence)

    arr = np.array([d["left_bases"], d["right_bases"]]).transpose()
    arr = np.unique(arr, axis=0)

    peaks["lbase"] = arr[:, 0]
    peaks["rbase"] = arr[:, 1]

    peaks["trace"] = [trace[l:r] for (l, r) in arr]

    if meta is not None:
        for key in meta.keys():
            peaks[key] = meta[key]

    return peaks

def find_peaks_block(block, indices, prominence, output):

    x0, y0 = indices
    if output is not None:
        output_path = output.joinpath(f"{x0}x{y0}.npy")
        if output_path.is_file():
            return 1

    _, bX, bY = block.shape

    results = []
    for x, y in itertools.product(range(bX), range(bY)):
        results.append(find_peaks(block[:, x, y], prominence=prominence,
                                  meta={"X":x0+x, "Y":y0+y}))

    if output is None:
        return results
    else:
        np.save(output_path, results)
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

        with LocalCluster() as lc:
            with Client(lc) as client:
                futures = []
                for (x0, y0) in itertools.product(range(0, X, steps), range(0, Y, steps)):

                    futures.append(client.submit(find_peaks_block,
                                                 self.data[:, x0:min(X, x0+steps), y0:min(Y, y0+steps)], (x0, y0),
                                                 prominence, output))

                res = client.gather(futures)

        output_path = self.path.with_suffix(".npy")

        if output is not None:
            res = [np.load(output.joinpath(file), allow_pickle=True)[()] for file in os.listdir(output.as_posix())]
            shutil.rmtree(output.as_posix())

        # combine results
        res = np.array(res)
        comb = {"pix_n":[]}
        for key in res[0, 0].keys():
            comb[key] = []

        for n, (i, j) in enumerate(itertools.product(range(res.shape[0]), range(res.shape[1]))):

            item = res[i, j]
            for l, r, tr in zip(item["lbase"], item["rbase"], item["trace"]):
                comb["pix_n"].append(n)
                comb["lbase"].append(l)
                comb["rbase"].append(r)
                comb["trace"].append(tr)
                comb["X"].append(item["X"])
                comb["Y"].append(item["Y"])

        np.save(output_path, comb)

        print("Saved results to: ", output_path)

if __name__ == "__main__":

    tau = Tau("/media/STORAGE/data/22A6x5-2.delta",
              # indices=(None, (64, 128), (64, 128))
              )

    print("run analysis ...")
    t0 = time()
    tau.run(prominence=5, steps=16,
            # output=Path("/home/janrei1@ad.cmm.se/Desktop/del/del/events")
            # output=Path("/home/janrei1@ad.cmm.se/Desktop/del/del/events")
            )

    t1 = time()
    print("Runtime: {:.2f}min".format((t1-t0)/60))
