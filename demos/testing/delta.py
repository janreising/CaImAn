import dask.array as da
import h5py as h5
# from os.path import basename, dirname, normpath, join
from pathlib import Path
import numpy as np
import tiledb
import os
from deprecated import deprecated
from scipy.ndimage import minimum_filter1d
from dask.distributed import Client
import tempfile
import shutil
import getopt
import sys
import time

@deprecated(reason="slow initial implementation")
def calculate_background(trace, window):

    half = int(window/2)

    for n in range(1, len(trace)):

        n0 = max(0, n-half)
        n1 = min(len(trace), n + half)

        prev_min = min(trace[n0:n])

        # current point is going down
        if trace[n] <= prev_min:
            trace[n] = trace[n]  # maybe percentile up to this point? would introduce lag

        # current point is going up
        else:
            post_min_pos = n + np.argmin(trace[n:n1])
            post_min = trace[post_min_pos]

            # there is a lower point
            if trace[n - 1] >= post_min:
                trace[n] = trace[n - 1]

            # there are only higher points within window
            else:
                trace[n] = trace[n - 1] + (post_min - trace[n - 1]) / (post_min_pos - (n - 1))

    return trace

@deprecated(reason="faster implementation but superseded by: calculate_background_even_faster")
def calculate_background_fast(trace, window):

    import pandas as pd

    # pad series for fast rolling window
    tr = pd.Series(np.pad(trace, window, mode='edge'))

    # use pandas to get rolling min
    tr = tr.rolling(window).min().values[window:]

    # take the max of 1xWindow shifted minimum
    tr_max = np.zeros((2, len(trace)))
    tr_max[0, :] = tr[:-window]
    tr_max[1, :] = tr[window:]
    tr_max = np.nanmax(tr_max, axis=0)

    return tr_max

def calculate_background_even_faster(trace, window):

        MIN = minimum_filter1d(np.pad(trace, pad_width=(0, window), mode='edge'), size=window+1, mode="nearest", origin=int(window/2))

        # take the max of 1xWindow shifted minimum
        tr_max = np.zeros((2, len(trace)))
        tr_max[0, :] = MIN[:-window]
        tr_max[1, :] = MIN[window:]
        tr_max = np.nanmax(tr_max, axis=0)

        return tr_max


class Delta:

    def __init__(self, input_data, loc=None, verbose=0):

        self.input_data = input_data
        self.loc = loc
        self.verbose = verbose
        self.chunksize = None

        # Get data size
        if input_data.endswith(".h5"):

            assert loc is not None, "please provide a dataset location as 'loc' parameter"
            with h5.File(input_data) as file:
                data = file[loc]
                self.Z, self.X, self.Y = data.shape

        elif input_data.endswith(".tdb"):

            with tiledb.open(input_data) as tdb:
                self.Z, self.X, self.Y = tdb.shape
                self.chunksize = [tdb.schema.domain.dim(i).tile for i in range(tdb.schema.domain.ndim)]

        else:
            self.Z, self.X, self.Y = None, None, None
            print("Unknown input data type: {}".format(input_data))

    def run(self, window=2000, steps=None, method='dF'):

        t0 = time.time()

        # convert to tiledb if necessary
        if self.input_data.endswith(".h5"):
            tileDBpath = self.save_h5_to_tiledb(self.input_data, loc=self.loc)
            delete_tiledb = True

        elif self.input_data.endswith(".tdb"):
            tileDBpath = self.input_data
            delete_tiledb = False

        else:
            self.vprint("unknown data type. Aborting!", urgency=0)
            return 0

        # calculate background parallelized
        self.vprint("calculating delta ...", 1)
        temp_dir = self.get_delta(tileDBpath, window, steps, method)

        # combine results
        self.vprint("combining results", 1)
        self.combine_results(temp_dir, method)

        # clean up
        if delete_tiledb: shutil.rmtree(tileDBpath)
        self.vprint("Runtime: {:.2f}".format(
            (time.time()-t0)/60), urgency=1)

    def vprint(self, msg, urgency=1):

        if urgency <= self.verbose:
            print("\t"*(urgency-1) + msg)

    def save_h5_to_tiledb(self, h5path, loc, tileDBpath=None, chunks=[-1, 'auto', 'auto']):

        # create new name if necessary
        if tileDBpath is None:
            tileDBpath = Path(h5path).with_suffix(".tdb").as_posix()

        # check if exists
        if os.path.isdir(tileDBpath):
            self.vprint("tileDB file already exists. Loading from {}".format(tileDBpath), urgency=0)

            with tiledb.open(tileDBpath, "r") as tdb:
                self.chunksize = [tdb.schema.domain.dim(i).tile for i in range(tdb.schema.domain.ndim)]

            return tileDBpath

        # save to tiledb
        with h5.File(h5path, "r") as file:
            data = da.from_array(file[loc], chunks=chunks)
            data.to_tiledb(tileDBpath)
            self.chunksize = data.chunksize

        self.vprint("tileDB saved to: {}".format(tileDBpath), urgency=2)

        return tileDBpath

    def calculate_delta(self, tileDBpath, dims, window, working_dir, method='dF'):

        methods = ['dF', 'background', 'dFF']
        assert method in methods, "please provide a valid method instead of {}: {}".format(method, methods)

        x0, x1, y0, y1 = dims
        save_path = "{}{}-{}x{}-{}.npy".format(working_dir, x0, x1, y0, y1)
        if os.path.isfile(save_path):
            print("precalculated range: {}-{} x {}-{}".format(x0, x1, y0, y1))
            return 1

        # reading data
        tdb = tiledb.open(tileDBpath, mode="r")
        data = tdb[:, x0:x1, y0:y1]
        tdb.close()

        # processing data
        Z, X, Y = data.shape
        res = np.zeros(data.shape, dtype="f4" if method == 'dFF' else "i2")
        for x in range(X):
            for y in range(Y):

                background = calculate_background_even_faster(data[:, x, y], window)

                if method == 'background':
                     res[:, x, y] = background
                elif method == 'dF':
                     res[:, x, y] = data[:, x, y] - background
                elif method == 'dFF':
                     res[:, x, y] = np.divide(data[:, x, y] - background, background)

        np.save(save_path, res)

        self.vprint("Finished range: {}-{} x {}-{}".format(x0, x1, y0, y1), urgency=2)

        return 1

    def get_delta(self, tileDBpath, window=2000, steps=None, method='dF'):

        # create temp directory
        temp_dir = tempfile.mkdtemp()

        # get steps
        if steps is None:
            _, cx, cy = self.chunksize
            assert cx == cy, "warning: chunksize is not equal ({}, {}) which will lead to inefficiency. Please provide a manual step size".format(cx, cy)
            steps = cx

        futures = []
        with Client() as client:
            for x in range(0, self.X, steps):
                for y in range(0, self.Y, steps):
                    futures.append(
                        client.submit(self.calculate_delta, #self,
                                      tileDBpath, [x, int(x+steps), y, int(y+steps)], window, temp_dir, method
                                                 )
            )

            self.vprint("#tasks: {}".format(len(futures)), urgency=1)
            client.gather(futures)

            return temp_dir

    def combine_results(self, output_dir, method, overwrite_existing=True):

        combined_delta = np.zeros((self.Z, self.X, self.Y),
              dtype="f4" if method == 'dFF' else "i2"
               )

        for r in os.listdir(output_dir):

            x, y = r.split(".")[0].split("x")
            x0, x1 = x.split("-")
            y0, y1 = y.split("-")

            x0, x1 = int(x0), int(x1)
            y0, y1 = int(y0), int(y1)

            combined_delta[:, x0:x1, y0:y1] = np.load(output_dir+r, allow_pickle=True)
            os.remove(output_dir+r)
        os.rmdir(output_dir)

        # save to h5 file if provided
        if self.input_data.endswith(".h5"):
            self.vprint("saving result to {}".format(self.input_data), urgency=1)
            with h5.File(self.input_data, "a") as f:

                new_loc = "dff/"+self.loc.split("/")[-1]

                if new_loc not in f:
                    f.create_dataset(new_loc, combined_delta.shape, dtype="i2", data=combined_delta, chunks=(100, 100, 100))
                elif overwrite_existing:
                    location = f[new_loc]
                    location[:] = combined_delta
                else:
                    self.vprint("{} already exists in {}. Please delete the dataset or add the 'overwrite_existing' flag.".format(
                        new_loc, Path(self.input_data).name
                    ))

        # save to tiledb
        tdb_path = Path(self.input_data).with_suffix(".delta").as_posix()
        self.vprint("saving delta to {}".format(tdb_path), urgency=1)

        if os.path.isdir(tdb_path):
            self.vprint("removing previous result ...", urgency=1)
            shutil.rmtree(tdb_path)

        da.from_array(combined_delta).to_tiledb(tdb_path, (100, 100, 100))


if __name__ == "__main__":

    input_file = None
    loc = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:l:", ["ifolder=", "loc="])
    except getopt.GetoptError:
        print("calpack.py -i <input_file>")
        sys.exit(2)

    loc=None
    for opt, arg in opts:
        if opt in ("-i", "--input_file"):
            input_file = arg

        if opt in ("-l", "--loc"):
            loc = arg

    d = Delta(input_file, loc=loc, verbose=5)
    d.run()
