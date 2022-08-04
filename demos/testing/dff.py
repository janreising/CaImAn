import random

import h5py as h5
import numpy as np
import sys, os, getopt, time
# import caiman as cm
from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
from itertools import repeat
import tiledb
import tifffile as tf
import pandas as pd
from scipy.ndimage import minimum_filter1d
from dask.distributed import Client, LocalCluster
from dask import array as da

def calculate_dFF(path, loc,
                  method="only_baseline", secsWindow = 5, quantilMin = 8):

    print(f"Calculating dFF for loc:{loc}")

    with h5.File(path, "r") as file:
        data = file[f"{loc}"]
        Z, X, Y = data.shape
        cz, cx, cy = data.chunks

    for x0  in tqdm(range(0, X, cx)):
        for y0 in range(0, Y, cy):

            x1 = min(x0+cx, X)
            y1 = min(y0+cy, Y)

            with h5.File(path, "r") as file:
                rec = cm.movie(file[f"{loc}"][:, x0:x1, y0:y1])
            rec = (rec + abs(np.min(rec)) + 1)

            dff, _ = rec.computeDFF(secsWindow=secsWindow, method=method, quantilMin=quantilMin)

            with h5.File(path, "a") as file:
                new_loc = loc.replace(loc.split("/")[0], "dff")
                if new_loc not in file:

                    chunks = (100, 100, 100) if Z > cz else None
                    data = file.create_dataset(f"{new_loc}", shape=(Z, X, Y), dtype="float32", chunks=chunks,
                                        compression="gzip", shuffle=True)
                else:
                    data = file[f"{new_loc}"]

                data[:, x0:x1, y0:y1] = dff[:, :, :]

####################
## better dFF/dFF ##
####################

def calculate_background(trace, window):

    # res = np.array(trace).astype(int)  # TODO inplace
    # res[0] = trace[0]   # TODO maybe forward/backward approach?

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

def calculate_background_fast(trace, window):

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

def dFF(file, dims, window, out, method='dF'):

    methods = ['dF', 'background', 'dFF']
    assert method in methods, "please provide a valid method instead of {}: {}".format(method, methods)

    x0, x1, y0, y1 = dims
    save_path = "{}{}-{}x{}-{}.npy".format(out, x0, x1, y0, y1)
    if os.path.isfile(save_path):
        print("precalculated range: {}-{} x {}-{}".format(x0, x1, y0, y1))
        return 1

    # reading data
    tdb = tiledb.open(file, mode="r")
    data = tdb[:, x0:x1, y0:y1]
    tdb.close()
    # print("loaded: {}-{} x {}-{}".format(x0, x1, y0, y1))

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


    # print("processed: {}-{} x {}-{}".format(x0, x1, y0, y1))

    # writing data
    # tdb = tiledb.open(file, mode="w")
    # tdb[:, x0:x1, y0:y1] = data[:]
    # tdb.close()

    np.save(save_path, res)

    print("Finished range: {}-{} x {}-{}".format(x0, x1, y0, y1))

    return 1

def save_to_tiledb(uri, data, chunks):

    dadata = da.from_array(data, chunks=chunks)
    dadata.to_tiledb(uri)

def get_dFF(file, loc=None, window=2000, steps=32, x_range=None, y_range=None, method='dF'):

    h5path = None
    if file.endswith(".tdb"):
        tdb = tiledb.open(file, mode="r")
        Z, X, Y = tdb.shape
        tdb.close()

    elif file.endswith(".h5"):

        assert loc is not None, "when providing .h5 file you also need to provide 'loc' argument"

        out_path = file.replace(".h5", "."+loc.replace("/", ".")+".tdb")

        if os.path.isdir(out_path):
            tdb = tiledb.open(out_path, mode="r")
            Z, X, Y = tdb.shape
            tdb.close()
            file = out_path
        else:

            h5path = file
            print("saving temp tdb file ...")
            with h5.File(file, "r") as h5file:
                data = h5file[loc]
                Z, X, Y = data.shape
                save_to_tiledb(out_path, data, (-1, steps, steps))

            file = out_path

    out = file[:-1] + file[-1].replace(os.sep, "") + ".dF/"
    if not os.path.isdir(out):
        os.mkdir(out)

    # TODO does this mean I don't need to save it differently?
    # TODO clean up
    futures = []
    with Client() as client:
        for x in range(max(0, x_range[0]), min(X, x_range[1]), steps):
            for y in range(max(0, y_range[0]), min(Y, y_range[1]), steps):
                futures.append(client.submit(dFF, file, [x, x+steps, y, y+steps], window, out, method))

        # random.shuffle(tasks)
        print("#tasks: ", len(futures))

        client.gather(futures)

    # with Pool(cpu_count()) as p:
    #     p.starmap(dFF, tasks)

    # combine
    print("combining results")


if __name__ == "__main__":

    input_file = None
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

    # assert os.path.isfile(input_file), "input_file is not a file: {}".format(input_file)

    t0 = time.time()
    # calculate_dFF(input_file, "cnmfe/ast")
    # calculate_dFF(input_file, "cnmfe/neu")

    get_dFF(input_file, loc=loc, window=2000, x_range=(0, 512), y_range=(0, 512), steps=64, method='dF')

    t1 = (time.time() - t0) / 60
    print("dFF finished in {:.2f}".format(t1))
