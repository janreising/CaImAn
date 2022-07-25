import random

import dask.array
import h5py as h5
import numpy as np
import sys, os, getopt, time
# import caiman as cm
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import repeat
import tiledb
import tifffile as tf
import pandas as pd
from scipy.ndimage import minimum_filter1d

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
    # print("received: {}-{} x {}-{}".format(x0, x1, y0, y1))
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

def get_dFF(file, window=2000, steps=32, x_range=None, y_range=None, method='dF'):

    tdb = tiledb.open(file, mode="r")
    Z, X, Y = tdb.shape
    tdb.close()

    out = file[:-1] + file[-1].replace(os.sep, "") + ".res/"
    if not os.path.isdir(out):
        os.mkdir(out)

    dims = []
    for x in range(max(0, x_range[0]), min(X, x_range[1]), steps):
        for y in range(max(0, y_range[0]), min(Y, y_range[1]), steps):
            dims.append([x, x+steps, y, y+steps])

    # for dim in dims:
    #     print(dim)
    # return 0

    tasks = list(zip(
        repeat(file), dims, repeat(window), repeat(out), repeat(method)
    ))
    random.shuffle(tasks)
    print("#tasks: ", len(tasks))

    with Pool(cpu_count()) as p:
        p.starmap(dFF, tasks)

    # combine
    print("combining results")
    res = np.zeros((Z, min(X, x_range[1]-x_range[0]), min(Y, y_range[1]-y_range[0])),
                   dtype="f4" if method == 'dFF' else "i2"
                   )
    print("RES: ", res.shape, res.dtype)
    for r in tqdm(os.listdir(out)):

        x, y = r.split(".")[0].split("x")
        x0, x1 = x.split("-")
        y0, y1 = y.split("-")

        x0, x1 = int(x0), int(x1)
        y0, y1 = int(y0), int(y1)

        # print("Pre: {}-{} [{}] x {}-{} [{}]".format(x0, x1, x1-x0, y0, y1, y1-y0))

        if x_range is not None:
            x0, x1 = x0-x_range[0], x1-x_range[0]

        if y_range is not None:
            y0, y1 = y0-y_range[0], y1-y_range[0]

        # print("Post: {}-{} [{}] x {}-{} [{}]".format(x0, x1, x1-x0, y0, y1, y1-y0))

        # loaded = np.load(out+r, allow_pickle=True)
        # print("{} <- {}".format(res[:, x0:x1, y0:y1].shape, loaded.shape))

        res[:, x0:x1, y0:y1] = np.load(out+r, allow_pickle=True)

    # if not os.path.isfile(file+".h5"):
    #     print("save to .h5")
    #     with h5.File(file+".h5", "w") as f:
    #         f.create_dataset("dFF/ast", res.shape, dtype=np.uintc, data=res, chunks=(100, 100, 100))

    print("save to tiff: ", file+".tiff")
    tf.imwrite(file+".tiff", res)

    print("Done")

if __name__ == "__main__":

    input_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:l:", ["ifolder=", "local="])
    except getopt.GetoptError:
        print("calpack.py -i <input_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input_file"):
            input_file = arg

    # assert os.path.isfile(input_file), "input_file is not a file: {}".format(input_file)

    t0 = time.time()
    # calculate_dFF(input_file, "cnmfe/ast")
    # calculate_dFF(input_file, "cnmfe/neu")

    get_dFF(input_file, window=2000, x_range=(0, 512), y_range=(0, 512), steps=64, method='dFF')

    t1 = (time.time() - t0) / 60
    print("dFF finished in {:.2f}".format(t1))
