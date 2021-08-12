import getopt
import os
import sys
import multiprocessing as mp

import h5py as h5
import numpy as np
from tqdm import tqdm
import tifffile as tf


class Downsampler():

    def __init__(self, path, base="data/", dwn="dwn/", verbose=0):

        self.path = path
        self.base = base
        self.dwn = dwn
        self.verbose = verbose

        with h5.File(self.path, "r") as file:
            self.locs = [self.base + loc for loc in list(file[self.base].keys())]

        self.convert_xyz_to_zxy()

    def downsample(self, xy_scale=1/2, z_scale=1/2):

        #TODO multiprocessing intra chunk with shared memory mmap(-1)

        assert xy_scale <= 1, "xy_scale needs to be smaller than 1"
        assert z_scale <= 1, "z_scale needs to be smaller than 1"

        assert xy_scale**-1 % 1 == 0, "xy_scale needs to be a proper ratio [1/2, 1/3, 1/4, ...]"
        assert xy_scale**-1 % 1 == 0, "z_scale needs to be a proper ratio [1/2, 1/3, 1/4, ...]"

        for loc in self.locs:

            with h5.File(self.path, "r") as file:

                # get dimensions
                data = file[loc]
                Z, X, Y = data.shape
                cZ, cX, cY = data.chunks

                if loc.replace(self.base, self.dwn) in file:
                    print(f"{loc} already downsampled. Skipping!")
                    continue

            # new dimensions
            assert X == Y, "X and Y axis not equally long"

            assert X*xy_scale % 1 == 0, "X axis is not divisable by factor: {:.3f}".format(X*xy_scale)
            assert Y*xy_scale % 1 == 0, "Y axis is not divisable by factor: {:.3f}".format(Y*xy_scale)
            assert Z*z_scale % 1 == 0, "Z axis is not divisable by factor: {:.3f}".format(Z*z_scale)

            assert cX * xy_scale % 1 == 0, "cX axis is not divisable by factor: {:.3f}".format(cX * xy_scale)
            assert cY * xy_scale % 1 == 0, "cY axis is not divisable by factor: {:.3f}".format(cY * xy_scale)
            assert cZ * z_scale % 1 == 0, "cZ axis is not divisable by factor: {:.3f}".format(cZ * z_scale)

            z, x, y = int(Z*z_scale), int(X*xy_scale), int(Y*xy_scale)

            # create shared memory
            out_path = self.path.replace("h5", "temp.mmmap")
            if os.path.isfile(out_path):
                os.remove(out_path)

            out = np.memmap(out_path, dtype="i2", mode="w+", shape=(z, x, y))
            out.flush()

            # create temp storages
            tasks = []
            for x0 in tqdm(range(0, X, cX)):
                for y0 in range(0, Y, cY):
                    for z0 in range(0, Z, cZ):

                        x1 = min(x0+cX, X)
                        y1 = min(y0+cY, Y)
                        z1 = min(z0+cZ, Z)

                        tasks.append([self.path, loc, out_path, z0, z1, x0, x1, y0, y1, xy_scale, z_scale, (z, x, y)])

            print(f"Processing {loc}")
            with mp.Pool(mp.cpu_count() - 2) as pool:
                r = list(tqdm(pool.imap(process_chunk, tasks), total=len(tasks)))

            # print(f"Saving tiff {loc}")
            # tiff_path = self.path+"{}.tiff".format(loc.replace("/", "-"))
            # print("Tiff path: ", tiff_path)
            # tf.imsave(tiff_path, data=out)

            loc_out = loc.replace("data", "dwn")
            print(f"Saving {loc_out}")
            with h5.File(self.path, "a") as file:
                data = file.create_dataset(loc_out, shape=(z, x, y), dtype="i2", chunks=(cZ, cX, cY), compression="gzip", shuffle=True)

                for z0 in tqdm(range(0, z, cX)):
                    for x0 in range(0, x, cY):
                        for y0 in range(0, y, cZ):

                            z1 = min(z0+cZ, z)
                            x1 = min(x0+cX, x)
                            y1 = min(y0+cY, y)

                            data[z0:z1, x0:x1, y0:y1] = out[z0:z1, x0:x1, y0:y1]

            # Delete temp file
            if os.path.isfile(out_path):
                os.remove(out_path)

    def convert_xyz_to_zxy(self, delete_original=True):

        # check if conversion is necessary
        with h5.File(self.path, "a") as file:

            if len(list(file.keys())) < 2:
                file.create_dataset("dummy", dtype="i2", shape=(1, 1, 1))

            key0 = list(file["data/"].keys())[0]
            d1, d2, d3 = file[f"data/{key0}"].shape
            if d2 == 1200 and d3 == 1200:
                if self.verbose > 1:
                    print("Expected data shape found (ZXY)")
                return True

        # convert data
        with h5.File(self.path, "a") as file:
            for loc in file["data/"].keys():

                if self.verbose > 0:
                    print(f"Converting channel {loc} from xyz to zxy")

                # get shape of original data set
                xyz = file[f"data/{loc}"]
                X, Y, Z = xyz.shape
                cx, cy, cz = xyz.chunks

                # create new dataset
                if f"zxy/{loc}" in file:
                    del file[f"zxy/{loc}"]

                zxy = file.create_dataset(f"zxy/{loc}", dtype="i2", shape=(Z, X, Y),
                                          compression="gzip", chunks=(cx, cy, cz), shuffle=True)

                # necessary for downstream processing
                if "dummy" not in file:
                    _ = file.create_dataset("dummy", dtype="i2", shape=(1, 1, 1))

                # transform and copy data to new shape
                for start in tqdm(range(0, Z, cz)):

                    stop = min(start+cz, Z)

                    temp = np.array(xyz[:, :, start:stop])
                    temp = np.swapaxes(temp, 0, 2)
                    temp = np.swapaxes(temp, 1, 2)

                    zxy[start:stop, :, :] = temp

        # clean up original data
        if delete_original:
            with h5.File(self.path, "a") as file:

                # remove
                del file["data/"]

                # move
                file.create_group('data')
                for key in file["zxy/"].keys():
                    file.move(f"zxy/{key}", f"data/{key}")
                del file["zxy"]


def process_chunk(task):

    path, loc, out_path, z0, z1, x0, x1, y0, y1, xy_scale, z_scale, shape = task

    out = np.memmap(out_path, shape=shape, dtype="i2", mode="r+")

    with h5.File(path, "r") as file:

        data = file[loc]
        cZ, cX, cY = data.chunks
        iz, ix, iy = int(z_scale ** -1), int(xy_scale ** -1), int(xy_scale ** -1)

        chunk = data[z0:z1, x0:x1, y0:y1]

        gZ, gX, gY = int(z0 * z_scale), int(x0 * xy_scale), int(y0 * xy_scale)

        for a0 in range(0, cZ, iz):
            for b0 in range(0, cX, ix):
                for c0 in range(0, cY, iy):
                    g0, g1, g2 = int(a0 * z_scale), int(b0 * xy_scale), int(c0 * xy_scale)
                    a1, b1, c1 = a0 + iz, b0 + ix, c0 + iz

                    out[gZ + g0, gX + g1, gY + g2] = np.mean(chunk[a0:a1, b0:b1, c0:c1])

    out.flush()

if __name__ == "__main__":

    input_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:", ["ifolder="])
    except getopt.GetoptError:
        print("calpack.py -i <input_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input_file"):
            input_file = arg

    assert os.path.isfile(input_file), "input_file is not a file: {}".format(input_file)

    print("InputFile: ", input_file)

    dw = Downsampler(input_file)
    dw.downsample()
