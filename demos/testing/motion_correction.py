import os, sys, psutil, shutil
# print(os.environ["PYTHONPATH"])
# print(os.environ["VIRTUAL_ENV"])
# sys.path.append("/home/j/janrei/Private/CaImAn/")
import numpy as np
import h5py as h5
import tifffile as tf
from tqdm import tqdm
import getopt
from skimage.transform import resize

import cv2
try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy.volparams import volparams

class CMotionCorrect():

    def __init__(self, path,
                 loc_out="mc/", on_server=True, verbose=0, delete_temp_files=True,
                 fr=10, pw_rigid=False, max_shifts=(50, 50), gSig_filt=(20, 20),
                 strides=(48, 48), overlaps=(24, 24), max_deviation_rigid=3, border_nan='copy'
                 ):

        self.path = path
        self.base, self.name, self.ext = self.deconstruct_path(path)
        # TODO check if exists
        # TODO change naming conventions
        self.loc_out = loc_out
        self.on_server = on_server
        self.delete_temp_files = delete_temp_files
        self.verbose = verbose

        self.fr = fr
        self.pw_rigid = pw_rigid
        self.max_shifts = max_shifts
        self.gSig_filt = gSig_filt
        self.strides = strides
        self.overlaps = overlaps
        self.max_deviation_rigid = max_deviation_rigid
        self.border_nan = border_nan

        self.files = []
        self.dimensions = []
        self.mmaps = []

    def run_motion_correction(self, ram_size_multiplier=5):

        ##################
        # File preparation
        if self.verbose > 0:
            print(f"Processing file: {self.base}{self.name}")

        # check array shape; convert if necessary
        self.convert_xyz_to_zxy()

        # decide whether to split files dependent on available RAM
        file_size = os.stat(self.path).st_size
        ram_size = psutil.virtual_memory().total
        if self.verbose > 0:
            print("{:.2f}GB:{:.2f}GB ({:.2f}%)".format(file_size / 1000 / 1024 / 1024,
                                                       ram_size / 1000 / 1024 / 1024,
                                                       file_size / ram_size * 100))

        split_files = False
        if ram_size < file_size * ram_size_multiplier:
            if self.verbose > 0:
                print("RAM not sufficient. Splitting file ...")
            split_files = True

        ##################
        # Process channels
        with h5.File(self.path, "r") as file:
            locs = [f"{key}" for key in list(file["data/"].keys())]

        for loc in locs:

            if self.verbose > 0:
                print("Processing location: ", repr(loc))

            # reset state
            self.files = []
            self.dimensions = []
            self.mmaps = []

            # check if mc already exists
            with h5.File(self.path, "r") as file:
                if f"mc/{loc}" in file:
                    if self.verbose > 0:
                        print("Motion Correction already exists. Skipping ...")
                        continue

            # split files if necessary
            if split_files:
                self.split_h5_file(f"data/{loc}")
            else:

                with h5.File(self.path, "r") as file:
                    self.dimensions.append(file[f"data/{loc}"].shape)
                self.files.append(self.path)


            # check if exists
            # TODO check if exists. Maybe complicated because the file names changes; code might have crashed previously
            # TODO actually impossible because different channels will use the same name for mmap files

            # Parameters
            opts_dict = {
                'fnames': self.files,
                'fr': self.fr,  # sample rate of the movie
                'pw_rigid': self.pw_rigid,  # flag for pw-rigid motion correction
                'max_shifts': self.max_shifts,  # 20, 20                             # maximum allowed rigid shift
                'gSig_filt': self.gSig_filt,
                # 10,10   # size of filter, in general gSig (see below),  # change if alg doesnt work
                'strides': self.strides,  # start a new patch for pw-rigid motion correction every x pixels
                'overlaps': self.overlaps,  # overlap between pathes (size of patch strides+overlaps)
                'max_deviation_rigid': self.max_deviation_rigid,  # maximum deviation allowed for patch with respect to rigid shifts
                'border_nan': self.border_nan,
            }

            opts = volparams(params_dict=opts_dict)

            ###################
            # Motion Correction

            # start cluster for parallel processing
            c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=7, single_thread=False)

            # Run correction
            if self.verbose > 0:
                print("Starting motion correction ...")
            mc = MotionCorrect(self.files, dview=dview, var_name_hdf5=f"data/{loc}", **opts.get_group('motion'))
            mc.motion_correct(save_movie=True)

            # stop cluster
            cm.stop_server(dview=dview)

            ####################
            # Convert mmap to h5
            self.get_mmaps()
            print("Converting mmap to h5 ...")
            self.save_memmap_to_h5(loc=f"mc/{loc}")

            #############
            # Save sample
            print("Saving sample ...")
            self.save_tiff(loc=f"mc/{loc}")

            ###################
            # delete temp files
            if self.delete_temp_files:

                if len(self.files) > 1:
                    for file in self.files:
                        os.remove(file)
                self.files = []
                self.dimensions = []

                for file in self.mmaps:
                    os.remove(file)
                self.mmaps = []

    @staticmethod
    def deconstruct_path(path):

        base = os.sep.join(path.split(os.sep)[:-1])
        name = path.split(os.sep)[-1].split(".")[0]
        ext = path.split(".")[-1]

        if base[-1] != os.sep:
            base = base + os.sep

        return base, name, ext

    def show_info(self):
        with h5.File(self.path, "r") as file:
            print(file.keys())
            print(file["data/ast"].shape)

    def dtqdm(self, iterator, position=0, leave=True):

        if self.on_server:
            return iterator
        else:
            return tqdm(iterator, position=position, leave=leave)

    def convert_xyz_to_zxy(self, delete_original=True):

        #TODO testing if this actually works

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
                zxy = file.create_dataset(f"zxy/{loc}", dtype="i2", shape=(Z, X, Y),
                                          compression="gzip", chunks=(cx, cy, cz), shuffle=True)

                # necessary for downstream processing
                if "dummy" not in zxy:
                    _ = zxy.create_dataset("dummy", dtype="i2", shape=(1, 1, 1))

                # transform and copy data to new shape
                for start in self.dtqdm(range(0, Z, cz)):

                    stop = min(start+cz, Z)

                    temp = np.array(xyz[:, :, start:stop])
                    temp = np.swapaxes(temp, 0, 2)
                    temp = np.swapaxes(temp, 1, 2)

                    zxy[start:stop, :, :] = temp

        # clean up original data
        if delete_original:
            with h5.File(self.path, "w") as file:

                # remove
                del file["data/"]

                # move
                file.create_group('data')
                for key in file["zxy/"].keys():
                    file.move(f"zxy/{key}", f"data/{key}")
                del file["zxy"]

    def split_h5_file(self, loc, split_file_size=2000):

        # Load file
        with h5.File(self.path, "r") as file:

            data = file[loc]

            Z, X, Y = data.shape

            names = []
            splits = int(os.stat(self.path).st_size / (split_file_size * 1024 * 1024))
            split_size = int(Z/splits)

            # create names for file split
            iterator = []
            for start in list(range(0, Z, split_size)):
                iterator.append([start, min(start+split_size, Z)])

            # combine last two splits if final file has too few frames
            if iterator[-1][1] - iterator[-1][0] < 100:
                iterator[-2][1] = Z
                iterator = iterator[:-1]

            # iterate over file splits
            c = 0
            for start, stop in self.dtqdm(iterator):

                name_out = f'{self.base}{c}-{self.name}0.h5'
                if not os.path.isfile(name_out):
                    chunk = data[start:stop, :, :]

                    temp = h5.File(name_out, "w")
                    chunk_drive = temp.create_dataset("/data/ast", shape=chunk.shape, dtype=data.dtype)
                    chunk_drive[:, :, :] = chunk
                    temp.create_dataset("/proc/dummy", shape=(1, 1, 1), dtype=data.dtype)

                c += 1
                self.files.append(name_out)
                self.dimensions.append(chunk_drive.shape)

        return True

    def save_memmap_to_h5(self, loc):

        # get output shape
        Z = sum([dim[0] for dim in self.dimensions])
        _, X, Y = self.dimensions[0]
        shape = (Z, X, Y)

        # combine mmap to full stack
        with h5.File(self.path, "a") as file:

            data = file.create_dataset(loc, shape=shape, dtype="i2",
                                       compression="gzip", chunks=(100, 100, 100), shuffle=True)

            # fill array
            c = 0
            mm = np.zeros(shape)
            for mmap in self.dtqdm(self.mmaps):
                mm = np.memmap(mmap, shape=shape, dtype=np.float32)
                data[c*Z:(c+1)*Z, :, :] = mm

                c += 1

        return True

    def get_mmaps(self):

        dir_files = os.listdir(self.base)

        if len(self.files) > 1:

            for p in self.file:
                start = p.replace(self.base, "").replace(".h5", "")

                for df in dir_files:
                    temp = df.split(os.sep)[-1]

                    if temp.startswith(start) and temp.endswith(".mmap"):
                        self.mmaps.append(f"{self.base}{df}")
                        continue

        else:
            for df in dir_files:
                temp = df.split(os.sep)[-1]

                if temp.startswith(self.name[:-2]) and temp.endswith(".mmap"):
                    self.mmaps.append(f"{self.base}{df}")
                    continue

    def save_tiff(self, loc, skip=None, downsize=0.5, subindices=None):

        with h5.File(self.path, "r") as file:

            arr = file[loc]
            loc_name = loc.replace("/", "-")
            out = f"{self.base}{self.name}_{loc_name}.tiff"

            Z, X, Y = arr.shape

            if skip is None:
                skip = int(Z/10)

            z0 = 0
            if subindices is not None:
                z0, z1 = subindices
                Z = min(Z, z1)

            z, x, y = int((Z-z0)/skip), int(X*downsize), int(Y*downsize)

            img_out = np.zeros((z-1, x, y))
            c = 0

            for i in self.dtqdm(range(z0, Z, skip)[:-1]):
                img = arr[i, :, :]

                if downsize != 1:
                    img = resize(img, (x, y))

                img_out[c, :, :] = img
                c += 1

            tf.imsave(out, img_out)

if __name__ == "__main__":

    input_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:", ["ifolder="])
    except getopt.GetoptError:
        print("calpack.py -i <input_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--ifolder"):
            input_file = arg

    assert os.path.isfile(input_file), "input_file is not a file: {}".format(input_file)

    print("InputFile: ", input_file)

    mc = CMotionCorrect(path=input_file, verbose=3, delete_temp_files=True)
    mc.run_motion_correction()



