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
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.cnmf import params as cnmfparams
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.initialization import downscale
from caiman.motion_correction import MotionCorrect

import warnings
warnings.filterwarnings("ignore")

class PreProcessor():

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

    def run_preprocess(self, ram_size_multiplier=7, locs=None, save_sample=False):

        ##################
        # File preparation
        if self.verbose > 0:
            print(f"Processing file: {self.base}{self.name}")

        # check array shape; convert if necessary
        self.convert_xyz_to_zxy()

        ##################
        # Process channels
        if locs is None:
            with h5.File(self.path, "r") as file:
                locs = [f"{key}" for key in list(file["data/"].keys())]

        if type(locs) == str:
            locs = [locs]

        for loc in locs:

            if self.verbose > 0:
                print("Processing location: ", repr(loc))

            # reset state
            self.files = []
            self.dimensions = []
            self.mmaps = []

            # start cluster for parallel processing
            c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
                                                             single_thread=False)

            ###################
            # MOTION CORRECTION

            # check if mc already exists
            skip_mc = False
            with h5.File(self.path, "r") as file:
                if f"mc/{loc}" in file:
                    if self.verbose > 0:
                        print("Motion Correction already exists. Skipping ...")
                        skip_mc = True

            if not skip_mc:

                if self.verbose > 0:
                    print("Starting motion correction ...")

                # decide whether to split files dependent on available RAM
                # file_size = os.stat(self.path).st_size
                with h5.File(self.path, "r") as file:
                    data = file[f"data/{loc}"]
                    z, x, y = data.shape
                    byte_num = np.dtype(data.dtype).itemsize
                    array_size = z * x * y * byte_num
                ram_size = psutil.virtual_memory().total
                if self.verbose > 0:
                    print("{:.2f}GB : {:.2f}GB ({:.2f}%)".format(array_size / 1000 / 1024 / 1024,
                                                               ram_size / 1000 / 1024 / 1024,
                                                               array_size / ram_size * 100))

                if ram_size < array_size * ram_size_multiplier:
                    if self.verbose > 0:
                        print("RAM not sufficient. Splitting ...")
                    self.split_h5_file(loc, ram_size_multiplier=ram_size_multiplier)
                else:
                    # since we are not splitting we need to manually
                    # save the dimensions for the next steps
                    with h5.File(self.path, "r") as file:
                        self.dimensions.append(file[f"data/{loc}"].shape)
                    self.files.append(self.path)



                # check if mmap already exists
                files_to_process = [file for file in self.files if self.mmap_exists(file) is None]

                if len(files_to_process) > 0:
                    # Parameters
                    opts_dict = {
                        'fnames': files_to_process,
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

                    # Run correction
                    if self.verbose > 0:
                        print("Starting motion correction ... [{}]".format(f"data/{loc}"))
                    mc = MotionCorrect(self.files, dview=dview, var_name_hdf5=f"data/{loc}", **opts.get_group('motion'))
                    mc.motion_correct(save_movie=True)

                ####################
                # Convert mmap to h5
                self.mmaps = [self.base + self.mmap_exists(file) for file in self.files]
                assert len(self.mmaps) == len(self.files), "Missing .mmap files"

                print("Converting mmap to h5 ...")
                self.save_memmap_to_h5(loc=f"mc/{loc}")

                # load memory mappable file
                if self.verbose > 0:
                    print("Creating Memory Map ...")

                # TODO this part should probably be:
                fname_new = cm.save_memmap_join(self.mmaps, base_name=f"mmap-{loc}_", dview=dview)

                # fname_mc = mc.fname_tot_els if self.pw_rigid else mc.fname_tot_rig
                # if self.pw_rigid:
                #     bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                #                                  np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
                # else:
                #     bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
                # fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                #                    border_to_0=bord_px)
                Yr, dims, T = cm.load_memmap(fname_new)
                images = Yr.T.reshape((T,) + dims, order='F')

            else:

                # load memory mappable file
                if self.verbose > 0:
                    print("Creating Memory Map ...")

                # check if mmap already exists
                with h5.File(self.path) as file:
                    Z, _, _ = file[f"data/{loc}"].shape

                fname_new = None
                for file in os.listdir(self.base):
                    if (file.startswith("memmap_")) and (file.endswith(".mmap")) and (f"frames_{Z}" in file):
                        fname_new = self.base + file
                        if self.verbose > 0:
                            print("Found memory map: {}".format(fname_new))
                        break

                if fname_new is None:

                    fname_new = cm.save_memmap([self.path], base_name=f"memmap-{loc}_", var_name_hdf5=f"mc/{loc}",
                               order='C', border_to_0=0, dview=dview)

                    # with h5.File(self.path, "r") as file:
                    #     data = file[f"mc/{loc}"]
                    #     Z, X, Y = data.shape
                    #     cz, cx, cy = data.chunks
                    #     fname_new = self.base + f"memmap__d1_{X}_d2_{Y}_d3_1_order_C_frames_{Z}_.mmap"
                    #     mmap = np.memmap(fname_new,
                    #                      dtype=np.float32, shape=(Z, X, Y), order='C', mode="w+")
                    #
                    #     for z0 in self.dtqdm(range(0, Z, cz)):
                    #         z1 = min(z0+cz, Z)
                    #
                    #         for x0 in range(0, X, cx):
                    #             x1 = min(x0+cx, X)
                    #
                    #             for y0 in range(0, Y, cy):
                    #                 y1 = min(y0+cy, Y)
                    #
                    #                 mmap[z0:z1, x0:x1, y0:y1] = data[z0:z1, x0:x1, y0:y1]


                Yr, dims, T = cm.load_memmap(fname_new)
                images = Yr.T.reshape((T,) + dims, order='F')

            #################
            # CNMFE DENOISING

            # %% Parameters for source extraction and deconvolution (CNMF-E algorithm)
            opts_dict = {'dims': dims,
                                'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': None, # upper bound on number of components per patch, in general None for 1p data
                                'gSig': (3, 3),  # gaussian width of a 2D gaussian kernel, which approximates a neuron
                                'gSiz': (13, 13),  # average diameter of a neuron, in general 4*gSig+1
                                'merge_thr': .7,  # merging threshold, max correlation allowed
                                'p': 1, # order of the autoregressive system
                                'tsub': 2, # downsampling factor in time for initialization,
                                      # increase if you have memory problems
                                'ssub': 1,  # downsampling factor in space for initialization,
                                    # increase if you have memory problems
                                    # you can pass them here as boolean vectors
                                'rf': 40,  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
                                'stride': 20,  # amount of overlap between the patches in pixels
                                        # (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
                                'only_init': True,  # set it to True to run CNMF-E
                                'nb': 0, # number of background components (rank) if positive,
                                      # else exact ring model with following settings
                                      # gnb= 0: Return background as b and W
                                      # gnb=-1: Return full rank background B
                                      # gnb<-1: Don't return background
                                'nb_patch': 0,  # number of background components (rank) per patch if gnb>0,
                                      # else it is set automatically
                                'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
                                'low_rank_background': None,  # None leaves background of each patch intact,
                                      # True performs global low-rank approximation if gnb>0
                                'update_background_components': True,
                                # sometimes setting to False improve the results
                                'min_corr': .8,  # min peak value from correlation image
                                'min_pnr': 10,  # min peak to noise ration from PNR image
                                'normalize_init': False,  # just leave as is
                                'center_psf': True,  # leave as is for 1 photon
                                'ssub_B': 2,  # additional downsampling factor in space for background
                                'ring_size_factor': 1.4,  # radius of ring is gSiz*ring_size_factor
                                'del_duplicates': True,  # whether to remove duplicates from initialization
                                'border_pix': 0, # number of pixels to not consider in the borders)
                                      }
            opts = cnmfparams.CNMFParams(params_dict=opts_dict)

            # # TODO DO WE NEED THIS?
            # # %% compute some summary images (correlation and peak to noise)
            # if self.verbose > 0:
            #     print("Calculating summary ...")
            # cn_filter, pnr = cm.summary_images.correlation_pnr(images[::correlation_skip],
            #                                                    gSig=opts_dict["gSig"][0], swap_dim=False)
            #
            # # inspect the summary images and set the parameters
            # inspect_correlation_pnr(cn_filter, pnr)
            # # print parameters set above, modify them if necessary based on summary images
            # print("Min correlation of Peak: {}\nMin peak to to noise ration: {}".format(
            #     opts_dict["min_corr"], opts_dict["min_pnr"]))

            # %% RUN CNMF ON PATCHES
            if self.verbose > 0:
                print("Running CNMF ...")
            cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=opts)
            cnm.fit(images)
            print("Fit successful!")

            # save result
            if self.verbose > 0:
                print("Reconstructing ...")
            rec = self.get_reconstructed(cnm.estimates, images)

            if self.verbose > 0:
                print("Saving reconstruction ...")

            with h5.File(self.path, "a") as file:
                data = file.create_dataset(f"cnmfe/{loc}", dtype="i2", shape=rec.shape)
                data[:, :, :] = rec

            # if self.verbose > 0:
            #     print("Calculating dFF ...")
            #
            # mov = cm.movie(rec)
            # mov_dff1, _ = (mov + abs(np.min(mov)) + 1).computeDFF(secsWindow=5, method='delta_f_over_sqrt_f')
            #
            # if self.verbose > 0:
            #     print("Saving dFF ...")
            #
            # with h5.File(self.path, "a") as file:
            #     data = file.create_dataset(f"dff/{loc}", dtype="i2", shape=rec.shape)
            #     data[:, :, :] = mov_dff1

            # stop cluster
            dview.terminate()
            cm.stop_server(dview=dview)

            #############
            # Save sample
            if save_sample:
                print("Saving samples ...")
                self.save_tiff(loc=f"mc/{loc}")
                self.save_tiff(loc=f"cnmfe/{loc}")
                self.save_tiff(loc=f"dff/{loc}")

            ###################
            # delete temp files
            #TODO this is not working -___________-
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
                if f"zxy/{loc}" in file:
                    del file[f"zxy/{loc}"]

                zxy = file.create_dataset(f"zxy/{loc}", dtype="i2", shape=(Z, X, Y),
                                          compression="gzip", chunks=(cx, cy, cz), shuffle=True)

                # necessary for downstream processing
                if "dummy" not in file:
                    _ = file.create_dataset("dummy", dtype="i2", shape=(1, 1, 1))

                # transform and copy data to new shape
                for start in self.dtqdm(range(0, Z, cz)):

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

    def split_h5_file(self, loc, ram_size_multiplier=5):

        # Load file
        with h5.File(self.path, "r") as file:

            data = file[f"data/{loc}"]

            Z, X, Y = data.shape

            array_size = Z * X * Y * np.dtype(data.dtype).itemsize
            ram_size = psutil.virtual_memory().total

            splits = max(2, int(array_size / (ram_size / ram_size_multiplier)))
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

                name_out = f'{self.base}{c}-{loc}-{self.name}0.h5'

                if not os.path.isfile(name_out):

                    chunk = data[start:stop, :, :]

                    with h5.File(name_out, "w") as temp:
                        chunk_drive = temp.create_dataset(f"data/{loc}", shape=chunk.shape, dtype=data.dtype)
                        chunk_drive[:, :, :] = chunk
                        temp.create_dataset("proc/dummy", shape=(1, 1, 1), dtype=data.dtype)
                        shape = chunk.shape

                else:
                    with h5.File(name_out, "r") as temp:
                        shape = temp[f"data/{loc}"].shape

                c += 1
                self.files.append(name_out)
                self.dimensions.append(shape)

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
            start, stop = (0, 0)
            print(self.mmaps)
            print(self.dimensions)
            for mmap, dim in self.dtqdm(zip(self.mmaps, self.dimensions)):
                _z, _x, _y = dim
                stop = start + _z
                mm = np.memmap(mmap, shape=dim, dtype=np.float32)
                data[start:stop, :, :] = mm

                start = stop

        return True

    def get_mmaps(self):

        dir_files = os.listdir(self.base)

        if len(self.files) > 1:

            for p in self.files:
                start = p.replace(self.base, "").replace(".h5", "")[-2]

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

    def mmap_exists(self, spath):

        spath = spath.replace(self.base, "").replace(".h5", "")[:-2]

        for fi in os.listdir(self.base):
            if fi.endswith(".mmap") and (spath in fi):
                return fi

        return None

    def save_tiff(self, loc, skip=1, downsize=0.5, subindices=None, max_frames=None):

        with h5.File(self.path, "r") as file:

            arr = file[loc]
            loc_name = loc.replace("/", "-")
            out = f"{self.base}{self.name}_{loc_name}.tiff"

            Z, X, Y = arr.shape

            if (max_frames is not None) and (int(Z/max_frames) > skip):
                skip = int(Z/max_frames)

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

    def get_reconstructed(self, estimates, imgs, include_bck=True):

        dims = imgs.shape[1:]

        AC = estimates.A.dot(estimates.C)
        Y_rec = AC.reshape(dims + (-1,), order='F')
        Y_rec = Y_rec.transpose([2, 0, 1])
        if estimates.W is not None:
            ssub_B = int(round(np.sqrt(np.prod(dims) / estimates.W.shape[0])))
            B = imgs.reshape((-1, np.prod(dims)), order='F').T - AC
            if ssub_B == 1:
                B = estimates.b0[:, None] + estimates.W.dot(B - estimates.b0[:, None])
            else:
                WB = estimates.W.dot(downscale(B.reshape(dims + (B.shape[-1],), order='F'),
                                               (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F'))
                Wb0 = estimates.W.dot(downscale(estimates.b0.reshape(dims, order='F'),
                                                (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
                B = estimates.b0.flatten('F')[:, None] + (np.repeat(
                    np.repeat((WB - Wb0).reshape(((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'),
                              ssub_B, 0), ssub_B, 1)[:dims[0], :dims[1]].reshape((-1, B.shape[-1]), order='F'))
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        elif estimates.b is not None and estimates.f is not None:
            B = estimates.b.dot(estimates.f)
            if 'matrix' in str(type(B)):
                B = B.toarray()
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        else:
            B = np.zeros_like(Y_rec)

        reconstructed = Y_rec + include_bck * B
        return reconstructed


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

    mc = PreProcessor(path=input_file, verbose=3, delete_temp_files=True, on_server=False)
    mc.run_preprocess(ram_size_multiplier=100)



