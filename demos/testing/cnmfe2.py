import os, sys
import h5py as h5
import numpy as np
import getopt
import time
import traceback

import caiman as cm
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.initialization import downscale
import tifffile as tf

import warnings
warnings.filterwarnings("ignore")

def main(path, loc, dview, n_processes, save_tiff=False, indices=None, ):

    print("Path: ", path)
    print("Loc: ", loc)

    # start the cluster
    # try:
    #     cm.stop_server()  # stop it if it was running
    # except():
    #     pass

    bord_px = 0     # because border_nan == 'copy' in motion correction

    # print("Saving mmap ...")
    # order = 'C'
    # mmap_name = cm.save_memmap([path], base_name='memmap_', var_name_hdf5=loc,
    #                            order=order, border_to_0=0,
    #                            slices=[indices, slice(0, 1200), slice(0, 1200)])
    #
    # # load memory mappable file
    # Yr, dims, T = cm.load_memmap(fname_new)
    # images = Yr.T.reshape((T,) + dims, order='F')

    print("Saving mmap ...")
    with h5.File(path) as file:
        print("Indices: {} - {}".format(indices.start, indices.stop))
        data = file[loc][indices.start:indices.stop, :, :]
        z, x, y = data.shape
        print(f"Loaded shape: {z}x{x}x{y}")

        # memmap__d1_1200_d2_1200_d3_1_order_C_frames_50_
        order = 'C'
        mmap_name = path + "_d1_{}_d2_{}_d3_{}_order_{}_frames_{}_{}.{}-{}.mmap".format(
            x, y, 1, order, z, loc.replace("/", "-"), indices.start, indices.stop)

        temp = np.memmap(mmap_name, dtype=np.float32, order=order, mode='w+', shape=data.shape)
        temp[:, :, :] = data

        # data = np.reshape(data, (x*y, z))
        # temp = np.memmap(mmap_name, dtype=np.float32, order=order, mode='w+', shape=(x*y, z))
        # temp[:] = data

        # del data
        # del temp

    # Yr, dims, T = cm.load_memmap(mmap_name)
    # images = Yr.T.reshape((T,) + dims, order=order)

    images = temp
    dims = (x, y)

    print("shape: {}".format(images.shape))

    # %% Parameters for source extraction and deconvolution (CNMF-E algorithm)

    p = 1  # order of the autoregressive system
    K = None  # upper bound on number of components per patch, in general None for 1p data
    gSig = (3, 3)  # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)  # average diameter of a neuron, in general 4*gSig+1
    Ain = None  # possibility to seed with predetermined binary masks
    merge_thr = .7  # merging threshold, max correlation allowed
    rf = 40  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20  # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2  # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1  # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0  # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0  # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = .8  # min peak value from correlation image
    min_pnr = 10  # min peak to noise ration from PNR image
    ssub_B = 2  # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    opts = params.CNMFParams(params_dict={'dims': dims,
                                          'method_init': 'corr_pnr',  # use this for 1 photon
                                          'K': K,
                                          'gSig': gSig,
                                          'gSiz': gSiz,
                                          'merge_thr': merge_thr,
                                          'p': p,
                                          'tsub': tsub,
                                          'ssub': ssub,
                                          'rf': rf,
                                          'stride': stride_cnmf,
                                          'only_init': True,  # set it to True to run CNMF-E
                                          'nb': gnb,
                                          'nb_patch': nb_patch,
                                          'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
                                          'low_rank_background': low_rank_background,
                                          'update_background_components': True,
                                          # sometimes setting to False improve the results
                                          'min_corr': min_corr,
                                          'min_pnr': min_pnr,
                                          'normalize_init': False,  # just leave as is
                                          'center_psf': True,  # leave as is for 1 photon
                                          'ssub_B': ssub_B,
                                          'ring_size_factor': ring_size_factor,
                                          'del_duplicates': True,  # whether to remove duplicates from initialization
                                          'border_pix': bord_px  # number of pixels to not consider in the borders)
                                          })
    try:
        # %% RUN CNMF ON PATCHES
        cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=None, params=opts)
        cnm.fit(images)
        print("Fit successful!")

        # save result
        rec = get_reconstructed(cnm.estimates, images)

        with h5.File(path, "a") as file:

            new_loc = loc.replace("mc", "cnmfe")
            if new_loc not in file:
                data = file.create_dataset(new_loc, dtype="i2", shape=file[loc].shape)
            else:
                data = file[new_loc]

            if indices is None:
                data[:, :, :] = rec
            else:
                data[indices.start:indices.stop, :, :] = rec

        if save_tiff:
            tf.imsave(path+"_"+loc.replace("/", "-")+"{}-{}".format(indices.start, indices.stop)+".tiff", rec)
            print("Sample saved!")

        # # save dFF
        # rec = cm.movie(rec)
        #
        # mov_dff1, _ = (rec + abs(np.min(rec)) + 1).computeDFF(secsWindow=5, method='delta_f_over_sqrt_f')
        # if save_tiff:
        #     tf.imsave(path+"_"+loc.replace("/", "-")+".dFF.tiff", mov_dff1)
        #
        # with h5.File(path, "a") as file:
        #     data = file.create_dataset(loc.replace("mc", "dff"), dtype="i2", shape=rec.shape)
        #     data[:, :, :] = mov_dff1

    except Exception as err:
        print(err)
        traceback.print_exc()

    finally:

        # cleanup
        os.remove(mmap_name)


def get_reconstructed(estimates, imgs, include_bck=True):

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

    # base = "/media/carmichael/LaCie SSD/JR/data/ca_imaging/28.06.21/slice4/"
    # name = "1-40X-loc1.h5"

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

    # main(path=input_file, loc="mc/ast", save_tiff=True, in_memory=True)
    # main(path=input_file, loc="mc/neu", save_tiff=True, in_memory=True)

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,  # TODO why is this so weird
                                                     single_thread=False)

    print("Cluster started!")

    try:
        steps = 500
        with h5.File(input_file) as file:
            data = file["mc/ast"]
            z, x, y = data.shape

        print(f"Shape: {x}x{y}x{z}")

        for z0 in range(0, z, steps):

            z1 = min(z, z0+steps)
            print(f"Processing {z0} to {z1}")

            main(path=input_file, loc="mc/ast", dview=dview, n_processes=n_processes,
                 save_tiff=True, indices=slice(z0, z1))
            main(path=input_file, loc="mc/neu", dview=dview, n_processes=n_processes,
                 save_tiff=True, indices=slice(z0, z1))

    finally:

        # stop cluster
        dview.terminate()
        cm.stop_server()

        print("Done")