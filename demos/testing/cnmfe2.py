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
from past.builtins import basestring
from tqdm import tqdm

from pbullet import Comm

import warnings
warnings.filterwarnings("ignore")

def main(path, loc, dview, n_processes, save_tiff=False, indices=None):

    print("Path: ", path)
    print("Loc: ", loc)

    print("Saving mmap ...")

    # mmap_name = cm.save_memmap([path], base_name='memmap_', var_name_hdf5=loc,
    #                            order='C', border_to_0=0, dview=None, slices=(indices, None, None))

    base_name = path.split(os.sep)[-1].replace(".zip.h5", "") + "_"
    mmap_name = save_memmap_h5(path, base_name=base_name, var_name_hdf5=loc,
                               order='C', slices=(indices, None, None))

    Yr, dims, T = cm.load_memmap(mmap_name)
    images = Yr.T.reshape((T,) + dims, order='C') # TODO can we get away with reshaping while we are saving?

    # %% Parameters for source extraction and deconvolution (CNMF-E algorithm)

    bord_px, use_cuda = 0, False

    p = 1  # order of the autoregressive system
    K = None  # upper bound on number of components per patch, in general None for 1p data
    gSig = (3, 3)  # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)  # average diameter of a neuron, in general 4*gSig+1
    Ain = None  # possibility to seed with predetermined binary masks
    merge_thr = .7  # merging threshold, max correlation allowed
    rf = 20 # 40  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80 #TODO change back
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
                                          'border_pix': bord_px,  # number of pixels to not consider in the borders)
                                          'use_cuda': use_cuda,
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
                data = file.create_dataset(new_loc, dtype="i2", shape=file[loc].shape,
                                           compression="gzip", chunks=(100, 100, 100), shuffle=True)
            else:
                data = file[new_loc]

            if indices is None:
                data[:, :, :] = rec
            else:
                data[indices.start:indices.stop, :, :] = rec

            print(f"Saved cnmfe result to {new_loc}")

        if save_tiff:
            tf.imwrite(path+"_"+loc.replace("/", "-")+"{}-{}".format(indices.start, indices.stop)+".tiff", rec)
            print("Sample saved!")

        # # save dFF
        # rec = cm.movie(rec)
        #
        # mov_dff1, _ = (rec + abs(np.min(rec)) + 1).computeDFF(secsWindow=5, method='only_baseline')
        # if save_tiff:
        #     tf.imsave(path+"_"+loc.replace("/", "-")+".dFF.tiff", mov_dff1)
        #
        # print("Saving dFF")
        # with h5.File(path, "a") as file:
        #
        #     new_loc = loc.replace("mc", "dff")
        #     if new_loc not in file:
        #         data = file.create_dataset(new_loc, dtype="i2", shape=file[loc].shape,
        #                                    compression="gzip", chunks=(100, 100, 100), shuffle=True)
        #     else:
        #         data = file[new_loc]
        #
        #     if indices is None:
        #         data[:, :, :] = mov_dff1
        #     else:
        #         data[indices.start:indices.stop, :, :] = mov_dff1[:]

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


def save_memmap(filenames,
                base_name='Yr',
                # resize_fact=(1, 1, 1),
                remove_init: int = 0,
                idx_xy= None,
                order: str = 'F',
                var_name_hdf5: str = 'mov',
                xy_shifts= None,
                dview=None,
                n_chunks: int = 100,
                slices=None) -> str:

    """ Efficiently write data from a list of tif files into a memory mappable file

    Args:
        filenames: list
            list of tif files or list of numpy arrays

        base_name: str
            the base used to build the file name. IT MUST NOT CONTAIN "_"

        resize_fact: tuple
            x,y, and z downsampling factors (0.5 means downsampled by a factor 2)

        remove_init: int
            number of frames to remove at the begining of each tif file
            (used for resonant scanning images if laser in rutned on trial by trial)

        idx_xy: tuple size 2 [or 3 for 3D data]
            for selecting slices of the original FOV, for instance
            idx_xy = (slice(150,350,None), slice(150,350,None))

        order: string
            whether to save the file in 'C' or 'F' order

        xy_shifts: list
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping

        is_3D: boolean
            whether it is 3D data

        add_to_movie: floating-point
            value to add to each image point, typically to keep negative values out.

        border_to_0: (undocumented)

        dview:       (undocumented)

        n_chunks:    (undocumented)

        slices: slice object or list of slice objects
            slice can be used to select portion of the movies in time and x,y
            directions. For instance
            slices = [slice(0,200),slice(0,100),slice(0,100)] will take
            the first 200 frames and the 100 pixels along x and y dimensions.
    Returns:
        fname_new: the name of the mapped file, the format is such that
            the name will contain the frame dimensions and the number of frames

    """
    if not isinstance(filenames, list):
        raise Exception('save_memmap: input should be a list of filenames')

    if slices is not None:
        slices = [slice(0, None) if sl is None else sl for sl in slices]

    Ttot = 0
    for idx, f in enumerate(filenames):

        if isinstance(f, (basestring, list)):
            Yr = cm.load(f, fr=1, in_memory=False, var_name_hdf5=var_name_hdf5)
            print(type(Yr), Yr.shape)
        else:
            Yr = cm.movie(f)

        if xy_shifts is not None:
            Yr = Yr.apply_shifts(xy_shifts, interpolation='cubic', remove_blanks=False)

        if slices is not None:
            Yr = Yr[tuple(slices)]
        else:
            if idx_xy is None:
                if remove_init > 0:
                    Yr = Yr[remove_init:]
            elif len(idx_xy) == 2:
                Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
            else:
                raise Exception('You need to set is_3D=True for 3D data)')
                Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

        # fx, fy, fz = resize_fact
        # if fx != 1 or fy != 1 or fz != 1:
        #     if 'movie' not in str(type(Yr)):
        #         Yr = cm.movie(Yr, fr=1)
        #     Yr = Yr.resize(fx=fx, fy=fy, fz=fz)

        T, dims = Yr.shape[0], Yr.shape[1:]
        Yr = np.transpose(Yr, list(range(1, len(dims) + 1)) + [0])
        Yr = np.reshape(Yr, (np.prod(dims), T), order='F')
        Yr = np.ascontiguousarray(Yr, dtype=np.float32) + np.float32(0.0001)

        if idx == 0:


            fname_tot = "{}_d1_{}_d2_{}_d3_{}_order_{}_frames_{}_.mmap".format(base_name,
                dims[0], dims[1], 1, order, T
            )

            print("Saving to: ", fname_tot)

            # if isinstance(f, str):
            #     fname_tot = caiman.paths.fn_relocated(os.path.join(os.path.split(f)[0], fname_tot))

            if len(filenames) > 1:
                # big_mov = np.memmap(caiman.paths.fn_relocated(fname_tot),
                #                     mode='w+',
                #                     dtype=np.float32,
                #                     shape=prepare_shape((np.prod(dims), T)),
                #                     order=order)
                # big_mov[:, Ttot:Ttot + T] = Yr
                # del big_mov

                print("Multiple files not implemented ...")
                sys.exit(2)

            else:
            #     logging.debug('SAVING WITH numpy.tofile()')
                Yr.tofile(fname_tot)
        else:
            big_mov = np.memmap(fname_tot,
                                dtype=np.float32,
                                mode='r+',
                                shape=prepare_shape((np.prod(dims), Ttot + T)),
                                order=order)

            big_mov[:, Ttot:Ttot + T] = Yr
            del big_mov

        sys.stdout.flush()
        # Ttot = Ttot + T
        #
        # fname_new = cm.paths.fn_relocated(fname_tot + f'_frames_{Ttot}_.mmap')
        # try:
        #     # need to explicitly remove destination on windows
        #     os.unlink(fname_new)
        # except OSError:
        #     pass
        # os.rename(fname_tot, fname_new)

    return fname_tot


def save_memmap_slim(filenames, base_name='Yr',
                     remove_init: int = 0, idx_xy=None, order: str = 'F',
                     var_name_hdf5: str = 'mov', xy_shifts=None, slices=None) -> str:

    """ Efficiently write data from a list of tif files into a memory mappable file

    Args:
        filenames: list
            list of tif files or list of numpy arrays

        base_name: str
            the base used to build the file name. IT MUST NOT CONTAIN "_"

        resize_fact: tuple
            x,y, and z downsampling factors (0.5 means downsampled by a factor 2)

        remove_init: int
            number of frames to remove at the begining of each tif file
            (used for resonant scanning images if laser in rutned on trial by trial)

        idx_xy: tuple size 2 [or 3 for 3D data]
            for selecting slices of the original FOV, for instance
            idx_xy = (slice(150,350,None), slice(150,350,None))

        order: string
            whether to save the file in 'C' or 'F' order

        xy_shifts: list
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping

        is_3D: boolean
            whether it is 3D data

        add_to_movie: floating-point
            value to add to each image point, typically to keep negative values out.

        border_to_0: (undocumented)

        dview:       (undocumented)

        n_chunks:    (undocumented)

        slices: slice object or list of slice objects
            slice can be used to select portion of the movies in time and x,y
            directions. For instance
            slices = [slice(0,200),slice(0,100),slice(0,100)] will take
            the first 200 frames and the 100 pixels along x and y dimensions.
    Returns:
        fname_new: the name of the mapped file, the format is such that
            the name will contain the frame dimensions and the number of frames

    """

    if isinstance(filenames, list):
        f = filenames[0]
    else:
        f = filenames

    root = os.sep.join(f.split(os.sep)[:-1]) + os.sep

    if slices is not None:
        slices = [slice(0, None) if sl is None else sl for sl in slices]

    Yr = cm.load(f, fr=1, in_memory=False, var_name_hdf5=var_name_hdf5)

    if xy_shifts is not None:
        Yr = Yr.apply_shifts(xy_shifts, interpolation='cubic', remove_blanks=False)

    if slices is not None:
        Yr = Yr[tuple(slices)]
    else:
        if idx_xy is None:
            if remove_init > 0:
                Yr = Yr[remove_init:]
        elif len(idx_xy) == 2:
            Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
        else:
            raise Exception('You need to set is_3D=True for 3D data)')
            Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

    T, dims = Yr.shape[0], Yr.shape[1:]
    Yr = np.transpose(Yr, list(range(1, len(dims) + 1)) + [0])
    Yr = np.reshape(Yr, (np.prod(dims), T), order='F')
    Yr = np.ascontiguousarray(Yr, dtype=np.float32) + np.float32(0.0001)

    fname_tot = "{}{}_d1_{}_d2_{}_d3_{}_order_{}_frames_{}_.mmap".format(root, base_name, dims[0], dims[1], 1, order, T)
    print(fname_tot)

    Yr.tofile(fname_tot)

    sys.stdout.flush()
    return fname_tot

def save_memmap_h5(filenames, base_name='Yr', order: str = 'F', var_name_hdf5: str = 'mov', slices=None) -> str:

    if isinstance(filenames, list):
        f = filenames[0]
    else:
        f = filenames

    with h5.File(f, "r") as file:

        data = file[var_name_hdf5]
        Z, X, Y = data.shape
        cz, cx, cy = data.chunks

        # indice selection
        slz0, slz1 = 0, Z
        slx0, slx1 = 0, X
        sly0, sly1 = 0, Y
        if slices is not None:
            slz, slx, sly = slices

            if slz is not None:
                slz0, slz1 = slz.start, slz.stop
            if slx is not None:
                slx0, slx1 = slx.start, slx.stop
            if sly is not None:
                sly0, sly1 = sly.start, sly.stop

        Z = min(slz1, Z)
        X = min(slx1, X)
        Y = min(sly1, Y)

        assert slz0 == 0 or slz0 % cz == 0, f"Please choose a Z slice that is 0 or a multiple of chunk size: {cz}"
        assert slx0 == 0 or slx0 % cx == 0, f"Please choose a X slice that is 0 or a multiple of chunk size: {cx}"
        assert sly0 == 0 or sly0 % cy == 0, f"Please choose a Y slice that is 0 or a multiple of chunk size: {cy}"

        Zlen, Xlen, Ylen = (Z-slz0, X-slx0, Y-sly0)

        # file name selection
        root = os.sep.join(f.split(os.sep)[:-1]) + os.sep
        fname_tot = "{}{}_d1_{}_d2_{}_d3_{}_order_{}_frames_{}_.mmap".format(root, base_name, Xlen, Ylen, 1,
                                                                             order, Zlen)

        out = np.memmap(fname_tot, dtype=np.float32, mode="w+", shape=(Xlen * Ylen, Zlen))

        for z0 in tqdm(range(slz0, Z, cz)):
            for x0 in range(slx0, X, cx):
                for y0 in range(sly0, Y, cy):

                    z1 = min(Z, z0 + cz)
                    x1 = min(X, x0 + cx)
                    y1 = min(Y, y0 + cy)

                    chunk = data[z0:z1, x0:x1, y0:y1]

                    chz, chx, chy = chunk.shape

                    for a0 in range(chz):
                        for c0 in range(chy):  # TODO change to cy (cx?); otherwise image is rotated and horizontally mirrored

                            col_section = chunk[a0, :, c0]

                            ind0 = int(x0 / cx * cx + y0 * Xlen + c0 * Xlen)
                            ind1 = ind0 + chx

                            indx0 = int((z0-slz0) / cz * cz + a0)
                            out[ind0:ind1, indx0] = col_section + np.float32(0.0001)

                    sys.stdout.flush()

    return fname_tot


if __name__ == "__main__":

    # base = "/media/carmichael/LaCie SSD/JR/data/ca_imaging/28.06.21/slice4/"
    # name = "1-40X-loc1.h5"

    input_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:l:", ["ifolder=", "local="])
    except getopt.GetoptError:
        print("calpack.py -i <input_file>")
        sys.exit(2)

    on_server = True
    for opt, arg in opts:
        if opt in ("-i", "--input_file"):
            input_file = arg

        if opt in ("-l"):
            on_server = False

    assert os.path.isfile(input_file), "input_file is not a file: {}".format(input_file)

    print("InputFile: ", input_file)

    # main(path=input_file, loc="mc/ast", save_tiff=True, in_memory=True)
    # main(path=input_file, loc="mc/neu", save_tiff=True, in_memory=True)

    n_processes = None
    if not on_server:
        n_processes = 6
        steps = 200
        use_cuda = False
    else:
        n_processes = None
        steps = 400
        use_cuda = False

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,  # TODO why is this so weird
                                                     single_thread=False)
    comm = Comm()
    print("Cluster started!")

    try:

        with h5.File(input_file) as file:
            data = file["mc/ast"]
            z, x, y = data.shape

        print(f"Shape (xyz): {x}x{y}x{z}")
        t0 = time.time()

        for z0 in range(0, z, steps):

            z1 = min(z, z0+steps)
            print(f"Processing {z0} to {z1}")

            main(path=input_file, loc="mc/ast", dview=dview, n_processes=n_processes, indices=slice(z0, z1))
            main(path=input_file, loc="mc/neu", dview=dview, n_processes=n_processes, indices=slice(z0, z1))

        # Finialization
        t1 = (time.time() - t0) / 60
        print("CMFE finished in {:.2f}".format(t1))
        comm.push_text("CMFE done!", f"CMFE done for {input_file}. It took {t1:.2f}min")

    except Exception as err:
        print(err)

        traceback.print_exc()

        # comm.push_text("Error in CMFE!", f"Exception in {input_file}")

    finally:

        # stop cluster
        dview.terminate()
        cm.stop_server()

        print("Done")
