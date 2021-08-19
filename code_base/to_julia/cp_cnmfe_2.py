
from builtins import object
from builtins import str

import cv2
import inspect
import logging
import numpy as np
import os
import psutil
import scipy
import sys
import glob
import pathlib

from typing import Any, List, Tuple, Union, Dict, Set
from past.utils import old_div
import tempfile
import shutil
from past.builtins import basestring
from scipy.sparse import csc_matrix
import h5py
import warnings
import itertools
import peakutils
import pickle
import time
from copy import copy, deepcopy
from collections import defaultdict
from contextlib import suppress
from scipy.linalg.lapack import dpotrf
from skimage.morphology import disk
import multiprocessing
import scipy.sparse as spr
from scipy.sparse import spdiags, diags
from scipy.linalg.lapack import dpotrf, dpotrs
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_closing
import platform
from skimage.transform import resize
from scipy.ndimage import center_of_mass
import numbers
import scipy.sparse as sp
from sklearn.decomposition import NMF

import cp_cnmfe as CNMFE_1
import cp_cnmfe_3 as CNMFE_3

from scipy.ndimage.morphology import generate_binary_structure, iterate_structure

from multiprocessing import current_process

_global_config = {
    'assume_finite': bool(os.environ.get('SKLEARN_ASSUME_FINITE', False)),
    'working_memory': int(os.environ.get('SKLEARN_WORKING_MEMORY', 1024)),
    'print_changed_only': True,
    'display': 'text',
}

try:
    profile
except:
    def profile(a): return a

# <------- CAIMAN CODE
import cp_cluster
import cp_motioncorrection
import cp_params
import cp_cnmfe
# CAIMAN CODE ------->

# from .estimates import Estimates
# from .initialization import initialize_components, compute_W
# from .map_reduce import run_CNMF_patches
# from .merging import merge_components
# from .params import CNMFParams
# from .pre_processing import preprocess_data
# from .spatial import update_spatial_components
# from .temporal import update_temporal_components, constrained_foopsi_parallel
# from .utilities import update_order
# from ... import mmapping
# from ...components_evaluation import estimate_components_quality
# from ...motion_correction import MotionCorrect
# from ...utils.utils import save_dict_to_hdf5, load_dict_from_hdf5
# from caiman import summary_images
# from caiman import cluster
# import caiman.paths


def _hsm(data):
    if data.size == 1:
        return data[0]
    elif data.size == 2:
        return data.mean()
    elif data.size == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return data[:2].mean()
        elif i2 > i1:
            return data[1:].mean()
        else:
            return data[1]
    else:

        wMin = np.inf
        N = old_div(data.size, 2) + data.size % 2

        for i in range(0, N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i

        return _hsm(data[j:j + N])

def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:

        def fnc(x):
            return mode_robust(x, dtype=dtype)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:

                wMin = np.inf
                N = data.size // 2 + data.size % 2

                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode

def estimator_html_repr(estimator):
    """Build a HTML representation of an estimator.

    Read more in the :ref:`User Guide <visualizing_composite_estimators>`.

    Parameters
    ----------
    estimator : estimator object
        The estimator to visualize.

    Returns
    -------
    html: str
        HTML representation of estimator.
    """
    # with closing(StringIO()) as out:
    #     container_id = "sk-" + str(uuid.uuid4())
    #     style_template = Template(_STYLE)
    #     style_with_id = style_template.substitute(id=container_id)
    #     out.write(f'<style>{style_with_id}</style>'
    #               f'<div id="{container_id}" class"sk-top-container">'
    #               '<div class="sk-container">')
    #     _write_estimator_html(out, estimator, estimator.__class__.__name__,
    #                           str(estimator), first_call=True)
    #     out.write('</div></div>')
    #
    #     html_output = out.getvalue()
    #     return html_output
    print("**to_julia** probably not important")
    return None

def get_config():
    """Retrieve current values for configuration set by :func:`set_config`

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    set_config : Set global scikit-learn configuration.
    """
    return _global_config.copy()

def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float."""
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError('Invalid beta_loss parameter: got %r instead '
                         'of one of %r, or a float.' %
                         (beta_loss, allowed_beta_loss.keys()))
    return beta_loss

def run_CNMF_patches(file_name, shape, params, gnb=1, dview=None,
                     memory_fact=1, border_pix=0, low_rank_background=True,
                     del_duplicates=False, indices=[slice(None)]*3):
    """Function that runs CNMF in patches

     Either in parallel or sequentially, and return the result for each.
     It requires that ipyparallel is running

     Will basically initialize everything in order to compute on patches then call a function in parallel that will
     recreate the cnmf object and fit the values.
     It will then recreate the full frame by listing all the fitted values together

    Args:
        file_name: string
            full path to an npy file (2D, pixels x time) containing the movie

        shape: tuple of three elements
            dimensions of the original movie across y, x, and time

        params:
            CNMFParms object containing all the parameters for the various algorithms

        gnb: int
            number of global background components

        backend: string
            'ipyparallel' or 'single_thread' or SLURM

        n_processes: int
            nuber of cores to be used (should be less than the number of cores started with ipyparallel)

        memory_fact: double
            unitless number accounting how much memory should be used.
            It represents the fration of patch processed in a single thread.
             You will need to try different values to see which one would work

        low_rank_background: bool
            if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)

        del_duplicates: bool
            if True keeps only neurons in each patch that are well centered within the patch.
            I.e. neurons that are closer to the center of another patch are removed to
            avoid duplicates, cause the other patch should already account for them.

    Returns:
        A_tot: matrix containing all the components from all the patches

        C_tot: matrix containing the calcium traces corresponding to A_tot

        sn_tot: per pixel noise estimate

        optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch

    Raises:
        Empty Exception
    """

    dims = shape[:-1]
    d = np.prod(dims)
    T = shape[-1]

    rf = params.get('patch', 'rf')
    if rf is None:
        rf = 16
    if np.isscalar(rf):
        rfs = [rf] * len(dims)
    else:
        rfs = rf

    stride = params.get('patch', 'stride')
    if stride is None:
        stride = 4
    if np.isscalar(stride):
        strides = [stride] * len(dims)
    else:
        strides = stride

    params_copy = deepcopy(params)
    npx_per_proc = np.prod(rfs) // memory_fact
    params_copy.set('preprocess', {'n_pixels_per_process': npx_per_proc})
    params_copy.set('spatial', {'n_pixels_per_process': npx_per_proc})
    params_copy.set('temporal', {'n_pixels_per_process': npx_per_proc})

    idx_flat, idx_2d = extract_patch_coordinates(
        dims, rfs, strides, border_pix=border_pix, indices=indices[1:])
    args_in = []
    patch_centers = []
    for id_f, id_2d in zip(idx_flat, idx_2d):
        #        print(id_2d)
        args_in.append((file_name, id_f, id_2d, params_copy))
        if del_duplicates:
            foo = np.zeros(d, dtype=bool)
            foo[id_f] = 1
            patch_centers.append(scipy.ndimage.center_of_mass(
                foo.reshape(dims, order='F')))
    logging.info('Patch size: {0}'.format(id_2d))
    st = time.time()
    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            file_res = dview.map_async(cnmf_patches, args_in).get(4294967)
        else:
            try:
                file_res = dview.map_sync(cnmf_patches, args_in)
                dview.results.clear()
            except:
                print('Something went wrong')
                raise
            finally:
                logging.info('Patch processing complete')

    else:
        file_res = list(map(cnmf_patches, args_in))

    logging.info('Elapsed time for processing patches: \
                 {0}s'.format(str(time.time() - st).split('.')[0]))
    # count components
    count = 0
    count_bgr = 0
    patch_id = 0
    num_patches = len(file_res)
    for jj, fff in enumerate(file_res):
        if fff is not None:
            idx_, shapes, A, b, C, f, S, bl, c1, neurons_sn, g, sn, _, YrA = fff
            for _ in range(np.shape(b)[-1]):
                count_bgr += 1

            A = A.tocsc()
            if del_duplicates:
                keep = []
                for ii in range(np.shape(A)[-1]):
                    neuron_center = (np.array(scipy.ndimage.center_of_mass(
                        A[:, ii].toarray().reshape(shapes, order='F'))) -
                        np.array(shapes) / 2. + np.array(patch_centers[jj]))
                    if np.argmin([np.linalg.norm(neuron_center - p) for p in
                                  np.array(patch_centers)]) == jj:
                        keep.append(ii)
                A = A[:, keep]
                file_res[jj][2] = A
                file_res[jj][4] = C[keep]
                if S is not None:
                    file_res[jj][6] = S[keep]
                    file_res[jj][7] = bl[keep]
                    file_res[jj][8] = c1[keep]
                    file_res[jj][9] = neurons_sn[keep]
                    file_res[jj][10] = g[keep]
                file_res[jj][-1] = YrA[keep]

            # for ii in range(np.shape(A)[-1]):
            #     new_comp = A[:, ii] / np.sqrt(A[:, ii].power(2).sum())
            #     if new_comp.sum() > 0:
            #         count += 1
            count += np.sum(A.sum(0) > 0)

            patch_id += 1

    # INITIALIZING
    nb_patch = params.get('patch', 'nb_patch')
    C_tot = np.zeros((count, T), dtype=np.float32)
    if params.get('init', 'center_psf'):
        S_tot = np.zeros((count, T), dtype=np.float32)
    else:
        S_tot = None
    YrA_tot = np.zeros((count, T), dtype=np.float32)
    F_tot = np.zeros((max(0, num_patches * nb_patch), T), dtype=np.float32)
    mask = np.zeros(d, dtype=np.uint8)
    sn_tot = np.zeros((d))

    f_tot, bl_tot, c1_tot, neurons_sn_tot, g_tot, idx_tot, id_patch_tot, shapes_tot = [
    ], [], [], [], [], [], [], []
    patch_id, empty, count_bgr, count = 0, 0, 0, 0
    idx_tot_B, idx_tot_A, a_tot, b_tot = [], [], [], []
    idx_ptr_B, idx_ptr_A = [0], [0]

    # instead of filling in the matrices, construct lists with their non-zero
    # entries and coordinates
    logging.info('Embedding patches results into whole FOV')
    for fff in file_res:
        if fff is not None:

            idx_, shapes, A, b, C, f, S, bl, c1, neurons_sn, g, sn, _, YrA = fff
            A = A.tocsc()

            # check A for nans, which result in corrupted outputs.  Better to fail here if any found
            nnan = np.isnan(A.data).sum()
            if nnan > 0:
                raise RuntimeError('found %d/%d nans in A, cannot continue' % (nnan, len(A.data)))

            sn_tot[idx_] = sn
            f_tot.append(f)
            bl_tot.append(bl)
            c1_tot.append(c1)
            neurons_sn_tot.append(neurons_sn)
            g_tot.append(g)
            idx_tot.append(idx_)
            shapes_tot.append(shapes)
            mask[idx_] += 1

            if scipy.sparse.issparse(b):
                b = scipy.sparse.csc_matrix(b)
                b_tot.append(b.data)
                idx_ptr_B += list(b.indptr[1:] - b.indptr[:-1])
                idx_tot_B.append(idx_[b.indices])
            else:
                for ii in range(np.shape(b)[-1]):
                    b_tot.append(b[:, ii])
                    idx_tot_B.append(idx_)
                    idx_ptr_B.append(len(idx_))
                    # F_tot[patch_id, :] = f[ii, :]
            count_bgr += b.shape[-1]
            if nb_patch >= 0:
                F_tot[patch_id * nb_patch:(patch_id + 1) * nb_patch] = f
            else:  # full background per patch
                F_tot = np.concatenate([F_tot, f])

            for ii in range(np.shape(A)[-1]):
                new_comp = A[:, ii]  # / np.sqrt(A[:, ii].power(2).sum())
                if new_comp.sum() > 0:
                    a_tot.append(new_comp.toarray().flatten())
                    idx_tot_A.append(idx_)
                    idx_ptr_A.append(len(idx_))
                    C_tot[count, :] = C[ii, :]
                    if params.get('init', 'center_psf'):
                        S_tot[count, :] = S[ii, :]
                    YrA_tot[count, :] = YrA[ii, :]
                    id_patch_tot.append(patch_id)
                    count += 1

            patch_id += 1
        else:
            empty += 1

    logging.debug('Skipped %d empty patches', empty)
    if count_bgr > 0:
        idx_tot_B = np.concatenate(idx_tot_B)
        b_tot = np.concatenate(b_tot)
        idx_ptr_B = np.cumsum(np.array(idx_ptr_B))
        B_tot = scipy.sparse.csc_matrix(
            (b_tot, idx_tot_B, idx_ptr_B), shape=(d, count_bgr))
    else:
        B_tot = scipy.sparse.csc_matrix((d, count_bgr), dtype=np.float32)

    if len(idx_tot_A):
        idx_tot_A = np.concatenate(idx_tot_A)
        a_tot = np.concatenate(a_tot)
        idx_ptr_A = np.cumsum(np.array(idx_ptr_A))
    A_tot = scipy.sparse.csc_matrix(
        (a_tot, idx_tot_A, idx_ptr_A), shape=(d, count), dtype=np.float32)

    C_tot = C_tot[:count, :]
    YrA_tot = YrA_tot[:count, :]
    F_tot = F_tot[:count_bgr]

    optional_outputs = dict()
    optional_outputs['b_tot'] = b_tot
    optional_outputs['f_tot'] = f_tot
    optional_outputs['bl_tot'] = bl_tot
    optional_outputs['c1_tot'] = c1_tot
    optional_outputs['neurons_sn_tot'] = neurons_sn_tot
    optional_outputs['g_tot'] = g_tot
    optional_outputs['S_tot'] = S_tot
    optional_outputs['idx_tot'] = idx_tot
    optional_outputs['shapes_tot'] = shapes_tot
    optional_outputs['id_patch_tot'] = id_patch_tot
    optional_outputs['B'] = B_tot
    optional_outputs['F'] = F_tot
    optional_outputs['mask'] = mask

    logging.info("Constructing background")

    Im = scipy.sparse.csr_matrix(
        (1. / (mask + np.finfo(np.float32).eps), (np.arange(d), np.arange(d))), dtype=np.float32)

    if not del_duplicates:
        A_tot = Im.dot(A_tot)

    if count_bgr == 0:
        b = None
        f = None
    elif low_rank_background is None:
        b = Im.dot(B_tot)
        f = F_tot
        logging.info("Leaving background components intact")
    elif low_rank_background:
        logging.info("Compressing background components with a low rank NMF")
        B_tot = Im.dot(B_tot)
        Bm = (B_tot)
        #f = np.r_[np.atleast_2d(np.mean(F_tot, axis=0)),
        #          np.random.rand(gnb - 1, T)]
        mdl = NMF(n_components=gnb, verbose=False, init='nndsvdar', tol=1e-10,
                  max_iter=100, shuffle=False, random_state=1)
        # Filter out nan components in the bg components
        nan_components = np.any(np.isnan(F_tot), axis=1)
        F_tot = F_tot[~nan_components, :]
        _ = mdl.fit_transform(F_tot).T
        Bm = Bm[:, ~nan_components]
        f = mdl.components_.squeeze()
        f = np.atleast_2d(f)
        for _ in range(100):
            f /= np.sqrt((f**2).sum(1)[:, None]) + np.finfo(np.float32).eps
            try:
                b = np.fmax(Bm.dot(F_tot.dot(f.T)).dot(
                    np.linalg.inv(f.dot(f.T))), 0)
            except np.linalg.LinAlgError:  # singular matrix
                b = np.fmax(Bm.dot(scipy.linalg.lstsq(f.T, F_tot.T)[0].T), 0)
            try:
                #f = np.linalg.inv(b.T.dot(b)).dot((Bm.T.dot(b)).T.dot(F_tot))
                f = np.linalg.solve(b.T.dot(b), (Bm.T.dot(b)).T.dot(F_tot))
            except np.linalg.LinAlgError:  # singular matrix
                f = scipy.linalg.lstsq(b, Bm.toarray())[0].dot(F_tot)

        nB = np.ravel(np.sqrt((b**2).sum(0)))
        b /= nB + np.finfo(np.float32).eps
        b = np.array(b, dtype=np.float32)
#        B_tot = scipy.sparse.coo_matrix(B_tot)
        f *= nB[:, None]
    else:
        logging.info('Removing overlapping background components \
                     from different patches')
        nA = np.ravel(np.sqrt(A_tot.power(2).sum(0)))
        A_tot /= nA
        A_tot = scipy.sparse.coo_matrix(A_tot)
        C_tot *= nA[:, None]
        YrA_tot *= nA[:, None]
        nB = np.ravel(np.sqrt(B_tot.power(2).sum(0)))
        B_tot /= nB
        B_tot = np.array(B_tot, dtype=np.float32)
#        B_tot = scipy.sparse.coo_matrix(B_tot)
        F_tot *= nB[:, None]

        processed_idx:Set = set([])
        # needed if a patch has more than 1 background component
        processed_idx_prev:Set = set([])
        for _b in np.arange(B_tot.shape[-1]):
            idx_mask = np.where(B_tot[:, _b])[0]
            idx_mask_repeat = processed_idx.intersection(idx_mask)
            if len(idx_mask_repeat) < len(idx_mask):
                processed_idx_prev = processed_idx
            else:
                idx_mask_repeat = processed_idx_prev.intersection(idx_mask)
            processed_idx = processed_idx.union(idx_mask)
            if len(idx_mask_repeat) > 0:
                B_tot[np.array(list(idx_mask_repeat), dtype=np.int), _b] = 0

        b = B_tot
        f = F_tot

        logging.info('using one background component per patch')

    logging.info("Constructing background DONE")

    return A_tot, C_tot, YrA_tot, b, f, sn_tot, optional_outputs

def cnmf_patches(args_in):
    """Function that is run for each patches

         Will be called

        Args:
            file_name: string
                full path to an npy file (2D, pixels x time) containing the movie

            shape: tuple of thre elements
                dimensions of the original movie across y, x, and time

            params:
                CNMFParms object containing all the parameters for the various algorithms

            rf: int
                half-size of the square patch in pixel

            stride: int
                amount of overlap between patches

            gnb: int
                number of global background components

            backend: string
                'ipyparallel' or 'single_thread' or SLURM

            n_processes: int
                nuber of cores to be used (should be less than the number of cores started with ipyparallel)

            memory_fact: double
                unitless number accounting how much memory should be used.
                It represents the fration of patch processed in a single thread.
                 You will need to try different values to see which one would work

            low_rank_background: bool
                if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)

        Returns:
            A_tot: matrix containing all the componenents from all the patches

            C_tot: matrix containing the calcium traces corresponding to A_tot

            sn_tot: per pixel noise estimate

            optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch

        Raises:
            Empty Exception
        """

    import logging
    # from . import cnmf
    file_name, idx_, shapes, params = args_in

    logger = logging.getLogger(__name__)
    name_log = os.path.basename(
        file_name[:-5]) + '_LOG_ ' + str(idx_[0]) + '_' + str(idx_[-1])
    # logger = logging.getLogger(name_log)
    # hdlr = logging.FileHandler('./' + name_log)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # hdlr.setFormatter(formatter)
    # logger.addHandler(hdlr)
    # logger.setLevel(logging.INFO)

    logger.debug(name_log + 'START')

    logger.debug(name_log + 'Read file')
    Yr, dims, timesteps = cp_motioncorrection.load_memmap(file_name)

    # slicing array (takes the min and max index in n-dimensional space and
    # cuts the box they define)
    # for 2d a rectangle/square, for 3d a rectangular cuboid/cube, etc.
    upper_left_corner = min(idx_)
    lower_right_corner = max(idx_)
    indices = np.unravel_index([upper_left_corner, lower_right_corner],
                               dims, order='F')  # indices as tuples
    slices = [slice(min_dim, max_dim + 1) for min_dim, max_dim in indices]
    # insert slice for timesteps, equivalent to :
    slices.insert(0, slice(timesteps))

    images = np.reshape(Yr.T, [timesteps] + list(dims), order='F')
    if params.get('patch', 'in_memory'):
        images = np.array(images[tuple(slices)], dtype=np.float32)
    else:
        images = images[slices]

    logger.debug(name_log+'file loaded')

    if (np.sum(np.abs(np.diff(images.reshape(timesteps, -1).T)))) > 0.1:

        opts = copy(params)
        opts.set('patch', {'n_processes': 1, 'rf': None, 'stride': None})
        for group in ('init', 'temporal', 'spatial'):
            opts.set(group, {'nb': params.get('patch', 'nb_patch')})
        for group in ('preprocess', 'temporal'):
            opts.set(group, {'p': params.get('patch', 'p_patch')})

        cnm = CNMF(n_processes=1, params=opts)

        cnm = cnm.fit(images)
        return [idx_, shapes, scipy.sparse.coo_matrix(cnm.estimates.A),
                cnm.estimates.b, cnm.estimates.C, cnm.estimates.f,
                cnm.estimates.S, cnm.estimates.bl, cnm.estimates.c1,
                cnm.estimates.neurons_sn, cnm.estimates.g, cnm.estimates.sn,
                cnm.params.to_dict(), cnm.estimates.YrA]
    else:
        raise RuntimeError('cannot merge if there are empty patches.')

        return None

def extract_patch_coordinates(dims: Tuple,
                              rf: Union[List, Tuple],
                              stride: Union[List[int], Tuple],
                              border_pix: int = 0,
                              indices=[slice(None)] * 2) -> Tuple[List, List]:
    """
    Partition the FOV in patches
    and return the indexed in 2D and 1D (flatten, order='F') formats

    Args:
        dims: tuple of int
            dimensions of the original matrix that will be divided in patches

        rf: tuple of int
            radius of receptive field, corresponds to half the size of the square patch

        stride: tuple of int
            degree of overlap of the patches
    """

    sl_start = [0 if sl.start is None else sl.start for sl in indices]
    sl_stop = [dim if sl.stop is None else sl.stop for (sl, dim) in zip(indices, dims)]
    sl_step = [1 for sl in indices]    # not used
    dims_large = dims
    dims = np.minimum(np.array(dims) - border_pix, sl_stop) - np.maximum(border_pix, sl_start)

    coords_flat = []
    shapes = []
    iters = [list(range(rf[i], dims[i] - rf[i], 2 * rf[i] - stride[i])) + [dims[i] - rf[i]] for i in range(len(dims))]

    coords = np.empty(list(map(len, iters)) + [len(dims)], dtype=np.object)
    for count_0, xx in enumerate(iters[0]):
        coords_x = np.arange(xx - rf[0], xx + rf[0] + 1)
        coords_x = coords_x[(coords_x >= 0) & (coords_x < dims[0])]
        coords_x += border_pix * 0 + np.maximum(sl_start[0], border_pix)

        for count_1, yy in enumerate(iters[1]):
            coords_y = np.arange(yy - rf[1], yy + rf[1] + 1)
            coords_y = coords_y[(coords_y >= 0) & (coords_y < dims[1])]
            coords_y += border_pix * 0 + np.maximum(sl_start[1], border_pix)

            if len(dims) == 2:
                idxs = np.meshgrid(coords_x, coords_y)

                coords[count_0, count_1] = idxs
                shapes.append(idxs[0].shape[::-1])

                coords_ = np.ravel_multi_index(idxs, dims_large, order='F')
                coords_flat.append(coords_.flatten())
            else:      # 3D data

                if border_pix > 0:
                    raise Exception(
                        'The parameter border pix must be set to 0 for 3D data since border removal is not implemented')

                for count_2, zz in enumerate(iters[2]):
                    coords_z = np.arange(zz - rf[2], zz + rf[2] + 1)
                    coords_z = coords_z[(coords_z >= 0) & (coords_z < dims[2])]
                    idxs = np.meshgrid(coords_x, coords_y, coords_z)
                    shps = idxs[0].shape
                    shapes.append([shps[1], shps[0], shps[2]])
                    coords[count_0, count_1, count_2] = idxs
                    coords_ = np.ravel_multi_index(idxs, dims, order='F')
                    coords_flat.append(coords_.flatten())

    for i, c in enumerate(coords_flat):
        assert len(c) == np.prod(shapes[i])

    return list(map(np.sort, coords_flat)), shapes

@profile
def compute_W(Y, A, C, dims, radius, data_fits_in_memory=True, ssub=1, tsub=1, parallel=False):
    """compute background according to ring model
    solves the problem
        min_{W,b0} ||X-W*X|| with X = Y - A*C - b0*1'
    subject to
        W(i,j) = 0 for each pixel j that is not in ring around pixel i
    Problem parallelizes over pixels i
    Fluctuating background activity is W*X, constant baselines b0.

    Args:
        Y: np.ndarray (2D or 3D)
            movie, raw data in 2D or 3D (pixels x time).
        A: np.ndarray or sparse matrix
            spatial footprint of each neuron.
        C: np.ndarray
            calcium activity of each neuron.
        dims: tuple
            x, y[, z] movie dimensions
        radius: int
            radius of ring
        data_fits_in_memory: [optional] bool
            If true, use faster but more memory consuming computation
        ssub: int
            spatial downscale factor
        tsub: int
            temporal downscale factor
        parallel: bool
            If true, use multiprocessing to process pixels in parallel

    Returns:
        W: scipy.sparse.csr_matrix (pixels x pixels)
            estimate of weight matrix for fluctuating background
        b0: np.ndarray (pixels,)
            estimate of constant background baselines
    """

    if current_process().name != 'MainProcess':
        # no parallelization over pixels if already processing patches in parallel
        parallel = False

    T = Y.shape[1]
    d1 = (dims[0] - 1) // ssub + 1
    d2 = (dims[1] - 1) // ssub + 1

    radius = int(round(radius / float(ssub)))
    ring = disk(radius + 1)
    ring[1:-1, 1:-1] -= disk(radius)
    ringidx = [i - radius - 1 for i in np.nonzero(ring)]

    def get_indices_of_pixels_on_ring(pixel):
        x = pixel % d1 + ringidx[0]
        y = pixel // d1 + ringidx[1]
        inside = (x >= 0) * (x < d1) * (y >= 0) * (y < d2)
        return x[inside] + y[inside] * d1

    b0 = np.array(Y.mean(1)) - A.dot(C.mean(1))

    if ssub > 1:
        ds_mat = CNMFE_1.decimation_matrix(dims, ssub)
        ds = lambda x: ds_mat.dot(x)
    else:
        ds = lambda x: x

    if data_fits_in_memory:
        if ssub == 1 and tsub == 1:
            X = Y - A.dot(C) - b0[:, None]
        else:
            X = decimate_last_axis(ds(Y), tsub) - \
                (ds(A).dot(decimate_last_axis(C, tsub)) if A.size > 0 else 0) - \
                ds(b0).reshape((-1, 1), order='F')

        def process_pixel(p):
            index = get_indices_of_pixels_on_ring(p)
            B = X[index]
            tmp = np.array(B.dot(B.T))
            tmp[np.diag_indices(len(tmp))] += np.trace(tmp) * 1e-5
            tmp2 = X[p]
            data = pd_solve(tmp, B.dot(tmp2))
            return index, data
    else:

        def process_pixel(p):
            index = get_indices_of_pixels_on_ring(p)
            if ssub == 1 and tsub == 1:
                B = Y[index] - A[index].dot(C) - b0[index, None]
            else:
                B = decimate_last_axis(ds(Y), tsub)[index] - \
                    (ds(A)[index].dot(decimate_last_axis(C, tsub)) if A.size > 0 else 0) - \
                    ds(b0).reshape((-1, 1), order='F')[index]
            tmp = np.array(B.dot(B.T))
            tmp[np.diag_indices(len(tmp))] += np.trace(tmp) * 1e-5
            if ssub == 1 and tsub == 1:
                tmp2 = Y[p] - A[p].dot(C).ravel() - b0[p]
            else:
                tmp2 = decimate_last_axis(ds(Y), tsub)[p] - \
                    (ds(A)[p].dot(decimate_last_axis(C, tsub)) if A.size > 0 else 0) - \
                    ds(b0).reshape((-1, 1), order='F')[p]
            data = pd_solve(tmp, B.dot(tmp2))
            return index, data

    Q = list((parmap if parallel else map)(process_pixel, range(d1 * d2)))
    indices, data = np.transpose(Q)
    indptr = np.concatenate([[0], np.cumsum(list(map(len, indices)))])
    indices = np.concatenate(indices)
    data = np.concatenate(data)
    return spr.csr_matrix((data, indices, indptr), dtype='float32'), b0.astype(np.float32)

def decimate_last_axis(y, sub):
    q = y.shape[-1] // sub
    r = y.shape[-1] % sub
    Y_ds = np.zeros(y.shape[:-1] + (q + (r > 0),), dtype=y.dtype)
    Y_ds[..., :q] = y[..., :q * sub].reshape(y.shape[:-1] + (-1, sub)).mean(-1)
    if r > 0:
        Y_ds[..., -1] = y[..., -r:].mean(-1)
    return Y_ds

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def pd_solve(a, b):
    """ Fast matrix solve for positive definite matrix a"""
    L, info = dpotrf(a)
    if info == 0:
        return dpotrs(L, b)[0]
    else:
        return np.linalg.solve(a, b)

def save_dict_to_hdf5(dic:Dict, filename:str, subdir:str='/') -> None:
    ''' Save dictionary to hdf5 file
    Args:
        dic: dictionary
            input (possibly nested) dictionary
        filename: str
            file name to save the dictionary to (in hdf5 format for now)
    '''

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, subdir, dic)

def recursively_save_dict_contents_to_group(h5file:h5py.File, path:str, dic:Dict) -> None:
    '''
    Args:
        h5file: hdf5 object
            hdf5 file where to store the dictionary
        path: str
            path within the hdf5 file structure
        dic: dictionary
            dictionary to save
    '''
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")

    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")

    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)
        if key == 'g':
            if item is None:
                item = 0
            logging.info(key + ' is an object type')
            try:
                item = np.array(list(item))
            except:
                item = np.asarray(item, dtype=np.float)
        if key == 'g_tot':
            item = np.asarray(item, dtype=np.float)
        if key in ['groups', 'idx_tot', 'ind_A', 'Ab_epoch', 'coordinates',
                   'loaded_model', 'optional_outputs', 'merged_ROIs', 'tf_in',
                   'tf_out', 'empty_merged']:
            logging.info('Key {} is not saved.'.format(key))
            continue

        if isinstance(item, (list, tuple)):
            if len(item) > 0 and all(isinstance(elem, str) for elem in item):
                item = np.string_(item)
            else:
                item = np.array(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, numpy.int32, and numpy.float64 types
        if isinstance(item, (np.int64, np.int32, np.float64, str, np.float, float, np.float32,int)):
            h5file[path + key] = item
            logging.debug('Saving {}'.format(key))
            if not h5file[path + key][()] == item:
                raise ValueError('Error while saving {}.'.format(key))
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            logging.debug('Saving {}'.format(key))
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S32')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key][()], item):
                raise ValueError('Error while saving {}.'.format(key))
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif 'sparse' in str(type(item)):
            logging.info(key + ' is sparse ****')
            h5file[path + key + '/data'] = item.tocsc().data
            h5file[path + key + '/indptr'] = item.tocsc().indptr
            h5file[path + key + '/indices'] = item.tocsc().indices
            h5file[path + key + '/shape'] = item.tocsc().shape
        # other types cannot be saved and will result in an error
        elif item is None or key == 'dview':
            h5file[path + key] = 'NoneType'
        elif key in ['dims', 'medw', 'sigma_smooth_snmf', 'dxy', 'max_shifts',
                     'strides', 'overlaps', 'gSig']:
            logging.info(key + ' is a tuple ****')
            h5file[path + key] = np.array(item)
        elif type(item).__name__ in ['CNMFParams', 'Estimates']: #  parameter object
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item.__dict__)
        else:
            raise ValueError(f"Cannot save {type(item)} type for key '{key}'.")

#%% remove components online
def remove_components_online(ind_rem, gnb, Ab, use_dense, Ab_dense, AtA, CY,
                             CC, M, N, noisyC, OASISinstances, C_on, exp_comps):

    """
    Remove components indexed by ind_r (indexing starts at zero)

    Args:
        ind_rem list
            indices of components to be removed (starting from zero)
        gnb int
            number of global background components
        Ab  csc_matrix
            matrix of components + background
        use_dense bool
            use dense representation
        Ab_dense ndarray
    """

    ind_rem.sort()
    ind_rem = [ind + gnb for ind in ind_rem[::-1]]
    ind_keep = list(set(range(Ab.shape[-1])) - set(ind_rem))
    ind_keep.sort()

    if use_dense:
        Ab_dense = np.delete(Ab_dense, ind_rem, axis=1)
    else:
        Ab_dense = []
    AtA = np.delete(AtA, ind_rem, axis=0)
    AtA = np.delete(AtA, ind_rem, axis=1)
    CY = np.delete(CY, ind_rem, axis=0)
    CC = np.delete(CC, ind_rem, axis=0)
    CC = np.delete(CC, ind_rem, axis=1)
    M -= len(ind_rem)
    N -= len(ind_rem)
    exp_comps -= len(ind_rem)
    noisyC = np.delete(noisyC, ind_rem, axis=0)
    for ii in ind_rem:
        del OASISinstances[ii - gnb]

    C_on = np.delete(C_on, ind_rem, axis=0)
    Ab = csc_matrix(Ab[:, ind_keep])
    ind_A = list(
        [(Ab.indices[Ab.indptr[ii]:Ab.indptr[ii+1]]) for ii in range(gnb, M)])
    groups = list(map(list, CNMFE_3.update_order(Ab)[0]))

    return Ab, Ab_dense, CC, CY, M, N, noisyC, OASISinstances, C_on, exp_comps, ind_A, groups, AtA

def parallel_dot_product(A: np.ndarray, b, block_size: int = 5000, dview=None, transpose=False,
                         num_blocks_per_run=20) -> np.ndarray:
    # todo: todocument
    """ Chunk matrix product between matrix and column vectors

    Args:
        A: memory mapped ndarray
            pixels x time

        b: time x comps
    """

    import pickle
    pars = []
    d1, d2 = np.shape(A)
    b = pickle.dumps(b)
    logging.debug(f'parallel dot product block size: {block_size}')

    if block_size < d1:
        for idx in range(0, d1 - block_size, block_size):
            idx_to_pass = list(range(idx, idx + block_size))
            pars.append([A.filename, idx_to_pass, b, transpose])

        if (idx + block_size) < d1:
            idx_to_pass = list(range(idx + block_size, d1))
            pars.append([A.filename, idx_to_pass, b, transpose])

    else:
        idx_to_pass = list(range(d1))
        pars.append([A.filename, idx_to_pass, b, transpose])

    logging.debug('Start product')
    b = pickle.loads(b)

    if transpose:
        output = np.zeros((d2, np.shape(b)[-1]), dtype=np.float32)
    else:
        output = np.zeros((d1, np.shape(b)[-1]), dtype=np.float32)

    if dview is None:
        if transpose:
            #            b = pickle.loads(b)
            logging.debug('Transposing')
            for _, pr in enumerate(pars):
                iddx, rs = dot_place_holder(pr)
                output = output + rs
        else:
            for _, pr in enumerate(pars):
                iddx, rs = dot_place_holder(pr)
                output[iddx] = rs

    else:
        for itera in range(0, len(pars), num_blocks_per_run):

            if 'multiprocessing' in str(type(dview)):
                results = dview.map_async(dot_place_holder, pars[itera:itera + num_blocks_per_run]).get(4294967)
            else:
                results = dview.map_sync(dot_place_holder, pars[itera:itera + num_blocks_per_run])

            logging.debug('Processed:' + str([itera, itera + len(results)]))

            if transpose:
                logging.debug('Transposing')

                for _, res in enumerate(results):
                    output += res[1]

            else:
                logging.debug('Filling')
                for res in results:
                    output[res[0]] = res[1]

            if 'multiprocessing' not in str(type(dview)):
                dview.clear()

    return output

def dot_place_holder(par: List) -> Tuple:
    # todo: todocument

    A_name, idx_to_pass, b_, transpose = par
    A_, _, _ = cp_motioncorrection.load_memmap(A_name)
    b_ = pickle.loads(b_).astype(np.float32)

    logging.debug((idx_to_pass[-1]))
    if 'sparse' in str(type(b_)):
        if transpose:
            #            outp = (b_.tocsr()[idx_to_pass].T.dot(
            #                A_[idx_to_pass])).T.astype(np.float32)
            outp = (b_.T.tocsc()[:, idx_to_pass].dot(A_[idx_to_pass])).T.astype(np.float32)
        else:
            outp = (b_.T.dot(A_[idx_to_pass].T)).T.astype(np.float32)
    else:
        if transpose:
            outp = A_[idx_to_pass].T.dot(b_[idx_to_pass]).astype(np.float32)
        else:
            outp = A_[idx_to_pass].dot(b_).astype(np.float32)

    del b_, A_
    return idx_to_pass, outp

def constrained_foopsi_parallel(arg_in):
    """ necessary for parallel computation of the function  constrained_foopsi

        the most likely discretized spike train underlying a fluorescence trace
    """

    Ytemp, nT, jj_, bl, c1, g, sn, argss = arg_in
    T = np.shape(Ytemp)[0]
    cc_, cb_, c1_, gn_, sn_, sp_, lam_ = constrained_foopsi(
        Ytemp, bl=bl, c1=c1, g=g, sn=sn, **argss)
    gd_ = np.max(np.real(np.roots(np.hstack((1, -gn_.T)))))
    gd_vec = gd_**list(range(T))

    C_ = cc_[:].T + cb_ + np.dot(c1_, gd_vec)
    Sp_ = sp_[:T].T
    Ytemp_ = Ytemp - C_.T

    return C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_, lam_

def constrained_foopsi(fluor, bl=None,  c1=None, g=None,  sn=None, p=None, method_deconvolution='oasis', bas_nonneg=True,
                       noise_range=[.25, .5], noise_method='logmexp', lags=5, fudge_factor=1.,
                       verbosity=False, solvers=None, optimize_g=0, s_min=None, **kwargs):
    """ Infer the most likely discretized spike train underlying a fluorescence trace

    It relies on a noise constrained deconvolution approach

    Args:
        fluor: np.ndarray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        bl: [optional] float
            Fluorescence baseline value. If no value is given, then bl is estimated
            from the data.

        c1: [optional] float
            value of calcium at time 0

        g: [optional] list,float
            Parameters of the AR process that models the fluorescence impulse response.
            Estimated from the data if no value is given

        sn: float, optional
            Standard deviation of the noise distribution.  If no value is given,
            then sn is estimated from the data.

        p: int
            order of the autoregression model

        method_deconvolution: [optional] string
            solution method for basis projection pursuit 'cvx' or 'cvxpy' or 'oasis'

        bas_nonneg: bool
            baseline strictly non-negative

        noise_range:  list of two elms
            frequency range for averaging noise PSD

        noise_method: string
            method of averaging noise PSD

        lags: int
            number of lags for estimating time constants

        fudge_factor: float
            fudge factor for reducing time constant bias

        verbosity: bool
             display optimization details

        solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers
            to be used with cvxpy, default is ['ECOS','SCS']

        optimize_g : [optional] int, only applies to method 'oasis'
            Number of large, isolated events to consider for optimizing g.
            If optimize_g=0 (default) the provided or estimated g is not further optimized.

        s_min : float, optional, only applies to method 'oasis'
            Minimal non-zero activity within each bin (minimal 'spike size').
            For negative values the threshold is abs(s_min) * sn * sqrt(1-g)
            If None (default) the standard L1 penalty is used
            If 0 the threshold is determined automatically such that RSS <= sn^2 T

    Returns:
        c: np.ndarray float
            The inferred denoised fluorescence signal at each time-bin.

        bl, c1, g, sn : As explained above

        sp: ndarray of float
            Discretized deconvolved neural activity (spikes)

        lam: float
            Regularization parameter
    Raises:
        Exception("You must specify the value of p")

        Exception('OASIS is currently only implemented for p=1 and p=2')

        Exception('Undefined Deconvolution Method')

    References:
        * Pnevmatikakis et al. 2016. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037
        * Machado et al. 2015. Cell 162(2):338-350

    \image: docs/img/deconvolution.png
    \image: docs/img/evaluationcomponent.png
    """

    if p is None:
        raise Exception("You must specify the value of p")

    if g is None or sn is None:
        # Estimate noise standard deviation and AR coefficients if they are not present
        g, sn = estimate_parameters(fluor, p=p, sn=sn, g=g, range_ff=noise_range,
                                    method=noise_method, lags=lags, fudge_factor=fudge_factor)
    lam = None
    if p == 0:
        c1 = 0
        g = np.array(0)
        bl = 0
        c = np.maximum(fluor, 0)
        sp = c.copy()

    else:  # choose a source extraction method
        if method_deconvolution == 'cvx':
            c, bl, c1, g, sn, sp = cvxopt_foopsi(
                fluor, b=bl, c1=c1, g=g, sn=sn, p=p, bas_nonneg=bas_nonneg, verbosity=verbosity)

        elif method_deconvolution == 'cvxpy':
            c, bl, c1, g, sn, sp = cvxpy_foopsi(
                fluor, g, sn, b=bl, c1=c1, bas_nonneg=bas_nonneg, solvers=solvers)

        elif method_deconvolution == 'oasis':
            from caiman.source_extraction.cnmf.oasis import constrained_oasisAR1
            penalty = 1 if s_min is None else 0
            if p == 1:
                if bl is None:
                    # Infer the most likely discretized spike train underlying an AR(1) fluorescence trace
                    # Solves the noise constrained sparse non-negative deconvolution problem
                    # min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g c_{t-1} >= 0
                    c, sp, bl, g, lam = constrained_oasisAR1(
                        fluor.astype(np.float32), g[0], sn, optimize_b=True, b_nonneg=bas_nonneg,
                        optimize_g=optimize_g, penalty=penalty, s_min=0 if s_min is None else s_min)
                else:
                    c, sp, _, g, lam = constrained_oasisAR1(
                        (fluor - bl).astype(np.float32), g[0], sn, optimize_b=False, penalty=penalty,
                        s_min=0 if s_min is None else s_min)

                c1 = c[0]

                # remove intial calcium to align with the other foopsi methods
                # it is added back in function constrained_foopsi_parallel of temporal.py
                c -= c1 * g**np.arange(len(fluor))
            elif p == 2:
                if bl is None:
                    c, sp, bl, g, lam = constrained_oasisAR2(
                        fluor.astype(np.float32), g, sn, optimize_b=True, b_nonneg=bas_nonneg,
                        optimize_g=optimize_g, penalty=penalty, s_min=s_min)
                else:
                    c, sp, _, g, lam = constrained_oasisAR2(
                        (fluor - bl).astype(np.float32), g, sn, optimize_b=False,
                        penalty=penalty, s_min=s_min)
                c1 = c[0]
                d = (g[0] + np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
                c -= c1 * d**np.arange(len(fluor))
            else:
                raise Exception(
                    'OASIS is currently only implemented for p=1 and p=2')
            g = np.ravel(g)

        else:
            raise Exception('Undefined Deconvolution Method')

    return c, bl, c1, g, sn, sp, lam

def cvxopt_foopsi(fluor, b, c1, g, sn, p, bas_nonneg, verbosity):
    """Solve the deconvolution problem using cvxopt and picos packages
    """
    try:
        from cvxopt import matrix, spmatrix, spdiag, solvers
        import picos
    except ImportError:
        raise ImportError(
            'Constrained Foopsi requires cvxopt and picos packages.')

    T = len(fluor)

    # construct deconvolution matrix  (sp = G*c)
    G = spmatrix(1., list(range(T)), list(range(T)), (T, T))

    for i in range(p):
        G = G + spmatrix(-g[i], np.arange(i + 1, T),
                         np.arange(T - i - 1), (T, T))

    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    gen_vec = G * matrix(np.ones(fluor.size))

    # Initialize variables in our problem
    prob = picos.Problem()

    # Define variables
    calcium_fit = prob.add_variable('calcium_fit', fluor.size)
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b = prob.add_variable('b', 1)
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)

        prob.add_constraint(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 = prob.add_variable('c1', 1)
        prob.add_constraint(c1 >= 0)
    else:
        flag_c1 = False

    # Add constraints
    prob.add_constraint(G * calcium_fit >= 0)
    res = abs(matrix(fluor.astype(float)) - calcium_fit - b *
              matrix(np.ones(fluor.size)) - matrix(gd_vec) * c1)
    prob.add_constraint(res < sn * np.sqrt(fluor.size))
    prob.set_objective('min', calcium_fit.T * gen_vec)

    # solve problem
    try:
        prob.solve(solver='mosek', verbose=verbosity)

    except ImportError:
        # warn('MOSEK is not installed. Spike inference may be VERY slow!')
        prob.solver_selection()
        prob.solve(verbose=verbosity)

    # if problem in infeasible due to low noise value then project onto the
    # cone of linear constraints with cvxopt
    if prob.status == 'prim_infeas_cer' or prob.status == 'dual_infeas_cer' or prob.status == 'primal infeasible':
        # warn('Original problem infeasible. Adjusting noise level and re-solving')
        # setup quadratic problem with cvxopt
        solvers.options['show_progress'] = verbosity
        ind_rows = list(range(T))
        ind_cols = list(range(T))
        vals = np.ones(T)
        if flag_b:
            ind_rows = ind_rows + list(range(T))
            ind_cols = ind_cols + [T] * T
            vals = np.concatenate((vals, np.ones(T)))
        if flag_c1:
            ind_rows = ind_rows + list(range(T))
            ind_cols = ind_cols + [T + cnt - 1] * T
            vals = np.concatenate((vals, np.squeeze(gd_vec)))
        P = spmatrix(vals, ind_rows, ind_cols, (T, T + cnt))
        H = P.T * P
        Py = P.T * matrix(fluor.astype(float))
        sol = solvers.qp(
            H, -Py, spdiag([-G, -spmatrix(1., list(range(cnt)), list(range(cnt)))]), matrix(0., (T + cnt, 1)))
        xx = sol['x']
        c = np.array(xx[:T])
        sp = np.array(G * matrix(c))
        c = np.squeeze(c)
        if flag_b:
            b = np.array(xx[T + 1]) + b_lb
        if flag_c1:
            c1 = np.array(xx[-1])
        sn = old_div(np.linalg.norm(fluor - c - c1 * gd_vec - b), np.sqrt(T))
    else:  # readout picos solution
        c = np.squeeze(calcium_fit.value)
        sp = np.squeeze(np.asarray(G * calcium_fit.value))
        if flag_b:
            b = np.squeeze(b.value)
        if flag_c1:
            c1 = np.squeeze(c1.value)

    return c, b, c1, g, sn, sp

def cvxpy_foopsi(fluor, g, sn, b=None, c1=None, bas_nonneg=True, solvers=None):
    """Solves the deconvolution problem using the cvxpy package and the ECOS/SCS library.

    Args:
        fluor: ndarray
            fluorescence trace

        g: list of doubles
            parameters of the autoregressive model, cardinality equivalent to p

        sn: double
            estimated noise level

        b: double
            baseline level. If None it is estimated.

        c1: double
            initial value of calcium. If None it is estimated.

        bas_nonneg: boolean
            should the baseline be estimated

        solvers: tuple of two strings
            primary and secondary solvers to be used. Can be choosen between ECOS, SCS, CVXOPT

    Returns:
        c: estimated calcium trace

        b: estimated baseline

        c1: esimtated initial calcium value

        g: esitmated parameters of the autoregressive model

        sn: estimated noise level

        sp: estimated spikes

    Raises:
        ImportError 'cvxpy solver requires installation of cvxpy. Not working in windows at the moment.'

        ValueError 'Problem solved suboptimally or unfeasible'
    """
    # todo: check the result and gen_vector vars
    try:
        import cvxpy as cvx

    except ImportError: # XXX Is the below still true?
        raise ImportError(
            'cvxpy solver requires installation of cvxpy. Not working in windows at the moment.')

    if solvers is None:
        solvers = ['ECOS', 'SCS']

    T = fluor.size

    # construct deconvolution matrix  (sp = G*c)
    G = scipy.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))

    for i, gi in enumerate(g):
        G = G + \
            scipy.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))

    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    gen_vec = G.dot(scipy.sparse.coo_matrix(np.ones((T, 1))))

    c = cvx.Variable(T)  # calcium at each time step
    constraints = []
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b = cvx.Variable(1)  # baseline value
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)
        constraints.append(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 = cvx.Variable(1)  # baseline value
        constraints.append(c1 >= 0)
    else:
        flag_c1 = False

    thrNoise = sn * np.sqrt(fluor.size)

    try:
        # minimize number of spikes
        objective = cvx.Minimize(cvx.norm(G * c, 1))
        constraints.append(G * c >= 0)
        constraints.append(
            cvx.norm(-c + fluor - b - gd_vec * c1, 2) <= thrNoise)  # constraints
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver=solvers[0])

        if not (prob.status == 'optimal' or prob.status == 'optimal_inaccurate'):
            raise ValueError('Problem solved suboptimally or unfeasible')

        print(('PROBLEM STATUS:' + prob.status))
        sys.stdout.flush()
    except (ValueError, cvx.SolverError):     # if solvers fail to solve the problem

        lam = old_div(sn, 500)
        constraints = constraints[:-1]
        objective = cvx.Minimize(cvx.norm(-c + fluor - b - gd_vec *
                                          c1, 2) + lam * cvx.norm(G * c, 1))
        prob = cvx.Problem(objective, constraints)

        try:  # in case scs was not installed properly
            try:
                print('TRYING AGAIN ECOS')
                sys.stdout.flush()
                result = prob.solve(solver=solvers[0])
            except:
                print((solvers[0] + ' DID NOT WORK TRYING ' + solvers[1]))
                result = prob.solve(solver=solvers[1])
        except:
            sys.stderr.write(
                '***** SCS solver failed, try installing and compiling SCS for much faster performance. '
                'Otherwise set the solvers in tempora_params to ["ECOS","CVXOPT"]')
            sys.stderr.flush()
            raise

        if not (prob.status == 'optimal' or prob.status == 'optimal_inaccurate'):
            print(('PROBLEM STATUS:' + prob.status))
            sp = fluor
            c = fluor
            b = 0
            c1 = 0
            return c, b, c1, g, sn, sp

    sp = np.squeeze(np.asarray(G * c.value))
    c = np.squeeze(np.asarray(c.value))
    if flag_b:
        b = np.squeeze(b.value)
    if flag_c1:
        c1 = np.squeeze(c1.value)

    return c, b, c1, g, sn, sp

def constrained_oasisAR2(y, g, sn, optimize_b=True, b_nonneg=True, optimize_g=0, decimate=5,
                         shift=100, window=None, tol=1e-9, max_iter=1, penalty=1, s_min=0):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min (s)_1 subject to (c-y)^2 = sn^2 T and s_t = c_t-g1 c_{t-1}-g2 c_{t-2} >= 0

    Args:
        y : array of float
            One dimensional array containing the fluorescence intensities (with baseline
            already subtracted) with one entry per time-bin.

        g : (float, float)
            Parameters of the AR(2) process that models the fluorescence impulse response.

        sn : float
            Standard deviation of the noise distribution.

        optimize_b : bool, optional, default True
            Optimize baseline if True else it is set to 0, see y.

        b_nonneg: bool, optional, default True
            Enforce strictly non-negative baseline if True.

        optimize_g : int, optional, default 0
            Number of large, isolated events to consider for optimizing g.
            No optimization if optimize_g=0.

        decimate : int, optional, default 5
            Decimation factor for estimating hyper-parameters faster on decimated data.

        shift : int, optional, default 100
            Number of frames by which to shift window from on run of NNLS to the next.

        window : int, optional, default None (200 or larger dependend on g)
            Window size.

        tol : float, optional, default 1e-9
            Tolerance parameter.

        max_iter : int, optional, default 1
            Maximal number of iterations.

        penalty : int, optional, default 1
            Sparsity penalty. 1: min (s)_1  0: min (s)_0

        s_min : float, optional, default 0
            Minimal non-zero activity within each bin (minimal 'spike size').
            For negative values the threshold is |s_min| * sn * sqrt(1-decay_constant)
            If 0 the threshold is determined automatically such that RSS <= sn^2 T

    Returns:
        c : array of float
            The inferred denoised fluorescence signal at each time-bin.

        s : array of float
            Discretized deconvolved neural activity (spikes).

        b : float
            Fluorescence baseline value.

        (g1, g2) : tuple of float
            Parameters of the AR(2) process that models the fluorescence impulse response.

        lam : float
            Sparsity penalty parameter lambda of dual problem.

    References:
        Friedrich J and Paninski L, NIPS 2016
        Friedrich J, Zhou P, and Paninski L, arXiv 2016
    """
    T = len(y)
    d = (g[0] + np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
    r = (g[0] - np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
    if window is None:
        window = int(min(T, max(200, -5 / np.log(d))))

    if not optimize_g:
        g11 = (np.exp(np.log(d) * np.arange(1, T + 1)) * np.arange(1, T + 1)) if d == r else \
            (np.exp(np.log(d) * np.arange(1, T + 1)) -
             np.exp(np.log(r) * np.arange(1, T + 1))) / (d - r)
        g12 = np.append(0, g[1] * g11[:-1])
        g11g11 = np.cumsum(g11 * g11)
        g11g12 = np.cumsum(g11 * g12)
        Sg11 = np.cumsum(g11)
        f_lam = 1 - g[0] - g[1]
    elif decimate == 0:  # need to run AR1 anyways for estimating AR coeffs
        decimate = 1
    thresh = sn * sn * T

    # get initial estimate of b and lam on downsampled data using AR1 model
    if decimate > 0:
        from caiman.source_extraction.cnmf.oasis import oasisAR1, constrained_oasisAR1
        _, s, b, aa, lam = constrained_oasisAR1(
            y[:len(y) // decimate * decimate].reshape(-1, decimate).mean(1),
            d**decimate, sn / np.sqrt(decimate),
            optimize_b=optimize_b, b_nonneg=b_nonneg, optimize_g=optimize_g)
        if optimize_g:
            from scipy.optimize import minimize
            d = aa**(1. / decimate)
            if decimate > 1:
                s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
            r = estimate_time_constant(s, 1, fudge_factor=.98)[0]
            g[0] = d + r
            g[1] = -d * r
            g11 = (np.exp(np.log(d) * np.arange(1, T + 1)) -
                   np.exp(np.log(r) * np.arange(1, T + 1))) / (d - r)
            g12 = np.append(0, g[1] * g11[:-1])
            g11g11 = np.cumsum(g11 * g11)
            g11g12 = np.cumsum(g11 * g12)
            Sg11 = np.cumsum(g11)
            f_lam = 1 - g[0] - g[1]
        elif decimate > 1:
            s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
        lam *= (1 - d**decimate) / f_lam

        # this window size seems necessary and sufficient
        possible_spikes = [x + np.arange(-2, 3)
                           for x in np.where(s > s.max() / 10.)[0]]
        ff = np.array(possible_spikes, dtype=np.int).ravel()
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        mask = np.zeros(T, dtype=bool)
        mask[ff] = True
    else:
        b = np.percentile(y, 15) if optimize_b else 0
        lam = 2 * sn * np.linalg.norm(g11)
        mask = None
    if b_nonneg:
        b = max(b, 0)

    # run ONNLS
    c, s = onnls(y - b, g, lam=lam, mask=mask,
                 shift=shift, window=window, tol=tol)

    if not optimize_b:  # don't optimize b, just the dual variable lambda
        for _ in range(max_iter - 1):
            res = y - c
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4 * thresh:
                break

            # calc shift dlam, here attributed to sparsity penalty
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * \
                np.exp(np.log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f - 1

                # if and elif correct last 2 time points for |s|_1 instead |c|_1
                if i == len(ls) - 2:  # last pool
                    tmp[f] = (1. / f_lam if l == 0 else
                              (Sg11[l] + g[1] / f_lam * g11[l - 1]
                               + (g[0] + g[1]) / f_lam * g11[l]
                               - g11g12[l] * tmp[f - 1]) / g11g11[l])
                # secondlast pool if last one has length 1
                elif i == len(ls) - 3 and ls[-2] == T - 1:
                    tmp[f] = (Sg11[l] + g[1] / f_lam * g11[l]
                              - g11g12[l] * tmp[f - 1]) / g11g11[l]
                else:  # all other pools
                    tmp[f] = (Sg11[l] - g11g12[l] * tmp[f - 1]) / g11g11[l]
                l += 1
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]

            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + np.sqrt(bb * bb - aa * cc)) / aa
            except:
                db = -bb / aa

            # perform shift
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask,
                         shift=shift, window=window, tol=tol)
            db = np.mean(y - c) - b
            b += db
            lam -= db / f_lam

    else:  # optimize b
        db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g[0] - g[1])
        g_converged = False
        for _ in range(max_iter - 1):
            res = y - c - b
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4 * thresh:
                break
            # calc shift db, here attributed to baseline
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * \
                np.exp(np.log(d) * np.arange(l))  # first pool
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f
                tmp[f] = (Sg11[l - 1] - g11g12[l - 1]
                          * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            tmp -= tmp.mean()
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + np.sqrt(bb * bb - aa * cc)) / aa
            except:
                db = -bb / aa

            # perform shift
            if b_nonneg:
                db = max(db, -b)
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask,
                         shift=shift, window=window, tol=tol)

            # update b and lam
            db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            lam -= db / f_lam

            # update g and b
            if optimize_g and (not g_converged):

                def getRSS(y, opt):
                    b, ld, lr = opt
                    if ld < lr:
                        return 1e3 * thresh
                    d, r = np.exp(ld), np.exp(lr)
                    g1, g2 = d + r, -d * r
                    tmp = b + onnls(y - b, [g1, g2], lam,
                                    mask=(s > 1e-2 * s.max()))[0] - y
                    return tmp.dot(tmp)

                result = minimize(lambda x: getRSS(y, x), (b, np.log(d), np.log(r)),
                                  bounds=((0 if b_nonneg else None, None),
                                          (None, -1e-4), (None, -1e-3)), method='L-BFGS-B',
                                  options={'gtol': 1e-04, 'maxiter': 10, 'ftol': 1e-05})
                if abs(result['x'][1] - np.log(d)) < 1e-3:
                    g_converged = True
                b, ld, lr = result['x']
                d, r = np.exp(ld), np.exp(lr)
                g = (d + r, -d * r)
                c, s = onnls(y - b, g, lam=lam, mask=mask,
                             shift=shift, window=window, tol=tol)

                # update b and lam
                db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                lam -= db

    if penalty == 0:  # get (locally optimal) L0 solution
        def c4smin(y, s, s_min):
            ls = np.append(np.where(s > s_min)[0], T)
            tmp = np.zeros_like(s)
            l = ls[0]  # first pool
            tmp[:l] = max(0, np.exp(np.log(d) * np.arange(l)).dot(y[:l]) * (1 - d * d)
                          / (1 - d**(2 * l))) * np.exp(np.log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):  # all other pools
                l = ls[i + 1] - f
                tmp[f] = (g11[:l].dot(y[f:f + l]) - g11g12[l - 1]
                          * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            return tmp

        if s_min == 0:
            spikesizes = np.sort(s[s > 1e-6])
            i = len(spikesizes) // 2
            l = 0
            u = len(spikesizes) - 1
            while u - l > 1:
                s_min = spikesizes[i]
                tmp = c4smin(y - b, s, s_min)
                res = y - b - tmp
                RSS = res.dot(res)
                if RSS < thresh or i == 0:
                    l = i
                    i = (l + u) // 2
                    res0 = tmp
                else:
                    u = i
                    i = (l + u) // 2
            if i > 0:
                c = res0
                s = np.append([0, 0], c[2:] - g[0] * c[1:-1] - g[1] * c[:-2])
        else:
            if s_min < 0:
                s_min = -s_min * sn * np.sqrt(1 - d)
            for factor in (.7, .8, .9, 1):
                c = c4smin(y - b, s, factor * s_min)
                s = np.append([0, 0], c[2:] - g[0] * c[1:-1] - g[1] * c[:-2])
        s[s < np.finfo(np.float32).eps] = 0

    return c, s, b, g, lam

def onnls(y, g, lam=0, shift=100, window=None, mask=None, tol=1e-9, max_iter=None):
    """ Infer the most likely discretized spike train underlying an AR(2) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    ``argmin_s 1/2|Ks-y|^2 + lam |s|_1`` for ``s>=0``

    Args:
        y : array of float, shape (T,)
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        g : array, shape (p,)
            if p in (1,2):
                Parameter(s) of the AR(p) process that models the fluorescence impulse response.
            else:
                Kernel that models the fluorescence impulse response.

        lam : float, optional, default 0
            Sparsity penalty parameter lambda.

        shift : int, optional, default 100
            Number of frames by which to shift window from on run of NNLS to the next.

        window : int, optional, default None (200 or larger dependend on g)
            Window size.

        mask : array of bool, shape (n,), optional, default (True,)*n
            Mask to restrict potential spike times considered.

        tol : float, optional, default 1e-9
            Tolerance parameter.

        max_iter : None or int, optional, default None
            Maximum number of iterations before termination.
            If None (default), it is set to window size.

    Returns:
        c : array of float, shape (T,)
            The inferred denoised fluorescence signal at each time-bin.

        s : array of float, shape (T,)
            Discretized deconvolved neural activity (spikes).

    References:
        Friedrich J and Paninski L, NIPS 2016
        Bro R and DeJong S, J Chemometrics 1997
    """

    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)
    if window is None:
        w = max(200, len(g) if len(g) > 2 else
                int(-5 / np.log(g[0] if len(g) == 1 else
                             (g[0] + np.sqrt(g[0] * g[0] + 4 * g[1])) / 2)))
    else:
        w = window
    w = min(T, w)
    shift = min(w, shift)
    K = np.zeros((w, w))

    if len(g) == 1:  # kernel for AR(1)
        _y = y - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        h = np.exp(np.log(g[0]) * np.arange(w))
        for i in range(w):
            K[i:, i] = h[:w - i]

    elif len(g) == 2:  # kernel for AR(2)
        _y = y - lam * (1 - g[0] - g[1])
        _y[-2] = y[-2] - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        d = (g[0] + np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
        r = (g[0] - np.sqrt(g[0] * g[0] + 4 * g[1])) / 2
        if d == r:
            h = np.exp(np.log(d) * np.arange(1, w + 1)) * np.arange(1, w + 1)
        else:
            h = (np.exp(np.log(d) * np.arange(1, w + 1)) -
                 np.exp(np.log(r) * np.arange(1, w + 1))) / (d - r)
        for i in range(w):
            K[i:, i] = h[:w - i]

    else:  # arbitrary kernel
        h = g
        for i in range(w):
            K[i:, i] = h[:w - i]
        a = np.linalg.inv(K).sum(0)
        _y = y - lam * a[0]
        _y[-w:] = y[-w:] - lam * a

    s = np.zeros(T)
    KK = K.T.dot(K)
    for i in range(0, max(1, T - w), shift):
        s[i:i + w] = _nnls(KK, K.T.dot(_y[i:i + w]), s[i:i + w], mask=mask[i:i + w],
                           tol=tol, max_iter=max_iter)[:w]

        # subtract contribution of spikes already committed to
        _y[i:i + w] -= K[:, :shift].dot(s[i:i + shift])
    s[i + shift:] = _nnls(KK[-(T - i - shift):, -(T - i - shift):],
                          K[:T - i - shift, :T - i -
                              shift].T.dot(_y[i + shift:]),
                          s[i + shift:], mask=mask[i + shift:])
    c = np.zeros_like(s)
    for t in np.where(s > tol)[0]:
        c[t:t + w] += s[t] * h[:min(w, T - t)]
    return c, s

def _nnls(KK, Ky, s=None, mask=None, tol=1e-9, max_iter=None):
    """
    Solve non-negative least squares problem
    ``argmin_s || Ks - y ||_2`` for ``s>=0``

    Args:
        KK : array, shape (n, n)
            Dot-product of design matrix K transposed and K, K'K

        Ky : array, shape (n,)
            Dot-product of design matrix K transposed and target vector y, K'y

        s : None or array, shape (n,), optional, default None
            Initialization of deconvolved neural activity.

        mask : array of bool, shape (n,), optional, default (True,)*n
            Mask to restrict potential spike times considered.

        tol : float, optional, default 1e-9
            Tolerance parameter.

        max_iter : None or int, optional, default None
            Maximum number of iterations before termination.
            If None (default), it is set to len(KK).

    Returns:
        s : array, shape (n,)
            Discretized deconvolved neural activity (spikes)

    References:
        Lawson C and Hanson RJ, SIAM 1987
        Bro R and DeJong S, J Chemometrics 1997
    """

    if mask is None:
        mask = np.ones(len(KK), dtype=bool)
    else:
        KK = KK[mask][:, mask]
        Ky = Ky[mask]
    if s is None:
        s = np.zeros(len(KK))
        l = Ky.copy()
        P = np.zeros(len(KK), dtype=bool)
    else:
        s = s[mask]
        P = s > 0
        l = Ky - KK[:, P].dot(s[P])
    i = 0
    if max_iter is None:
        max_iter = len(KK)
    for i in range(max_iter):  # max(l) is checked at the end, should do at least one iteration
        w = np.argmax(l)
        P[w] = True

        try:  # likely unnnecessary try-except-clause for robustness sake
            #mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
            mu = np.linalg.solve(KK[P][:, P], Ky[P])
        except:
            #mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
            mu = np.linalg.solve(KK[P][:, P] + tol * np.eye(P.sum()), Ky[P])
            print(r'added $\epsilon$I to avoid singularity')
        while len(mu > 0) and min(mu) < 0:
            a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
            s[P] += a * (mu - s[P])
            P[s <= tol] = False
            try:
                #mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
                mu = np.linalg.solve(KK[P][:, P], Ky[P])
            except:
                #mu = np.linalg.inv(KK[P][:, P] + tol *
                #                   np.eye(P.sum())).dot(Ky[P])
                mu = np.linalg.solve(KK[P][:, P] + tol * np.eye(P.sum()), Ky[P])
                print(r'added $\epsilon$I to avoid singularity')
        s[P] = mu.copy()
        l = Ky - KK[:, P].dot(s[P])
        if max(l) < tol:
            break
    tmp = np.zeros(len(mask))
    tmp[mask] = s
    return tmp

def estimate_parameters(fluor, p=2, sn=None, g=None, range_ff=[0.25, 0.5],
                        method='logmexp', lags=5, fudge_factor=1.):
    """
    Estimate noise standard deviation and AR coefficients if they are not present

    Args:
        p: positive integer
            order of AR system

        sn: float
            noise standard deviation, estimated if not provided.

        lags: positive integer
            number of additional lags where he autocovariance is computed

        range_ff : (1,2) array, nonnegative, max value <= 0.5
            range of frequency (x Nyquist rate) over which the spectrum is averaged

        method: string
            method of averaging: Mean, median, exponentiated mean of logvalues (default)

        fudge_factor: float (0< fudge_factor <= 1)
            shrinkage factor to reduce bias
    """

    if sn is None:
        sn = GetSn(fluor, range_ff, method)

    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor, p, sn, lags, fudge_factor)

    return g, sn


def estimate_parameters(fluor, p=2, sn=None, g=None, range_ff=[0.25, 0.5],
                        method='logmexp', lags=5, fudge_factor=1.):
    """
    Estimate noise standard deviation and AR coefficients if they are not present

    Args:
        p: positive integer
            order of AR system

        sn: float
            noise standard deviation, estimated if not provided.

        lags: positive integer
            number of additional lags where he autocovariance is computed

        range_ff : (1,2) array, nonnegative, max value <= 0.5
            range of frequency (x Nyquist rate) over which the spectrum is averaged

        method: string
            method of averaging: Mean, median, exponentiated mean of logvalues (default)

        fudge_factor: float (0< fudge_factor <= 1)
            shrinkage factor to reduce bias
    """

    if sn is None:
        sn = GetSn(fluor, range_ff, method)

    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor, p, sn, lags, fudge_factor)

    return g, sn

def estimate_time_constant(fluor, p=2, sn=None, lags=5, fudge_factor=1.):
    """
    Estimate AR model parameters through the autocovariance function

    Args:
        fluor        : nparray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        p            : positive integer
            order of AR system

        sn           : float
            noise standard deviation, estimated if not provided.

        lags         : positive integer
            number of additional lags where he autocovariance is computed

        fudge_factor : float (0< fudge_factor <= 1)
            shrinkage factor to reduce bias

    Returns:
        g       : estimated coefficients of the AR process
    """

    if sn is None:
        sn = GetSn(fluor)

    lags += p
    xc = CNMFE_1.axcov(fluor, lags)
    xc = xc[:, np.newaxis]

    A = scipy.linalg.toeplitz(xc[lags + np.arange(lags)],
                              xc[lags + np.arange(p)]) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[lags + 1:], rcond=None)[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = old_div((gr + gr.conjugate()), 2.)
    np.random.seed(45) # We want some variability below, but it doesn't have to be random at
                       # runtime. A static seed captures our intent, while still not disrupting
                       # the desired identical results from runs.
    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()

def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    """
    Estimate noise power through the power spectral density over the range of large frequencies

    Args:
        fluor    : nparray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        range_ff : (1,2) array, nonnegative, max value <= 0.5
            range of frequency (x Nyquist rate) over which the spectrum is averaged

        method   : string
            method of averaging: Mean, median, exponentiated mean of logvalues (default)

    Returns:
        sn       : noise standard deviation
    """

    ff, Pxx = scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(old_div(Pxx_ind, 2))),
        'median': lambda Pxx_ind: np.sqrt(np.median(old_div(Pxx_ind, 2))),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(old_div(Pxx_ind, 2)))))
    }[method](Pxx_ind)

    return sn

# definitions for demixed time series extraction and denoising/deconvolving
@profile
def HALS4activity(Yr, A, noisyC, AtA=None, iters=5, tol=1e-3, groups=None,
                  order=None):
    """Solves C = argmin_C ||Yr-AC|| using block-coordinate decent. Can use
    groups to update non-overlapping components in parallel or a specified
    order.

    Args:
        Yr : np.array (possibly memory mapped, (x,y,[,z]) x t)
            Imaging data reshaped in matrix format

        A : scipy.sparse.csc_matrix (or np.array) (x,y,[,z]) x # of components)
            Spatial components and background

        noisyC : np.array  (# of components x t)
            Temporal traces (including residuals plus background)

        AtA : np.array, optional (# of components x # of components)
            A.T.dot(A) Overlap matrix of shapes A.

        iters : int, optional
            Maximum number of iterations.

        tol : float, optional
            Change tolerance level

        groups : list of sets
            grouped components to be updated simultaneously

        order : list
            Update components in that order (used if nonempty and groups=None)

    Returns:
        C : np.array (# of components x t)
            solution of HALS

        noisyC : np.array (# of components x t)
            solution of HALS + residuals, i.e, (C + YrA)
    """

    AtY = A.T.dot(Yr)
    num_iters = 0
    C_old = np.zeros_like(noisyC)
    C = noisyC.copy()
    if AtA is None:
        AtA = A.T.dot(A)
    AtAd = AtA.diagonal() + np.finfo(np.float32).eps

    # faster than np.linalg.norm
    def norm(c): return np.sqrt(c.ravel().dot(c.ravel()))
    while (norm(C_old - C) >= tol * norm(C_old)) and (num_iters < iters):
        C_old[:] = C
        if groups is None:
            if order is None:
                order = list(range(AtY.shape[0]))
            for m in order:
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / AtAd[m]
                C[m] = np.maximum(noisyC[m], 0)
        else:
            for m in groups:
                noisyC[m] = C[m] + ((AtY[m] - AtA[m].dot(C)).T/AtAd[m]).T
                C[m] = np.maximum(noisyC[m], 0)
        num_iters += 1
    return C, noisyC

