
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
# import matlotlib.pyplot as plt

from typing import Any, List, Tuple, Union, Dict, Set, Optional
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
from scipy.sparse import csgraph, lil_matrix
from scipy.linalg import eig
from scipy.ndimage.filters import correlate
from sklearn.decomposition import NMF
from sklearn.utils.extmath import randomized_svd
from scipy.ndimage import convolve
import scipy.ndimage as nd
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation
from scipy.stats import norm

import cp_cnmfe as CNMFE_1
import cp_cnmfe_2 as CNMFE_2

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

def update_order(A, new_a=None, prev_list=None, method='greedy'):
    '''Determines the update order of the temporal components given the spatial
    components by creating a nest of random approximate vertex covers

     Args:
         A:    np.ndarray
              matrix of spatial components (d x K)
         new_a: sparse array
              spatial component that is added, in order to efficiently update the orders in online scenarios
         prev_list: list of list
              orders from previous iteration, you need to pass if new_a is not None

     Returns:
         O:  list of sets
             list of subsets of components. The components of each subset can be updated in parallel
         lo: list
             length of each subset

    Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''
    K = np.shape(A)[-1]
    if new_a is None and prev_list is None:

        if method == 'greedy':
            prev_list, count_list = update_order_greedy(A, flag_AA=False)
        else:
            prev_list, count_list = update_order_random(A, flag_AA=False)
        return prev_list, count_list

    else:

        if new_a is None or prev_list is None:
            raise Exception(
                'In the online update order you need to provide both new_a and prev_list')

        counter = 0

        AA = A.T.dot(new_a)
        for group in prev_list:
            if AA[list(group)].sum() == 0:
                group.append(K)
                counter += 1
                break

        if counter == 0:
            if prev_list is not None:
                prev_list = list(prev_list)
                prev_list.append([K])

        count_list = [len(gr) for gr in prev_list]

        return prev_list, count_list

def update_order_random(A, flag_AA=True):
    """Determies the update order of temporal components using
    randomized partitions of non-overlapping components
    """

    K = np.shape(A)[-1]
    if flag_AA:
        AA = A.copy()
    else:
        AA = A.T.dot(A)

    AA.setdiag(0)
    F = (AA) > 0
    F = F.toarray()
    rem_ind = np.arange(K)
    O = []
    lo = []
    while len(rem_ind) > 0:
        L = np.sort(app_vertex_cover(F[rem_ind, :][:, rem_ind]))
        if L.size:
            ord_ind = set(rem_ind) - set(rem_ind[L])
            rem_ind = rem_ind[L]
        else:
            ord_ind = set(rem_ind)
            rem_ind = []

        O.append(ord_ind)
        lo.append(len(ord_ind))

    return O[::-1], lo[::-1]

def update_order_greedy(A, flag_AA=True):
    """Determines the update order of the temporal components

    this, given the spatial components using a greedy method
    Basically we can update the components that are not overlapping, in parallel

    Args:
        A:  sparse crc matrix
            matrix of spatial components (d x K)
        OR:
            A.T.dot(A) matrix (d x d) if flag_AA = true

        flag_AA: boolean (default true)

     Returns:
         parllcomp:   list of sets
             list of subsets of components. The components of each subset can be updated in parallel

         len_parrllcomp:  list
             length of each subset

    Author:
        Eftychios A. Pnevmatikakis, Simons Foundation, 2017
    """
    K = np.shape(A)[-1]
    parllcomp:List = []
    for i in range(K):
        new_list = True
        for ls in parllcomp:
            if flag_AA:
                if A[i, ls].nnz == 0:
                    ls.append(i)
                    new_list = False
                    break
            else:
                if (A[:, i].T.dot(A[:, ls])).nnz == 0:
                    ls.append(i)
                    new_list = False
                    break

        if new_list:
            parllcomp.append([i])
    len_parrllcomp = [len(ls) for ls in parllcomp]
    return parllcomp, len_parrllcomp

def app_vertex_cover(A):
    """ Finds an approximate vertex cover for a symmetric graph with adjacency matrix A.

    Args:
        A:  boolean 2d array (K x K)
            Adjacency matrix. A is boolean with diagonal set to 0

    Returns:
        L:   A vertex cover of A

    Authors:
    Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    """

    L = []
    while A.any():
        nz = np.nonzero(A)[0]          # find non-zero edges
        u = nz[np.random.randint(0, len(nz))]
        A[u, :] = False
        A[:, u] = False
        L.append(u)

    return np.asarray(L)

def HALS4shapes(Yr, A, C, iters=2):
    K = A.shape[-1]
    ind_A = A > 0
    U = C.dot(Yr.T)
    V = C.dot(C.T)
    V_diag = V.diagonal() + np.finfo(float).eps
    for _ in range(iters):
        for m in range(K):  # neurons
            ind_pixels = np.squeeze(ind_A[:, m])
            A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                       ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                        V_diag[m]), 0, np.inf)

    return A

def update_temporal_components(Y, A, b, Cin, fin, bl=None, c1=None, g=None, sn=None, nb=1, ITER=2, block_size_temp=5000, num_blocks_per_run_temp=20, debug=False, dview=None, **kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.

    Args:
        Y: np.ndarray (2D)
            input data with time in the last axis (d x T)

        A: sparse matrix (crc format)
            matrix of temporal components (d x K)

        b: ndarray (dx1)
            current estimate of background component

        Cin: np.ndarray
            current estimate of temporal components (K x T)

        fin: np.ndarray
            current estimate of temporal background (vector of length T)

        g:  np.ndarray
            Global time constant (not used)

        bl: np.ndarray
           baseline for fluorescence trace for each column in A

        c1: np.ndarray
           initial concentration for each column in A

        g:  np.ndarray
           discrete time constant for each column in A

        sn: np.ndarray
           noise level for each column in A

        nb: [optional] int
            Number of background components

        ITER: positive integer
            Maximum number of block coordinate descent loops.

        method_foopsi: string
            Method of deconvolution of neural activity. constrained_foopsi is the only method supported at the moment.

        n_processes: int
            number of processes to use for parallel computation.
             Should be less than the number of processes started with ipcluster.

        backend: 'str'
            single_thread no parallelization
            ipyparallel, parallelization using the ipyparallel cluster.
            You should start the cluster (install ipyparallel and then type
            ipcluster -n 6, where 6 is the number of processes).
            SLURM: using SLURM scheduler

        memory_efficient: Bool
            whether or not to optimize for memory usage (longer running times). necessary with very large datasets

        kwargs: dict
            all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation).
             Some useful parameters are

        p: int
            order of the autoregression model

        method: [optional] string
            solution method for constrained foopsi. Choices are
                'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
                'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)

        solvers: list string
                primary and secondary (if problem unfeasible for approx solution)
                 solvers to be used with cvxpy, default is ['ECOS','SCS']

    Note:
        The temporal components are updated in parallel by default by forming of sequence of vertex covers.

    Returns:
        C:   np.ndarray
                matrix of temporal components (K x T)

        A:   np.ndarray
                updated A

        b:   np.array
                updated estimate

        f:   np.array
                vector of temporal background (length T)

        S:   np.ndarray
                matrix of merged deconvolved activity (spikes) (K x T)

        bl:  float
                same as input

        c1:  float
                same as input

        sn:  float
                same as input

        g:   float
                same as input

        YrA: np.ndarray
                matrix of spatial component filtered raw data, after all contributions have been removed.
                YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)

        lam: np.ndarray
            Automatically tuned sparsity parameter
    """

    if 'p' not in kwargs or kwargs['p'] is None:
        raise Exception("You have to provide a value for p")

    # INITIALIZATION OF VARS
    d, T = np.shape(Y)
    nr = np.shape(A)[-1]
    if b is not None:
        # if b.shape[0] < b.shape[1]:
        #     b = b.T
        nb = b.shape[1]
    if bl is None:
        bl = np.repeat(None, nr)

    if c1 is None:
        c1 = np.repeat(None, nr)

    if g is None:
        g = np.repeat(None, nr)

    if sn is None:
        sn = np.repeat(None, nr)

    A = scipy.sparse.hstack((A, b)).tocsc()
    S = np.zeros(np.shape(Cin))
    Cin = np.vstack((Cin, fin))
    C = Cin.copy()
    nA = np.ravel(A.power(2).sum(axis=0)) + np.finfo(np.float32).eps

    logging.info('Generating residuals')
#    dview_res = None if block_size >= 500 else dview
    if 'memmap' in str(type(Y)):
        bl_siz1 = d // (np.maximum(num_blocks_per_run_temp - 1, 1))
        bl_siz2 = int(psutil.virtual_memory().available/(num_blocks_per_run_temp + 1) - 4*A.nnz) // int(4*T)
        # block_size_temp
        YA = CNMFE_2.parallel_dot_product(Y, A.tocsr(), dview=dview, block_size=min(bl_siz1, bl_siz2),
                                  transpose=True, num_blocks_per_run=num_blocks_per_run_temp) * diags(1. / nA);
    else:
        YA = (A.T.dot(Y).T) * diags(1. / nA)
    AA = ((A.T.dot(A)) * diags(1. / nA)).tocsr()
    YrA = YA - AA.T.dot(Cin).T
    # creating the patch of components to be computed in parallel
    parrllcomp, len_parrllcomp = update_order_greedy(AA[:nr, :][:, :nr])
    logging.info("entering the deconvolution ")
    C, S, bl, YrA, c1, sn, g, lam = update_iteration(parrllcomp, len_parrllcomp, nb, C, S, bl, nr,
                                                     ITER, YrA, c1, sn, g, Cin, T, nA, dview, debug, AA, kwargs)
    ff = np.where(np.sum(C, axis=1) == 0)  # remove empty components
    if np.size(ff) > 0:  # Eliminating empty temporal components
        ff = ff[0]
        logging.info('removing {0} empty spatial component(s)'.format(len(ff)))
        keep = list(range(A.shape[1]))
        for i in ff:
            keep.remove(i)

        A = A[:, keep]
        C = np.delete(C, list(ff), 0)
        YrA = np.delete(YrA, list(ff), 1)
        S = np.delete(S, list(ff), 0)
        sn = np.delete(sn, list(ff))
        g = np.delete(g, list(ff))
        bl = np.delete(bl, list(ff))
        c1 = np.delete(c1, list(ff))
        lam = np.delete(lam, list(ff))

        background_ff = list(filter(lambda i: i > 0, ff - nr))
        nr = nr - (len(ff) - len(background_ff))

    b = A[:, nr:].toarray()
    A = csc_matrix(A[:, :nr])
    f = C[nr:, :]
    C = C[:nr, :]
    YrA = np.array(YrA[:, :nr]).T
    return C, A, b, f, S, bl, c1, sn, g, YrA, lam


def update_iteration(parrllcomp, len_parrllcomp, nb, C, S, bl, nr,
                     ITER, YrA, c1, sn, g, Cin, T, nA, dview, debug, AA, kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.

    Args:
        YrA: np.ndarray (2D)
            input data with time in the last axis (d x T)

        AA: sparse matrix (crc format)
            matrix of temporal components (d x K)

        Cin: np.ndarray
            current estimate of temporal components (K x T)

        g:  np.ndarray
            Global time constant (not used)

        bl: np.ndarray
           baseline for fluorescence trace for each column in A

        c1: np.ndarray
           initial concentration for each column in A

        g:  np.ndarray
           discrete time constant for each column in A

        sn: np.ndarray
           noise level for each column in A

        nb: [optional] int
            Number of background components

        ITER: positive integer
            Maximum number of block coordinate descent loops.

        backend: 'str'
            single_thread no parallelization
            ipyparallel, parallelization using the ipyparallel cluster.
            You should start the cluster (install ipyparallel and then type
            ipcluster -n 6, where 6 is the number of processes).
            SLURM: using SLURM scheduler

        memory_efficient: Bool
            whether or not to optimize for memory usage (longer running times). necessary with very large datasets

        **kwargs: dict
            all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation).
             Some useful parameters are

        p: int
            order of the autoregression model

        method: [optional] string
            solution method for constrained foopsi. Choices are
                'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
                'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)

        solvers: list string
            primary and secondary (if problem unfeasible for approx solution)
            solvers to be used with cvxpy, default is ['ECOS','SCS']

    Note:
        The temporal components are updated in parallel by default by forming of sequence of vertex covers.

    Returns:
        C:   np.ndarray
            matrix of temporal components (K x T)

        S:   np.ndarray
            matrix of merged deconvolved activity (spikes) (K x T)

        bl:  float
            same as input

        c1:  float
            same as input

        g:   float
            same as input

        sn:  float
            same as input

        YrA: np.ndarray
            matrix of spatial component filtered raw data, after all contributions have been removed.
            YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)
"""

    lam = np.repeat(None, nr)
    for _ in range(ITER):

        for count, jo_ in enumerate(parrllcomp):
            # INITIALIZE THE PARAMS
            jo = np.array(list(jo_))
            Ytemp = YrA[:, jo.flatten()] + Cin[jo, :].T
            Ctemp = np.zeros((np.size(jo), T))
            Stemp = np.zeros((np.size(jo), T))
            nT = nA[jo]
            args_in = [(np.squeeze(np.array(Ytemp[:, jj])), nT[jj], jj, None,
                        None, None, None, kwargs) for jj in range(len(jo))]
            # computing the most likely discretized spike train underlying a fluorescence trace
            if 'multiprocessing' in str(type(dview)):
                results = dview.map_async(
                    CNMFE_2.constrained_foopsi_parallel, args_in).get(4294967)

            elif dview is not None and platform.system() != 'Darwin':
                if debug:
                    results = dview.map_async(
                        CNMFE_2.constrained_foopsi_parallel, args_in)
                    results.get()
                    for outp in results.stdout:
                        print((outp[:-1]))
                        sys.stdout.flush()
                    for outp in results.stderr:
                        print((outp[:-1]))
                        sys.stderr.flush()
                else:
                    results = dview.map_sync(
                        CNMFE_2.constrained_foopsi_parallel, args_in)

            else:
                results = list(map(CNMFE_2.constrained_foopsi_parallel, args_in))
            # unparsing and updating the result
            for chunk in results:
                C_, Sp_, Ytemp_, cb_, c1_, sn_, gn_, jj_, lam_ = chunk
                Ctemp[jj_, :] = C_[None, :]
                Stemp[jj_, :] = Sp_[None, :]
                bl[jo[jj_]] = cb_
                c1[jo[jj_]] = c1_
                sn[jo[jj_]] = sn_
                g[jo[jj_]] = gn_.T if kwargs['p'] > 0 else []
                lam[jo[jj_]] = lam_

            YrA -= AA[jo, :].T.dot(Ctemp - C[jo, :]).T
            C[jo, :] = Ctemp.copy()
            S[jo, :] = Stemp
            logging.info("{0} ".format(np.sum(len_parrllcomp[:count + 1])) +
                         "out of total {0} temporal components ".format(nr) +
                         "updated")

        for ii in np.arange(nr, nr + nb):
            cc = np.maximum(YrA[:, ii] + Cin[ii], -np.Inf)
            YrA -= AA[ii, :].T.dot((cc - Cin[ii])[None, :]).T
            C[ii, :] = cc

        if dview is not None and not('multiprocessing' in str(type(dview))):
            dview.results.clear()

        try:
            if scipy.linalg.norm(Cin - C, 'fro') <= 1e-3*scipy.linalg.norm(C, 'fro'):
                logging.info("stopping: overall temporal component not changing" +
                             " significantly")
                break
            else:  # we keep Cin and do the iteration once more
                Cin = C.copy()
        except ValueError:
            logging.warning("Aborting updating of temporal components due" +
                            " to possible numerical issues.")
            C = Cin.copy()
            break

    return C, S, bl, YrA, c1, sn, g, lam


def update_spatial_components(Y, C=None, f=None, A_in=None, sn=None, dims=None,
                              min_size=3, max_size=8, dist=3,
                              normalize_yyt_one=True, method_exp='dilate',
                              expandCore=None, dview=None, n_pixels_per_process=128,
                              medw=(3, 3), thr_method='max', maxthr=0.1,
                              nrgthr=0.9999, extract_cc=True, b_in=None,
                              se=np.ones((3, 3), dtype=np.int),
                              ss=np.ones((3, 3), dtype=np.int), nb=1,
                              method_ls='lasso_lars', update_background_components=True,
                              low_rank_background=True, block_size_spat=1000,
                              num_blocks_per_run_spat=20):
    """update spatial footprints and background through Basis Pursuit Denoising

    for each pixel i solve the problem
        [A(i,:),b(i)] = argmin sum(A(i,:))
    subject to
        || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

    for each pixel the search is limited to a few spatial components

    Args:
        Y: np.ndarray (2D or 3D)
            movie, raw data in 2D or 3D (pixels x time).

        C: np.ndarray
            calcium activity of each neuron.

        f: np.ndarray
            temporal profile  of background activity.

        A_in: np.ndarray
            spatial profile of background activity. If A_in is boolean then it defines the spatial support of A.
            Otherwise it is used to determine it through determine_search_location

        b_in: np.ndarray
            you can pass background as input, especially in the case of one background per patch, since it will update using hals

        dims: [optional] tuple
            x, y[, z] movie dimensions

        min_size: [optional] int

        max_size: [optional] int

        dist: [optional] int

        sn: [optional] float
            noise associated with each pixel if known

        backend [optional] str
            'ipyparallel', 'single_thread'
            single_thread:no parallelization. It can be used with small datasets.
            ipyparallel: uses ipython clusters and then send jobs to each of them
            SLURM: use the slurm scheduler

        n_pixels_per_process: [optional] int
            number of pixels to be processed by each thread

        method: [optional] string
            method used to expand the search for pixels 'ellipse' or 'dilate'

        expandCore: [optional]  scipy.ndimage.morphology
            if method is dilate this represents the kernel used for expansion

        dview: view on ipyparallel client
                you need to create an ipyparallel client and pass a view on the processors (client = Client(), dview=client[:])

        medw, thr_method, maxthr, nrgthr, extract_cc, se, ss: [optional]
            Parameters for components post-processing. Refer to spatial.threshold_components for more details

        nb: [optional] int
            Number of background components

        method_ls:
            method to perform the regression for the basis pursuit denoising.
                 'nnls_L0'. Nonnegative least square with L0 penalty
                 'lasso_lars' lasso lars function from scikit learn

            normalize_yyt_one: bool
                whether to normalize the C and A matrices so that diag(C*C.T) are ones

        update_background_components:bool
            whether to update the background components in the spatial phase

        low_rank_background:bool
            whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
            (to be used with one background per patch)


    Returns:
        A: np.ndarray
             new estimate of spatial footprints

        b: np.ndarray
            new estimate of spatial background

        C: np.ndarray
             temporal components (updated only when spatial components are completely removed)

        f: np.ndarray
            same as f_in except if empty component deleted.

    Raises:
        Exception 'You need to define the input dimensions'

        Exception 'Dimension of Matrix Y must be pixels x time'

        Exception 'Dimension of Matrix C must be neurons x time'

        Exception 'Dimension of Matrix f must be background comps x time '

        Exception 'Either A or C need to be determined'

        Exception 'Dimension of Matrix A must be pixels x neurons'

        Exception 'You need to provide estimate of C and f'

        Exception 'Not implemented consistently'

        Exception "Failed to delete: " + folder
    """
    #logging.info('Initializing update of Spatial Components')

    if expandCore is None:
        expandCore = iterate_structure(
            generate_binary_structure(2, 1), 2).astype(int)

    if dims is None:
        raise Exception('You need to define the input dimensions')

    # shape transformation and tests
    Y, A_in, C, f, n_pixels_per_process, rank_f, d, T = test(
        Y, A_in, C, f, n_pixels_per_process, nb)

    start_time = time.time()
    logging.info('Computing support of spatial components')
    # we compute the indicator from distance indicator
    ind2_, nr, C, f, b_, A_in = computing_indicator(
        Y, A_in, b_in, C, f, nb, method_exp, dims, min_size, max_size, dist, expandCore, dview)

    # remove components that have a nan
    ff = np.where(np.isnan(np.sum(C, axis=1)))
    if np.size(ff) > 0:
        logging.info("Eliminating nan components: {}".format(ff))
        ff = ff[0]
        A_in = csc_column_remove(A_in, list(ff))
        C = np.delete(C, list(ff), 0)

    # remove empty components
    ff = np.where(np.sum(C, axis=1)==0)
    if np.size(ff) > 0:
        logging.info("Eliminating empty components: {}".format(ff))
        ff = ff[0]
        A_in = csc_column_remove(A_in, list(ff))
        C = np.delete(C, list(ff), 0)

    if normalize_yyt_one and C is not None:
        C = np.array(C)
        nr_C = np.shape(C)[0]
        d_ = scipy.sparse.lil_matrix((nr_C, nr_C))
        d_.setdiag(np.sqrt(np.sum(C ** 2, 1)))
        A_in = A_in * d_
        C = C/(np.sqrt((C**2).sum(1))[:, np.newaxis] + np.finfo(np.float32).eps)

    if b_in is None:
        b_in = b_

    logging.info('Memory mapping')
    # we create a memory map file if not already the case, we send Cf, a
    # matrix that include background components
    C_name, Y_name, folder = creatememmap(Y, np.vstack((C, f)), dview)

    # we create a pixel group array (chunks for the cnmf)for the parallelization of the process
    logging.info('Updating Spatial Components using lasso lars')
    cct = np.diag(C.dot(C.T))
    pixel_groups = []
    for i in range(0, np.prod(dims) - n_pixels_per_process + 1, n_pixels_per_process):
        pixel_groups.append([Y_name, C_name, sn, ind2_[i:i + n_pixels_per_process], list(
            range(i, i + n_pixels_per_process)), method_ls, cct, ])
    if i + n_pixels_per_process < np.prod(dims):
        pixel_groups.append([Y_name, C_name, sn, ind2_[(i + n_pixels_per_process):np.prod(dims)], list(
            range(i + n_pixels_per_process, np.prod(dims))), method_ls, cct])
    #A_ = scipy.sparse.lil_matrix((d, nr + np.size(f, 0)))
    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            parallel_result = dview.map_async(
                regression_ipyparallel, pixel_groups).get(4294967)
        else:
            parallel_result = dview.map_sync(
                regression_ipyparallel, pixel_groups)
            dview.results.clear()
    else:
        parallel_result = list(map(regression_ipyparallel, pixel_groups))
    data:List = []
    rows:List = []
    cols:List = []
    for chunk in parallel_result:
        for pars in chunk:
            px, idxs_, a = pars
            #A_[px, idxs_] = a
            nz = np.where(a>0)[0]
            data.extend(a[nz])
            rows.extend(len(nz)*[px])
            cols.extend(idxs_[nz])
    A_ = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(d, nr + np.size(f, 0)))

    logging.info("thresholding components")
    A_ = threshold_components(A_, dims, dview=dview, medw=medw, thr_method=thr_method,
                              maxthr=maxthr, nrgthr=nrgthr, extract_cc=extract_cc, se=se, ss=ss)
    #ff = np.where(np.sum(A_, axis=0) == 0)  # remove empty components
    ff = np.asarray(A_.sum(0) == 0).nonzero()[1]
    if np.size(ff) > 0:
        logging.info('removing {0} empty spatial component(s)'.format(ff.shape[0]))
        if any(ff < nr):
            A_ = csc_column_remove(A_, list(ff[ff < nr]))
            C = np.delete(C, list(ff[ff < nr]), 0)
            ff -= nr
            nr = nr - len(ff[ff < nr])
        else:
            ff -= nr
        if update_background_components:
            background_ff = list(filter(lambda i: i >= 0, ff))
            f = np.delete(f, background_ff, 0)
            if b_in is not None:
                b_in = np.delete(b_in, background_ff, 1)

    A_ = A_[:, :nr]
    if update_background_components:
        A_ = csr_matrix(A_)
        logging.info("Computing residuals")
        if 'memmap' in str(type(Y)):
            bl_siz1 = Y.shape[0] // (num_blocks_per_run_spat - 1)
            bl_siz2 = psutil.virtual_memory().available // (4*Y.shape[-1]*(num_blocks_per_run_spat + 1))
            Y_resf = parallel_dot_product(Y, f.T, dview=dview, block_size=min(bl_siz1, bl_siz2), num_blocks_per_run=num_blocks_per_run_spat) - \
                A_.dot(C[:nr].dot(f.T))
        else:
            # Y*f' - A*(C*f')
            Y_resf = np.dot(Y, f.T) - A_.dot(C[:nr].dot(f.T))

        if b_in is None:
            # update baseline based on residual
            #b = np.fmax(Y_resf.dot(np.linalg.inv(f.dot(f.T))), 0)
            b = np.fmax(np.linalg.solve(f.dot(f.T), Y_resf.T), 0).T
        else:
            ind_b = [np.where(_b)[0] for _b in b_in.T]
            b = HALS4shape_bckgrnd(Y_resf, b_in, f, ind_b)

    else:
        if b_in is None:
            raise Exception(
                'If you set the update_background_components to True you have to pass them as input to update_spatial')
        # try:
        #    b = np.delete(b_in, background_ff, 0)
        # except NameError:
        b = b_in
    # print(("--- %s seconds ---" % (time.time() - start_time)))
    logging.info('Updating done in ' +
                 '{0}s'.format(str(time.time() - start_time).split(".")[0]))
    try:  # clean up
        # remove temporary file created
        logging.info("Removing created tempfiles")
        shutil.rmtree(folder)
    except:
        raise Exception("Failed to delete: " + folder)

    return csc_matrix(A_), b, C, f

def test(Y, A_in, C, f, n_pixels_per_process, nb):
    """test the shape of each matrix, reshape it, and test the number of pixels per process

        if it doesn't follow the rules it will throw an exception that should not be caught by spatial.

        Args:
            Y: np.ndarray (2D or 3D)
                movie, raw data in 2D or 3D (pixels x time).

            C: np.ndarray
                calcium activity of each neuron.

            f: np.ndarray
                temporal profile  of background activity.

            A_in: np.ndarray
                spatial profile of background activity. If A_in is boolean then it defines the spatial support of A.
                Otherwise it is used to determine it through determine_search_location

            n_pixels_per_process: [optional] int
                number of pixels to be processed by each thread

        Returns:
            same:
                but reshaped and tested

        Raises:
            Exception 'You need to define the input dimensions'
            Exception 'Dimension of Matrix Y must be pixels x time'
            Exception 'Dimension of Matrix C must be neurons x time'
            Exception 'Dimension of Matrix f must be background comps x time'
            Exception 'Either A or C need to be determined'
            Exception 'Dimension of Matrix A must be pixels x neurons'
            Exception 'You need to provide estimate of C and f'
            Exception 'Not implemented consistently'
            Exception "Failed to delete: " + folder
        """
    if Y.ndim < 2 and not isinstance(Y, basestring):
        Y = np.atleast_2d(Y)
        if Y.shape[1] == 1:
            raise Exception('Dimension of Matrix Y must be pixels x time')

    if C is not None:
        C = np.atleast_2d(C)
        if C.shape[1] == 1:
            raise Exception('Dimension of Matrix C must be neurons x time')

    if f is not None:
        f = np.atleast_2d(f)
        if f.shape[1] == 1:
            raise Exception(
                'Dimension of Matrix f must be background comps x time ')
    else:
        f = np.zeros((0, Y.shape[1]), dtype=np.float32)

    if (A_in is None) and (C is None):
        raise Exception('Either A or C need to be determined')

    if A_in is not None:
        if len(A_in.shape) == 1:
            A_in = np.atleast_2d(A_in).T
            if A_in.shape[0] == 1:
                raise Exception(
                    'Dimension of Matrix A must be pixels x neurons ')

    [d, T] = np.shape(Y)

    if A_in is None:
        A_in = np.ones((d, np.shape(C)[1]), dtype=bool)

    if n_pixels_per_process > d:
        print('The number of pixels per process (n_pixels_per_process)'
              ' is larger than the total number of pixels!! Decreasing suitably.')
        n_pixels_per_process = d

    if f is not None:
        nb = f.shape[0]

    return Y, A_in, C, f, n_pixels_per_process, nb, d, T

def regression_ipyparallel(pars):
    """update spatial footprints and background through Basis Pursuit Denoising

       for each pixel i solve the problem
           [A(i,:),b(i)] = argmin sum(A(i,:))
       subject to
           || Y(i,:) - A(i,:)*C + b(i)*f || <= sn(i)*sqrt(T);

       for each pixel the search is limited to a few spatial components

       Args:
           C_name: string
                memmap C

           Y_name: string
                memmap Y

           idxs_Y: np.array
               indices of the Calcium traces for each computed components

           idxs_C: np.array
               indices of the Calcium traces for each computed components

           method_least_square:
               method to perform the regression for the basis pursuit denoising.
                    'nnls_L0'. Nonnegative least square with L0 penalty
                    'lasso_lars' lasso lars function from scikit learn

       Returns:
           px: np.ndarray
                positions o the regression

           idxs_C: np.ndarray
               indices of the Calcium traces for each computed components

           a: learned weight

       Raises:
           Exception 'Least Square Method not found!
       """

    # /!\ need to import since it is run from within the server
    import numpy as np
    import sys
    import gc
    from sklearn import linear_model

    Y_name, C_name, noise_sn, idxs_C, idxs_Y, method_least_square, cct = pars
    # we load from the memmap file
    if isinstance(Y_name, basestring):
        Y, _, _ = cp_motioncorrection.load_memmap(Y_name)
        Y = np.array(Y[idxs_Y, :])
    else:
        Y = Y_name[idxs_Y, :]
    if isinstance(C_name, basestring):
        C = np.load(C_name, mmap_mode='r')
        C = np.array(C)
    else:
        C = C_name

    _, T = np.shape(C)  # initialize values
    As = []
    for y, px, idx_px_from_0 in zip(Y, idxs_Y, range(len(idxs_C))):
        c = C[idxs_C[idx_px_from_0], :]
        idx_only_neurons = idxs_C[idx_px_from_0]
        if len(idx_only_neurons) > 0:
            cct_ = cct[idx_only_neurons[idx_only_neurons < len(cct)]]
        else:
            cct_ = []

        # skip if no components OR pixel has 0 activity
        if np.size(c) > 0 and noise_sn[px] > 0:
            sn = noise_sn[px] ** 2 * T
            if method_least_square == 'lasso_lars_old':
                raise Exception("Obsolete parameter") # Old code, support was removed

            elif method_least_square == 'nnls_L0':  # Nonnegative least square with L0 penalty
                a = nnls_L0(c.T, y, 1.2 * sn)

            elif method_least_square == 'lasso_lars':  # lasso lars function from scikit learn
                lambda_lasso = 0 if np.size(cct_) == 0 else \
                    .5 * noise_sn[px] * np.sqrt(np.max(cct_)) / T
                clf = linear_model.LassoLars(alpha=lambda_lasso, positive=True,
                                             fit_intercept=True)
#                clf = linear_model.Lasso(alpha=lambda_lasso, positive=True,
#                                         fit_intercept=True, normalize=True,
#                                         selection='random')
                a_lrs = clf.fit(np.array(c.T), np.ravel(y))
                a = a_lrs.coef_

            else:
                raise Exception(
                    'Least Square Method not found!' + method_least_square)

            if not np.isscalar(a):
                a = a.T

            As.append((px, idxs_C[idx_px_from_0], a))

    if isinstance(Y_name, basestring):
        del Y
    if isinstance(C_name, basestring):
        del C
    if isinstance(Y_name, basestring):
        gc.collect()

    return As

def nnls_L0(X, Yp, noise):
    """
    Nonnegative least square with L0 penalty

    It will basically call the scipy function with some tests
    we want to minimize :
    min|| Yp-W_lam*X||**2 <= noise
    with ||W_lam||_0  penalty
    and W_lam >0

    Args:
        X: np.array
            the input parameter ((the regressor

        Y: np.array
            ((the regressand

    Returns:
        W_lam: np.array
            the learned weight matrices ((Models

    """
    W_lam, RSS = scipy.optimize.nnls(X, np.ravel(Yp))
    RSS = RSS * RSS
    if RSS > noise:  # hard noise constraint problem infeasible
        return W_lam

    while 1:
        eliminate = []
        for i in np.where(W_lam[:-1] > 0)[0]:  # W_lam[:-1] to skip background
            mask = W_lam > 0
            mask[i] = 0
            Wtmp, tmp = scipy.optimize.nnls(X * mask, np.ravel(Yp))
            if tmp * tmp < noise:
                eliminate.append([i, tmp])
        if eliminate == []:
            return W_lam
        else:
            W_lam[eliminate[np.argmin(np.array(eliminate)[:, 1])][0]] = 0

def creatememmap(Y, Cf, dview):
    """memmap the C and Y objects in parallel

       the memmapped object will be read during parallelized computation such as the regression function

       Args:
           Y: np.ndarray (2D or 3D)
               movie, raw data in 2D or 3D (pixels x time).

           Cf: np.ndarray
               calcium activity of each neuron + background components

       Returns:
           C_name: string
                the memmapped name of Cf

           Y_name: string
                the memmapped name of Y

           Raises:
           Exception 'Not implemented consistently'
           """
    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        tmpf = os.environ.get('SLURM_SUBMIT_DIR')
        # print(f'cluster temporary folder: {tmpf}')
        folder = tempfile.mkdtemp(dir=tmpf)
    else:
        folder = tempfile.mkdtemp()

    if dview is None:
        Y_name = Y
        C_name = Cf
    else:
        C_name = os.path.join(folder, 'C_temp.npy')
        np.save(C_name, Cf)

        if isinstance(Y, np.core.memmap):  # if input file is already memory mapped then find the filename
            Y_name = Y.filename
        # if not create a memory mapped version (necessary for parallelization)
        elif isinstance(Y, basestring) or dview is None:
            Y_name = Y
        else:
            Y_name = os.path.join(folder, 'Y_temp.npy')
            np.save(Y_name, Y)
            Y, _, _, _ = cp_motioncorrection.load_memmap(Y_name)
            raise Exception('Not implemented consistently')
    return C_name, Y_name, folder

def threshold_components(A, dims, medw=None, thr_method='max', maxthr=0.1, nrgthr=0.9999, extract_cc=True,
                         se=None, ss=None, dview=None):
    """
    Post-processing of spatial components which includes the following steps

    (i) Median filtering
    (ii) Thresholding
    (iii) Morphological closing of spatial support
    (iv) Extraction of largest connected component ( to remove small unconnected pixel )

    Args:
        A:      np.ndarray
            2d matrix with spatial components

        dims:   tuple
            dimensions of spatial components

        medw: [optional] tuple
            window of median filter

        thr_method: [optional] string
            Method of thresholding:
                'max' sets to zero pixels that have value less than a fraction of the max value
                'nrg' keeps the pixels that contribute up to a specified fraction of the energy

        maxthr: [optional] scalar
            Threshold of max value

        nrgthr: [optional] scalar
            Threshold of energy

        extract_cc: [optional] bool
            Flag to extract connected components (might want to turn to False for dendritic imaging)

        se: [optional] np.intarray
            Morphological closing structuring element

        ss: [optional] np.intarray
            Binary element for determining connectivity

    Returns:
        Ath: np.ndarray
            2d matrix with spatial components thresholded
    """
    if medw is None:
        medw = (3,) * len(dims)
    if se is None:
        se = np.ones((3,) * len(dims), dtype='uint8')
    if ss is None:
        ss = np.ones((3,) * len(dims), dtype='uint8')
    # dims and nm of neurones
    d, nr = np.shape(A)
    # instanciation of A thresh.
    #Ath = np.zeros((d, nr))
    pars = []
    # for each neurons
    A_1 = scipy.sparse.csc_matrix(A)
    for i in range(nr):
        pars.append([A_1[:, i], i, dims,
                     medw, d, thr_method, se, ss, maxthr, nrgthr, extract_cc])

    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            res = dview.map_async(
                threshold_components_parallel, pars).get(4294967)
        else:
            res = dview.map_async(threshold_components_parallel, pars)
    else:
        res = list(map(threshold_components_parallel, pars))

    res.sort(key=lambda x: x[1])
    indices:List = []
    indptr = [0]
    data:List = []
    for r in res:
        At, i = r
        indptr.append(indptr[-1]+At.indptr[-1])
        indices.extend(At.indices.tolist())
        data.extend(At.data.tolist())

    Ath = csc_matrix((data, indices, indptr), shape=(d, nr))
    return Ath

def threshold_components_parallel(pars):
    """
       Post-processing of spatial components which includes the following steps

       (i) Median filtering
       (ii) Thresholding
       (iii) Morphological closing of spatial support
       (iv) Extraction of largest connected component ( to remove small unconnected pixel )
       /!\ need to be called through the function threshold components

       Args:
           [parsed] - A list of actual parameters:
               A:      np.ndarray
               2d matrix with spatial components

               dims:   tuple
                   dimensions of spatial components

               medw: [optional] tuple
                   window of median filter

               thr_method: [optional] string
                   Method of thresholding:
                       'max' sets to zero pixels that have value less than a fraction of the max value
                       'nrg' keeps the pixels that contribute up to a specified fraction of the energy

               maxthr: [optional] scalar
                   Threshold of max value

               nrgthr: [optional] scalar
                   Threshold of energy

               extract_cc: [optional] bool
                   Flag to extract connected components (might want to turn to False for dendritic imaging)

               se: [optional] np.intarray
                   Morphological closing structuring element

               ss: [optional] np.intarray
                   Binary element for determining connectivity

       Returns:
           Ath: np.ndarray
               2d matrix with spatial components thresholded
       """

    A_i, i, dims, medw, d, thr_method, se, ss, maxthr, nrgthr, extract_cc = pars
    A_i = A_i.toarray()
    # we reshape this one dimension column of the 2d components into the 2D that
    A_temp = np.reshape(A_i, dims[::-1])
    # we apply a median filter of size medw
    A_temp = median_filter(A_temp, medw)
    if thr_method == 'max':
        BW = (A_temp > maxthr * np.max(A_temp))
    elif thr_method == 'nrg':
        Asor = np.sort(np.squeeze(np.reshape(A_temp, (d, 1))))[::-1]
        temp = np.cumsum(Asor ** 2)
        ff = np.squeeze(np.where(temp < nrgthr * temp[-1]))
        if ff.size > 0:
            ind = ff if ff.ndim == 0 else ff[-1]
            A_temp[A_temp < Asor[ind]] = 0
            BW = (A_temp >= Asor[ind])
        else:
            BW = np.zeros_like(A_temp)
    # we want to remove the components that are valued 0 in this now 1d matrix
    Ath = np.squeeze(np.reshape(A_temp, (d, 1)))
    Ath2 = np.zeros((d))
    # we do that to have a full closed structure even if the values have been trehsolded
    BW = binary_closing(BW.astype(np.int), structure=se)

    # if we have deleted the element
    if BW.max() == 0:
        return csr_matrix(Ath2), i
    #
    # we want to extract the largest connected component ( to remove small unconnected pixel )
    if extract_cc:
        # # we extract each future as independent with the cross structuring element
        # labeled_array, num_features = label(BW, structure=ss)
        # labeled_array = np.squeeze(np.reshape(labeled_array, (d, 1)))
        # nrg = np.zeros((num_features, 1))
        # # we extract the energy for each component
        # for j in range(num_features):
        #     nrg[j] = np.sum(Ath[labeled_array == j + 1] ** 2)
        # indm = np.argmax(nrg)
        # Ath2[labeled_array == indm + 1] = Ath[labeled_array == indm + 1]
        print("**to_julia** probably not needed")
        sys.exit(2)

    else:
        BW = BW.flatten()
        Ath2[BW] = Ath[BW]

    return csr_matrix(Ath2), i

def csc_column_remove(A, ind):
    """ Removes specified columns for a scipy.sparse csc_matrix
    Args:
        A: scipy.sparse.csc_matrix
            Input matrix
        ind: iterable[int]
            list or np.array with columns to be removed
    """
    d1, d2 = A.shape
    if 'csc_matrix' not in str(type(A)): # FIXME
        logging.warning("Original matrix not in csc_format. Converting it" + " anyway.")
        A = scipy.sparse.csc_matrix(A)
    indptr = A.indptr
    ind_diff = np.diff(A.indptr).tolist()
    ind_sort = sorted(ind, reverse=True)
    data_list = [A.data[indptr[i]:indptr[i + 1]] for i in range(d2)]
    indices_list = [A.indices[indptr[i]:indptr[i + 1]] for i in range(d2)]
    for i in ind_sort:
        del data_list[i]
        del indices_list[i]
        del ind_diff[i]
    indptr_final = np.cumsum([0] + ind_diff)
    data_final = [item for sublist in data_list for item in sublist]
    indices_final = [item for sublist in indices_list for item in sublist]
    A = scipy.sparse.csc_matrix((data_final, indices_final, indptr_final), shape=[d1, d2 - len(ind)])
    return A

def HALS4shape_bckgrnd(Y_resf, B, F, ind_B, iters=5):
    K = B.shape[-1]
    U = Y_resf.T
    V = F.dot(F.T)
    for _ in range(iters):
        for m in range(K):  # neurons
            ind_pixels = ind_B[m]

            B[ind_pixels, m] = np.clip(B[ind_pixels, m] +
                                       ((U[m, ind_pixels] - V[m].dot(B[ind_pixels].T)) /
                                        V[m, m]), 0, np.inf)
    return B

def computing_indicator(Y, A_in, b, C, f, nb, method, dims, min_size, max_size, dist, expandCore, dview):
    """compute the indices of the distance from the cm to search for the spatial component (calling determine_search_location)

    does this by following an ellipse from the cm or doing a step by step dilatation around the cm
    if it doesn't follow the rules it will throw an exception that is not supposed to be caught by spatial.


    Args:
        Y: np.ndarray (2D or 3D)
               movie, raw data in 2D or 3D (pixels x time).

        C: np.ndarray
               calcium activity of each neuron.

        f: np.ndarray
               temporal profile  of background activity.

        A_in: np.ndarray
               spatial profile of background activity. If A_in is boolean then it defines the spatial support of A.
               Otherwise it is used to determine it through determine_search_location

        n_pixels_per_process: [optional] int
               number of pixels to be processed by each thread

        min_size: [optional] int

        max_size: [optional] int

        dist: [optional] int

        dims: [optional] tuple
                x, y[, z] movie dimensions

        method: [optional] string
                method used to expand the search for pixels 'ellipse' or 'dilate'

        expandCore: [optional]  scipy.ndimage.morphology
                if method is dilate this represents the kernel used for expansion


    Returns:
        same:
            but reshaped and tested

    Raises:
        Exception 'You need to define the input dimensions'

        Exception 'Dimension of Matrix Y must be pixels x time'

        Exception 'Dimension of Matrix C must be neurons x time'

        Exception 'Dimension of Matrix f must be background comps x time'

        Exception 'Either A or C need to be determined'

        Exception 'Dimension of Matrix A must be pixels x neurons'

        Exception 'You need to provide estimate of C and f'

        Exception 'Not implemented consistently'

        Exception 'Failed to delete: " + folder'
           """

    if A_in.dtype == bool:
        dist_indicator = A_in.copy()
        print("spatial support for each components given by the user")
        # we compute C,B,f,Y if we have boolean for A matrix
        if C is None:  # if C is none we approximate C, b and f from the binary mask
            dist_indicator_av = old_div(dist_indicator.astype(
                'float32'), np.sum(dist_indicator.astype('float32'), axis=0))
            px = (np.sum(dist_indicator, axis=1) > 0)
            not_px = 1 - px
            if Y.shape[-1] < 30000:
                f = Y[not_px, :].mean(0)
            else:  # memory mapping fails here for some reasons
                print('estimating f')
                f = 0
                for xxx in not_px:
                    f = (f + Y[xxx]) / 2

            f = np.atleast_2d(f)

            Y_resf = np.dot(Y, f.T)
            b = np.maximum(Y_resf, 0) / (np.linalg.norm(f)**2)
            C = np.maximum(csr_matrix(dist_indicator_av.T).dot(
                Y) - dist_indicator_av.T.dot(b).dot(f), 0)
            A_in = scipy.sparse.coo_matrix(A_in.astype(np.float32))
            nr, _ = np.shape(C)  # number of neurons
            ind2_ = [np.hstack((np.where(iid_)[0], nr + np.arange(f.shape[0])))
                     if np.size(np.where(iid_)[0]) > 0 else [] for iid_ in dist_indicator]

    else:
        if C is None:
            raise Exception('You need to provide estimate of C and f')

        nr, _ = np.shape(C)  # number of neurons

        if b is None:
            dist_indicator = determine_search_location(
                A_in, dims, method=method, min_size=min_size, max_size=max_size, dist=dist, expandCore=expandCore, dview=dview)
        else:
            dist_indicator = determine_search_location(
                scipy.sparse.hstack([A_in, scipy.sparse.coo_matrix(b)]), dims, method=method, min_size=min_size, max_size=max_size, dist=dist, expandCore=expandCore,
                dview=dview)

        ind2_ = [np.where(iid_.squeeze())[0]  for iid_ in dist_indicator.astype(np.bool).toarray()]
        ind2_ = [iid_ if (np.size(iid_) > 0) and (np.min(iid_) < nr) else [] for iid_ in ind2_]

    return ind2_, nr, C, f, b, A_in

def determine_search_location(A, dims, method='ellipse', min_size=3, max_size=8, dist=3,
                              expandCore=iterate_structure(generate_binary_structure(2, 1), 2).astype(int), dview=None):
    """
    compute the indices of the distance from the cm to search for the spatial component

    does this by following an ellipse from the cm or doing a step by step dilatation around the cm

    Args:
        A[:, i]: the A of each components

        dims:
            the dimension of each A's ( same usually )

         dist:
             computed distance matrix

         dims: [optional] tuple
             x, y[, z] movie dimensions

         method: [optional] string
             method used to expand the search for pixels 'ellipse' or 'dilate'

         expandCore: [optional]  scipy.ndimage.morphology
             if method is dilate this represents the kernel used for expansion

        min_size: [optional] int

        max_size: [optional] int

        dist: [optional] int

        dims: [optional] tuple
             x, y[, z] movie dimensions

    Returns:
        dist_indicator: np.ndarray
            distance from the cm to search for the spatial footprint

    Raises:
        Exception 'You cannot pass empty (all zeros) components!'
    """

    from scipy.ndimage.morphology import grey_dilation

    # we initialize the values
    if len(dims) == 2:
        d1, d2 = dims
    elif len(dims) == 3:
        d1, d2, d3 = dims
    d, nr = np.shape(A)
    A = csc_matrix(A)
#    dist_indicator = scipy.sparse.lil_matrix((d, nr),dtype= np.float32)
#    dist_indicator = scipy.sparse.csc_matrix((d, nr), dtype=np.float32)

    if method == 'ellipse':
        Coor = dict()
        # we create a matrix of size A.x of each pixel coordinate in A.y and inverse
        if len(dims) == 2:
            Coor['x'] = np.kron(np.ones(d2), list(range(d1)))
            Coor['y'] = np.kron(list(range(d2)), np.ones(d1))
        elif len(dims) == 3:
            Coor['x'] = np.kron(np.ones(d3 * d2), list(range(d1)))
            Coor['y'] = np.kron(
                np.kron(np.ones(d3), list(range(d2))), np.ones(d1))
            Coor['z'] = np.kron(list(range(d3)), np.ones(d2 * d1))
        if not dist == np.inf:  # determine search area for each neuron
            cm = np.zeros((nr, len(dims)))  # vector for center of mass
            Vr:List = []  # cell(nr,1);
            dist_indicator = []
            pars = []
            # for each dim
            for i, c in enumerate(['x', 'y', 'z'][:len(dims)]):
                # mass center in this dim = (coor*A)/sum(A)
                cm[:, i] = old_div(
                    np.dot(Coor[c], A[:, :nr].todense()), A[:, :nr].sum(axis=0))

            # parallelizing process of the construct ellipse function
            for i in range(nr):
                pars.append([Coor, cm[i], A[:, i], Vr, dims,
                             dist, max_size, min_size, d])
            if dview is None:
                res = list(map(construct_ellipse_parallel, pars))
            else:
                if 'multiprocessing' in str(type(dview)):
                    res = dview.map_async(
                        construct_ellipse_parallel, pars).get(4294967)
                else:
                    res = dview.map_sync(construct_ellipse_parallel, pars)
            for r in res:
                dist_indicator.append(r)

            dist_indicator = scipy.sparse.coo_matrix((np.asarray(dist_indicator)).squeeze().T)

        else:
            raise Exception('Not implemented')
            dist_indicator = True * np.ones((d, nr))

    elif method == 'dilate':
        indptr = [0]
        indices:List = []
        data = []
        if dview is None:
            for i in range(nr):
                A_temp = np.reshape(A[:, i].toarray(), dims[::-1])
                if len(expandCore) > 0:
                    if len(expandCore.shape) < len(dims):  # default for 3D
                        expandCore = iterate_structure(
                            generate_binary_structure(len(dims), 1), 2).astype(int)
                    A_temp = grey_dilation(A_temp, footprint=expandCore)
                else:
                    A_temp = grey_dilation(A_temp, [1] * len(dims))

                nz = np.where(np.squeeze(np.reshape(A_temp, (d, 1)))[:, None] > 0)[0].tolist()
                indptr.append(indptr[-1] + len(nz))
                indices.extend(nz)
                data.extend(len(nz)*[True])
                #dist_indicator[:, i] = scipy.sparse.coo_matrix(np.squeeze(np.reshape(A_temp, (d, 1)))[:, None] > 0)
            dist_indicator = csc_matrix((data, indices, indptr), shape=(d, nr))

        else:
            logging.info('dilate in parallel...')
            pars = []
            for i in range(nr):
                pars.append([A[:, i], dims, expandCore, d])

            if 'multiprocessing' in str(type(dview)):
                parallel_result = dview.map_async(
                    construct_dilate_parallel, pars).get(4294967)
            else:
                parallel_result = dview.map_sync(
                    construct_dilate_parallel, pars)
                dview.results.clear()

            i = 0
            for res in parallel_result:
                indptr.append(indptr[-1] + len(res.row))
                indices.extend(res.row)
                data.extend(len(res.row)*[True])
                #dist_indicator[:, i] = res
                i += 1
            dist_indicator = csc_matrix((data, indices, indptr), shape=(d, nr))

    else:
        raise Exception('Not implemented')
        dist_indicator = True * np.ones((d, nr))

    return csc_matrix(dist_indicator)

def construct_ellipse_parallel(pars):
    """update spatial footprints and background through Basis Pursuit Denoising


    """
    Coor, cm, A_i, Vr, dims, dist, max_size, min_size, d = pars
    dist_cm = coo_matrix(np.hstack([Coor[c].reshape(-1, 1) - cm[k]
                                    for k, c in enumerate(['x', 'y', 'z'][:len(dims)])]))
    Vr.append(dist_cm.T * spdiags(A_i.toarray().squeeze(),
                                  0, d, d) * dist_cm / A_i.sum(axis=0))

    if np.sum(np.isnan(Vr)) > 0:
        raise Exception('You cannot pass empty (all zeros) components!')

    D, V = eig(Vr[-1])

    dkk = [np.min((max_size ** 2, np.max((min_size ** 2, dd.real))))
           for dd in D]

    # search indexes for each component
    return np.sqrt(np.sum([old_div((dist_cm * V[:, k]) ** 2, dkk[k]) for k in range(len(dkk))], 0)) <= dist

def merge_components(Y, A, b, C, R, f, S, sn_pix, temporal_params,
                     spatial_params, dview=None, thr=0.85, fast_merge=True,
                     mx=1000, bl=None, c1=None, sn=None, g=None,
                     merge_parallel=False, max_merge_area=None):

    """ Merging of spatially overlapping components that have highly correlated temporal activity

    The correlation threshold for merging overlapping components is user specified in thr

    Args:
        Y: np.ndarray
            residual movie after subtracting all found components
            (Y_res = Y - A*C - b*f) (d x T)

        A: sparse matrix
            matrix of spatial components (d x K)

        b: np.ndarray
             spatial background (vector of length d)

        C: np.ndarray
             matrix of temporal components (K x T)

        R: np.ndarray
             array of residuals (K x T)

        f:     np.ndarray
             temporal background (vector of length T)

        S:     np.ndarray
             matrix of deconvolved activity (spikes) (K x T)

        sn_pix: ndarray
             noise standard deviation for each pixel

        temporal_params: dictionary
             all the parameters that can be passed to the
             update_temporal_components function

        spatial_params: dictionary
             all the parameters that can be passed to the
             update_spatial_components function

        thr:   scalar between 0 and 1
             correlation threshold for merging (default 0.85)

        mx:    int
             maximum number of merging operations (default 50)

        sn_pix:    nd.array
             noise level for each pixel (vector of length d)

        fast_merge: bool
            if true perform rank 1 merging, otherwise takes best neuron

        bl:
             baseline for fluorescence trace for each row in C
        c1:
             initial concentration for each row in C
        g:
             discrete time constant for each row in C
        sn:
             noise level for each row in C

        merge_parallel: bool
             perform merging in parallel

        max_merge_area: int
            maximum area (in pixels) of merged components,
            used to determine whether to merge

    Returns:
        A:     sparse matrix
                matrix of merged spatial components (d x K)

        C:     np.ndarray
                matrix of merged temporal components (K x T)

        nr:    int
            number of components after merging

        merged_ROIs: list
            index of components that have been merged

        S:     np.ndarray
                matrix of merged deconvolved activity (spikes) (K x T)

        bl: float
            baseline for fluorescence trace

        c1: float
            initial concentration

        g:  float
            discrete time constant

        sn: float
            noise level

        R:  np.ndarray
            residuals
    Raises:
        Exception "The number of elements of bl\c1\g\sn must match the number of components"
    """

    #tests and initialization
    nr = A.shape[1]
    A = csc_matrix(A)
    if bl is not None and len(bl) != nr:
        raise Exception(
            "The number of elements of bl must match the number of components")
    if c1 is not None and len(c1) != nr:
        raise Exception(
            "The number of elements of c1 must match the number of components")
    if sn is not None and len(sn) != nr:
        raise Exception(
            "The number of elements of sn must match the number of components")
    if g is not None and len(g) != nr:
        raise Exception(
            "The number of elements of g must match the number of components")
    if R is None:
        R = np.zeros_like(C)

    [d, t] = np.shape(Y)

    # % find graph of overlapping spatial components
    A_corr = scipy.sparse.triu(A.T * A)
    A_corr.setdiag(0)
    A_corr = A_corr.tocsc()
    FF2 = A_corr > 0
    C_corr = scipy.sparse.lil_matrix(A_corr.shape)
    for ii in range(nr):
        overlap_indices = A_corr[ii, :].nonzero()[1]
        if len(overlap_indices) > 0:
            # we chesk the correlation of the calcium traces for eahc overlapping components
            corr_values = [scipy.stats.pearsonr(C[ii, :], C[jj, :])[
                0] for jj in overlap_indices]
            C_corr[ii, overlap_indices] = corr_values

    FF1 = (C_corr + C_corr.T) > thr
    FF3 = FF1.multiply(FF2)

    nb, connected_comp = csgraph.connected_components(
        FF3)  # % extract connected components

    p = temporal_params['p']
    list_conxcomp_initial = []
    for i in range(nb):  # we list them
        if np.sum(connected_comp == i) > 1:
            list_conxcomp_initial.append((connected_comp == i).T)
    list_conxcomp = np.asarray(list_conxcomp_initial).T

    if list_conxcomp.ndim > 1:
        cor = np.zeros((np.shape(list_conxcomp)[1], 1))
        for i in range(np.size(cor)):
            fm = np.where(list_conxcomp[:, i])[0]
            for j1 in range(np.size(fm)):
                for j2 in range(j1 + 1, np.size(fm)):
                    cor[i] = cor[i] + C_corr[fm[j1], fm[j2]]

#        if not fast_merge:
#            Y_res = Y - A.dot(C) #residuals=background=noise
        if np.size(cor) > 1:
            # we get the size (indices)
            ind = np.argsort(np.squeeze(cor))[::-1]
        else:
            ind = [0]

        nbmrg = min((np.size(ind), mx))   # number of merging operations

        if merge_parallel:
            merged_ROIs = [np.where(list_conxcomp[:, ind[i]])[0] for i in range(nbmrg)]
            Acsc_mats = [csc_matrix(A[:, merged_ROI]) for merged_ROI in merged_ROIs]
            Ctmp_mats = [C[merged_ROI] + R[merged_ROI] for merged_ROI in merged_ROIs]
            C_to_norms = [np.sqrt(np.ravel(Acsc.power(2).sum(
                    axis=0)) * np.sum(Ctmp ** 2, axis=1)) for (Acsc, Ctmp) in zip(Acsc_mats, Ctmp_mats)]
            indxs = [np.argmax(C_to_norm) for C_to_norm in C_to_norms]
            g_idxs = [merged_ROI[indx] for (merged_ROI, indx) in zip(merged_ROIs, indxs)]
            fms = [fast_merge]*nbmrg
            tps = [temporal_params]*nbmrg
            gs = [g]*nbmrg

            if dview is None:
               merge_res = list(map(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
            elif 'multiprocessin' in str(type(dview)):
               merge_res = list(dview.map(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
            else:
               merge_res = list(dview.map_sync(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
               dview.results.clear()
            #merge_res = list(dview.map(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
            bl_merged = np.array([res[0] for res in merge_res])
            c1_merged = np.array([res[1] for res in merge_res])
            A_merged = csc_matrix(scipy.sparse.vstack([csc_matrix(res[2]) for res in merge_res]).T)
            C_merged = np.vstack([res[3] for res in merge_res])
            g_merged = np.vstack([res[4] for res in merge_res])
            sn_merged = np.array([res[5] for res in merge_res])
            S_merged = np.vstack([res[6] for res in merge_res])
            R_merged = np.vstack([res[7] for res in merge_res])
        else:
            # we initialize the values
            A_merged = lil_matrix((d, nbmrg))
            C_merged = np.zeros((nbmrg, t))
            R_merged = np.zeros((nbmrg, t))
            S_merged = np.zeros((nbmrg, t))
            bl_merged = np.zeros((nbmrg, 1))
            c1_merged = np.zeros((nbmrg, 1))
            sn_merged = np.zeros((nbmrg, 1))
            g_merged = np.zeros((nbmrg, p))
            merged_ROIs = []
            for i in range(nbmrg):
                merged_ROI = np.where(list_conxcomp[:, ind[i]])[0]
                logging.info('Merging components {}'.format(merged_ROI))
                merged_ROIs.append(merged_ROI)
                Acsc = A.tocsc()[:, merged_ROI]
                Ctmp = np.array(C)[merged_ROI, :] + np.array(R)[merged_ROI, :]
                C_to_norm = np.sqrt(np.ravel(Acsc.power(2).sum(
                    axis=0)) * np.sum(Ctmp ** 2, axis=1))
                indx = np.argmax(C_to_norm)
                g_idx = [merged_ROI[indx]]
                bm, cm, computedA, computedC, gm, sm, ss, yra = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx,
                                                                                indx, temporal_params)

                A_merged[:, i] = csr_matrix(computedA).T
                C_merged[i, :] = computedC
                R_merged[i, :] = yra
                S_merged[i, :] = ss[:t]
                bl_merged[i] = bm
                c1_merged[i] = cm
                sn_merged[i] = sm
                g_merged[i, :] = gm

        empty = np.ravel((C_merged.sum(1) == 0) + (A_merged.sum(0) == 0))
        if np.any(empty):
            A_merged = A_merged[:, ~empty]
            C_merged = C_merged[~empty]
            R_merged = R_merged[~empty]
            S_merged = S_merged[~empty]
            bl_merged = bl_merged[~empty]
            c1_merged = c1_merged[~empty]
            sn_merged = sn_merged[~empty]
            g_merged = g_merged[~empty]

        if len(merged_ROIs) > 0:
            # we want to remove merged neuron from the initial part and replace them with merged ones
            neur_id = np.unique(np.hstack(merged_ROIs))
            good_neurons = np.setdiff1d(list(range(nr)), neur_id)
            A = scipy.sparse.hstack((A.tocsc()[:, good_neurons], A_merged.tocsc()))
            C = np.vstack((C[good_neurons, :], C_merged))
            # we continue for the variables
            if S is not None:
                S = np.vstack((S[good_neurons, :], S_merged))
            if R is not None:
                R = np.vstack((R[good_neurons, :], R_merged))
            if bl is not None:
                bl = np.hstack((bl[good_neurons], np.array(bl_merged).flatten()))
            if c1 is not None:
                c1 = np.hstack((c1[good_neurons], np.array(c1_merged).flatten()))
            if sn is not None:
                sn = np.hstack((sn[good_neurons], np.array(sn_merged).flatten()))
            if g is not None:
#                g = np.vstack((np.vstack(g)[good_neurons], g_merged))
                g = np.vstack(g)[good_neurons]
                if g.shape[1] == 0:
                    g = np.zeros((len(good_neurons), g_merged.shape[1]))
                g = np.vstack((g, g_merged))

            nr = nr - len(neur_id) + len(C_merged)

    else:
        logging.info('No more components merged!')
        merged_ROIs = []
        empty = []

    return A, C, nr, merged_ROIs, S, bl, c1, sn, g, empty, R

def merge_iter(a):
    Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx, indx, temporal_params = a
    res = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx,
                          indx, temporal_params)
    return res

def merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx, indx, temporal_params):
    if fast_merge:
        # we normalize the values of different A's to be able to compare them efficiently. we then sum them

        computedA = Acsc.dot(C_to_norm)
        for _ in range(10):
            computedC = np.maximum((Acsc.T.dot(computedA)).dot(Ctmp) /
                                   (computedA.T.dot(computedA)), 0)
            nc = computedC.T.dot(computedC)
            if nc == 0:
                break
            computedA = np.maximum(Acsc.dot(Ctmp.dot(computedC.T)) / nc, 0)

#        computedA = Acsc.dot(scipy.sparse.diags(
#            C_to_norm, 0, (len(C_to_norm), len(C_to_norm)))).sum(axis=1)
#
#        # we operate a rank one NMF, refining it multiple times (see cnmf code_base )
#        for _ in range(10):
#            computedC = np.maximum(Acsc.T.dot(computedA).T.dot(
#                Ctmp) / (computedA.T * computedA), 0)
#            if computedC * computedC.T == 0:
#                break
#            computedA = np.maximum(
#                Acsc.dot(Ctmp.dot(computedC.T)) / (computedC * computedC.T), 0)
    else:
        logging.info('Simple merging ny taking best neuron')
        computedC = Ctmp[indx]
        computedA = Acsc[:, indx]
    # then we de-normalize them using A_to_norm
    A_to_norm = np.sqrt(computedA.T.dot(computedA)) #/Acsc.power(2).sum(0).max())
    computedA /= A_to_norm
    computedC *= A_to_norm
    # r = (computedA.T.dot(Acsc.dot(Ctmp)))/(computedA.T.dot(computedA)) - computedC
    r = ((Acsc.T.dot(computedA)).dot(Ctmp))/(computedA.T.dot(computedA)) - computedC
    # we then compute the traces ( deconvolution ) to have a clean c and noise in the background
    c_in =  np.array(computedC+r).squeeze()
    if g is not None:
        deconvC, bm, cm, gm, sm, ss, lam_ = CNMFE_2.constrained_foopsi(
            c_in, g=g_idx, **temporal_params)
    else:
        deconvC, bm, cm, gm, sm, ss, lam_ = CNMFE_2.constrained_foopsi(
            c_in, g=None, **temporal_params)
    return bm, cm, computedA, deconvC, gm, sm, ss, c_in - deconvC

def initialize_components(Y, K=30, gSig=[5, 5], gSiz=None, ssub=1, tsub=1, nIter=5, maxIter=5, nb=1,
                          kernel=None, use_hals=True, normalize_init=True, img=None, method_init='greedy_roi',
                          max_iter_snmf=500, alpha_snmf=10e2, sigma_smooth_snmf=(.5, .5, .5),
                          perc_baseline_snmf=20, options_local_NMF=None, rolling_sum=False,
                          rolling_length=100, sn=None, options_total=None,
                          min_corr=0.8, min_pnr=10, seed_method='auto', ring_size_factor=1.5,
                          center_psf=False, ssub_B=2, init_iter=2, remove_baseline = True,
                          SC_kernel='heat', SC_sigma=1, SC_thr=0, SC_normalize=True, SC_use_NN=False,
                          SC_nnn=20, lambda_gnmf=1):
    """
    Initalize components. This function initializes the spatial footprints, temporal components,
    and background which are then further refined by the CNMF iterations. There are four
    different initialization methods depending on the data you're processing:
        'greedy_roi': GreedyROI method used in standard 2p processing (default)
        'corr_pnr': GreedyCorr method used for processing 1p data
        'sparse_nmf': Sparse NMF method suitable for dendritic/axonal imaging
        'graph_nmf': Graph NMF method also suitable for dendritic/axonal imaging

    The GreedyROI method by default is not using the RollingGreedyROI method. This can
    be changed through the binary flag 'rolling_sum'.

    All the methods can be used for volumetric data except 'corr_pnr' which is only
    available for 2D data.

    It is also by default followed by hierarchical alternative least squares (HALS) NMF.
    Optional use of spatio-temporal downsampling to boost speed.

    Args:
        Y: np.ndarray
            d1 x d2 [x d3] x T movie, raw data.

        K: [optional] int
            number of neurons to extract (default value: 30). Maximal number for method 'corr_pnr'.

        tau: [optional] list,tuple
            standard deviation of neuron size along x and y [and z] (default value: (5,5).

        gSiz: [optional] list,tuple
            size of kernel (default 2*tau + 1).

        nIter: [optional] int
            number of iterations for shape tuning (default 5).

        maxIter: [optional] int
            number of iterations for HALS algorithm (default 5).

        ssub: [optional] int
            spatial downsampling factor recommended for large datasets (default 1, no downsampling).

        tsub: [optional] int
            temporal downsampling factor recommended for long datasets (default 1, no downsampling).

        kernel: [optional] np.ndarray
            User specified kernel for greedyROI
            (default None, greedy ROI searches for Gaussian shaped neurons)

        use_hals: [optional] bool
            Whether to refine components with the hals method

        normalize_init: [optional] bool
            Whether to normalize_init data before running the initialization

        img: optional [np 2d array]
            Image with which to normalize. If not present use the mean + offset

        method_init: {'greedy_roi', 'corr_pnr', 'sparse_nmf', 'graph_nmf', 'pca_ica'}
            Initialization method (default: 'greedy_roi')

        max_iter_snmf: int
            Maximum number of sparse NMF iterations

        alpha_snmf: scalar
            Sparsity penalty

        rolling_sum: boolean
            Detect new components based on a rolling sum of pixel activity (default: False)

        rolling_length: int
            Length of rolling window (default: 100)

        center_psf: Boolean
            True indicates centering the filtering kernel for background
            removal. This is useful for data with large background
            fluctuations.

        min_corr: float
            minimum local correlation coefficients for selecting a seed pixel.

        min_pnr: float
            minimum peak-to-noise ratio for selecting a seed pixel.

        seed_method: str {'auto', 'manual', 'semi'}
            methods for choosing seed pixels
            'semi' detects K components automatically and allows to add more manually
            if running as notebook 'semi' and 'manual' require a backend that does not
            inline figures, e.g. %matplotlib tk

        ring_size_factor: float
            it's the ratio between the ring radius and neuron diameters.

        nb: integer
            number of background components for approximating the background using NMF model

        sn: ndarray
            per pixel noise

        options_total: dict
            the option dictionary

        ssub_B: int, optional
            downsampling factor for 1-photon imaging background computation

        init_iter: int, optional
            number of iterations for 1-photon imaging initialization

    Returns:
        Ain: np.ndarray
            (d1 * d2 [ * d3]) x K , spatial filter of each neuron.

        Cin: np.ndarray
            T x K , calcium activity of each neuron.

        center: np.ndarray
            K x 2 [or 3] , inferred center of each neuron.

        bin: np.ndarray
            (d1 * d2 [ * d3]) x nb, initialization of spatial background.

        fin: np.ndarray
            nb x T matrix, initalization of temporal background

    Raises:
        Exception "Unsupported method"

        Exception 'You need to define arguments for local NMF'

    """
    method = method_init
    if method == 'local_nmf':
        tsub_lnmf = tsub
        ssub_lnmf = ssub
        tsub = 1
        ssub = 1

    if gSiz is None:
        gSiz = 2 * (np.asarray(gSig) + .5).astype(int) + 1

    d, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    # rescale according to downsampling factor
    gSig = np.asarray(gSig, dtype=float) / ssub
    gSiz = np.round(np.asarray(gSiz) / ssub).astype(np.int)

    if normalize_init is True:
        logging.info('Variance Normalization')
        if img is None:
            img = np.mean(Y, axis=-1)
            img += np.median(img)
            img += np.finfo(np.float32).eps

        Y = old_div(Y, np.reshape(img, d + (-1,), order='F'))
        alpha_snmf /= np.mean(img)
    else:
        Y = np.array(Y)

    # spatial downsampling

    if ssub != 1 or tsub != 1:

        if method == 'corr_pnr':
            logging.info("Spatial/Temporal downsampling 1-photon")
            # this icrements the performance against ground truth and solves border problems
            Y_ds = downscale(Y, tuple([ssub] * len(d) + [tsub]), opencv=False)
        else:
            logging.info("Spatial/Temporal downsampling 2-photon")
            # this icrements the performance against ground truth and solves border problems
            Y_ds = downscale(Y, tuple([ssub] * len(d) + [tsub]), opencv=True)
#            mean_val = np.mean(Y)
#            Y_ds = downscale_local_mean(Y, tuple([ssub] * len(d) + [tsub]), cval=mean_val)
    else:
        Y_ds = Y

    ds = Y_ds.shape[:-1]
    if nb > min(np.prod(ds), Y_ds.shape[-1]):
        nb = -1

    logging.info('Roi Initialization...')
    if method == 'greedy_roi':
        Ain, Cin, _, b_in, f_in = greedyROI(
            Y_ds, nr=K, gSig=gSig, gSiz=gSiz, nIter=nIter, kernel=kernel, nb=nb,
            rolling_sum=rolling_sum, rolling_length=rolling_length, seed_method=seed_method)

        if use_hals:
            logging.info('Refining Components using HALS NMF iterations')
            Ain, Cin, b_in, f_in = hals(
                Y_ds, Ain, Cin, b_in, f_in, maxIter=maxIter)
    elif method == 'corr_pnr':
        Ain, Cin, _, b_in, f_in, extra_1p = greedyROI_corr(
            Y, Y_ds, max_number=K, gSiz=gSiz[0], gSig=gSig[0], min_corr=min_corr, min_pnr=min_pnr,
            ring_size_factor=ring_size_factor, center_psf=center_psf, options=options_total,
            sn=sn, nb=nb, ssub=ssub, ssub_B=ssub_B, init_iter=init_iter, seed_method=seed_method)

    # elif method == 'sparse_nmf':
    #     Ain, Cin, _, b_in, f_in = sparseNMF(
    #         Y_ds, nr=K, nb=nb, max_iter_snmf=max_iter_snmf, alpha=alpha_snmf,
    #         sigma_smooth=sigma_smooth_snmf, remove_baseline=remove_baseline, perc_baseline=perc_baseline_snmf)
    #
    # elif method == 'compressed_nmf':
    #     Ain, Cin, _, b_in, f_in = compressedNMF(
    #         Y_ds, nr=K, nb=nb, max_iter_snmf=max_iter_snmf,
    #         sigma_smooth=sigma_smooth_snmf, remove_baseline=remove_baseline, perc_baseline=perc_baseline_snmf)
    #
    # elif method == 'graph_nmf':
    #     Ain, Cin, _, b_in, f_in = graphNMF(
    #         Y_ds, nr=K, nb=nb, max_iter_snmf=max_iter_snmf, lambda_gnmf=lambda_gnmf,
    #         sigma_smooth=sigma_smooth_snmf, remove_baseline=remove_baseline,
    #         perc_baseline=perc_baseline_snmf, SC_kernel=SC_kernel,
    #         SC_sigma=SC_sigma, SC_use_NN=SC_use_NN, SC_nnn=SC_nnn,
    #         SC_normalize=SC_normalize, SC_thr=SC_thr)

    # elif method == 'pca_ica':
    #     Ain, Cin, _, b_in, f_in = ICA_PCA(
    #         Y_ds, nr=K, sigma_smooth=sigma_smooth_snmf, truncate=2, fun='logcosh', tol=1e-10,
    #         max_iter=max_iter_snmf, remove_baseline=True, perc_baseline=perc_baseline_snmf, nb=nb)
    #
    # elif method == 'local_nmf':
    #     # todo check this unresolved reference
    #     from SourceExtraction.CNMF4Dendrites import CNMF4Dendrites
    #     from SourceExtraction.AuxilaryFunctions import GetCentersData
    #     # Get initialization for components center
    #     # print(Y_ds.transpose([2, 0, 1]).shape)
    #     if options_local_NMF is None:
    #         raise Exception('You need to define arguments for local NMF')
    #     else:
    #         NumCent = options_local_NMF.pop('NumCent', None)
    #         # Max number of centers to import from Group Lasso intialization - if 0,
    #         # we don't run group lasso
    #         cent = GetCentersData(Y_ds.transpose([2, 0, 1]), NumCent)
    #         sig = Y_ds.shape[:-1]
    #         # estimate size of neuron - bounding box is 3 times this size. If larger
    #         # then data, we have no bounding box.
    #         cnmf_obj = CNMF4Dendrites(
    #             sig=sig, verbose=True, adaptBias=True, **options_local_NMF)
    #
    #     # Define CNMF parameters
    #     _, _, _ = cnmf_obj.fit(
    #         np.array(Y_ds.transpose([2, 0, 1]), dtype=np.float), cent)
    #
    #     Ain = cnmf_obj.A
    #     Cin = cnmf_obj.C
    #     b_in = cnmf_obj.b
    #     f_in = cnmf_obj.f

    else:

        print(method)
        raise Exception("Unsupported initialization method")

    K = np.shape(Ain)[-1]

    if Ain.size > 0 and not center_psf and ssub != 1:

        Ain = np.reshape(Ain, ds + (K,), order='F')

        if len(ds) == 2:
            Ain = resize(Ain, d + (K,))

        else:  # resize only deals with 2D images, hence apply resize twice
            Ain = np.reshape([resize(a, d[1:] + (K,))
                              for a in Ain], (ds[0], d[1] * d[2], K), order='F')
            Ain = resize(Ain, (d[0], d[1] * d[2], K))

        Ain = np.reshape(Ain, (np.prod(d), K), order='F')

    sparse_b = spr.issparse(b_in)
    if (nb > 0 or nb == -1) and (ssub != 1 or tsub != 1):
        b_in = np.reshape(b_in.toarray() if sparse_b else b_in, ds + (-1,), order='F')

        if len(ds) == 2:
            b_in = resize(b_in, d + (b_in.shape[-1],))
        else:
            b_in = np.reshape([resize(b, d[1:] + (b_in.shape[-1],))
                               for b in b_in], (ds[0], d[1] * d[2], -1), order='F')
            b_in = resize(b_in, (d[0], d[1] * d[2], b_in.shape[-1]))

        b_in = np.reshape(b_in, (np.prod(d), -1), order='F')
        if sparse_b:
            b_in = spr.csc_matrix(b_in)

        f_in = resize(np.atleast_2d(f_in), [b_in.shape[-1], T])

    if Ain.size > 0:
        Cin = resize(Cin, [K, T])
        center = np.asarray(
            [center_of_mass(a.reshape(d, order='F')) for a in Ain.T])
    else:
        Cin = np.empty((K, T), dtype=np.float32)
        center = []

    if normalize_init is True:
        if Ain.size > 0:
            Ain = Ain * np.reshape(img, (np.prod(d), -1), order='F')
        if sparse_b:
            b_in = spr.diags(img.ravel(order='F')).dot(b_in)
        else:
            b_in = b_in * np.reshape(img, (np.prod(d), -1), order='F')
    if method == 'corr_pnr' and ring_size_factor is not None:
        return scipy.sparse.csc_matrix(Ain), Cin, b_in, f_in, center, extra_1p
    else:
        return scipy.sparse.csc_matrix(Ain), Cin, b_in, f_in, center

def hals(Y, A, C, b, f, bSiz=3, maxIter=5):
    """ Hierarchical alternating least square method for solving NMF problem

    Y = A*C + b*f

    Args:
       Y:      d1 X d2 [X d3] X T, raw data.
           It will be reshaped to (d1*d2[*d3]) X T in this
           function

       A:      (d1*d2[*d3]) X K, initial value of spatial components

       C:      K X T, initial value of temporal components

       b:      (d1*d2[*d3]) X nb, initial value of background spatial component

       f:      nb X T, initial value of background temporal component

       bSiz:   int or tuple of int
        blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will
        be convolved with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0.

       maxIter: maximum iteration of iterating HALS.

    Returns:
        the updated A, C, b, f

    Authors:
        Johannes Friedrich, Andrea Giovannucci

    See Also:
        http://proceedings.mlr.press/v39/kimura14.pdf
    """

    # smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    nb = b.shape[1]  # number of background components
    if bSiz is not None:
        if isinstance(bSiz, (int, float)):
             bSiz = [bSiz] * len(dims)
        ind_A = nd.filters.uniform_filter(np.reshape(A,
                dims + (K,), order='F'), size=bSiz + [0])
        ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
    else:
        ind_A = A>1e-10

    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels

    def HALS4activity(Yr, A, C, iters=2):
        U = A.T.dot(Yr)
        V = A.T.dot(A) + np.finfo(A.dtype).eps
        for _ in range(iters):
            for m in range(len(U)):  # neurons and background
                C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                               V[m, m], 0, np.inf)
        return C

    def HALS4shape(Yr, A, C, iters=2):
        U = C.dot(Yr.T)
        V = C.dot(C.T) + np.finfo(C.dtype).eps
        for _ in range(iters):
            for m in range(K):  # neurons
                ind_pixels = np.squeeze(ind_A[:, m].toarray())
                A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                           ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                            V[m, m]), 0, np.inf)
            for m in range(nb):  # background
                A[:, K + m] = np.clip(A[:, K + m] + ((U[K + m] - V[K + m].dot(A.T)) /
                                                     V[K + m, K + m]), 0, np.inf)
        return A

    Ab = np.c_[A, b]
    Cf = np.r_[C, f.reshape(nb, -1)]
    for _ in range(maxIter):
        Cf = HALS4activity(np.reshape(
            Y, (np.prod(dims), T), order='F'), Ab, Cf)
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)

    return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)

def greedyROI_corr(Y, Y_ds, max_number=None, gSiz=None, gSig=None, center_psf=True,
                   min_corr=None, min_pnr=None, seed_method='auto',
                   min_pixel=3, bd=0, thresh_init=2, ring_size_factor=None, nb=1, options=None,
                   sn=None, save_video=False, video_name='initialization.mp4', ssub=1,
                   ssub_B=2, init_iter=2):
    """
    initialize neurons based on pixels' local correlations and peak-to-noise ratios.

    Args:
        *** see init_neurons_corr_pnr for descriptions of following input arguments ***
        data:
        max_number:
        gSiz:
        gSig:
        center_psf:
        min_corr:
        min_pnr:
        seed_method:
        min_pixel:
        bd:
        thresh_init:
        swap_dim:
        save_video:
        video_name:
        *** see init_neurons_corr_pnr for descriptions of above input arguments ***

        ring_size_factor: float
            it's the ratio between the ring radius and neuron diameters.
        ring_model: Boolean
            True indicates using ring model to estimate the background
            components.
        nb: integer
            number of background components for approximating the background using NMF model
            for nb=0 the exact background of the ringmodel (b0 and W) is returned
            for nb=-1 the full rank background B is returned
            for nb<-1 no background is returned
        ssub_B: int, optional
            downsampling factor for 1-photon imaging background computation
        init_iter: int, optional
            number of iterations for 1-photon imaging initialization
    """
    if min_corr is None or min_pnr is None:
        raise Exception(
            'Either min_corr or min_pnr are None. Both of them must be real numbers.')

    logging.info('One photon initialization (GreedyCorr)')
    o = options['temporal_params'].copy()
    o['s_min'] = None
    if o['p'] > 1:
        o['p'] = 1
    A, C, _, _, center = init_neurons_corr_pnr(
        Y_ds, max_number=max_number, gSiz=gSiz, gSig=gSig,
        center_psf=center_psf, min_corr=min_corr,
        min_pnr=min_pnr * np.sqrt(np.size(Y) / np.size(Y_ds)),
        seed_method=seed_method, deconvolve_options=o,
        min_pixel=min_pixel, bd=bd, thresh_init=thresh_init,
        swap_dim=True, save_video=save_video, video_name=video_name)

    dims = Y.shape[:2]
    T = Y.shape[-1]
    d1, d2, total_frames = Y_ds.shape
    tsub = int(round(float(T) / total_frames))
    B = Y_ds.reshape((-1, total_frames), order='F') - A.dot(C)

    if ring_size_factor is not None:
        # background according to ringmodel
        logging.info('Computing ring model background')
        W, b0 = CNMFE_2.compute_W(Y_ds.reshape((-1, total_frames), order='F'),
                          A, C, (d1, d2), ring_size_factor * gSiz, ssub=ssub_B)

        def compute_B(b0, W, B):  # actually computes -B to efficiently compute Y-B in place
            if ssub_B == 1:
                B = -b0[:, None] - W.dot(B - b0[:, None])  # "-B"
            else:
                B = -b0[:, None] - (np.repeat(np.repeat(W.dot(
                    downscale(B.reshape((d1, d2, B.shape[-1]), order='F'),
                              (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F') -
                    downscale(b0.reshape((d1, d2), order='F'),
                              (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
                    .reshape(((d1 - 1) // ssub_B + 1, (d2 - 1) // ssub_B + 1, -1), order='F'),
                    ssub_B, 0), ssub_B, 1)[:d1, :d2].reshape((-1, B.shape[-1]), order='F'))  # "-B"
            return B

        B = compute_B(b0, W, B)  # "-B"
        B += Y_ds.reshape((-1, total_frames), order='F')  # "Y-B"

        logging.info('Updating spatial components')
        A, _, C, _ = update_spatial_components(
            B, C=C, f=np.zeros((0, total_frames), np.float32), A_in=A,
            sn=np.sqrt(downscale((sn**2).reshape(dims, order='F'),
                                 tuple([ssub] * len(dims))).ravel() / tsub) / ssub,
            b_in=np.zeros((d1 * d2, 0), np.float32),
            dview=None, dims=(d1, d2), **options['spatial_params'])
        logging.info('Updating temporal components')
        C, A = update_temporal_components(
            B, spr.csc_matrix(A, dtype=np.float32),
            np.zeros((d1 * d2, 0), np.float32),
            C, np.zeros((0, total_frames), np.float32),
            dview=None, bl=None, c1=None, sn=None, g=None, **o)[:2]

        # find more neurons in residual
        # print('Compute Residuals')
        for i in range(init_iter - 1):
            if max_number is not None:
                max_number -= A.shape[-1]
            if max_number is not 0:
                if i == init_iter-2 and seed_method.lower()[:4] == 'semi':
                    seed_method, min_corr, min_pnr = 'manual', 0, 0
                logging.info('Searching for more neurons in the residual')
                A_R, C_R, _, _, center_R = init_neurons_corr_pnr(
                    (B - A.dot(C)).reshape(Y_ds.shape, order='F'),
                    max_number=max_number, gSiz=gSiz, gSig=gSig,
                    center_psf=center_psf, min_corr=min_corr, min_pnr=min_pnr,
                    seed_method=seed_method, deconvolve_options=o,
                    min_pixel=min_pixel, bd=bd, thresh_init=thresh_init,
                    swap_dim=True, save_video=save_video, video_name=video_name)
                A = spr.coo_matrix(np.concatenate((A.toarray(), A_R), 1))
                C = np.concatenate((C, C_R), 0)

        # 1st iteration on decimated data
        logging.info('Merging components')
        A, C = merge_components(
            B, A, [], C, None, [], C, [], o, options['spatial_params'],
            dview=None, thr=options['merging']['merge_thr'], mx=np.Inf, fast_merge=True)[:2]
        A = A.astype(np.float32)
        C = C.astype(np.float32)
        logging.info('Updating spatial components')
        A, _, C, _ = update_spatial_components(
            B, C=C, f=np.zeros((0, total_frames), np.float32), A_in=A,
            sn=np.sqrt(downscale((sn**2).reshape(dims, order='F'),
                                 tuple([ssub] * len(dims))).ravel() / tsub) / ssub,
            b_in=np.zeros((d1 * d2, 0), np.float32),
            dview=None, dims=(d1, d2), **options['spatial_params'])
        A = A.astype(np.float32)
        logging.info('Updating temporal components')
        C, A = update_temporal_components(
            B, spr.csc_matrix(A),
            np.zeros((d1 * d2, 0), np.float32),
            C, np.zeros((0, total_frames), np.float32),
            dview=None, bl=None, c1=None, sn=None, g=None, **o)[:2]

        logging.info('Recomputing background')
        # background according to ringmodel
        W, b0 = CNMFE_2.compute_W(Y_ds.reshape((-1, total_frames), order='F'),
                          A, C, (d1, d2), ring_size_factor * gSiz, ssub=ssub_B)

        # 2nd iteration on non-decimated data
        K = C.shape[0]
        if T > total_frames:
            C = np.repeat(C, tsub, 1)[:, :T]
            Ys = (Y if ssub == 1 else downscale(
                Y, (ssub, ssub, 1))).reshape((-1, T), order='F')
            # N.B: upsampling B in space is fine, but upsampling in time doesn't work well,
            # cause the error in upsampled background can be of similar size as neural signal
            B = Ys - A.dot(C)
        else:
            B = Y_ds.reshape((-1, T), order='F') - A.dot(C)
        B = compute_B(b0, W, B)  # "-B"
        if nb > 0 or nb == -1:
            B0 = -B
        if ssub > 1:
            B = np.reshape(B, (d1, d2, -1), order='F')
            B = (np.repeat(np.repeat(B, ssub, 0), ssub, 1)[:dims[0], :dims[1]]
                 .reshape((-1, T), order='F'))
            A = A.toarray().reshape((d1, d2, K), order='F')
            A = spr.csc_matrix(np.repeat(np.repeat(A, ssub, 0), ssub, 1)[:dims[0], :dims[1]]
                               .reshape((np.prod(dims), K), order='F'))
        B += Y.reshape((-1, T), order='F')  # "Y-B"

        logging.info('Merging components')
        A, C = merge_components(
            B, A, [], C, None, [], C, [], o, options['spatial_params'],
            dview=None, thr=options['merging']['merge_thr'], mx=np.Inf, fast_merge=True)[:2]
        A = A.astype(np.float32)
        C = C.astype(np.float32)
        logging.info('Updating spatial components')
        options['spatial_params']['se'] = np.ones((1,) * len((d1, d2)), dtype=np.uint8)
        A, _, C, _ = update_spatial_components(
            B, C=C, f=np.zeros((0, T), np.float32), A_in=A, sn=sn,
            b_in=np.zeros((np.prod(dims), 0), np.float32),
            dview=None, dims=dims, **options['spatial_params'])
        logging.info('Updating temporal components')
        C, A, b__, f__, S, bl, c1, neurons_sn, g1, YrA, lam__ = \
            update_temporal_components(
                B, spr.csc_matrix(A, dtype=np.float32),
                np.zeros((np.prod(dims), 0), np.float32), C, np.zeros((0, T), np.float32),
                dview=None, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

        A = A.toarray()
        if nb > 0 or nb == -1:
            B = B0

    use_NMF = True
    if nb == -1:
        logging.info('Returning full background')
        b_in = spr.eye(len(B), dtype='float32')
        f_in = B
    elif nb > 0:
        logging.info('Estimate low rank background (rank = {0})'.format(nb))
        print(nb)
        if use_NMF:
            model = NMF(n_components=nb, init='nndsvdar')
            b_in = model.fit_transform(np.maximum(B, 0))
            # f_in = model.components_.squeeze()
            f_in = np.linalg.lstsq(b_in, B)[0]
        else:
            b_in, s_in, f_in = spr.linalg.svds(B, k=nb)
            f_in *= s_in[:, np.newaxis]
    else:
        b_in = np.empty((A.shape[0], 0))
        f_in = np.empty((0, T))
        if nb == 0:
            logging.info('Returning background as b0 and W')
            return (A, C, center.T, b_in.astype(np.float32), f_in.astype(np.float32),
                    (S.astype(np.float32), bl, c1, neurons_sn, g1, YrA, lam__,
                     W, b0))
        else:
            logging.info("Not returning background")
    return (A, C, center.T, b_in.astype(np.float32), f_in.astype(np.float32),
            None if ring_size_factor is None else
            (S.astype(np.float32), bl, c1, neurons_sn, g1, YrA, lam__))

def init_neurons_corr_pnr(data, max_number=None, gSiz=15, gSig=None,
                          center_psf=True, min_corr=0.8, min_pnr=10,
                          seed_method='auto', deconvolve_options=None,
                          min_pixel=3, bd=1, thresh_init=2, swap_dim=True,
                          save_video=False, video_name='initialization.mp4',
                          background_filter='disk'):
    """
    using greedy method to initialize neurons by selecting pixels with large
    local correlation and large peak-to-noise ratio
    Args:
        data: np.ndarray (3D)
            the data used for initializing neurons. its dimension can be
            d1*d2*T or T*d1*d2. If it's the latter, swap_dim should be
            False; otherwise, True.
        max_number: integer
            maximum number of neurons to be detected. If None, then the
            algorithm will stop when all pixels are below the thresholds.
        gSiz: float
            average diameter of a neuron
        gSig: float number or a vector with two elements.
            gaussian width of the gaussian kernel used for spatial filtering.
        center_psf: Boolean
            True indicates centering the filtering kernel for background
            removal. This is useful for data with large background
            fluctuations.
        min_corr: float
            minimum local correlation coefficients for selecting a seed pixel.
        min_pnr: float
            minimum peak-to-noise ratio for selecting a seed pixel.
        seed_method: str {'auto', 'manual'}
            methods for choosing seed pixels
            if running as notebook 'manual' requires a backend that does not
            inline figures, e.g. %matplotlib tk
        deconvolve_options: dict
            all options for deconvolving temporal traces.
        min_pixel: integer
            minimum number of nonzero pixels for one neuron.
        bd: integer
            pixels that are bd pixels away from the boundary will be ignored for initializing neurons.
        thresh_init: float
            pixel values smaller than thresh_init*noise will be set as 0
            when computing the local correlation image.
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab
            format)
        save_video: Boolean
            save the initialization procedure if it's True
        video_name: str
            name of the video to be saved.

    Returns:
        A: np.ndarray (d1*d2*T)
            spatial components of all neurons
        C: np.ndarray (K*T)
            nonnegative and denoised temporal components of all neurons
        C_raw: np.ndarray (K*T)
            raw calcium traces of all neurons
        S: np.ndarray (K*T)
            deconvolved calcium traces of all neurons
        center: np.ndarray
            center localtions of all neurons
    """

    if swap_dim:
        d1, d2, total_frames = data.shape
        data_raw = np.transpose(data, [2, 0, 1])
    else:
        total_frames, d1, d2 = data.shape
        data_raw = data

    data_filtered = data_raw.copy()
    if gSig:
        # spatially filter data
        if not isinstance(gSig, list):
            gSig = [gSig, gSig]
        ksize = tuple([int(2 * i) * 2 + 1 for i in gSig])
        # create a spatial filter for removing background

        if center_psf:
            if background_filter == 'box':
                for idx, img in enumerate(data_filtered):
                    data_filtered[idx, ] = cv2.GaussianBlur(
                        img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1) \
                        - cv2.boxFilter(img, ddepth=-1, ksize=ksize, borderType=1)
            else:
                psf = cv2.getGaussianKernel(ksize[0], gSig[0], cv2.CV_32F).dot(
                    cv2.getGaussianKernel(ksize[1], gSig[1], cv2.CV_32F).T)
                ind_nonzero = psf >= psf[0].max()
                psf -= psf[ind_nonzero].mean()
                psf[~ind_nonzero] = 0
                for idx, img in enumerate(data_filtered):
                    data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)
        else:
            for idx, img in enumerate(data_filtered):
                data_filtered[idx, ] = cv2.GaussianBlur(img, ksize=ksize, sigmaX=gSig[0],
                                                        sigmaY=gSig[1], borderType=1)

    # compute peak-to-noise ratio
    data_filtered -= data_filtered.mean(axis=0)
    data_max = np.max(data_filtered, axis=0)
    noise_pixel = get_noise_fft(data_filtered.T, noise_method='mean')[0].T
    pnr = np.divide(data_max, noise_pixel)

    # remove small values and only keep pixels with large fluorescence signals
    tmp_data = np.copy(data_filtered)
    tmp_data[tmp_data < thresh_init * noise_pixel] = 0
    # compute correlation image
    cn = local_correlations_fft(tmp_data, swap_dim=False)
    del(tmp_data)
#    cn[np.isnan(cn)] = 0  # remove abnormal pixels

    # make required copy here, after memory intensive computation of cn
    data_raw = data_raw.copy()

    # screen seed pixels as neuron centers
    v_search = cn * pnr
    v_search[(cn < min_corr) | (pnr < min_pnr)] = 0
    ind_search = (v_search <= 0)  # indicate whether the pixel has
    # been searched before. pixels with low correlations or low PNRs are
    # ignored directly. ind_search[i]=0 means the i-th pixel is still under
    # consideration of being a seed pixel

    # pixels near the boundaries are ignored because of artifacts
    ind_bd = np.zeros(shape=(d1, d2)).astype(
        np.bool)  # indicate boundary pixels
    if bd > 0:
        ind_bd[:bd, :] = True
        ind_bd[-bd:, :] = True
        ind_bd[:, :bd] = True
        ind_bd[:, -bd:] = True

    ind_search[ind_bd] = 1

    # creating variables for storing the results
    if not max_number:
        # maximum number of neurons
        max_number = np.int32((ind_search.size - ind_search.sum()) / 5)
    Ain = np.zeros(shape=(max_number, d1, d2),
                   dtype=np.float32)  # neuron shapes
    Cin = np.zeros(shape=(max_number, total_frames),
                   dtype=np.float32)  # de-noised traces
    Sin = np.zeros(shape=(max_number, total_frames),
                   dtype=np.float32)  # spiking # activity
    Cin_raw = np.zeros(shape=(max_number, total_frames),
                       dtype=np.float32)  # raw traces
    center = np.zeros(shape=(2, max_number))  # neuron centers

    num_neurons = 0  # number of initialized neurons
    continue_searching = max_number > 0
    min_v_search = min_corr * min_pnr
    [ii, jj] = np.meshgrid(range(d2), range(d1))
    pixel_v = ((ii * 10 + jj) * 1e-5).astype(np.float32)

    if save_video:
        # FFMpegWriter = animation.writers['ffmpeg']
        # metadata = dict(title='Initialization procedure', artist='CaImAn',
        #                 comment='CaImAn is cool!')
        # writer = FFMpegWriter(fps=2, metadata=metadata)
        # # visualize the initialization procedure.
        # fig = plt.figure(figsize=(12, 8), facecolor=(0.9, 0.9, 0.9))
        # # with writer.saving(fig, "initialization.mp4", 150):
        # writer.setup(fig, video_name, 150)
        #
        # ax_cn = plt.subplot2grid((2, 3), (0, 0))
        # ax_cn.imshow(cn)
        # ax_cn.set_title('Correlation')
        # ax_cn.set_axis_off()
        #
        # ax_pnr_cn = plt.subplot2grid((2, 3), (0, 1))
        # ax_pnr_cn.imshow(cn * pnr)
        # ax_pnr_cn.set_title('Correlation*PNR')
        # ax_pnr_cn.set_axis_off()
        #
        # ax_cn_box = plt.subplot2grid((2, 3), (0, 2))
        # ax_cn_box.imshow(cn)
        # ax_cn_box.set_xlim([54, 63])
        # ax_cn_box.set_ylim([54, 63])
        # ax_cn_box.set_title('Correlation')
        # ax_cn_box.set_axis_off()
        #
        # ax_traces = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        # ax_traces.set_title('Activity at the seed pixel')
        #
        # writer.grab_frame()
        print("**to_julia** probably not necessary")
        sys.exit(2)

    all_centers = []
    while continue_searching:
        if seed_method.lower() == 'manual':
            # manually pick seed pixels
            fig = plt.figure(figsize=(14,6))
            ax = plt.axes([.03, .05, .96, .22])
            sc_all = []
            sc_select = []
            for i in range(3):
                plt.axes([.01+.34*i, .3, .3, .61])
                sc_all.append(plt.scatter([],[], color='g'))
                sc_select.append(plt.scatter([],[], color='r'))
                title = ('corr*pnr', 'correlation (corr)', 'peak-noise-ratio (pnr)')[i]
                img = (v_search, cn, pnr)[i]
                plt.imshow(img, interpolation=None, vmin=np.percentile(img[~np.isnan(img)], 1),
                           vmax=np.percentile(img[~np.isnan(img)], 99), cmap='gray')
                if len(all_centers):
                    plt.scatter(*np.transpose(all_centers), c='b')
                plt.axis('off')
                plt.title(title)
            plt.suptitle('Click to add component. Click again on it to remove it. Press any key to update figure. Add more components, or press any key again when done.')
            centers = []

            def key_press(event):
                plt.close(fig)

            def onclick(event):
                new_center = int(round(event.xdata)), int(round(event.ydata))
                if new_center in centers:
                    centers.remove(new_center)
                else:
                    centers.append(new_center)
                print(centers)
                ax.clear()
                if len(centers):
                    ax.plot(data_filtered[:, centers[-1][1], centers[-1][0]], c='r')
                    for sc in sc_all:
                        sc.set_offsets(centers)
                    for sc in sc_select:
                        sc.set_offsets(centers[-1:])
                else:
                    for sc in sc_all:
                        sc.set_offsets(np.zeros((0,2)))
                    for sc in sc_select:
                        sc.set_offsets(np.zeros((0,2)))
                plt.draw()

            cid = fig.canvas.mpl_connect('key_press_event', key_press)
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show(block=True)

            if centers == []:
                break
            all_centers += centers
            csub_max, rsub_max = np.transpose(centers)
            tmp_kernel = np.ones(shape=tuple([int(round(gSiz / 4.))] * 2))
            v_max = cv2.dilate(v_search, tmp_kernel)
            local_max = v_max[rsub_max, csub_max]
            ind_local_max = local_max.argsort()[::-1]

        else:
            # local maximum, for identifying seed pixels in following steps
            v_search[(cn < min_corr) | (pnr < min_pnr)] = 0
            # add an extra value to avoid repeated seed pixels within one ROI.
            v_search = cv2.medianBlur(v_search, 3) + pixel_v
            v_search[ind_search] = 0
            tmp_kernel = np.ones(shape=tuple([int(round(gSiz / 4.))] * 2))
            v_max = cv2.dilate(v_search, tmp_kernel)

            # automatically select seed pixels as the local maximums
            v_max[(v_search != v_max) | (v_search < min_v_search)] = 0
            v_max[ind_search] = 0
            [rsub_max, csub_max] = v_max.nonzero()  # subscript of seed pixels
            local_max = v_max[rsub_max, csub_max]
            n_seeds = len(local_max)  # number of candidates
            if n_seeds == 0:
                # no more candidates for seed pixels
                break
            else:
                # order seed pixels according to their corr * pnr values
                ind_local_max = local_max.argsort()[::-1]
            img_vmax = np.median(local_max)

        # try to initialization neurons given all seed pixels
        for ith_seed, idx in enumerate(ind_local_max):
            r = rsub_max[idx]
            c = csub_max[idx]
            ind_search[r, c] = True  # this pixel won't be searched
            if v_search[r, c] < min_v_search:
                # skip this pixel if it's not sufficient for being a seed pixel
                continue

            # roughly check whether this is a good seed pixel
            # y0 = data_filtered[:, r, c]
            # if np.max(y0) < thresh_init * noise_pixel[r, c]:
            #     v_search[r, c] = 0
            #     continue
            y0 = np.diff(data_filtered[:, r, c])
            if y0.max() < 3 * y0.std():
                v_search[r, c] = 0
                continue

            # if Ain[:, r, c].sum() > 0 and np.max([scipy.stats.pearsonr(y0, cc)[0]
            #                                       for cc in Cin_raw[Ain[:, r, c] > 0]]) > .7:
            #     v_search[r, c] = 0
            #     continue

            # crop a small box for estimation of ai and ci
            r_min = max(0, r - gSiz)
            r_max = min(d1, r + gSiz + 1)
            c_min = max(0, c - gSiz)
            c_max = min(d2, c + gSiz + 1)
            nr = r_max - r_min
            nc = c_max - c_min
            patch_dims = (nr, nc)  # patch dimension
            data_raw_box = \
                data_raw[:, r_min:r_max, c_min:c_max].reshape(-1, nr * nc)
            data_filtered_box = \
                data_filtered[:, r_min:r_max, c_min:c_max].reshape(-1, nr * nc)
            # index of the seed pixel in the cropped box
            ind_ctr = np.ravel_multi_index((r - r_min, c - c_min),
                                           dims=(nr, nc))

            # neighbouring pixels to update after initializing one neuron
            r2_min = max(0, r - 2 * gSiz)
            r2_max = min(d1, r + 2 * gSiz + 1)
            c2_min = max(0, c - 2 * gSiz)
            c2_max = min(d2, c + 2 * gSiz + 1)

            if save_video:
                # ax_pnr_cn.cla()
                # ax_pnr_cn.imshow(v_search, vmin=0, vmax=img_vmax)
                # ax_pnr_cn.set_title('Neuron %d' % (num_neurons + 1))
                # ax_pnr_cn.set_axis_off()
                # ax_pnr_cn.plot(csub_max[ind_local_max[ith_seed:]], rsub_max[
                #     ind_local_max[ith_seed:]], '.r', ms=5)
                # ax_pnr_cn.plot(c, r, 'or', markerfacecolor='red')
                #
                # ax_cn_box.imshow(cn[r_min:r_max, c_min:c_max], vmin=0, vmax=1)
                # ax_cn_box.set_title('Correlation')
                #
                # ax_traces.cla()
                # ax_traces.plot(y0)
                # ax_traces.set_title('The fluo. trace at the seed pixel')
                #
                # writer.grab_frame()
                print("**to_julia** probably not necessary")

            [ai, ci_raw, ind_success] = extract_ac(data_filtered_box,
                                                   data_raw_box, ind_ctr, patch_dims)

            if (np.sum(ai > 0) < min_pixel) or (not ind_success):
                # bad initialization. discard and continue
                continue
            else:
                # cheers! good initialization.
                center[:, num_neurons] = [c, r]
                Ain[num_neurons, r_min:r_max, c_min:c_max] = ai
                Cin_raw[num_neurons] = ci_raw.squeeze()
                if deconvolve_options['p']:
                    # deconvolution
                    ci, baseline, c1, _, _, si, _ = \
                        CNMFE_2.constrained_foopsi(ci_raw, **deconvolve_options)
                    if ci.sum() == 0:
                        continue
                    Cin[num_neurons] = ci
                    Sin[num_neurons] = si
                else:
                    # no deconvolution
                    ci = ci_raw.copy()
                    ci[ci < 0] = 0
                    if ci.sum() == 0:
                        continue
                    Cin[num_neurons] = ci.squeeze()

                if save_video:
                    # # mark the seed pixel on the correlation image
                    # ax_cn.plot(c, r, '.r')
                    #
                    # ax_cn_box.cla()
                    # ax_cn_box.imshow(ai)
                    # ax_cn_box.set_title('Spatial component')
                    #
                    # ax_traces.cla()
                    # ax_traces.plot(ci_raw)
                    # ax_traces.plot(ci, 'r')
                    # ax_traces.set_title('Temporal component')
                    #
                    # writer.grab_frame()
                    print("**to_julia** probably not necessary")
                    sys.exit(2)

                # avoid searching nearby pixels
                ind_search[r_min:r_max, c_min:c_max] += (ai > ai.max() / 2)

                # remove the spatial-temporal activity of the initialized
                # and update correlation image & PNR image
                # update the raw data
                data_raw[:, r_min:r_max, c_min:c_max] -= \
                    ai[np.newaxis, ...] * ci[..., np.newaxis, np.newaxis]

                if gSig:
                    # spatially filter the neuron shape
                    tmp_img = Ain[num_neurons, r2_min:r2_max, c2_min:c2_max]
                    if center_psf:
                        if background_filter == 'box':
                            ai_filtered = cv2.GaussianBlur(tmp_img, ksize=ksize, sigmaX=gSig[0],
                                                           sigmaY=gSig[1], borderType=1) \
                                - cv2.boxFilter(tmp_img, ddepth=-1, ksize=ksize, borderType=1)
                        else:
                            ai_filtered = cv2.filter2D(tmp_img, -1, psf, borderType=1)
                    else:
                        ai_filtered = cv2.GaussianBlur(tmp_img, ksize=ksize, sigmaX=gSig[0],
                                                       sigmaY=gSig[1], borderType=1)
                    # update the filtered data
                    data_filtered[:, r2_min:r2_max, c2_min:c2_max] -= \
                        ai_filtered[np.newaxis, ...] * ci[..., np.newaxis, np.newaxis]
                    data_filtered_box = data_filtered[:, r2_min:r2_max, c2_min:c2_max].copy()
                else:
                    data_filtered_box = data_raw[:, r2_min:r2_max, c2_min:c2_max].copy()

                # update PNR image
                # data_filtered_box -= data_filtered_box.mean(axis=0)
                max_box = np.max(data_filtered_box, axis=0)
                noise_box = noise_pixel[r2_min:r2_max, c2_min:c2_max]
                pnr_box = np.divide(max_box, noise_box)
                pnr[r2_min:r2_max, c2_min:c2_max] = pnr_box
                pnr_box[pnr_box < min_pnr] = 0

                # update correlation image
                data_filtered_box[data_filtered_box <
                                  thresh_init * noise_box] = 0
                cn_box = local_correlations_fft(
                    data_filtered_box, swap_dim=False)
                cn_box[np.isnan(cn_box) | (cn_box < 0)] = 0
                cn[r_min:r_max, c_min:c_max] = cn_box[
                    (r_min - r2_min):(r_max - r2_min), (c_min - c2_min):(c_max - c2_min)]
                cn_box[cn_box < min_corr] = 0
                cn_box = cn[r2_min:r2_max, c2_min:c2_max]

                # update v_search
                v_search[r2_min:r2_max, c2_min:c2_max] = cn_box * pnr_box
                v_search[ind_search] = 0
                # avoid searching nearby pixels
                # v_search[r_min:r_max, c_min:c_max] *= (ai < np.max(ai) / 2.)

                # increase the number of detected neurons
                num_neurons += 1  #
                if num_neurons == max_number:
                    continue_searching = False
                    break
                else:
                    if num_neurons % 100 == 1:
                        logging.info('{0} neurons have been initialized'.format(num_neurons - 1))

    logging.info('In total, {0} neurons were initialized.'.format(num_neurons))
    # A = np.reshape(Ain[:num_neurons], (-1, d1 * d2)).transpose()
    A = np.reshape(Ain[:num_neurons], (-1, d1 * d2), order='F').transpose()
    C = Cin[:num_neurons]
    C_raw = Cin_raw[:num_neurons]
    S = Sin[:num_neurons]
    center = center[:, :num_neurons]

    if save_video:
        # plt.close()
        # writer.finish()
        print("**to_julia** probably not necessary")
        sys.exit(2)

    return A, C, C_raw, S, center

def extract_ac(data_filtered, data_raw, ind_ctr, patch_dims):
    # parameters
    min_corr_neuron = 0.9  # 7
    max_corr_bg = 0.3
    data_filtered = data_filtered.copy()

    # compute the temporal correlation between each pixel and the seed pixel
    data_filtered -= data_filtered.mean(axis=0)  # data centering
    tmp_std = np.sqrt(np.sum(data_filtered ** 2, axis=0))  # data
    # normalization
    tmp_std[tmp_std == 0] = 1
    data_filtered /= tmp_std
    y0 = data_filtered[:, ind_ctr]  # fluorescence trace at the center
    tmp_corr = np.dot(y0.reshape(1, -1), data_filtered)  # corr. coeff. with y0
    # pixels in the central area of neuron
    ind_neuron = (tmp_corr > min_corr_neuron).squeeze()
    # pixels outside of neuron's ROI
    ind_bg = (tmp_corr < max_corr_bg).squeeze()

    # extract temporal activity
    ci = np.mean(data_filtered[:, ind_neuron], axis=1)
    # initialize temporal activity of the neural
    if ci.dot(ci) == 0:  # avoid empty results
        return None, None, False

    # roughly estimate the background fluctuation
    y_bg = np.median(data_raw[:, ind_bg], axis=1).reshape(-1, 1)\
        if np.any(ind_bg) else np.ones((len(ci), 1), np.float32)
    # extract spatial components
    X = np.concatenate([ci.reshape(-1, 1), y_bg, np.ones(y_bg.shape, np.float32)], 1)
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, data_raw)
    try:
        #ai = np.linalg.inv(XX).dot(Xy)[0]
        # ai = np.linalg.solve(XX, Xy)[0]
        ai = pd_solve(XX, Xy)[0]
    except:
        ai = scipy.linalg.lstsq(XX, Xy)[0][0]
    ai = ai.reshape(patch_dims)
    ai[ai < 0] = 0

    # post-process neuron shape
    ai = circular_constraint(ai)
    ai = connectivity_constraint(ai)

    # remove baseline
    # ci -= np.median(ci)
    sn = get_noise_welch(ci)
    y_diff = np.concatenate([[-1], np.diff(ci)])
    b = np.median(ci[(y_diff >= 0) * (y_diff < sn)])
    ci -= b

    # return results
    return ai, ci, True

def get_noise_welch(Y, noise_range=[0.25, 0.5], noise_method='logmexp',
                    max_num_samples_fft=3072):
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Args:
        Y: np.ndarray
            Input movie data with time in the last axis

        noise_range: np.ndarray [2 x 1] between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
            default: [0.25,0.5]

        noise method: string
            method of averaging the noise.
            Choices:
                'mean': Mean
                'median': Median
                'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        sn: np.ndarray
            Noise level for each pixel
    """
    T = Y.shape[-1]
    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
                            Y[..., np.int(T // 2 - max_num_samples_fft / 3 / 2):
                            np.int(T // 2 + max_num_samples_fft / 3 / 2)],
                            Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = np.shape(Y)[-1]
    ff, Pxx = scipy.signal.welch(Y)
    Pxx = Pxx[..., (ff >= noise_range[0]) & (ff <= noise_range[1])]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx, -1) / 2),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx, -1) / 2),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx / 2), -1)))
    }[noise_method](Pxx)
    return sn

def connectivity_constraint(img_original, thr=.01, sz=5):
    """remove small nonzero pixels and disconnected components"""
    img = img_original.copy()
    ai_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((sz, sz), np.uint8))
    tmp = ai_open > img.max() * thr
    l, _ = label(tmp)
    img[l != l.ravel()[np.argmax(img)]] = 0
    return img

def circular_constraint(img_original):
    img = img_original.copy()
    nr, nc = img.shape
    [rsub, csub] = img.nonzero()
    if len(rsub) == 0:
        return img

    rmin = np.min(rsub)
    rmax = np.max(rsub) + 1
    cmin = np.min(csub)
    cmax = np.max(csub) + 1

    if (rmax - rmin) * (cmax - cmin) <= 1:
        return img

    if rmin == 0 and rmax == nr and cmin == 0 and cmax == nc:
        ind_max = np.argmax(img)
        y0, x0 = np.unravel_index(ind_max, [nr, nc])
        vmax = img[y0, x0]
        x, y = np.meshgrid(np.arange(nc), np.arange(nr))
        try:
            fy, fx = np.gradient(img)
        except ValueError:
            f = np.gradient(img.ravel()).reshape(nr, nc)
            (fy, fx) = (f, 0) if nc == 1 else (0, f)
        ind = ((fx * (x0 - x) + fy * (y0 - y) < 0) & (img < vmax / 3))
        img[ind] = 0

        # # remove isolated pixels
        l, _ = label(img)
        ind = binary_dilation(l == l[y0, x0])
        img[~ind] = 0
    else:
        tmp_img = circular_constraint(img[rmin:rmax, cmin:cmax])
        img[rmin:rmax, cmin:cmax] = tmp_img

    return img

def pd_solve(a, b):
    """ Fast matrix solve for positive definite matrix a"""
    L, info = dpotrf(a)
    if info == 0:
        return dpotrs(L, b)[0]
    else:
        return np.linalg.solve(a, b)

def get_noise_fft(Y, noise_range=[0.25, 0.5], noise_method='logmexp', max_num_samples_fft=3072,
                  opencv=True):
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Args:
        Y: np.ndarray
            Input movie data with time in the last axis

        noise_range: np.ndarray [2 x 1] between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
            default: [0.25,0.5]

        noise method: string
            method of averaging the noise.
            Choices:
                'mean': Mean
                'median': Median
                'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        sn: np.ndarray
            Noise level for each pixel
    """
    T = Y.shape[-1]
    # Y=np.array(Y,dtype=np.float64)

    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
                            Y[..., np.int(T // 2 - max_num_samples_fft / 3 / 2)
                                          :np.int(T // 2 + max_num_samples_fft / 3 / 2)],
                            Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = np.shape(Y)[-1]

    # we create a map of what is the noise on the FFT space
    ff = np.arange(0, 0.5 + 1. / T, 1. / T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1, ind2)
    # we compute the mean of the noise spectral density s
    if Y.ndim > 1:
        if opencv:
            import cv2
            try:
                cv2.setNumThreads(0)
            except:
                pass
            psdx_list = []
            for y in Y.reshape(-1, T):
                dft = cv2.dft(y, flags=cv2.DFT_COMPLEX_OUTPUT).squeeze()[
                    :len(ind)][ind]
                psdx_list.append(np.sum(1. / T * dft * dft, 1))
            psdx = np.reshape(psdx_list, Y.shape[:-1] + (-1,))
        else:
            xdft = np.fft.rfft(Y, axis=-1)
            xdft = xdft[..., ind[:xdft.shape[-1]]]
            psdx = 1. / T * abs(xdft)**2
        psdx *= 2
        sn = mean_psd(psdx, method=noise_method)

    else:
        xdft = np.fliplr(np.fft.rfft(Y))
        psdx = 1. / T * (xdft**2)
        psdx[1:] *= 2
        sn = mean_psd(psdx[ind[:psdx.shape[0]]], method=noise_method)

    return sn, psdx

def mean_psd(y, method='logmexp'):
    """
    Averaging the PSD

    Args:
        y: np.ndarray
             PSD values

        method: string
            method of averaging the noise.
            Choices:
             'mean': Mean
             'median': Median
             'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
        mp: array
            mean psd
    """

    if method == 'mean':
        mp = np.sqrt(np.mean(old_div(y, 2), axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(old_div(y, 2), axis=-1))
    else:
        mp = np.log(old_div((y + 1e-10), 2))
        mp = np.mean(mp, axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)

    return mp

def local_correlations_fft(Y,
                           eight_neighbours: bool = True,
                           swap_dim: bool = True,
                           opencv: bool = True,
                           rolling_window=None) -> np.ndarray:
    """Computes the correlation image for the input dataset Y using a faster FFT based method

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format

        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively

        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

        opencv: Boolean
            If True process using open cv method

        rolling_window: (undocumented)

    Returns:
        Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    if rolling_window is None:
        Y -= np.mean(Y, axis=0)
        Ystd = np.std(Y, axis=0)
        Ystd[Ystd == 0] = np.inf
        Y /= Ystd
    else:
        Ysum = np.cumsum(Y, axis=0)
        Yrm = (Ysum[rolling_window:] - Ysum[:-rolling_window]) / rolling_window
        Y[:rolling_window] -= Yrm[0]
        Y[rolling_window:] -= Yrm
        del Yrm, Ysum
        Ystd = np.cumsum(Y**2, axis=0)
        Yrst = np.sqrt((Ystd[rolling_window:] - Ystd[:-rolling_window]) / rolling_window)
        Yrst[Yrst == 0] = np.inf
        Y[:rolling_window] /= Yrst[0]
        Y[rolling_window:] /= Yrst
        del Ystd, Yrst

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            # yapf: disable
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                          dtype='float32')
            # yapf: enable
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = np.stack([cv2.filter2D(img, -1, sz, borderType=0) for img in Y])
        MASK = cv2.filter2D(np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = convolve(np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')

    YYconv = Yconv * Y
    del Y, Yconv
    if rolling_window is None:
        Cn = np.mean(YYconv, axis=0) / MASK
    else:
        YYconv_cs = np.cumsum(YYconv, axis=0)
        del YYconv
        YYconv_rm = (YYconv_cs[rolling_window:] - YYconv_cs[:-rolling_window]) / rolling_window
        del YYconv_cs
        Cn = YYconv_rm / MASK

    return Cn

def compressedNMF(Y_ds, nr, r_ov=10, max_iter_snmf=500,
                  sigma_smooth=(.5, .5, .5), remove_baseline=False,
                  perc_baseline=20, nb=1, truncate=2, tol=1e-3):
    m = scipy.ndimage.gaussian_filter(np.transpose(
            Y_ds, np.roll(np.arange(Y_ds.ndim), 1)), sigma=sigma_smooth,
            mode='nearest', truncate=truncate)
    if remove_baseline:
        logging.info('REMOVING BASELINE')
        bl = np.percentile(m, perc_baseline, axis=0)
        m = np.maximum(0, m - bl)
    else:
        logging.info('NOT REMOVING BASELINE')
        bl = np.zeros(m.shape[1:])

    T, dims = m.shape[0], m.shape[1:]
    d = np.prod(dims)
    yr = np.reshape(m, [T, d], order='F')
#    L = randomized_range_finder(yr, nr + r_ov, 3)
#    R = randomized_range_finder(yr.T, nr + r_ov, 3)
#    Yt = L.T.dot(yr).dot(R)
#    c_in, a_in = compressive_nmf(Yt, L, R.T, nr)
#    C_in = L.dot(c_in)
#    A_in = a_in.dot(R.T)
#    A_in = A_in.T
#    C_in = C_in.T
    A, C, USV = nnsvd_init(yr, nr, r_ov=r_ov)
    W_r = np.random.randn(d, nr + r_ov)
    W_l = np.random.randn(T, nr + r_ov)
    US = USV[0]*USV[1]
    YYt = US.dot(USV[2].dot(USV[2].T)).dot(US.T)
#    YYt = yr.dot(yr.T)

    B = YYt.dot(YYt.dot(US.dot(USV[2].dot(W_r))))
    PC, _ = np.linalg.qr(B)

    B = USV[2].T.dot(US.T.dot(YYt.dot(YYt.dot(W_l))))
    PA, _ = np.linalg.qr(B)
#    mdl = NMF(n_components=nr, verbose=False, init='nndsvd', tol=1e-10,
#              max_iter=1)
#    C = mdl.fit_transform(yr).T
#    A = mdl.components_.T

    yrPA = yr.dot(PA)
    yrPC = PC.T.dot(yr)
    for it in range(max_iter_snmf):

        C__ = C.copy()
        A__ = A.copy()
        C_ = C.dot(PC)
        A_ = PA.T.dot(A)

        C = C*(yrPA.dot(A_)/(C.T.dot(A_.T.dot(A_))+np.finfo(C.dtype).eps)).T
        A = A*(yrPC.T.dot(C_.T))/(A.dot(C_.dot(C_.T)) +  np.finfo(C.dtype).eps)
        nA = np.sqrt((A**2).sum(0))
        A /= nA
        C *= nA[:, np.newaxis]
        if (np.linalg.norm(C - C__)/np.linalg.norm(C__) < tol) & (np.linalg.norm(A - A__)/np.linalg.norm(A__) < tol):
            logging.info('Graph NMF converged after {} iterations'.format(it+1))
            break
    A_in = A
    C_in = C

    m1 = yr.T - A_in.dot(C_in) + np.maximum(0, bl.flatten(order='F'))[:, np.newaxis]
    model = NMF(n_components=nb, init='random',
                random_state=0, max_iter=max_iter_snmf)
    b_in = model.fit_transform(np.maximum(m1, 0)).astype(np.float32)
    f_in = model.components_.astype(np.float32)
    center = com(A_in, *dims)

    return A_in, C_in, center, b_in, f_in

def com(A: np.ndarray, d1: int, d2: int, d3: Optional[int] = None) -> np.array:
    """Calculation of the center of mass for spatial components

     Args:
         A:   np.ndarray
              matrix of spatial components (d x K)

         d1:  int
              number of pixels in x-direction

         d2:  int
              number of pixels in y-direction

         d3:  int
              number of pixels in z-direction

     Returns:
         cm:  np.ndarray
              center of mass for spatial components (K x 2 or 3)
    """

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)

    if d3 is None:
        Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                          np.outer(np.arange(d2), np.ones(d1)).ravel()],
                         dtype=A.dtype)
    else:
        Coor = np.matrix([
            np.outer(np.ones(d3),
                     np.outer(np.ones(d2), np.arange(d1)).ravel()).ravel(),
            np.outer(np.ones(d3),
                     np.outer(np.arange(d2), np.ones(d1)).ravel()).ravel(),
            np.outer(np.arange(d3),
                     np.outer(np.ones(d2), np.ones(d1)).ravel()).ravel()
        ],
                         dtype=A.dtype)

    cm = (Coor * A / A.sum(axis=0)).T
    return np.array(cm)

def nnsvd_init(X, n_components, r_ov=10, eps=1e-6, random_state=42):
    # NNDSVD initialization from scikit learn package (modified)
    U, S, V = randomized_svd(X, n_components + r_ov, random_state=random_state)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    C = W.T
    A = H.T
    return A[:, 1:n_components], C[:n_components], (U, S, V) #


def com(A: np.ndarray, d1: int, d2: int, d3: Optional[int] = None) -> np.array:
    """Calculation of the center of mass for spatial components

     Args:
         A:   np.ndarray
              matrix of spatial components (d x K)

         d1:  int
              number of pixels in x-direction

         d2:  int
              number of pixels in y-direction

         d3:  int
              number of pixels in z-direction

     Returns:
         cm:  np.ndarray
              center of mass for spatial components (K x 2 or 3)
    """

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)

    if d3 is None:
        Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                          np.outer(np.arange(d2), np.ones(d1)).ravel()],
                         dtype=A.dtype)
    else:
        Coor = np.matrix([
            np.outer(np.ones(d3),
                     np.outer(np.ones(d2), np.arange(d1)).ravel()).ravel(),
            np.outer(np.ones(d3),
                     np.outer(np.arange(d2), np.ones(d1)).ravel()).ravel(),
            np.outer(np.arange(d3),
                     np.outer(np.ones(d2), np.ones(d1)).ravel()).ravel()
        ],
                         dtype=A.dtype)

    cm = (Coor * A / A.sum(axis=0)).T
    return np.array(cm)


def greedyROI(Y, nr=30, gSig=[5, 5], gSiz=[11, 11], nIter=5, kernel=None, nb=1,
              rolling_sum=False, rolling_length=100, seed_method='auto'):
    """
    Greedy initialization of spatial and temporal components using spatial Gaussian filtering

    Args:
        Y: np.array
            3d or 4d array of fluorescence data with time appearing in the last axis.

        nr: int
            number of components to be found

        gSig: scalar or list of integers
            standard deviation of Gaussian kernel along each axis

        gSiz: scalar or list of integers
            size of spatial component

        nIter: int
            number of iterations when refining estimates

        kernel: np.ndarray
            User specified kernel to be used, if present, instead of Gaussian (default None)

        nb: int
            Number of background components

        rolling_max: boolean
            Detect new components based on a rolling sum of pixel activity (default: True)

        rolling_length: int
            Length of rolling window (default: 100)

        seed_method: str {'auto', 'manual', 'semi'}
            methods for choosing seed pixels
            'semi' detects nr components automatically and allows to add more manually
            if running as notebook 'semi' and 'manual' require a backend that does not
            inline figures, e.g. %matplotlib tk

    Returns:
        A: np.array
            2d array of size (# of pixels) x nr with the spatial components. Each column is
            ordered columnwise (matlab format, order='F')

        C: np.array
            2d array of size nr X T with the temporal components

        center: np.array
            2d array of size nr x 2 [ or 3] with the components centroids

    Author:
        Eftychios A. Pnevmatikakis and Andrea Giovannucci based on a matlab implementation by Yuanjun Gao
            Simons Foundation, 2015

    See Also:
        http://www.cell.com/neuron/pdf/S0896-6273(15)01084-3.pdf


    """
    logging.info("Greedy initialization of spatial and temporal components using spatial Gaussian filtering")
    d = np.shape(Y)
    Y[np.isnan(Y)] = 0
    med = np.median(Y, axis=-1)
    Y = Y - med[..., np.newaxis]
    gHalf = np.array(gSiz) // 2
    gSiz = 2 * gHalf + 1
    # we initialize every values to zero
    if seed_method.lower() == 'manual':
        nr = 0
    A = np.zeros((np.prod(d[0:-1]), nr), dtype=np.float32)
    C = np.zeros((nr, d[-1]), dtype=np.float32)
    center = np.zeros((nr, Y.ndim - 1), dtype='uint16')

    rho = imblur(Y, sig=gSig, siz=gSiz, nDimBlur=Y.ndim - 1, kernel=kernel)
    if rolling_sum:
        logging.info('Using rolling sum for initialization (RollingGreedyROI)')
        rolling_filter = np.ones(
            (rolling_length), dtype=np.float32) / rolling_length
        rho_s = scipy.signal.lfilter(rolling_filter, 1., rho**2)
        v = np.amax(rho_s, axis=-1)
    else:
        logging.info('Using total sum for initialization (GreedyROI)')
        v = np.sum(rho**2, axis=-1)

    if seed_method.lower() != 'manual':
        for k in range(nr):
            # we take the highest value of the blurred total image and we define it as
            # the center of the neuron
            ind = np.argmax(v)
            ij = np.unravel_index(ind, d[0:-1])
            for c, i in enumerate(ij):
                center[k, c] = i

            # we define a squared size around it
            ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                     for c in range(len(ij))]
            # we create an array of it (fl like) and compute the trace like the pixel ij trough time
            dataTemp = np.array(
                Y[tuple([slice(*a) for a in ijSig])].copy(), dtype=np.float32)
            traceTemp = np.array(np.squeeze(rho[ij]), dtype=np.float32)

            coef, score = finetune(dataTemp, traceTemp, nIter=nIter)
            C[k, :] = np.squeeze(score)
            dataSig = coef[..., np.newaxis] * \
                score.reshape([1] * (Y.ndim - 1) + [-1])
            xySig = np.meshgrid(*[np.arange(s[0], s[1])
                                  for s in ijSig], indexing='xy')
            arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze()
                            for s in xySig], dtype=np.int)
            indices = np.ravel_multi_index(arr, d[0:-1], order='F')

            A[indices, k] = np.reshape(
                coef, (1, np.size(coef)), order='C').squeeze()
            Y[tuple([slice(*a) for a in ijSig])] -= dataSig.copy()
            if k < nr - 1 or seed_method.lower() != 'auto':
                Mod = [[np.maximum(ij[c] - 2 * gHalf[c], 0),
                        np.minimum(ij[c] + 2 * gHalf[c] + 1, d[c])] for c in range(len(ij))]
                ModLen = [m[1] - m[0] for m in Mod]
                Lag = [ijSig[c] - Mod[c][0] for c in range(len(ij))]
                dataTemp = np.zeros(ModLen)
                dataTemp[tuple([slice(*a) for a in Lag])] = coef
                dataTemp = imblur(dataTemp[..., np.newaxis],
                                  sig=gSig, siz=gSiz, kernel=kernel)
                temp = dataTemp * score.reshape([1] * (Y.ndim - 1) + [-1])
                rho[tuple([slice(*a) for a in Mod])] -= temp.copy()
                if rolling_sum:
                    rho_filt = scipy.signal.lfilter(
                        rolling_filter, 1., rho[tuple([slice(*a) for a in Mod])]**2)
                    v[tuple([slice(*a) for a in Mod])] = np.amax(rho_filt, axis=-1)
                else:
                    v[tuple([slice(*a) for a in Mod])] = \
                        np.sum(rho[tuple([slice(*a) for a in Mod])]**2, axis=-1)
        center = center.tolist()
    else:
        center = []

    if seed_method.lower() in ('manual', 'semi'):
        # manually pick seed pixels
        while True:
            fig = plt.figure(figsize=(13, 12))
            ax = plt.axes([.04, .04, .95, .18])
            sc_all = []
            sc_select = []
            plt.axes([0, .25, 1, .7])
            sc_all.append(plt.scatter([], [], color='g'))
            sc_select.append(plt.scatter([], [], color='r'))
            plt.imshow(v, interpolation=None, vmin=np.percentile(v[~np.isnan(v)], 1),
                       vmax=np.percentile(v[~np.isnan(v)], 99), cmap='gray')
            if len(center):
                plt.scatter(*np.transpose(center)[::-1], c='b')
            plt.axis('off')
            plt.suptitle(
                'Click to add component. Click again on it to remove it. Press any key to update figure. Add more components, or press any key again when done.')
            centers = []

            def key_press(event):
                plt.close(fig)

            def onclick(event):
                new_center = int(round(event.xdata)), int(round(event.ydata))
                if new_center in centers:
                    centers.remove(new_center)
                else:
                    centers.append(new_center)
                print(centers)
                ax.clear()
                if len(centers):
                    ax.plot(Y[centers[-1][1], centers[-1][0]], c='r')
                    for sc in sc_all:
                        sc.set_offsets(centers)
                    for sc in sc_select:
                        sc.set_offsets(centers[-1:])
                else:
                    for sc in sc_all:
                        sc.set_offsets(np.zeros((0,2)))
                    for sc in sc_select:
                        sc.set_offsets(np.zeros((0,2)))
                plt.draw()

            cid = fig.canvas.mpl_connect('key_press_event', key_press)
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show(block=True)

            if centers == []:
                break
            centers = np.array(centers)[:,::-1].tolist()
            center += centers

            # we initialize every values to zero
            A_ = np.zeros((np.prod(d[0:-1]), len(centers)), dtype=np.float32)
            C_ = np.zeros((len(centers), d[-1]), dtype=np.float32)
            for k, ij in enumerate(centers):
                # we define a squared size around it
                ijSig = [[np.maximum(ij[c] - gHalf[c], 0), np.minimum(ij[c] + gHalf[c] + 1, d[c])]
                         for c in range(len(ij))]
                # we create an array of it (fl like) and compute the trace like the pixel ij trough time
                dataTemp = np.array(
                    Y[tuple([slice(*a) for a in ijSig])].copy(), dtype=np.float32)
                traceTemp = np.array(np.squeeze(rho[tuple(ij)]), dtype=np.float32)

                coef, score = finetune(dataTemp, traceTemp, nIter=nIter)
                C_[k, :] = np.squeeze(score)
                dataSig = coef[..., np.newaxis] * \
                    score.reshape([1] * (Y.ndim - 1) + [-1])
                xySig = np.meshgrid(*[np.arange(s[0], s[1])
                                      for s in ijSig], indexing='xy')
                arr = np.array([np.reshape(s, (1, np.size(s)), order='F').squeeze()
                                for s in xySig], dtype=np.int)
                indices = np.ravel_multi_index(arr, d[0:-1], order='F')

                A_[indices, k] = np.reshape(
                    coef, (1, np.size(coef)), order='C').squeeze()
                Y[tuple([slice(*a) for a in ijSig])] -= dataSig.copy()

                Mod = [[np.maximum(ij[c] - 2 * gHalf[c], 0),
                        np.minimum(ij[c] + 2 * gHalf[c] + 1, d[c])] for c in range(len(ij))]
                ModLen = [m[1] - m[0] for m in Mod]
                Lag = [ijSig[c] - Mod[c][0] for c in range(len(ij))]
                dataTemp = np.zeros(ModLen)
                dataTemp[tuple([slice(*a) for a in Lag])] = coef
                dataTemp = imblur(dataTemp[..., np.newaxis],
                                  sig=gSig, siz=gSiz, kernel=kernel)
                temp = dataTemp * score.reshape([1] * (Y.ndim - 1) + [-1])
                rho[tuple([slice(*a) for a in Mod])] -= temp.copy()
                if rolling_sum:
                    rho_filt = scipy.signal.lfilter(
                        rolling_filter, 1., rho[tuple([slice(*a) for a in Mod])]**2)
                    v[tuple([slice(*a) for a in Mod])] = np.amax(rho_filt, axis=-1)
                else:
                    v[tuple([slice(*a) for a in Mod])] = \
                        np.sum(rho[tuple([slice(*a) for a in Mod])]**2, axis=-1)
            A = np.concatenate([A, A_], 1)
            C = np.concatenate([C, C_])

    res = np.reshape(Y, (np.prod(d[0:-1]), d[-1]),
                     order='F') + med.flatten(order='F')[:, None]
#    model = NMF(n_components=nb, init='random', random_state=0)
    model = NMF(n_components=nb, init='nndsvdar')
    b_in = model.fit_transform(np.maximum(res, 0)).astype(np.float32)
    f_in = model.components_.astype(np.float32)

    return A, C, np.array(center, dtype='uint16'), b_in, f_in

def finetune(Y, cin, nIter=5):
    """compute a initialized version of A and C

    Args:
        Y:  D1*d2*T*K patches

        c: array T*K
            the inital calcium traces

        nIter: int
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
    a: array (d1,D2) the computed A as l2(Y*C)/Y*C

    c: array(T) C as the sum of As on x*y axis
"""
    debug_ = False
    if debug_:
        import os
        f = open('_LOG_1_' + str(os.getpid()), 'w+')
        f.write('Y:' + str(np.mean(Y)) + '\n')
        f.write('cin:' + str(np.mean(cin)) + '\n')
        f.close()

    # we compute the multiplication of patches per traces ( non negatively )
    for _ in range(nIter):
        a = np.maximum(np.dot(Y, cin), 0)
        a = old_div(a, np.sqrt(np.sum(a**2)) +
                    np.finfo(np.float32).eps)  # compute the l2/a
        # c as the variation of thoses patches
        cin = np.sum(Y * a[..., np.newaxis], tuple(np.arange(Y.ndim - 1)))

    return a, cin

def imblur(Y, sig=5, siz=11, nDimBlur=None, kernel=None, opencv=True):
    """
    Spatial filtering with a Gaussian or user defined kernel

    The parameters are specified in GreedyROI

    Args:
        Y: np.ndarray
            d1 x d2 [x d3] x T movie, raw data.

        sig: [optional] list,tuple
            half size of neurons

        siz: [optional] list,tuple
            size of kernel (default 2*tau + 1).

        nDimBlur: [optional]
            if you want to specify the number of dimension

        kernel: [optional]
            if you want to specify a kernel

        opencv: [optional]
            if you want to process to the blur using open cv method

    Returns:
        the blurred image
    """
    # TODO: document (jerem)
    if kernel is None:
        if nDimBlur is None:
            nDimBlur = Y.ndim - 1
        else:
            nDimBlur = np.min((Y.ndim, nDimBlur))

        if np.isscalar(sig):
            sig = sig * np.ones(nDimBlur)

        if np.isscalar(siz):
            siz = siz * np.ones(nDimBlur)

        X = Y.copy()
        if opencv and nDimBlur == 2:
            if X.ndim > 2:
                # if we are on a video we repeat for each frame
                for frame in range(X.shape[-1]):
                    if sys.version_info >= (3, 0):
                        X[:, :, frame] = cv2.GaussianBlur(X[:, :, frame], tuple(
                            siz), sig[0], None, sig[1], cv2.BORDER_CONSTANT)
                    else:
                        X[:, :, frame] = cv2.GaussianBlur(X[:, :, frame], tuple(siz), sig[
                                                          0], sig[1], cv2.BORDER_CONSTANT, 0)

            else:
                if sys.version_info >= (3, 0):
                    X = cv2.GaussianBlur(
                        X, tuple(siz), sig[0], None, sig[1], cv2.BORDER_CONSTANT)
                else:
                    X = cv2.GaussianBlur(
                        X, tuple(siz), sig[0], sig[1], cv2.BORDER_CONSTANT, 0)
        else:
            for i in range(nDimBlur):
                h = np.exp(
                    old_div(-np.arange(-np.floor(old_div(siz[i], 2)),
                                       np.floor(old_div(siz[i], 2)) + 1)**2, (2 * sig[i]**2)))
                h /= np.sqrt(h.dot(h))
                shape = [1] * len(Y.shape)
                shape[i] = -1
                X = correlate(X, h.reshape(shape), mode='constant')

    else:
        X = correlate(Y, kernel[..., np.newaxis], mode='constant')
        # for t in range(np.shape(Y)[-1]):
        #    X[:,:,t] = correlate(Y[:,:,t],kernel,mode='constant', cval=0.0)

    return X

def downscale(Y, ds, opencv=False):
    """downscaling without zero padding
    faster version of skimage.transform._warps.block_reduce(Y, ds, np.nanmean, np.nan)"""

    from caiman.base.movies import movie
    d = Y.ndim
    if opencv and (d in [2, 3]):
        if d == 2:
            Y = Y[..., None]
            ds = tuple(ds) + (1,)
        else:
            Y_ds = movie(Y.transpose(2, 0, 1)).resize(fx=1. / ds[0], fy=1. / ds[1], fz=1. / ds[2],
                                                      interpolation=cv2.INTER_AREA).transpose(1, 2, 0)
        logging.info('Downscaling using OpenCV')
    else:
        if d > 3:
            # raise NotImplementedError
            # slower and more memory intensive version using skimage
            from skimage.transform._warps import block_reduce
            return block_reduce(Y, ds, np.nanmean, np.nan)
        elif d == 1:
            return decimate_last_axis(Y, ds)
        elif d == 2:
            Y = Y[..., None]
            ds = tuple(ds) + (1,)

        if d == 3 and Y.shape[-1] > 1 and ds[0] == ds[1]:
            ds_mat = CNMFE_1.decimation_matrix(Y.shape[:2], ds[0])
            Y_ds = ds_mat.dot(Y.reshape((-1, Y.shape[-1]), order='F')).reshape(
                (1 + (Y.shape[0] - 1) // ds[0], 1 + (Y.shape[1] - 1) // ds[0], -1), order='F')
            if ds[2] > 1:
                Y_ds = decimate_last_axis(Y_ds, ds[2])
        else:
            q = np.array(Y.shape) // np.array(ds)
            r = np.array(Y.shape) % np.array(ds)
            s = q * np.array(ds)
            Y_ds = np.zeros(q + (r > 0), dtype=Y.dtype)
            Y_ds[:q[0], :q[1], :q[2]] = (Y[:s[0], :s[1], :s[2]]
                                         .reshape(q[0], ds[0], q[1], ds[1], q[2], ds[2])
                                         .mean(1).mean(2).mean(3))
            if r[0]:
                Y_ds[-1, :q[1], :q[2]] = (Y[-r[0]:, :s[1], :s[2]]
                                          .reshape(r[0], q[1], ds[1], q[2], ds[2])
                                          .mean(0).mean(1).mean(2))
                if r[1]:
                    Y_ds[-1, -1, :q[2]] = (Y[-r[0]:, -r[1]:, :s[2]]
                                           .reshape(r[0], r[1], q[2], ds[2])
                                           .mean(0).mean(0).mean(1))
                    if r[2]:
                        Y_ds[-1, -1, -1] = Y[-r[0]:, -r[1]:, -r[2]:].mean()
                if r[2]:
                    Y_ds[-1, :q[1], -1] = (Y[-r[0]:, :s[1]:, -r[2]:]
                                           .reshape(r[0], q[1], ds[1], r[2])
                                           .mean(0).mean(1).mean(1))
            if r[1]:
                Y_ds[:q[0], -1, :q[2]] = (Y[:s[0], -r[1]:, :s[2]]
                                          .reshape(q[0], ds[0], r[1], q[2], ds[2])
                                          .mean(1).mean(1).mean(2))
                if r[2]:
                    Y_ds[:q[0], -1, -1] = (Y[:s[0]:, -r[1]:, -r[2]:]
                                           .reshape(q[0], ds[0], r[1], r[2])
                                           .mean(1).mean(1).mean(1))
            if r[2]:
                Y_ds[:q[0], :q[1], -1] = (Y[:s[0], :s[1], -r[2]:]
                                          .reshape(q[0], ds[0], q[1], ds[1], r[2])
                                          .mean(1).mean(2).mean(2))
    return Y_ds if d == 3 else Y_ds[:, :, 0]

def decimate_last_axis(y, sub):
    q = y.shape[-1] // sub
    r = y.shape[-1] % sub
    Y_ds = np.zeros(y.shape[:-1] + (q + (r > 0),), dtype=y.dtype)
    Y_ds[..., :q] = y[..., :q * sub].reshape(y.shape[:-1] + (-1, sub)).mean(-1)
    if r > 0:
        Y_ds[..., -1] = y[..., -r:].mean(-1)
    return Y_ds


class Estimates(object):
    """
    Class for storing and reusing the analysis results and performing basic
    processing and plotting operations.
    """
    def __init__(self, A=None, b=None, C=None, f=None, R=None, dims=None):
        """Class for storing the variables related to the estimates of spatial footprints, temporal traces,
        deconvolved neural activity, and background. Quality metrics are also stored. The class has methods
        for evaluating the quality of each component, DF/F normalization and some basic plotting.

        Args:
            A:  scipy.sparse.csc_matrix (dimensions: # of pixels x # components)
                set of spatial footprints. Each footprint is represented in a column of A, flattened with order = 'F'

            C:  np.ndarray (dimensions: # of components x # of timesteps)
                set of temporal traces (each row of C corresponds to a trace)

            f:  np.ndarray (dimensions: # of background components x # of timesteps)
                set of temporal background components

            b:  np.ndarray or scipy.sparse.csc_matrix (dimensions: # of pixels x # of background components)
                set of spatial background components, flattened with order = 'F'

            R:  np.ndarray (dimensions: # of components x # of timesteps)
                set of trace residuals

            YrA:    np.ndarray (dimensions: # of components x # of timesteps)
                set of trace residuals

            S:  np.ndarray (dimensions: # of components x # of timesteps)
                set of deconvolved neural activity traces

            F_dff:  np.ndarray (dimensions: # of components x # of timesteps)
                set of DF/F normalized activity traces (only for 2p)

            W:  scipy.sparse.coo_matrix (dimensions: # of pixels x # of pixels)
                Ring model matrix (used in 1p processing with greedy_pnr for background computation)

            b0: np.ndarray (dimensions: # of pixels)
                constant baseline for each pixel

            sn: np.ndarray (dimensions: # of pixels)
                noise std for each pixel

            g:  list (length: # of components)
                time constants for each trace

            bl: list (length: # of components)
                constant baseline for each trace

            c1: list (length: # of components)
                initial value for each trace

            neurons_sn: list (length: # of components)
                noise std for each trace

            center: list (length: # of components)
                centroid coordinate for each spatial footprint

            coordinates: list (length: # of components)
                contour plot for each spatial footprint

            idx_components: list
                indices of accepted components

            idx_components_bad: list
                indices of rejected components

            SNR_comp: np.ndarray
                trace SNR for each component

            r_values: np.ndarray
                space correlation for each component

            cnn_preds: np.ndarray
                CNN predictions for each component

            ecc: np.ndarray
                eccentricity values
        """
        # variables related to the estimates of traces, footprints, deconvolution and background
        self.A = A
        self.C = C
        self.f = f
        self.b = b
        self.R = R
        self.W = None
        self.b0 = None
        self.YrA = None

        self.S = None
        self.sn = None
        self.g = None
        self.bl = None
        self.c1 = None
        self.neurons_sn = None
        self.lam = None

        self.center = None

        self.merged_ROIs = None
        self.coordinates = None
        self.F_dff = None

        self.idx_components = None
        self.idx_components_bad = None
        self.SNR_comp = None
        self.r_values = None
        self.cnn_preds = None
        self.ecc = None

        # online

        self.noisyC = None
        self.C_on = None
        self.Ab = None
        self.Cf = None
        self.OASISinstances = None
        self.CY = None
        self.CC = None
        self.Ab_dense = None
        self.Yr_buf = None
        self.mn = None
        self.vr = None
        self.ind_new = None
        self.rho_buf = None
        self.AtA = None
        self.AtY_buf = None
        self.sv = None
        self.groups = None

        self.dims = dims
        self.shifts:List = []

        self.A_thr = None
        self.discarded_components = None



    def plot_contours(self, img=None, idx=None, thr_method='max',
                      thr=0.2, display_numbers=True, params=None,
                      cmap='viridis'):
        """view contours of all spatial footprints.

        Args:
            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)
            idx :   list
                list of accepted components
            thr_method : str
                thresholding method for computing contours ('max', 'nrg')
                if list of coordinates self.coordinates is None, i.e. not already computed
            thr : float
                threshold value
                only effective if self.coordinates is None, i.e. not already computed
            display_numbers :   bool
                flag for displaying the id number of each contour
            params : params object
                set of dictionary containing the various parameters
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if img is None:
            img = np.reshape(np.array(self.A.mean(1)), self.dims, order='F')
        if self.coordinates is None:  # not hasattr(self, 'coordinates'):
            self.coordinates = caiman.utils.visualization.get_contours(self.A, img.shape, thr=thr, thr_method=thr_method)
        plt.figure()
        if params is not None:
            plt.suptitle('min_SNR=%1.2f, rval_thr=%1.2f, use_cnn=%i'
                         %(params.quality['min_SNR'],
                           params.quality['rval_thr'],
                           int(params.quality['use_cnn'])))
        if idx is None:
            caiman.utils.visualization.plot_contours(self.A, img, coordinates=self.coordinates,
                                                     display_numbers=display_numbers,
                                                     cmap=cmap)
        else:
            if not isinstance(idx, list):
                idx = idx.tolist()
            coor_g = [self.coordinates[cr] for cr in idx]
            bad = list(set(range(self.A.shape[1])) - set(idx))
            coor_b = [self.coordinates[cr] for cr in bad]
            plt.subplot(1, 2, 1)
            caiman.utils.visualization.plot_contours(self.A[:, idx], img,
                                                     coordinates=coor_g,
                                                     display_numbers=display_numbers,
                                                     cmap=cmap)
            plt.title('Accepted Components')
            bad = list(set(range(self.A.shape[1])) - set(idx))
            plt.subplot(1, 2, 2)
            caiman.utils.visualization.plot_contours(self.A[:, bad], img,
                                                     coordinates=coor_b,
                                                     display_numbers=display_numbers,
                                                     cmap=cmap)
            plt.title('Rejected Components')
        return self

    def plot_contours_nb(self, img=None, idx=None, thr_method='max',
                         thr=0.2, params=None, line_color='white', cmap='viridis'):
        """view contours of all spatial footprints (notebook environment).

        Args:
            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)
            idx :   list
                list of accepted components
            thr_method : str
                thresholding method for computing contours ('max', 'nrg')
                if list of coordinates self.coordinates is None, i.e. not already computed
            thr : float
                threshold value
                only effective if self.coordinates is None, i.e. not already computed
            params : params object
                set of dictionary containing the various parameters
        """
        try:
            import bokeh
            if 'csc_matrix' not in str(type(self.A)):
                self.A = scipy.sparse.csc_matrix(self.A)
            if self.dims is None:
                self.dims = img.shape
            if img is None:
                img = np.reshape(np.array(self.A.mean(1)), self.dims, order='F')
            if self.coordinates is None:  # not hasattr(self, 'coordinates'):
                self.coordinates = caiman.utils.visualization.get_contours(self.A,
                                        self.dims, thr=thr, thr_method=thr_method)
            if idx is None:
                p = caiman.utils.visualization.nb_plot_contour(img, self.A, self.dims[0],
                                self.dims[1], coordinates=self.coordinates,
                                thr_method=thr_method, thr=thr, show=False,
                                line_color=line_color, cmap=cmap)
                p.title.text = 'Contour plots of found components'
                if params is not None:
                    p.xaxis.axis_label = '''\
                    min_SNR={min_SNR}, rval_thr={rval_thr}, use_cnn={use_cnn}\
                    '''.format(min_SNR=params.quality['min_SNR'],
                               rval_thr=params.quality['rval_thr'],
                               use_cnn=params.quality['use_cnn'])
                bokeh.plotting.show(p)
            else:
                if not isinstance(idx, list):
                    idx = idx.tolist()
                coor_g = [self.coordinates[cr] for cr in idx]
                bad = list(set(range(self.A.shape[1])) - set(idx))
                coor_b = [self.coordinates[cr] for cr in bad]
                p1 = caiman.utils.visualization.nb_plot_contour(img, self.A[:, idx],
                                self.dims[0], self.dims[1], coordinates=coor_g,
                                thr_method=thr_method, thr=thr, show=False,
                                line_color=line_color, cmap=cmap)
                p1.plot_width = 450
                p1.plot_height = 450 * self.dims[0] // self.dims[1]
                p1.title.text = "Accepted Components"
                if params is not None:
                    p1.xaxis.axis_label = '''\
                    min_SNR={min_SNR}, rval_thr={rval_thr}, use_cnn={use_cnn}\
                    '''.format(min_SNR=params.quality['min_SNR'],
                               rval_thr=params.quality['rval_thr'],
                               use_cnn=params.quality['use_cnn'])
                bad = list(set(range(self.A.shape[1])) - set(idx))
                p2 = caiman.utils.visualization.nb_plot_contour(img, self.A[:, bad],
                                self.dims[0], self.dims[1], coordinates=coor_b,
                                thr_method=thr_method, thr=thr, show=False,
                                line_color=line_color, cmap=cmap)
                p2.plot_width = 450
                p2.plot_height = 450 * self.dims[0] // self.dims[1]
                p2.title.text = 'Rejected Components'
                if params is not None:
                    p2.xaxis.axis_label = '''\
                    min_SNR={min_SNR}, rval_thr={rval_thr}, use_cnn={use_cnn}\
                    '''.format(min_SNR=params.quality['min_SNR'],
                               rval_thr=params.quality['rval_thr'],
                               use_cnn=params.quality['use_cnn'])
                bokeh.plotting.show(bokeh.layouts.row(p1, p2))
        except:
            print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")
            print("Using non-interactive plot as fallback")
            self.plot_contours(img=img, idx=idx, crd=crd, thr_method=thr_method,
                               thr=thr, params=params, cmap=cmap)
        return self

    def view_components(self, Yr=None, img=None, idx=None):
        """view spatial and temporal components interactively

        Args:
            Yr :    np.ndarray
                movie in format pixels (d) x frames (T)

            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)

            idx :   list
                list of components to be plotted
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)

        plt.ion()
        nr, T = self.C.shape
        if self.R is None:
            self.R = self.YrA
        if self.R.shape != (nr, T):
            if self.YrA is None:
                self.compute_residuals(Yr)
            else:
                self.R = self.YrA

        if img is None:
            img = np.reshape(np.array(self.A.mean(axis=1)), self.dims, order='F')

        if idx is None:
            caiman.utils.visualization.view_patches_bar(Yr, self.A, self.C,
                    self.b, self.f, self.dims[0], self.dims[1], YrA=self.R, img=img)
        else:
            caiman.utils.visualization.view_patches_bar(Yr, self.A.tocsc()[:,idx],
                                                        self.C[idx], self.b, self.f,
                                                        self.dims[0], self.dims[1], YrA=self.R[idx], img=img)
        return self

    def nb_view_components(self, Yr=None, img=None, idx=None,
                           denoised_color=None, cmap='jet', thr=0.99):
        """view spatial and temporal components interactively in a notebook

        Args:
            Yr :    np.ndarray
                movie in format pixels (d) x frames (T)

            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)

            idx :   list
                list of components to be plotted

            thr: double
                threshold regulating the extent of the displayed patches

            denoised_color: string or None
                color name (e.g. 'red') or hex color code (e.g. '#F0027F')

            cmap: string
                name of colormap (e.g. 'viridis') used to plot image_neurons
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)

        plt.ion()
        nr, T = self.C.shape
        if self.R is None:
            self.R = self.YrA
        if self.R.shape != [nr, T]:
            if self.YrA is None:
                self.compute_residuals(Yr)
            else:
                self.R = self.YrA

        if img is None:
            img = np.reshape(np.array(self.A.mean(axis=1)), self.dims, order='F')

        if idx is None:
            caiman.utils.visualization.nb_view_patches(Yr, self.A, self.C,
                    self.b, self.f, self.dims[0], self.dims[1], YrA=self.R, image_neurons=img,
                    thr=thr, denoised_color=denoised_color, cmap=cmap)
        else:
            caiman.utils.visualization.nb_view_patches(Yr, self.A.tocsc()[:,idx],
                                                        self.C[idx], self.b, self.f,
                                                        self.dims[0], self.dims[1], YrA=self.R[idx], image_neurons=img,
                                                        thr=thr, denoised_color=denoised_color, cmap=cmap)
        return self

    def hv_view_components(self, Yr=None, img=None, idx=None,
                           denoised_color=None, cmap='viridis'):
        """view spatial and temporal components interactively in a notebook

        Args:
            Yr :    np.ndarray
                movie in format pixels (d) x frames (T)

            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)

            idx :   list
                list of components to be plotted

            denoised_color: string or None
                color name (e.g. 'red') or hex color code (e.g. '#F0027F')

            cmap: string
                name of colormap (e.g. 'viridis') used to plot image_neurons
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)

        plt.ion()
        nr, T = self.C.shape
        if self.R is None:
            self.R = self.YrA
        if self.R.shape != [nr, T]:
            if self.YrA is None:
                self.compute_residuals(Yr)
            else:
                self.R = self.YrA

        if img is None:
            img = np.reshape(np.array(self.A.mean(axis=1)), self.dims, order='F')

        if idx is None:
            hv_plot = caiman.utils.visualization.hv_view_patches(
                Yr, self.A, self.C, self.b, self.f, self.dims[0], self.dims[1],
                YrA=self.R, image_neurons=img, denoised_color=denoised_color,
                cmap=cmap)
        else:
            hv_plot = caiman.utils.visualization.hv_view_patches(
                Yr, self.A.tocsc()[:, idx], self.C[idx], self.b, self.f,
                self.dims[0], self.dims[1], YrA=self.R[idx], image_neurons=img,
                denoised_color=denoised_color, cmap=cmap)
        return hv_plot

    def nb_view_components_3d(self, Yr=None, image_type='mean', dims=None,
                              max_projection=False, axis=0,
                              denoised_color=None, cmap='jet', thr=0.9):
        """view spatial and temporal components interactively in a notebook
        (version for 3d data)

        Args:
            Yr :    np.ndarray
                movie in format pixels (d) x frames (T) (only required to
                compute the correlation image)


            dims: tuple of ints
                dimensions of movie (x, y and z)

            image_type: 'mean'|'max'|'corr'
                image to be overlaid to neurons (average of shapes,
                maximum of shapes or nearest neigbor correlation of raw data)

            max_projection: bool
                plot max projection along specified axis if True, o/w plot layers

            axis: int (0, 1 or 2)
                axis along which max projection is performed or layers are shown

            thr: scalar between 0 and 1
                Energy threshold for computing contours

            denoised_color: string or None
                color name (e.g. 'red') or hex color code (e.g. '#F0027F')

            cmap: string
                name of colormap (e.g. 'viridis') used to plot image_neurons

        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if dims is None:
            dims = self.dims
        plt.ion()
        nr, T = self.C.shape
        if self.R is None:
            self.R = self.YrA
        if self.R.shape != [nr, T]:
            if self.YrA is None:
                self.compute_residuals(Yr)
            else:
                self.R = self.YrA

        caiman.utils.visualization.nb_view_patches3d(self.YrA, self.A, self.C,
                    dims=dims, image_type=image_type, Yr=Yr,
                    max_projection=max_projection, axis=axis, thr=thr,
                    denoised_color=denoised_color, cmap=cmap)

        return self

    def make_color_movie(self, imgs, q_max=99.75, q_min=2, gain_res=1,
                         magnification=1, include_bck=True,
                         frame_range=slice(None, None, None),
                         bpx=0, save_movie=False, display=True,
                         movie_name='results_movie_color.avi',
                         opencv_code='H264'):
        """
        Displays a color movie where each component is given an arbitrary
        color. Will be merged with play_movie soon. Check that function for
        arg definitions.
        """
        dims = imgs.shape[1:]
        cols_c = np.random.rand(self.C.shape[0], 1, 3)
        cols_f = np.ones((self.f.shape[0], 1, 3))/8
        Cs = np.vstack((np.expand_dims(self.C[:, frame_range], -1)*cols_c,
                        np.expand_dims(self.f[:, frame_range], -1)*cols_f))
        AC = np.tensordot(np.hstack((self.A.toarray(), self.b)), Cs, axes=(1, 0))
        AC = AC.reshape((dims) + (-1, 3)).transpose(2, 0, 1, 3)

        AC /= np.percentile(AC, 99.75, axis=(0, 1, 2))
        mov = caiman.movie(np.concatenate((np.repeat(np.expand_dims(imgs[frame_range]/np.percentile(imgs[:1000], 99.75), -1), 3, 3),
                                           AC), axis=2))
        if not display:
            return mov

        mov.play(q_min=q_min, q_max=q_max, magnification=magnification,
                 save_movie=save_movie, movie_name=movie_name)

        return mov


    def play_movie(self, imgs, q_max=99.75, q_min=2, gain_res=1,
                   magnification=1, include_bck=True,
                   frame_range=slice(None, None, None),
                   bpx=0, thr=0., save_movie=False,
                   movie_name='results_movie.avi',
                   display=True, opencv_codec='H264',
                   use_color=False, gain_color=4, gain_bck=0.2):
        """
        Displays a movie with three panels (original data (left panel),
        reconstructed data (middle panel), residual (right panel))

        Args:
            imgs: np.array (possibly memory mapped, t,x,y[,z])
                Imaging data

            q_max: float (values in [0, 100], default: 99.75)
                percentile for maximum plotting value

            q_min: float (values in [0, 100], default: 1)
                percentile for minimum plotting value

            gain_res: float (1)
                amplification factor for residual movie

            magnification: float (1)
                magnification factor for whole movie

            include_bck: bool (True)
                flag for including background in original and reconstructed movie

            frame_range: range or slice or list (default: slice(None))
                display only a subset of frames

            bpx: int (default: 0)
                number of pixels to exclude on each border

            thr: float (values in [0, 1[) (default: 0)
                threshold value for contours, no contours if thr=0

            save_movie: bool (default: False)
                flag to save an avi file of the movie

            movie_name: str (default: 'results_movie.avi')
                name of saved file

            display: bool (default: True)
                flag for playing the movie (to stop the movie press 'q')

            opencv_codec: str (default: 'H264')
                FourCC video codec for saving movie. Check http://www.fourcc.org/codecs.php

            use_color: bool (default: False)
                flag for making a color movie. If True a random color will be assigned
                for each of the components

            gain_color: float (default: 4)
                amplify colors in the movie to make them brighter

            gain_bck: float (default: 0.2)
                dampen background in the movie to expose components (applicable
                only when color is used.)

        Returns:
            mov: The concatenated output movie
        """
        dims = imgs.shape[1:]
        if 'movie' not in str(type(imgs)):
            imgs = caiman.movie(imgs[frame_range])
        else:
            imgs = imgs[frame_range]

        if use_color:
            cols_c = np.random.rand(self.C.shape[0], 1, 3)*gain_color
            Cs = np.expand_dims(self.C[:, frame_range], -1)*cols_c
            #AC = np.tensordot(np.hstack((self.A.toarray(), self.b)), Cs, axes=(1, 0))
            Y_rec_color = np.tensordot(self.A.toarray(), Cs, axes=(1, 0))
            Y_rec_color = Y_rec_color.reshape((dims) + (-1, 3), order='F').transpose(2, 0, 1, 3)

        AC = self.A.dot(self.C[:, frame_range])
        Y_rec = AC.reshape(dims + (-1,), order='F')
        Y_rec = Y_rec.transpose([2, 0, 1])
        if self.W is not None:
            ssub_B = int(round(np.sqrt(np.prod(dims) / self.W.shape[0])))
            B = imgs.reshape((-1, np.prod(dims)), order='F').T - AC
            if ssub_B == 1:
                B = self.b0[:, None] + self.W.dot(B - self.b0[:, None])
            else:
                WB = self.W.dot(downscale(B.reshape(dims + (B.shape[-1],), order='F'),
                              (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F'))
                Wb0 = self.W.dot(downscale(self.b0.reshape(dims, order='F'),
                              (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
                B = self.b0.flatten('F')[:, None] + (np.repeat(np.repeat((WB - Wb0).reshape(((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'),
                                     ssub_B, 0), ssub_B, 1)[:dims[0], :dims[1]].reshape((-1, B.shape[-1]), order='F'))
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        elif self.b is not None and self.f is not None:
            B = self.b.dot(self.f[:, frame_range])
            if 'matrix' in str(type(B)):
                B = B.toarray()
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        else:
            B = np.zeros_like(Y_rec)
        if bpx > 0:
            B = B[:, bpx:-bpx, bpx:-bpx]
            Y_rec = Y_rec[:, bpx:-bpx, bpx:-bpx]
            imgs = imgs[:, bpx:-bpx, bpx:-bpx]

        Y_res = imgs - Y_rec - B
        if use_color:
            if bpx > 0:
                Y_rec_color = Y_rec_color[:, bpx:-bpx, bpx:-bpx]
            mov = caiman.concatenate((np.repeat(np.expand_dims(imgs - (not include_bck) * B, -1), 3, 3),
                                      Y_rec_color + include_bck * np.expand_dims(B*gain_bck, -1),
                                      np.repeat(np.expand_dims(Y_res * gain_res, -1), 3, 3)), axis=2)
        else:
            mov = caiman.concatenate((imgs[frame_range] - (not include_bck) * B,
                                      Y_rec + include_bck * B, Y_res * gain_res), axis=2)
        if not display:
            return mov

        if thr > 0:
            import cv2
            if save_movie:
                fourcc = cv2.VideoWriter_fourcc(*opencv_codec)
                out = cv2.VideoWriter(movie_name, fourcc, 30.0,
                                      tuple([int(magnification*s) for s in mov.shape[1:][::-1]]))
            contours = []
            for a in self.A.T.toarray():
                a = a.reshape(dims, order='F')
                if bpx > 0:
                    a = a[bpx:-bpx, bpx:-bpx]
                # a = cv2.GaussianBlur(a, (9, 9), .5)
                if magnification != 1:
                    a = cv2.resize(a, None, fx=magnification, fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
                ret, thresh = cv2.threshold(a, thr * np.max(a), 1., 0)
                contour, hierarchy = cv2.findContours(
                    thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours.append(contour)
                contours.append(list([c + np.array([[a.shape[1], 0]]) for c in contour]))
                contours.append(list([c + np.array([[2 * a.shape[1], 0]]) for c in contour]))

            maxmov = np.nanpercentile(mov[0:10], q_max) if q_max < 100 else np.nanmax(mov)
            minmov = np.nanpercentile(mov[0:10], q_min) if q_min > 0 else np.nanmin(mov)
            for iddxx, frame in enumerate(mov):
                if magnification != 1:
                    frame = cv2.resize(frame, None, fx=magnification, fy=magnification,
                                       interpolation=cv2.INTER_LINEAR)
                frame = np.clip((frame - minmov) * 255. / (maxmov - minmov), 0, 255)
                if frame.ndim < 3:
                    frame = np.repeat(frame[..., None], 3, 2)
                for contour in contours:
                    cv2.drawContours(frame, contour, -1, (0, 255, 255), 1)
                cv2.imshow('frame', frame.astype('uint8'))
                if save_movie:
                    out.write(frame.astype('uint8'))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            if save_movie:
                out.release()
            cv2.destroyAllWindows()

        else:
            mov.play(q_min=q_min, q_max=q_max, magnification=magnification,
                     save_movie=save_movie, movie_name=movie_name)

        return mov

    def compute_background(self, Yr):
        """compute background (has big memory requirements)

         Args:
             Yr :    np.ndarray
                 movie in format pixels (d) x frames (T)
            """
        logging.warning("Computing the full background has big memory requirements!")
        if self.f is not None:  # low rank background
            return self.b.dot(self.f)
        else:  # ring model background
            ssub_B = np.round(np.sqrt(Yr.shape[0] / self.W.shape[0])).astype(int)
            if ssub_B == 1:
                return self.b0[:, None] + self.W.dot(Yr - self.A.dot(self.C) - self.b0[:, None])
            else:
                ds_mat = decimation_matrix(self.dims, ssub_B)
                B = ds_mat.dot(Yr) - ds_mat.dot(self.A).dot(self.C) - ds_mat.dot(self.b0)[:, None]
                B = self.W.dot(B).reshape(((self.dims[0] - 1) // ssub_B + 1,
                                           (self.dims[1] - 1) // ssub_B + 1, -1), order='F')
                B = self.b0[:, None] + np.repeat(np.repeat(B, ssub_B, 0), ssub_B, 1
                                                 )[:self.dims[0], :self.dims[1]].reshape(
                    (-1, B.shape[-1]), order='F')
                return B

    def compute_residuals(self, Yr):
        """compute residual for each component (variable R)

         Args:
             Yr :    np.ndarray
                 movie in format pixels (d) x frames (T)
        """
        if len(Yr.shape) > 2:
            Yr = np.reshape(Yr.transpose(1,2,0), (-1, Yr.shape[0]), order='F')
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if 'array' not in str(type(self.b)):
            self.b = self.b.toarray()
        if 'array' not in str(type(self.C)):
            self.C = self.C.toarray()
        if 'array' not in str(type(self.f)):
            self.f = self.f.toarray()

        Ab = scipy.sparse.hstack((self.A, self.b)).tocsc()
        nA2 = np.ravel(Ab.power(2).sum(axis=0)) + np.finfo(np.float32).eps
        nA2_inv_mat = scipy.sparse.spdiags(
            1. / nA2, 0, nA2.shape[0], nA2.shape[0])
        Cf = np.vstack((self.C, self.f))
        if 'numpy.ndarray' in str(type(Yr)):
            YA = (Ab.T.dot(Yr)).T * nA2_inv_mat
        else:
            YA = caiman.mmapping.parallel_dot_product(Yr, Ab, dview=self.dview,
                        block_size=2000, transpose=True, num_blocks_per_run=5) * nA2_inv_mat

        AA = Ab.T.dot(Ab) * nA2_inv_mat
        self.R = (YA - (AA.T.dot(Cf)).T)[:, :self.A.shape[-1]].T

        return self

    def detrend_df_f(self, quantileMin=8, frames_window=500,
                     flag_auto=True, use_fast=False, use_residuals=True,
                     detrend_only=False):
        """Computes DF/F normalized fluorescence for the extracted traces. See
        caiman.source.extraction.utilities.detrend_df_f for details

        Args:
            quantile_min: float
                quantile used to estimate the baseline (values in [0,100])

            frames_window: int
                number of frames for computing running quantile

            flag_auto: bool
                flag for determining quantile automatically (different for each
                trace)

            use_fast: bool
                flag for using approximate fast percentile filtering

            use_residuals: bool
                flag for using non-deconvolved traces in DF/F calculation

            detrend_only: bool (False)
                flag for only subtracting baseline and not normalizing by it.
                Used in 1p data processing where baseline fluorescence cannot
                be determined.

        Returns:
            self: CNMF object
                self.F_dff contains the DF/F normalized traces
        """

        if self.C is None or self.C.shape[0] == 0:
            logging.warning("There are no components for DF/F extraction!")
            return self

        if use_residuals:
            if self.R is None:
                if self.YrA is None:
                    R = None
                else:
                    R = self.YrA
            else:
                R = self.R
        else:
            R = None

        self.F_dff = detrend_df_f(self.A, self.b, self.C, self.f, self.YrA,
                                  quantileMin=quantileMin,
                                  frames_window=frames_window,
                                  flag_auto=flag_auto, use_fast=use_fast,
                                  detrend_only=detrend_only)
        return self

    def normalize_components(self):
        """ Normalizes components such that spatial components have l_2 norm 1
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if 'array' not in str(type(self.C)):
            self.C = self.C.toarray()
        if 'array' not in str(type(self.f)) and self.f is not None:
            self.f = self.f.toarray()

        nA = np.sqrt(np.ravel(self.A.power(2).sum(axis=0)))
        nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
        nA_inv_mat = scipy.sparse.spdiags(1. / (nA + np.finfo(np.float32).eps), 0, nA.shape[0], nA.shape[0])
        self.A = self.A * nA_inv_mat
        self.C = nA_mat * self.C
        if self.YrA is not None:
            self.YrA = nA_mat * self.YrA
        if self.R is not None:
            self.R = nA_mat * self.R
        if self.bl is not None:
            self.bl = nA * self.bl
        if self.c1 is not None:
            self.c1 = nA * self.c1
        if self.neurons_sn is not None:
            self.neurons_sn = nA * self.neurons_sn

        if self.f is not None:  # 1p with exact ring-model
            nB = np.sqrt(np.ravel((self.b.power(2) if scipy.sparse.issparse(self.b)
                         else self.b**2).sum(axis=0)))
            nB_mat = scipy.sparse.spdiags(nB, 0, nB.shape[0], nB.shape[0])
            nB_inv_mat = scipy.sparse.spdiags(1. / (nB + np.finfo(np.float32).eps), 0, nB.shape[0], nB.shape[0])
            self.b = self.b * nB_inv_mat
            self.f = nB_mat * self.f
        return self

    def select_components(self, idx_components=None, use_object=False, save_discarded_components=True):
        """Keeps only a selected subset of components and removes the rest.
        The subset can be either user defined with the variable idx_components
        or read from the estimates object. The flag use_object determines this
        choice. If no subset is present then all components are kept.

        Args:
            idx_components: list
                indices of components to be kept

            use_object: bool
                Flag to use self.idx_components for reading the indices.

            save_discarded_components: bool
                whether to save the components from initialization so that they
                can be restored using the restore_discarded_components method

        Returns:
            self: Estimates object
        """
        if use_object:
            idx_components = self.idx_components
            idx_components_bad = self.idx_components_bad
        else:
            idx_components_bad = np.setdiff1d(np.arange(self.A.shape[-1]), idx_components)

        if idx_components is not None:
            if save_discarded_components and self.discarded_components is None:
                self.discarded_components = Estimates()

            for field in ['C', 'S', 'YrA', 'R', 'F_dff', 'g', 'bl', 'c1', 'neurons_sn',
                          'lam', 'cnn_preds', 'SNR_comp', 'r_values', 'coordinates']:
                if getattr(self, field) is not None:
                    if isinstance(getattr(self, field), list):
                        setattr(self, field, np.array(getattr(self, field)))
                    if len(getattr(self, field)) == self.A.shape[-1]:
                        if save_discarded_components:
                            setattr(self.discarded_components, field,
                                    getattr(self, field)[idx_components_bad]
                                    if getattr(self.discarded_components, field) is None else
                                    np.concatenate([getattr(self.discarded_components, field),
                                                    getattr(self, field)[idx_components_bad]]))
                        setattr(self, field, getattr(self, field)[idx_components])
                    else:
                        print('*** Variable ' + field +
                              ' has not the same number of components as A ***')

            for field in ['A', 'A_thr']:
                if getattr(self, field) is not None:
                    if 'sparse' in str(type(getattr(self, field))):
                        if save_discarded_components:
                            if getattr(self.discarded_components, field) is None:
                                setattr(self.discarded_components, field,
                                    getattr(self, field).tocsc()[:, idx_components_bad])
                            else:
                                caiman.source_extraction.cnmf.online_cnmf.csc_append(
                                    getattr(self.discarded_components, field),
                                    getattr(self, field).tocsc()[:, idx_components_bad])
                        setattr(self, field, getattr(self, field).tocsc()[:, idx_components])
                    else:
                        if save_discarded_components:
                            setattr(self.discarded_components, field,
                                getattr(self, field)[:, idx_components_bad]
                                    if getattr(self.discarded_components, field) is None else
                                    np.concatenate([getattr(self.discarded_components, field),
                                        getattr(self, field)[:, idx_components_bad]], axis=-1))
                        setattr(self, field, getattr(self, field)[:, idx_components])

            self.nr = len(idx_components)

            if save_discarded_components:
                if not hasattr(self.discarded_components, 'nr'):
                    self.discarded_components.nr = 0
                self.discarded_components.nr += len(idx_components_bad)
                self.discarded_components.dims = self.dims

            self.idx_components = None
            self.idx_components_bad = None

        return self

    def restore_discarded_components(self):
        ''' Recover components that are filtered out with the select_components method
        '''
        if self.discarded_components is not None:
            for field in ['C', 'S', 'YrA', 'R', 'F_dff', 'g', 'bl', 'c1', 'neurons_sn', 'lam', 'cnn_preds','SNR_comp','r_values','coordinates']:
                if getattr(self, field) is not None:
                    if isinstance(getattr(self, field), list):
                        setattr(self, field, np.array(getattr(self, field)))
                    if len(getattr(self, field)) == self.A.shape[-1]:
                        setattr(self, field, np.concatenate([getattr(self, field), getattr(self.discarded_components, field)], axis=0))
                        setattr(self.discarded_components, field, None)
                    else:
                        logging.warning('Variable ' + field + ' could not be \
                                        restored as it does not have the same \
                                        number of components as A')

            for field in ['A', 'A_thr']:
                print(field)
                if getattr(self, field) is not None:
                    if 'sparse' in str(type(getattr(self, field))):
                        setattr(self, field, scipy.sparse.hstack([getattr(self, field).tocsc(),getattr(self.discarded_components, field).tocsc()]))
                    else:
                        setattr(self, field,np.concatenate([getattr(self, field), getattr(self.discarded_components, field)], axis=0))

                    setattr(self.discarded_components, field, None)

            self.nr = self.A.shape[-1]

    def evaluate_components_CNN(self, params, neuron_class=1):
        """Estimates the quality of inferred spatial components using a
        pretrained CNN classifier.

        Args:
            params: params object
                see .params for details
            neuron_class: int
                class label for neuron shapes
        Returns:
            self: Estimates object
                self.idx_components contains the indeced of components above
                the required treshold.
        """
        dims = params.get('data', 'dims')
        gSig = params.get('init', 'gSig')
        min_cnn_thr = params.get('quality', 'min_cnn_thr')
        predictions = evaluate_components_CNN(self.A, dims, gSig)[0]
        self.cnn_preds = predictions[:, neuron_class]
        self.idx_components = np.where(self.cnn_preds >= min_cnn_thr)[0]
        return self

    def evaluate_components(self, imgs, params, dview=None):
        """Computes the quality metrics for each component and stores the
        indices of the components that pass user specified thresholds. The
        various thresholds and parameters can be passed as inputs. If left
        empty then they are read from self.params.quality']

        Args:
            imgs: np.array (possibly memory mapped, t,x,y[,z])
                Imaging data

            params: params object
                Parameters of the algorithm. The parameters in play here are
                contained in the subdictionary params.quality:

                min_SNR: float
                    trace SNR threshold

                rval_thr: float
                    space correlation threshold

                use_cnn: bool
                    flag for using the CNN classifier

                min_cnn_thr: float
                    CNN classifier threshold

        Returns:
            self: estimates object
                self.idx_components: np.array
                    indices of accepted components
                self.idx_components_bad: np.array
                    indices of rejected components
                self.SNR_comp: np.array
                    SNR values for each temporal trace
                self.r_values: np.array
                    space correlation values for each component
                self.cnn_preds: np.array
                    CNN classifier values for each component
        """
        dims = imgs.shape[1:]
        opts = params.get_group('quality')
        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
            estimate_components_quality_auto(imgs, self.A, self.C, self.b, self.f, self.YrA,
                                             params.get('data', 'fr'),
                                             params.get('data', 'decay_time'),
                                             params.get('init', 'gSig'),
                                             dims, dview=dview,
                                             min_SNR=opts['min_SNR'],
                                             r_values_min=opts['rval_thr'],
                                             use_cnn=opts['use_cnn'],
                                             thresh_cnn_min=opts['min_cnn_thr'],
                                             thresh_cnn_lowest=opts['cnn_lowest'],
                                             r_values_lowest=opts['rval_lowest'],
                                             min_SNR_reject=opts['SNR_lowest'])
        self.idx_components = idx_components.astype(int)
        self.idx_components_bad = idx_components_bad.astype(int)
        if np.any(np.isnan(r_values)):
            logging.warning('NaN values detected for space correlation in {}'.format(np.where(np.isnan(r_values))[0]) +
                            '. Changing their value to -1.')
            r_values = np.where(np.isnan(r_values), -1, r_values)
        if np.any(np.isnan(SNR_comp)):
            logging.warning('NaN values detected for trace SNR in {}'.format(np.where(np.isnan(SNR_comp))[0]) +
                            '. Changing their value to 0.')
            SNR_comp = np.where(np.isnan(SNR_comp), 0, SNR_comp)
        self.SNR_comp = SNR_comp
        self.r_values = r_values
        self.cnn_preds = cnn_preds
        if opts['use_ecc']:
            self.ecc = compute_eccentricity(self.A, dims)
            idx_ecc = np.where(self.ecc < opts['max_ecc'])[0]
            self.idx_components_bad = np.union1d(self.idx_components_bad,
                                                 np.setdiff1d(self.idx_components,
                                                              idx_ecc))
            self.idx_components = np.intersect1d(self.idx_components, idx_ecc)
        return self

    def filter_components(self, imgs, params, new_dict={}, dview=None, select_mode='All'):
        """Filters components based on given thresholds without re-computing
        the quality metrics. If the quality metrics are not present then it
        calls self.evaluate components.

        Args:
            imgs: np.array (possibly memory mapped, t,x,y[,z])
                Imaging data

            params: params object
                Parameters of the algorithm

            select_mode: str
                Can be 'All' (no subselection is made, but quality filtering is performed),
                'Accepted' (subselection of accepted components, a field named self.accepted_list must exist),
                'Rejected' (subselection of rejected components, a field named self.rejected_list must exist),
                'Unassigned' (both fields above need to exist)

            new_dict: dict
                New dictionary with parameters to be called. The dictionary
                modifies the params.quality subdictionary in the following
                entries:
                    min_SNR: float
                        trace SNR threshold

                    SNR_lowest: float
                        minimum required trace SNR

                    rval_thr: float
                        space correlation threshold

                    rval_lowest: float
                        minimum required space correlation

                    use_cnn: bool
                        flag for using the CNN classifier

                    min_cnn_thr: float
                        CNN classifier threshold

                    cnn_lowest: float
                        minimum required CNN threshold

                    gSig_range: list
                        gSig scale values for CNN classifier

        Returns:
            self: estimates object
                self.idx_components: np.array
                    indices of accepted components
                self.idx_components_bad: np.array
                    indices of rejected components
                self.SNR_comp: np.array
                    SNR values for each temporal trace
                self.r_values: np.array
                    space correlation values for each component
                self.cnn_preds: np.array
                    CNN classifier values for each component
        """
        dims = imgs.shape[1:]
        params.set('quality', new_dict)

        opts = params.get_group('quality')
        flag = [a is None for a in [self.r_values, self.SNR_comp, self.cnn_preds]]

        if any(flag):
            self.evaluate_components(imgs, params, dview=dview)
        else:
            self.idx_components, self.idx_components_bad, self.cnn_preds = \
            select_components_from_metrics(self.A, dims, params.get('init', 'gSig'),
                                           self.r_values, self.SNR_comp,
                                           predictions=self.cnn_preds,
                                           r_values_min=opts['rval_thr'],
                                           r_values_lowest=opts['rval_lowest'],
                                           min_SNR=opts['min_SNR'],
                                           min_SNR_reject=opts['SNR_lowest'],
                                           thresh_cnn_min=opts['min_cnn_thr'],
                                           thresh_cnn_lowest=opts['cnn_lowest'],
                                           use_cnn=opts['use_cnn'],
                                           gSig_range=opts['gSig_range'])
            if opts['use_ecc']:
                idx_ecc = np.where(self.ecc < opts['max_ecc'])[0]
                self.idx_components_bad = np.union1d(self.idx_components_bad,
                                                     np.setdiff1d(self.idx_components,
                                                                  idx_ecc))
                self.idx_components = np.intersect1d(self.idx_components, idx_ecc)

        if select_mode == 'Accepted':
           self.idx_components = np.array(np.intersect1d(self.idx_components,self.accepted_list))
        elif select_mode == 'Rejected':
           self.idx_components = np.array(np.intersect1d(self.idx_components,self.rejected_list))
        elif select_mode == 'Unassigned':
           self.idx_components = np.array(np.setdiff1d(self.idx_components,np.union1d(self.rejected_list,self.accepted_list)))

        self.idx_components_bad = np.array(np.setdiff1d(range(len(self.SNR_comp)),self.idx_components))

        return self

    def deconvolve(self, params, dview=None, dff_flag=False):
        ''' performs deconvolution on the estimated traces using the parameters
        specified in params. Deconvolution on detrended and normalized (DF/F)
        traces can be performed by setting dff_flag=True. In this case the
        results of the deconvolution are stored in F_dff_dec and S_dff

        Args:
            params: params object
                Parameters of the algorithm
            dff_flag: bool (True)
                Flag for deconvolving the DF/F traces

        Returns:
            self: estimates object
        '''

        F = self.C + self.YrA
        args = dict()
        args['p'] = params.get('preprocess', 'p')
        args['method_deconvolution'] = params.get('temporal', 'method_deconvolution')
        args['bas_nonneg'] = params.get('temporal', 'bas_nonneg')
        args['noise_method'] = params.get('temporal', 'noise_method')
        args['s_min'] = params.get('temporal', 's_min')
        args['optimize_g'] = params.get('temporal', 'optimize_g')
        args['noise_range'] = params.get('temporal', 'noise_range')
        args['fudge_factor'] = params.get('temporal', 'fudge_factor')

        args_in = [(F[jj], None, jj, None, None, None, None,
                    args) for jj in range(F.shape[0])]

        if 'multiprocessing' in str(type(dview)):
            results = dview.map_async(
                constrained_foopsi_parallel, args_in).get(4294967)
        elif dview is not None:
            results = dview.map_sync(constrained_foopsi_parallel, args_in)
        else:
            results = list(map(constrained_foopsi_parallel, args_in))

        results = list(zip(*results))

        order = list(results[7])
        self.C = np.stack([results[0][i] for i in order])
        self.S = np.stack([results[1][i] for i in order])
        self.bl = [results[3][i] for i in order]
        self.c1 = [results[4][i] for i in order]
        self.g = [results[6][i] for i in order]
        self.neurons_sn = [results[5][i] for i in order]
        self.lam = [results[8][i] for i in order]
        self.YrA = F - self.C

        if dff_flag:
            if self.F_dff is None:
                logging.warning('The F_dff field is empty. Run the method' +
                                ' estimates.detrend_df_f before attempting' +
                                ' to deconvolve.')
            else:
                args_in = [(self.F_dff[jj], None, jj, 0, 0, self.g[jj], None,
                        args) for jj in range(F.shape[0])]

                if 'multiprocessing' in str(type(dview)):
                    results = dview.map_async(
                        constrained_foopsi_parallel, args_in).get(4294967)
                elif dview is not None:
                    results = dview.map_sync(constrained_foopsi_parallel,
                                             args_in)
                else:
                    results = list(map(constrained_foopsi_parallel, args_in))

                results = list(zip(*results))
                order = list(results[7])
                self.F_dff_dec = np.stack([results[0][i] for i in order])
                self.S_dff = np.stack([results[1][i] for i in order])

    def merge_components(self, Y, params, mx=50, fast_merge=True,
                         dview=None, max_merge_area=None):
            """merges components
            """
            self.A, self.C, self.nr, self.merged_ROIs, self.S, \
            self.bl, self.c1, self.neurons_sn, self.g, empty_merged, \
            self.YrA =\
                merge_components(Y, self.A, self.b, self.C, self.YrA,
                                 self.f, self.S, self.sn, params.get_group('temporal'),
                                 params.get_group('spatial'), dview=dview,
                                 bl=self.bl, c1=self.c1, sn=self.neurons_sn,
                                 g=self.g, thr=params.get('merging', 'merge_thr'), mx=mx,
                                 fast_merge=fast_merge, merge_parallel=params.get('merging', 'merge_parallel'),
                                 max_merge_area=max_merge_area)

    def manual_merge(self, components, params):
        ''' merge a given list of components. The indices
        of components are pythonic, i.e., they start from 0. Moreover,
        the indices refer to the absolute indices, i.e., the indices before
        splitting the components in accepted and rejected. If you want to e.g.
        merge components 0 from idx_components and 9 from idx_components_bad
        you will to set
        ```
        components = [[self.idx_components[0], self.idx_components_bad[9]]]
        ```

        Args:
            components: list
                list of components to be merged. Each element should be a
                tuple, list or np.array of the components to be merged. No
                duplicates are allowed. If you're merging only one pair (or
                set) of components, then use a list with a single (list)
                element
            params: params object

        Returns:
            self: estimates object
        '''

        ln = np.sum(np.array([len(comp) for comp in components]))
        ids = set.union(*[set(comp) for comp in components])
        if ln != len(ids):
            raise Exception('The given list contains duplicate entries')

        p = params.temporal['p']
        nbmrg = len(components)   # number of merging operations
        d = self.A.shape[0]
        T = self.C.shape[1]
        # we initialize the values
        A_merged = scipy.sparse.lil_matrix((d, nbmrg))
        C_merged = np.zeros((nbmrg, T))
        R_merged = np.zeros((nbmrg, T))
        S_merged = np.zeros((nbmrg, T))
        bl_merged = np.zeros((nbmrg, 1))
        c1_merged = np.zeros((nbmrg, 1))
        sn_merged = np.zeros((nbmrg, 1))
        g_merged = np.zeros((nbmrg, p))
        merged_ROIs = []

        for i in range(nbmrg):
            merged_ROI = list(set(components[i]))
            logging.info('Merging components {}'.format(merged_ROI))
            merged_ROIs.append(merged_ROI)

            Acsc = self.A.tocsc()[:, merged_ROI]
            Ctmp = np.array(self.C[merged_ROI]) + np.array(self.YrA[merged_ROI])

            C_to_norm = np.sqrt(np.ravel(Acsc.power(2).sum(
                axis=0)) * np.sum(Ctmp ** 2, axis=1))
            indx = np.argmax(C_to_norm)
            g_idx = [merged_ROI[indx]]
            fast_merge = True
            bm, cm, computedA, computedC, gm, \
            sm, ss, yra = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge,
                                          None, g_idx, indx, params.temporal)

            A_merged[:, i] = computedA[:, np.newaxis]
            C_merged[i, :] = computedC
            R_merged[i, :] = yra
            S_merged[i, :] = ss[:T]
            bl_merged[i] = bm
            c1_merged[i] = cm
            sn_merged[i] = sm
            g_merged[i, :] = gm

        empty = np.ravel((C_merged.sum(1) == 0) + (A_merged.sum(0) == 0))
        nbmrg -= sum(empty)
        if np.any(empty):
            A_merged = A_merged[:, ~empty]
            C_merged = C_merged[~empty]
            R_merged = R_merged[~empty]
            S_merged = S_merged[~empty]
            bl_merged = bl_merged[~empty]
            c1_merged = c1_merged[~empty]
            sn_merged = sn_merged[~empty]
            g_merged = g_merged[~empty]

        neur_id = np.unique(np.hstack(merged_ROIs))
        nr = self.C.shape[0]
        good_neurons = np.setdiff1d(list(range(nr)), neur_id)
        if self.idx_components is not None:
            new_indices = list(range(len(good_neurons),
                                     len(good_neurons) + nbmrg))

            mapping_mat = np.zeros(nr)
            mapping_mat[good_neurons] = np.arange(len(good_neurons), dtype=int)
            gn_ = good_neurons.tolist()
            new_idx = [mapping_mat[i] for i in self.idx_components if i in gn_]
            new_idx_bad = [mapping_mat[i] for i in self.idx_components_bad if i in gn_]
            new_idx.sort()
            new_idx_bad.sort()
            self.idx_components = np.array(new_idx + new_indices, dtype=int)
            self.idx_components_bad = np.array(new_idx_bad, dtype=int)

        self.A = scipy.sparse.hstack((self.A.tocsc()[:, good_neurons],
                                      A_merged.tocsc()))
        self.C = np.vstack((self.C[good_neurons, :], C_merged))
        # we continue for the variables
        if self.YrA is not None:
            self.YrA = np.vstack((self.YrA[good_neurons, :], R_merged))
            self.R = self.YrA
        if self.S is not None:
            self.S = np.vstack((self.S[good_neurons, :], S_merged))
        if self.bl is not None:
            self.bl = np.hstack((self.bl[good_neurons],
                                 np.array(bl_merged).flatten()))
        if self.c1 is not None:
            self.c1 = np.hstack((self.c1[good_neurons],
                                 np.array(c1_merged).flatten()))
        if self.neurons_sn is not None:
            self.neurons_sn = np.hstack((self.neurons_sn[good_neurons],
                                 np.array(sn_merged).flatten()))
        if self.g is not None:
            self.g = np.vstack((np.vstack(self.g)[good_neurons], g_merged))
        self.nr = nr - len(neur_id) + len(C_merged)
        if self.coordinates is not None:
            self.coordinates = caiman.utils.visualization.get_contours(self.A,\
                                self.dims, thr_method='max', thr='0.2')

    def threshold_spatial_components(self, maxthr=0.25, dview=None):
        ''' threshold spatial components. See parameters of
        spatial.threshold_components

        @param medw:
        @param thr_method:
        @param maxthr:
        @param extract_cc:
        @param se:
        @param ss:
        @param dview:
        @return:
        '''

        if self.A_thr is None:
            A_thr = threshold_components(self.A, self.dims,  maxthr=maxthr, dview=dview,
                                         medw=None, thr_method='max', nrgthr=0.99,
                                         extract_cc=True, se=None, ss=None)

            self.A_thr = A_thr
        else:
            print('A_thr already computed. If you want to recompute set self.A_thr to None')

    def remove_small_large_neurons(self, min_size_neuro, max_size_neuro,
                                   select_comp=False):
        ''' remove neurons that are too large or too small

    	Args:
            min_size_neuro: int
                min size in pixels
            max_size_neuro: int
                max size in pixels
            select_comp: bool
                remove components that are too small/large from main estimates
                fields. See estimates.selecte_components() for more details.

        Returns:
            neurons_to_keep: np.array
                indeces of components with size within the acceptable range
        '''
        if self.A_thr is None:
            raise Exception('You need to compute thresolded components before calling remove_duplicates: use the threshold_components method')

        A_gt_thr_bin = self.A_thr.toarray() > 0
        size_neurons_gt = A_gt_thr_bin.sum(0)
        neurons_to_keep = np.where((size_neurons_gt > min_size_neuro) & (size_neurons_gt < max_size_neuro))[0]
#        self.select_components(idx_components=neurons_to_keep)
        if self.idx_components is None:
            self.idx_components = np.arange(self.A.shape[-1])
        self.idx_components = np.intersect1d(self.idx_components, neurons_to_keep)
        self.idx_components_bad = np.setdiff1d(np.arange(self.A.shape[-1]), self.idx_components)
        if select_comp:
            self.select_components(use_object=True)
        return neurons_to_keep



    def remove_duplicates(self, predictions=None, r_values=None, dist_thr=0.1,
                          min_dist=10, thresh_subset=0.6, plot_duplicates=False,
                          select_comp=False):
        ''' remove neurons that heavily overlap and might be duplicates.

        Args:
            predictions
            r_values
            dist_thr
            min_dist
            thresh_subset
            plot_duplicates
        '''
        if self.A_thr is None:
            raise Exception('You need to compute thresolded components before calling remove_duplicates: use the threshold_components method')

        A_gt_thr_bin = (self.A_thr.toarray() > 0).reshape([self.dims[0], self.dims[1], -1], order='F').transpose([2, 0, 1]) * 1.

        duplicates_gt, indices_keep_gt, indices_remove_gt, D_gt, overlap_gt = detect_duplicates_and_subsets(
            A_gt_thr_bin,predictions=predictions, r_values=r_values, dist_thr=dist_thr, min_dist=min_dist,
            thresh_subset=thresh_subset)
        logging.info('Number of duplicates: {}'.format(len(duplicates_gt)))
        if len(duplicates_gt) > 0:
            if plot_duplicates:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(A_gt_thr_bin[np.array(duplicates_gt).flatten()].sum(0))
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(A_gt_thr_bin[np.array(indices_keep_gt)[:]].sum(0))
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.imshow(A_gt_thr_bin[np.array(indices_remove_gt)[:]].sum(0))
                plt.colorbar()
                plt.pause(1)

            components_to_keep = np.delete(np.arange(self.A.shape[-1]), indices_remove_gt)

        else:
            components_to_keep = np.arange(self.A.shape[-1])

        if self.idx_components is None:
            self.idx_components = np.arange(self.A.shape[-1])
        self.idx_components = np.intersect1d(self.idx_components, components_to_keep)
        self.idx_components_bad = np.setdiff1d(np.arange(self.A.shape[-1]), self.idx_components)
        if select_comp:
            self.select_components(use_object=True)

        return components_to_keep

    def masks_2_neurofinder(self, dataset_name):
        """Return masks to neurofinder format
        """
        if self.A_thr is None:
            raise Exception(
                'You need to compute thresolded components before calling this method: use the threshold_components method')
        bin_masks = self.A_thr.reshape([self.dims[0], self.dims[1], -1], order='F').transpose([2, 0, 1])
        return nf_masks_to_neurof_dict(bin_masks, dataset_name)

    def save_NWB(self,
                 filename,
                 imaging_plane_name=None,
                 imaging_series_name=None,
                 sess_desc='CaImAn Results',
                 exp_desc=None,
                 identifier=None,
                 imaging_rate=30.,
                 starting_time=0.,
                 session_start_time=None,
                 excitation_lambda=488.0,
                 imaging_plane_description='some imaging plane description',
                 emission_lambda=520.0,
                 indicator='OGB-1',
                 location='brain',
                 raw_data_file=None):
        """writes NWB file

        Args:
            filename: str

            imaging_plane_name: str, optional

            imaging_series_name: str, optional

            sess_desc: str, optional

            exp_desc: str, optional

            identifier: str, optional

            imaging_rate: float, optional
                default: 30 (Hz)

            starting_time: float, optional
                default: 0.0 (seconds)

            location: str, optional

            session_start_time: datetime.datetime, optional
                Only required for new files

            excitation_lambda: float

            imaging_plane_description: str

            emission_lambda: float

            indicator: str

            location: str
        """

        from pynwb import NWBHDF5IO, TimeSeries, NWBFile
        from pynwb.base import Images
        from pynwb.image import GrayscaleImage
        from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, ImageSeries
        from pynwb.device import Device
        import os

        if identifier is None:
            import uuid
            identifier = uuid.uuid1().hex

        if '.nwb' != os.path.splitext(filename)[-1].lower():
            raise Exception("Wrong filename")

        if not os.path.isfile(filename):  # if the file doesn't exist create new and add the original data path
            print('filename does not exist. Creating new NWB file with only estimates output')

            nwbfile = NWBFile(sess_desc, identifier, session_start_time, experiment_description=exp_desc)
            device = Device('imaging_device')
            nwbfile.add_device(device)
            optical_channel = OpticalChannel('OpticalChannel',
                                             'main optical channel',
                                             emission_lambda=emission_lambda)
            nwbfile.create_imaging_plane(name='ImagingPlane',
                                         optical_channel=optical_channel,
                                         description=imaging_plane_description,
                                         device=device,
                                         excitation_lambda=excitation_lambda,
                                         imaging_rate=imaging_rate,
                                         indicator=indicator,
                                         location=location)
            if raw_data_file:
                nwbfile.add_acquisition(ImageSeries(name='TwoPhotonSeries',
                                                    external_file=[raw_data_file],
                                                    format='external',
                                                    rate=imaging_rate,
                                                    starting_frame=[0]))
            with NWBHDF5IO(filename, 'w') as io:
                io.write(nwbfile)

        time.sleep(4)  # ensure the file is fully closed before opening again in append mode
        logging.info('Saving the results in the NWB file...')

        with NWBHDF5IO(filename, 'r+') as io:
            nwbfile = io.read()
            # Add processing results

            # Create the module as 'ophys' unless it is taken and append 'ophysX' instead
            ophysmodules = [s[5:] for s in list(nwbfile.modules) if s.startswith('ophys')]
            if any('' in s for s in ophysmodules):
                if any([s for s in ophysmodules if s.isdigit()]):
                    nummodules = max([int(s) for s in ophysmodules if s.isdigit()])+1
                    print('ophys module previously created, writing to ophys'+str(nummodules)+' instead')
                    mod = nwbfile.create_processing_module('ophys'+str(nummodules), 'contains caiman estimates for '
                                                                                    'the main imaging plane')
                else:
                    print('ophys module previously created, writing to ophys1 instead')
                    mod = nwbfile.create_processing_module('ophys1', 'contains caiman estimates for the main '
                                                                     'imaging plane')
            else:
                mod = nwbfile.create_processing_module('ophys', 'contains caiman estimates for the main imaging plane')

            img_seg = ImageSegmentation()
            mod.add(img_seg)
            fl = Fluorescence()
            mod.add_data_interface(fl)
#            mot_crct = MotionCorrection()
#            mod.add_data_interface(mot_crct)

            # Add the ROI-related stuff
            if imaging_plane_name is not None:
                imaging_plane = nwbfile.imaging_planes[imaging_plane_name]
            else:
                if len(nwbfile.imaging_planes) == 1:
                    imaging_plane = list(nwbfile.imaging_planes.values())[0]
                else:
                    raise Exception('There is more than one imaging plane in the file, you need to specify the name'
                                    ' via the "imaging_plane_name" parameter')

            if imaging_series_name is not None:
                image_series = nwbfile.acquisition[imaging_series_name]
            else:
                if not len(nwbfile.acquisition):
                    image_series = None
                elif len(nwbfile.acquisition) == 1:
                    image_series = list(nwbfile.acquisition.values())[0]
                else:
                    raise Exception('There is more than one imaging plane in the file, you need to specify the name'
                                    ' via the "imaging_series_name" parameter')

            ps = img_seg.create_plane_segmentation(
                name='PlaneSegmentation',
                description='CNMF_ROIs',
                imaging_plane=imaging_plane,
                reference_images=image_series)

            ps.add_column('r', 'description of r values')
            ps.add_column('snr', 'signal to noise ratio')
            ps.add_column('accepted', 'in accepted list')
            ps.add_column('rejected', 'in rejected list')
            if self.cnn_preds:
                ps.add_column('cnn', 'description of CNN')
            if self.idx_components:
                ps.add_column('keep', 'in idx_components')

            # Add ROIs
            for i in range(self.A.shape[-1]):
                add_roi_kwargs = dict(image_mask=self.A.T[i].T.toarray().reshape(self.dims),
                                      r=self.r_values[i], snr=self.SNR_comp[i], accepted=False, rejected=False)
                if hasattr(self, 'accepted_list'):
                    add_roi_kwargs.update(accepted=i in self.accepted_list)
                if hasattr(self, 'rejected_list'):
                    add_roi_kwargs.update(rejected=i in self.rejected_list)
                if self.cnn_preds:
                    add_roi_kwargs.update(cnn=self.cnn_preds[i])
                if self.idx_components:
                    add_roi_kwargs.update(keep=i in self.idx_components)

                ps.add_roi(**add_roi_kwargs)

            for bg in self.b.T:  # Backgrounds
                add_bg_roi_kwargs = dict(image_mask=bg.reshape(self.dims), r=np.nan, snr=np.nan, accepted=False,
                                         rejected=False)
                if 'keep' in ps.colnames:
                    add_bg_roi_kwargs.update(keep=False)
                if 'cnn' in ps.colnames:
                    add_bg_roi_kwargs.update(cnn=np.nan)
                ps.add_roi(**add_bg_roi_kwargs)

            # Add Traces
            n_rois = self.A.shape[-1]
            n_bg = len(self.f)
            rt_region_roi = ps.create_roi_table_region(
                'ROIs', region=list(range(n_rois)))

            rt_region_bg = ps.create_roi_table_region(
                'Background', region=list(range(n_rois, n_rois+n_bg)))

            timestamps = np.arange(self.f.shape[1]) / imaging_rate + starting_time

            # Neurons
            fl.create_roi_response_series(name='RoiResponseSeries', data=self.C.T, rois=rt_region_roi, unit='lumens',
                                          timestamps=timestamps)
            # Background
            fl.create_roi_response_series(name='Background_Fluorescence_Response', data=self.f.T, rois=rt_region_bg,
                                          unit='lumens', timestamps=timestamps)

            mod.add(TimeSeries(name='residuals', description='residuals', data=self.YrA.T, timestamps=timestamps,
                               unit='NA'))
            if hasattr(self, 'Cn'):
                images = Images('summary_images')
                images.add_image(GrayscaleImage(name='local_correlations', data=self.Cn))

                # Add MotionCorrection
    #            create_corrected_image_stack(corrected, original, xy_translation, name='CorrectedImageStack')
            io.write(nwbfile)


