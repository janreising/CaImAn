
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


import cp_cnmfe_2 as CNMFE_2
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



class CNMF(object):
    """  Source extraction using constrained non-negative matrix factorization.

    The general class which is used to produce a factorization of the Y matrix being the video
    it computes it using all the files inside of cnmf folder.
    Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
    of the structure of the class

    it is calling everyfunction from the cnmf folder
    you can find out more at how the functions are called and how they are laid out at the ipython notebook

    See Also:
    @url http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
    .. image:: docs/img/quickintro.png
    @author andrea giovannucci
    """
    def __init__(self, n_processes, k=5, gSig=[4, 4], gSiz=None, merge_thresh=0.8, p=2, dview=None,
                 Ain=None, Cin=None, b_in=None, f_in=None, do_merge=True,
                 ssub=2, tsub=2, p_ssub=1, p_tsub=1, method_init='greedy_roi', alpha_snmf=100,
                 rf=None, stride=None, memory_fact=1, gnb=1, nb_patch=1, only_init_patch=False,
                 method_deconvolution='oasis', n_pixels_per_process=4000, block_size_temp=5000, num_blocks_per_run_temp=20,
                 block_size_spat=5000, num_blocks_per_run_spat=20,
                 check_nan=True, skip_refinement=False, normalize_init=True, options_local_NMF=None,
                 minibatch_shape=100, minibatch_suff_stat=3,
                 update_num_comps=True, rval_thr=0.9, thresh_fitness_delta=-20,
                 thresh_fitness_raw=None, thresh_overlap=.5,
                 max_comp_update_shape=np.inf, num_times_comp_updated=np.inf,
                 batch_update_suff_stat=False, s_min=None,
                 remove_very_bad_comps=False, border_pix=0, low_rank_background=True,
                 update_background_components=True, rolling_sum=True, rolling_length=100,
                 min_corr=.85, min_pnr=20, ring_size_factor=1.5,
                 center_psf=False, use_dense=True, deconv_flag=True,
                 simultaneously=False, n_refit=0, del_duplicates=False, N_samples_exceptionality=None,
                 max_num_added=3, min_num_trial=2, thresh_CNN_noisy=0.5,
                 fr=30, decay_time=0.4, min_SNR=2.5, ssub_B=2, init_iter=2,
                 sniper_mode=False, use_peak_max=False, test_both=False,
                 expected_comps=500, max_merge_area=None, params=None):
        """
        Constructor of the CNMF method

        Args:
            n_processes: int
               number of processed used (if in parallel this controls memory usage)

            k: int
               number of neurons expected per FOV (or per patch if patches_pars is  None)

            gSig: tuple
                expected half size of neurons

            merge_thresh: float
                merging threshold, max correlation allowed

            dview: Direct View object
                for parallelization purposes when using ipyparallel

            p: int
                order of the autoregressive process used to estimate deconvolution

            Ain: ndarray
                if know, it is the initial estimate of spatial filters

            ssub: int
                downsampleing factor in space

            tsub: int
                 downsampling factor in time

            p_ssub: int
                downsampling factor in space for patches

            method_init: str
               can be greedy_roi or sparse_nmf

            alpha_snmf: float
                weight of the sparsity regularization

            p_tsub: int
                 downsampling factor in time for patches

            rf: int
                half-size of the patches in pixels. rf=25, patches are 50x50

            gnb: int
                number of global background components

            nb_patch: int
                number of background components per patch

            stride: int
                amount of overlap between the patches in pixels

            memory_fact: float
                unitless number accounting how much memory should be used. You will
                 need to try different values to see which one would work the default is OK for a 16 GB system

            N_samples_fitness: int
                number of samples over which exceptional events are computed (See utilities.evaluate_components)

            only_init_patch= boolean
                only run initialization on patches

            method_deconvolution = 'oasis' or 'cvxpy'
                method used for deconvolution. Suggested 'oasis' see
                Friedrich J, Zhou P, Paninski L. Fast Online Deconvolution of Calcium Imaging Data.
                PLoS Comput Biol. 2017; 13(3):e1005423.

            n_pixels_per_process: int.
                Number of pixels to be processed in parallel per core (no patch mode). Decrease if memory problems

            block_size: int.
                Number of pixels to be used to perform residual computation in blocks. Decrease if memory problems

            num_blocks_per_run_spat: int
                In case of memory problems you can reduce this numbers, controlling the number of blocks processed in parallel during residual computing

            num_blocks_per_run_temp: int
                In case of memory problems you can reduce this numbers, controlling the number of blocks processed in parallel during residual computing

            check_nan: Boolean.
                Check if file contains NaNs (costly for very large files so could be turned off)

            skip_refinement:
                Bool. If true it only performs one iteration of update spatial update temporal instead of two

            normalize_init=Bool.
                Differences in intensities on the FOV might caus troubles in the initialization when patches are not used,
                 so each pixels can be normalized by its median intensity

            options_local_NMF:
                experimental, not to be used

            remove_very_bad_comps:Bool
                whether to remove components with very low values of component quality directly on the patch.
                 This might create some minor imprecisions.
                However benefits can be considerable if done because if many components (>2000) are created
                and joined together, operation that causes a bottleneck

            border_pix:int
                number of pixels to not consider in the borders

            low_rank_background:bool
                if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)
                 In the False case all the nonzero elements of the background components are updated using hals (to be used with one background per patch)

            update_background_components:bool
                whether to update the background components during the spatial phase

            min_corr: float
                minimal correlation peak for 1-photon imaging initialization

            min_pnr: float
                minimal peak  to noise ratio for 1-photon imaging initialization

            ring_size_factor: float
                it's the ratio between the ring radius and neuron diameters.

                    max_comp_update_shape:
                             threshold number of components after which selective updating starts (using the parameter num_times_comp_updated)

                num_times_comp_updated:
                number of times each component is updated. In inf components are updated at every initbatch time steps

            expected_comps: int
                number of expected components (try to exceed the expected)

            deconv_flag : bool, optional
                If True, deconvolution is also performed using OASIS

            simultaneously : bool, optional
                If true, demix and denoise/deconvolve simultaneously. Slower but can be more accurate.

            n_refit : int, optional
                Number of pools (cf. oasis.pyx) prior to the last one that are refitted when
                simultaneously demixing and denoising/deconvolving.

            N_samples_exceptionality : int, optional
                Number of consecutives intervals to be considered when to_julia new neuron candidates

            del_duplicates: bool
                whether to delete the duplicated created in initialization

            max_num_added : int, optional
                maximum number of components to be added at each step in OnACID

            min_num_trial : int, optional
                minimum numbers of attempts to include a new components in OnACID

            thresh_CNN_noisy: float
                threshold on the per patch CNN classifier for online algorithm

            ssub_B: int, optional
                downsampleing factor for 1-photon imaging background computation

            init_iter: int, optional
                number of iterations for 1-photon imaging initialization

            max_merge_area: int, optional
                maximum area (in pixels) of merged components, used to determine whether to merge components during fitting process
        """

        self.dview = dview

        # these are movie properties that will be refactored into the Movie object
        self.dims = None
        self.empty_merged = None

        # these are member variables related to the CNMF workflow
        self.skip_refinement = skip_refinement
        self.remove_very_bad_comps = remove_very_bad_comps

        if params is None:
            self.params = cp_params.CNMFParams(
                border_pix=border_pix, del_duplicates=del_duplicates, low_rank_background=low_rank_background,
                memory_fact=memory_fact, n_processes=n_processes, nb_patch=nb_patch, only_init_patch=only_init_patch,
                p_ssub=p_ssub, p_tsub=p_tsub, remove_very_bad_comps=remove_very_bad_comps, rf=rf, stride=stride,
                check_nan=check_nan, n_pixels_per_process=n_pixels_per_process,
                k=k, center_psf=center_psf, gSig=gSig, gSiz=gSiz,
                init_iter=init_iter, method_init=method_init, min_corr=min_corr,  min_pnr=min_pnr,
                gnb=gnb, normalize_init=normalize_init, options_local_NMF=options_local_NMF,
                ring_size_factor=ring_size_factor, rolling_length=rolling_length, rolling_sum=rolling_sum,
                ssub=ssub, ssub_B=ssub_B, tsub=tsub,
                block_size_spat=block_size_spat, num_blocks_per_run_spat=num_blocks_per_run_spat,
                block_size_temp=block_size_temp, num_blocks_per_run_temp=num_blocks_per_run_temp,
                update_background_components=update_background_components,
                method_deconvolution=method_deconvolution, p=p, s_min=s_min,
                do_merge=do_merge, merge_thresh=merge_thresh,
                decay_time=decay_time, fr=fr, min_SNR=min_SNR, rval_thr=rval_thr,
                N_samples_exceptionality=N_samples_exceptionality, batch_update_suff_stat=batch_update_suff_stat,
                expected_comps=expected_comps, max_comp_update_shape=max_comp_update_shape, max_num_added=max_num_added,
                min_num_trial=min_num_trial, minibatch_shape=minibatch_shape, minibatch_suff_stat=minibatch_suff_stat,
                n_refit=n_refit, num_times_comp_updated=num_times_comp_updated, simultaneously=simultaneously,
                sniper_mode=sniper_mode, test_both=test_both, thresh_CNN_noisy=thresh_CNN_noisy,
                thresh_fitness_delta=thresh_fitness_delta, thresh_fitness_raw=thresh_fitness_raw, thresh_overlap=thresh_overlap,
                update_num_comps=update_num_comps, use_dense=use_dense, use_peak_max=use_peak_max, alpha_snmf=alpha_snmf,
                max_merge_area=max_merge_area
            )
        else:
            self.params = params
            params.set('patch', {'n_processes': n_processes})

        self.estimates = CNMFE_3.Estimates(A=Ain, C=Cin, b=b_in, f=f_in,
                                   dims=self.params.data['dims'])

    def fit_file(self, indices=None, include_eval=False):
        """
        This method packages the analysis pipeline (motion correction, memory
        mapping, patch based CNMF processing and component evaluation) in a
        single method that can be called on a specific (sequence of) file(s).
        It is assumed that the CNMF object already contains a params object
        where the location of the files and all the relevant parameters have
        been specified. The method will perform the last step, i.e. component
        evaluation, if the flag "include_eval" is set to `True`.

        Args:
            motion_correct (bool)
                flag for performing motion correction
            indices (list of slice objects)
                perform analysis only on a part of the FOV
            include_eval (bool)
                flag for performing component evaluation
        Returns:
            cnmf object with the current estimates
        """
        if indices is None:
            indices = (slice(None), slice(None))
        fnames = self.params.get('data', 'fnames')
        if os.path.exists(fnames[0]):
            _, extension = os.path.splitext(fnames[0])[:2]
            extension = extension.lower()
        else:
            logging.warning("Error: File not found, with file list:\n" + fnames[0])
            raise Exception('File not found!')

        base_name = pathlib.Path(fnames[0]).stem + "_memmap_"
        if extension == '.mmap':
            fname_new = fnames[0]
            Yr, dims, T = cp_motioncorrection.load_memmap(fnames[0])
            if np.isfortran(Yr):
                raise Exception('The file should be in C order (see save_memmap function)')
        else:
            fname_new = cp_motioncorrection.save_memmap(fnames, base_name=base_name, order='C')
            Yr, dims, T = cp_motioncorrection.load_memmap(fname_new)

        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        self.mmap_file = fname_new
        if not include_eval:
            return self.fit(images, indices=indices)

        fit_cnm = self.fit(images, indices=indices)
        Cn = local_correlations(images[::max(T//1000, 1)], swap_dim=False)
        Cn[np.isnan(Cn)] = 0
        fit_cnm.save(fname_new[:-5]+'_init.hdf5')
        #fit_cnm.params.change_params({'p': self.params.get('preprocess', 'p')})
        # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
        cnm2 = fit_cnm.refit(images, dview=self.dview)
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=self.dview)
        # update object with selected components
        #cnm2.estimates.select_components(use_object=True)
        # Extract DF/F values
        cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
        cnm2.estimates.Cn = Cn
        cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')

        cp_motioncorrection.stop_server(dview=self.dview)
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        return cnm2


    def refit(self, images, dview=None):
        """
        Refits the data using CNMF initialized from a previous interation

        Args:
            images
            dview
        Returns:
            cnm
                A new CNMF object
        """
        from copy import deepcopy
        cnm = CNMF(self.params.patch['n_processes'], params=self.params, dview=dview)
        cnm.params.patch['rf'] = None
        cnm.params.patch['only_init'] = False
        estimates = deepcopy(self.estimates)
        estimates.select_components(use_object=True)
        estimates.coordinates = None
        cnm.estimates = estimates
        cnm.mmap_file = self.mmap_file
        return cnm.fit(images)

    def fit(self, images, indices=(slice(None), slice(None))):
        """
        This method uses the cnmf algorithm to find sources in data.
        it is calling every function from the cnmf folder
        you can find out more at how the functions are called and how they are laid out at the ipython notebook

        Args:
            images : mapped np.ndarray of shape (t,x,y[,z]) containing the images that vary over time.

            indices: list of slice objects along dimensions (x,y[,z]) for processing only part of the FOV

        Returns:
            self: updated using the cnmf algorithm with C,A,S,b,f computed according to the given initial values

        Raises:
        Exception 'You need to provide a memory mapped file as input if you use patches!!'

        See Also:
        ..image::docs/img/quickintro.png

        http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3

        """
        # Todo : to compartment
        if isinstance(indices, slice):
            indices = [indices]
        if isinstance(indices, tuple):
            indices = list(indices)
        indices = [slice(None)] + indices
        if len(indices) < len(images.shape):
            indices = indices + [slice(None)]*(len(images.shape) - len(indices))
        dims_orig = images.shape[1:]
        dims_sliced = images[tuple(indices)].shape[1:]
        is_sliced = (dims_orig != dims_sliced)
        if self.params.get('patch', 'rf') is None and (is_sliced or 'ndarray' in str(type(images))):
            images = images[tuple(indices)]
            self.dview = None
            logging.info("Parallel processing in a single patch "
                            "is not available for loaded in memory or sliced" +
                            " data.")

        T = images.shape[0]
        self.params.set('online', {'init_batch': T})
        self.dims = images.shape[1:]
        #self.params.data['dims'] = images.shape[1:]
        Y = np.transpose(images, list(range(1, len(self.dims) + 1)) + [0])
        Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
        if np.isfortran(Yr):
            raise Exception('The file is in F order, it should be in C order (see save_memmap function)')

        logging.info((T,) + self.dims)

        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        try:
            Y.filename = images.filename
            Yr.filename = images.filename
            self.mmap_file = images.filename
        except AttributeError:  # if no memmapping cause working with small data
            pass

        # update/set all options that depend on data dimensions
        # number of rows, columns [and depths]
        self.params.set('spatial', {'medw': (3,) * len(self.dims),
                                    'se': np.ones((3,) * len(self.dims), dtype=np.uint8),
                                    'ss': np.ones((3,) * len(self.dims), dtype=np.uint8)
                                    })

        logging.info(('Using ' + str(self.params.get('patch', 'n_processes')) + ' processes'))
        if self.params.get('preprocess', 'n_pixels_per_process') is None:
            avail_memory_per_process = psutil.virtual_memory()[
                1] / 2.**30 / self.params.get('patch', 'n_processes')
            mem_per_pix = 3.6977678498329843e-09
            npx_per_proc = np.int(avail_memory_per_process / 8. / mem_per_pix / T)
            npx_per_proc = np.int(np.minimum(npx_per_proc, np.prod(self.dims) // self.params.get('patch', 'n_processes')))
            self.params.set('preprocess', {'n_pixels_per_process': npx_per_proc})

        self.params.set('spatial', {'n_pixels_per_process': self.params.get('preprocess', 'n_pixels_per_process')})

        logging.info('using ' + str(self.params.get('preprocess', 'n_pixels_per_process')) + ' pixels per process')
        logging.info('using ' + str(self.params.get('spatial', 'block_size_spat')) + ' block_size_spat')
        logging.info('using ' + str(self.params.get('temporal', 'block_size_temp')) + ' block_size_temp')

        if self.params.get('patch', 'rf') is None:  # no patches
            logging.info('preprocessing ...')
            Yr = self.preprocess(Yr)
            if self.estimates.A is None:
                logging.info('initializing ...')
                self.initialize(Y)

            if self.params.get('patch', 'only_init'):  # only return values after initialization
                if not (self.params.get('init', 'method_init') == 'corr_pnr' and
                    self.params.get('init', 'ring_size_factor') is not None):
                    self.compute_residuals(Yr)
                    self.estimates.bl = None
                    self.estimates.c1 = None
                    self.estimates.neurons_sn = None


                if self.remove_very_bad_comps:
                    logging.info('removing bad components : ')
                    final_frate = 10
                    r_values_min = 0.5  # threshold on space consistency
                    fitness_min = -15  # threshold on time variability
                    fitness_delta_min = -15
                    Npeaks = 10
                    traces = np.array(self.C)
                    logging.info('estimating the quality...')
                    idx_components, idx_components_bad, fitness_raw,\
                        fitness_delta, r_values = estimate_components_quality(
                            traces, Y, self.estimates.A, self.estimates.C, self.estimates.b, self.estimates.f,
                            final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
                            fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, return_all=True, N=5)

                    logging.info(('Keeping ' + str(len(idx_components)) +
                           ' and discarding  ' + str(len(idx_components_bad))))
                    self.estimates.C = self.estimates.C[idx_components]
                    self.estimates.A = self.estimates.A[:, idx_components] # type: ignore # not provable that self.initialise provides a value
                    self.estimates.YrA = self.estimates.YrA[idx_components]

                self.estimates.normalize_components()

                return self

            logging.info('update spatial ...')
            self.update_spatial(Yr, use_init=True)

            logging.info('update temporal ...')
            if not self.skip_refinement:
                # set this to zero for fast updating without deconvolution
                self.params.set('temporal', {'p': 0})
            else:
                self.params.set('temporal', {'p': self.params.get('preprocess', 'p')})
                logging.info('deconvolution ...')

            self.update_temporal(Yr)

            if not self.skip_refinement:
                logging.info('refinement...')
                if self.params.get('merging', 'do_merge'):
                    logging.info('merging components ...')
                    self.merge_comps(Yr, mx=50, fast_merge=True, max_merge_area=self.params.get('merging', 'max_merge_area'))

                logging.info('Updating spatial ...')

                self.update_spatial(Yr, use_init=False)
                # set it back to original value to perform full deconvolution
                self.params.set('temporal', {'p': self.params.get('preprocess', 'p')})
                logging.info('update temporal ...')
                self.update_temporal(Yr, use_init=False)
            # else:
            #     todo : ask for those..
                # C, f, S, bl, c1, neurons_sn, g1, YrA, lam = self.estimates.C, self.estimates.f, self.estimates.S, self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, self.estimates.g, self.estimates.YrA, self.estimates.lam

            # embed in the whole FOV
            if is_sliced:
                FOV = np.zeros(dims_orig, order='C')
                FOV[indices[1:]] = 1
                FOV = FOV.flatten(order='F')
                ind_nz = np.where(FOV>0)[0].tolist()
                self.estimates.A = self.estimates.A.tocsc()
                A_data = self.estimates.A.data
                A_ind = np.array(ind_nz)[self.estimates.A.indices]
                A_ptr = self.estimates.A.indptr
                A_FOV = scipy.sparse.csc_matrix((A_data, A_ind, A_ptr),
                                                shape=(FOV.shape[0], self.estimates.A.shape[-1]))
                b_FOV = np.zeros((FOV.shape[0], self.estimates.b.shape[-1]))
                b_FOV[ind_nz] = self.estimates.b
                self.estimates.A = A_FOV
                self.estimates.b = b_FOV

        else:  # use patches
            if self.params.get('patch', 'stride') is None:
                self.params.set('patch', {'stride': np.int(self.params.get('patch', 'rf') * 2 * .1)})
                logging.info(
                    ('Setting the stride to 10% of 2*rf automatically:' + str(self.params.get('patch', 'stride'))))

            if not isinstance(images, np.memmap):
                raise Exception(
                    'You need to provide a memory mapped file as input if you use patches!!')

            self.estimates.A, self.estimates.C, self.estimates.YrA, self.estimates.b, self.estimates.f, \
                self.estimates.sn, self.estimates.optional_outputs = CNMFE_2.run_CNMF_patches(
                    images.filename, self.dims + (T,), self.params,
                    dview=self.dview, memory_fact=self.params.get('patch', 'memory_fact'),
                    gnb=self.params.get('init', 'nb'), border_pix=self.params.get('patch', 'border_pix'),
                    low_rank_background=self.params.get('patch', 'low_rank_background'),
                    del_duplicates=self.params.get('patch', 'del_duplicates'),
                    indices=indices)

            self.estimates.bl, self.estimates.c1, self.estimates.g, self.estimates.neurons_sn = None, None, None, None
            logging.info("merging")
            self.estimates.merged_ROIs = [0]


            if self.params.get('init', 'center_psf'):  # merge taking best neuron
                if self.params.get('patch', 'nb_patch') > 0:

                    while len(self.estimates.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.Inf, fast_merge=True)

                    logging.info("update temporal")
                    self.update_temporal(Yr, use_init=False)

                    self.params.set('spatial', {'se': np.ones((1,) * len(self.dims), dtype=np.uint8)})
                    logging.info('update spatial ...')
                    self.update_spatial(Yr, use_init=False)

                    logging.info("update temporal")
                    self.update_temporal(Yr, use_init=False)
                else:
                    while len(self.estimates.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.Inf, fast_merge=True)
                        #if len(self.estimates.merged_ROIs) > 0:
                            #not_merged = np.setdiff1d(list(range(len(self.estimates.YrA))),
                            #                          np.unique(np.concatenate(self.estimates.merged_ROIs)))
                            #self.estimates.YrA = np.concatenate([self.estimates.YrA[not_merged],
                            #                           np.array([self.estimates.YrA[m].mean(0) for ind, m in enumerate(self.estimates.merged_ROIs) if not self.empty_merged[ind]])])
                    if self.params.get('init', 'nb') == 0:
                        self.estimates.W, self.estimates.b0 = CNMFE_2.compute_W(
                            Yr, self.estimates.A.toarray(), self.estimates.C, self.dims,
                            self.params.get('init', 'ring_size_factor') *
                            self.params.get('init', 'gSiz')[0],
                            ssub=self.params.get('init', 'ssub_B'))
                    if len(self.estimates.C):
                        self.deconvolve()
                        self.estimates.C = self.estimates.C.astype(np.float32)
                    else:
                        self.estimates.S = self.estimates.C
            else:
                while len(self.estimates.merged_ROIs) > 0:
                    self.merge_comps(Yr, mx=np.Inf)

                logging.info("update temporal")
                self.update_temporal(Yr, use_init=False)

        self.estimates.normalize_components()
        return self

    def save(self, filename):
        '''save object in hdf5 file format

        Args:
            filename: str
                path to the hdf5 file containing the saved object
        '''

        if '.hdf5' in filename:
            filename = cp_motioncorrection.fn_relocated(filename)
            CNMFE_2.save_dict_to_hdf5(self.__dict__, filename)
        else:
            raise Exception("File extension not supported for cnmf.save")

    def remove_components(self, ind_rm):
        """
        Remove a specified list of components from the CNMF object.

        Args:
            ind_rm :    list
                        indices of components to be removed
        """

        self.estimates.Ab, self.estimates.Ab_dense, self.estimates.CC, self.estimates.CY, self.M,\
            self.N, self.estimates.noisyC, self.estimates.OASISinstances, self.estimates.C_on,\
            expected_comps, self.ind_A,\
            self.estimates.groups, self.estimates.AtA = CNMFE_2.remove_components_online(
                ind_rm, self.params.get('init', 'nb'), self.estimates.Ab,
                self.params.get('online', 'use_dense'), self.estimates.Ab_dense,
                self.estimates.AtA, self.estimates.CY, self.estimates.CC, self.M, self.N,
                self.estimates.noisyC, self.estimates.OASISinstances, self.estimates.C_on,
                self.params.get('online', 'expected_comps'))
        self.params.set('online', {'expected_comps': expected_comps})

    def compute_residuals(self, Yr):
        """
        Compute residual trace for each component (variable YrA).
        WARNING: At the moment this method is valid only for the 2p processing
        pipeline

         Args:
             Yr :    np.ndarray
                     movie in format pixels (d) x frames (T)
        """
        block_size, num_blocks_per_run = self.params.get('temporal', 'block_size_temp'), self.params.get('temporal', 'num_blocks_per_run_temp')
        if 'csc_matrix' not in str(type(self.estimates.A)):
            self.estimates.A = scipy.sparse.csc_matrix(self.estimates.A)
        if 'array' not in str(type(self.estimates.b)):
            self.estimates.b = self.estimates.b.toarray()
        if 'array' not in str(type(self.estimates.C)):
            self.estimates.C = self.estimates.C.estimates.toarray()
        if 'array' not in str(type(self.estimates.f)):
            self.estimates.f = self.estimates.f.toarray()

        Ab = scipy.sparse.hstack((self.estimates.A, self.estimates.b)).tocsc()
        nA2 = np.ravel(Ab.power(2).sum(axis=0))
        nA2_inv_mat = scipy.sparse.spdiags(
            1. / (nA2 + np.finfo(np.float32).eps), 0, nA2.shape[0], nA2.shape[0])
        Cf = np.vstack((self.estimates.C, self.estimates.f))
        if 'numpy.ndarray' in str(type(Yr)):
            YA = (Ab.T.dot(Yr)).T * nA2_inv_mat
        else:
            YA = CNMFE_2.parallel_dot_product(Yr, Ab, dview=self.dview, block_size=block_size,
                                           transpose=True, num_blocks_per_run=num_blocks_per_run) * nA2_inv_mat

        AA = Ab.T.dot(Ab) * nA2_inv_mat
        self.estimates.YrA = (YA - (AA.T.dot(Cf)).T)[:, :self.estimates.A.shape[-1]].T
        self.estimates.R = self.estimates.YrA

        return self


    def deconvolve(self, p=None, method_deconvolution=None, bas_nonneg=None,
                   noise_method=None, optimize_g=0, s_min=None, **kwargs):
        """Performs deconvolution on already extracted traces using
        constrained foopsi.
        """

        p = self.params.get('preprocess', 'p') if p is None else p
        method_deconvolution = (self.params.get('temporal', 'method_deconvolution')
                if method_deconvolution is None else method_deconvolution)
        bas_nonneg = (self.params.get('temporal', 'bas_nonneg')
                      if bas_nonneg is None else bas_nonneg)
        noise_method = (self.params.get('temporal', 'noise_method')
                        if noise_method is None else noise_method)
        s_min = self.params.get('temporal', 's_min') if s_min is None else s_min

        F = self.estimates.C + self.estimates.YrA
        args = dict()
        args['p'] = p
        args['method_deconvolution'] = method_deconvolution
        args['bas_nonneg'] = bas_nonneg
        args['noise_method'] = noise_method
        args['s_min'] = s_min
        args['optimize_g'] = optimize_g
        args['noise_range'] = self.params.get('temporal', 'noise_range')
        args['fudge_factor'] = self.params.get('temporal', 'fudge_factor')

        args_in = [(F[jj], None, jj, None, None, None, None,
                    args) for jj in range(F.shape[0])]

        if 'multiprocessing' in str(type(self.dview)):
            results = self.dview.map_async(
                CNMFE_2.constrained_foopsi_parallel, args_in).get(4294967)
        elif self.dview is not None:
            results = self.dview.map_sync(CNMFE_2.constrained_foopsi_parallel, args_in)
        else:
            results = list(map(CNMFE_2.constrained_foopsi_parallel, args_in))

        if sys.version_info >= (3, 0):
            results = list(zip(*results))
        else:  # python 2
            results = zip(*results)

        order = list(results[7])
        self.estimates.C = np.stack([results[0][i] for i in order])
        self.estimates.S = np.stack([results[1][i] for i in order])
        self.estimates.bl = [results[3][i] for i in order]
        self.estimates.c1 = [results[4][i] for i in order]
        self.estimates.g = [results[6][i] for i in order]
        self.estimates.neurons_sn = [results[5][i] for i in order]
        self.estimates.lam = [results[8][i] for i in order]
        self.estimates.YrA = F - self.estimates.C
        return self

    def HALS4traces(self, Yr, groups=None, use_groups=False, order=None,
                    update_bck=True, bck_non_neg=True, **kwargs):
        """Solves C, f = argmin_C ||Yr-AC-bf|| using block-coordinate decent.
        Can use groups to update non-overlapping components in parallel or a
        specified order.

        Args:
            Yr : np.array (possibly memory mapped, (x,y,[,z]) x t)
                Imaging data reshaped in matrix format

            groups : list of sets
                grouped components to be updated simultaneously

            use_groups : bool
                flag for using groups

            order : list
                Update components in that order (used if nonempty and groups=None)

            update_bck : bool
                Flag for updating temporal background components

            bck_non_neg : bool
                Require temporal background to be non-negative

        Returns:
            self (updated values for self.estimates.C, self.estimates.f, self.estimates.YrA)
        """
        if update_bck:
            Ab = scipy.sparse.hstack([self.estimates.b, self.estimates.A]).tocsc()
            try:
                Cf = np.vstack([self.estimates.f, self.estimates.C + self.estimates.YrA])
            except():
                Cf = np.vstack([self.estimates.f, self.estimates.C])
        else:
            Ab = self.estimates.A
            try:
                Cf = self.estimates.C + self.estimates.YrA
            except():
                Cf = self.estimates.C
            Yr = Yr - self.estimates.b.dot(self.estimates.f)
        if (groups is None) and use_groups:
            groups = list(map(list, CNMFE_3.update_order(Ab)[0]))
        self.estimates.groups = groups
        C, noisyC = CNMFE_2.HALS4activity(Yr, Ab, Cf, groups=self.estimates.groups, order=order,
                                  **kwargs) # FIXME: this function is not defined in this scope
        if update_bck:
            if bck_non_neg:
                self.estimates.f = C[:self.params.get('init', 'nb')]
            else:
                self.estimates.f = noisyC[:self.params.get('init', 'nb')]
            self.estimates.C = C[self.params.get('init', 'nb'):]
            self.estimates.YrA = noisyC[self.params.get('init', 'nb'):] - self.estimates.C
        else:
            self.estimates.C = C
            self.estimates.YrA = noisyC - self.estimates.C
        return self

    def HALS4footprints(self, Yr, update_bck=True, num_iter=2):
        """Uses hierarchical alternating least squares to update shapes and
        background

        Args:
            Yr: np.array (possibly memory mapped, (x,y,[,z]) x t)
                Imaging data reshaped in matrix format

            update_bck: bool
                flag for updating spatial background components

            num_iter: int
                number of iterations

        Returns:
            self (updated values for self.estimates.A and self.estimates.b)
        """
        if update_bck:
            Ab = np.hstack([self.estimates.b, self.estimates.A.toarray()])
            try:
                Cf = np.vstack([self.estimates.f, self.estimates.C + self.estimates.YrA])
            except():
                Cf = np.vstack([self.estimates.f, self.estimates.C])
        else:
            Ab = self.estimates.A.toarray()
            try:
                Cf = self.estimates.C + self.estimates.YrA
            except():
                Cf = self.estimates.C
            Yr = Yr - self.estimates.b.dot(self.estimates.f)
        Ab = HALS4shapes(Yr, Ab, Cf, iters=num_iter) # FIXME: this function is not defined in this scope
        if update_bck:
            self.estimates.A = scipy.sparse.csc_matrix(Ab[:, self.params.get('init', 'nb'):])
            self.estimates.b = Ab[:, :self.params.get('init', 'nb')]
        else:
            self.estimates.A = scipy.sparse.csc_matrix(Ab)

        return self

    def update_temporal(self, Y, use_init=True, **kwargs):
        """Updates temporal components

        Args:
            Y:  np.array (d1*d2) x T
                input data

        """
        lc = locals()
        pr = inspect.signature(self.update_temporal)
        params = [k for k, v in pr.parameters.items() if '=' in str(v)]
        kw2 = {k: lc[k] for k in params}
        try:
            kwargs_new = {**kw2, **kwargs}
        except():  # python 2.7
            kwargs_new = kw2.copy()
            kwargs_new.update(kwargs)
        self.params.set('temporal', kwargs_new)


        self.estimates.C, self.estimates.A, self.estimates.b, self.estimates.f, self.estimates.S, \
        self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, \
        self.estimates.g, self.estimates.YrA, self.estimates.lam = CNMFE_3.update_temporal_components(
                Y, self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, dview=self.dview,
                **self.params.get_group('temporal'))
        self.estimates.R = self.estimates.YrA
        return self

    def update_spatial(self, Y, use_init=True, **kwargs):
        """Updates spatial components

        Args:
            Y:  np.array (d1*d2) x T
                input data
            use_init: bool
                use Cin, f_in for computing A, b otherwise use C, f

        Returns:
            self
                modified values self.estimates.A, self.estimates.b possibly self.estimates.C, self.estimates.f
        """
        lc = locals()
        pr = inspect.signature(self.update_spatial)
        params = [k for k, v in pr.parameters.items() if '=' in str(v)]
        kw2 = {k: lc[k] for k in params}
        try:
            kwargs_new = {**kw2, **kwargs}
        except():  # python 2.7
            kwargs_new = kw2.copy()
            kwargs_new.update(kwargs)
        self.params.set('spatial', kwargs_new)
        for key in kwargs_new:
            if hasattr(self, key):
                setattr(self, key, kwargs_new[key])
        self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f =\
            CNMFE_3.update_spatial_components(Y, C=self.estimates.C, f=self.estimates.f, A_in=self.estimates.A,
                                      b_in=self.estimates.b, dview=self.dview,
                                      sn=self.estimates.sn, dims=self.dims, **self.params.get_group('spatial'))

        return self

    def merge_comps(self, Y, mx=50, fast_merge=True, max_merge_area=None):
        """merges components
        """
        self.estimates.A, self.estimates.C, self.estimates.nr, self.estimates.merged_ROIs, self.estimates.S, \
        self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, self.estimates.g, self.empty_merged, \
        self.estimates.YrA =\
            CNMFE_3.merge_components(Y, self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.YrA,
                             self.estimates.f, self.estimates.S, self.estimates.sn, self.params.get_group('temporal'),
                             self.params.get_group('spatial'), dview=self.dview,
                             bl=self.estimates.bl, c1=self.estimates.c1, sn=self.estimates.neurons_sn,
                             g=self.estimates.g, thr=self.params.get('merging', 'merge_thr'), mx=mx,
                             fast_merge=fast_merge, merge_parallel=self.params.get('merging', 'merge_parallel'),
                             max_merge_area=max_merge_area)

        return self

    def initialize(self, Y, **kwargs):
        """Component initialization
        """
        self.params.set('init', kwargs)
        estim = self.estimates
        if (self.params.get('init', 'method_init') == 'corr_pnr' and
                self.params.get('init', 'ring_size_factor') is not None):
            estim.A, estim.C, estim.b, estim.f, estim.center, \
                extra_1p = CNMFE_3.initialize_components(
                    Y, sn=estim.sn, options_total=self.params.to_dict(),
                    **self.params.get_group('init'))
            try:
                estim.S, estim.bl, estim.c1, estim.neurons_sn, \
                    estim.g, estim.YrA, estim.lam = extra_1p
            except:
                estim.S, estim.bl, estim.c1, estim.neurons_sn, \
                    estim.g, estim.YrA, estim.lam, estim.W, estim.b0 = extra_1p
        else:
            estim.A, estim.C, estim.b, estim.f, estim.center =\
                CNMFE_3.initialize_components(Y, sn=estim.sn, options_total=self.params.to_dict(),
                                      **self.params.get_group('init'))

        self.estimates = estim

        return self

    def preprocess(self, Yr):
        """
        Examines data to remove corrupted pixels and computes the noise level
        estimate for each pixel.

        Args:
            Yr: np.array (or memmap.array)
                2d array of data (pixels x timesteps) typically in memory
                mapped form
        """
        Yr, self.estimates.sn, self.estimates.g, self.estimates.psx = preprocess_data(
            Yr, dview=self.dview, **self.params.get_group('preprocess'))
        return Yr

def preprocess_data(Y, sn=None, dview=None, n_pixels_per_process=100,
                    noise_range=[0.25, 0.5], noise_method='logmexp',
                    compute_g=False, p=2, lags=5, include_noise=False,
                    pixels=None, max_num_samples_fft=3000, check_nan=True):
    """
    Performs the pre-processing operations described above.

    Args:
        Y: ndarray
            input movie (n_pixels x Time). Can be also memory mapped file.

        n_processes: [optional] int
            number of processes/threads to use concurrently

        n_pixels_per_process: [optional] int
            number of pixels to be simultaneously processed by each process

        p: positive integer
            order of AR process, default: 2

        lags: positive integer
            number of lags in the past to consider for determining time constants. Default 5

        include_noise: Boolean
            Flag to include pre-estimated noise value when determining time constants. Default: False

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
        Y: ndarray
             movie preprocessed (n_pixels x Time). Can be also memory mapped file.
        g:  np.ndarray (p x 1)
            Discrete time constants
        psx: ndarray
            position of thoses pixels
        sn_s: ndarray (memory mapped)
            file where to store the results of computation.
    """

    if check_nan:
        Y, coor = interpolate_missing_data(Y)

    if sn is None:
        if dview is None:
            sn, psx = get_noise_fft(Y, noise_range=noise_range, noise_method=noise_method,
                                    max_num_samples_fft=max_num_samples_fft)
        else:
            sn, psx = get_noise_fft_parallel(Y, n_pixels_per_process=n_pixels_per_process, dview=dview,
                                             noise_range=noise_range, noise_method=noise_method,
                                             max_num_samples_fft=max_num_samples_fft)
    else:
        psx = None

    if compute_g:
        g = estimate_time_constant(Y, sn, p=p, lags=lags,
                                   include_noise=include_noise, pixels=pixels)
    else:
        g = None

    # psx  # no need to keep psx in memory as long a we don't use it elsewhere
    return Y, sn, g, None

def interpolate_missing_data(Y):
    """
    Interpolate any missing data using nearest neighbor interpolation.
    Missing data is identified as entries with values NaN

    Args:
        Y   np.ndarray (3D)
            movie, raw data in 3D format (d1 x d2 x T)

    Returns:
        Y   np.ndarray (3D)
            movie, data with interpolated entries (d1 x d2 x T)
        coordinate list
            list of interpolated coordinates

    Raises:
        Exception 'The algorithm has not been tested with missing values (NaNs). Remove NaNs and rerun the algorithm.'
    """
    coor = []
    logging.info('Checking for missing data entries (NaN)')
    if np.any(np.isnan(Y)):
        logging.info('Interpolating missing data')
        for idx, row in enumerate(Y):
            nans = np.where(np.isnan(row))[0]
            n_nans = np.where(~np.isnan(row))[0]
            coor.append((idx, nans))
            Y[idx, nans] = np.interp(nans, n_nans, row[n_nans])
        raise Exception(
            'The algorithm has not been tested with missing values (NaNs). Remove NaNs and rerun the algorithm.')

    return Y, coor

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

def get_noise_fft_parallel(Y, n_pixels_per_process=100, dview=None, **kwargs):
    """parallel version of get_noise_fft.

    Args:
        Y: ndarray
            input movie (n_pixels x Time). Can be also memory mapped file.

        n_processes: [optional] int
            number of processes/threads to use concurrently

        n_pixels_per_process: [optional] int
            number of pixels to be simultaneously processed by each process

        backend: [optional] string
            the type of concurrency to be employed. only 'multithreading' for the moment

        **kwargs: [optional] dict
            all the parameters passed to get_noise_fft

    Returns:
        sn: ndarray(double)
            noise associated to each pixel
    """
    folder = tempfile.mkdtemp()

    # Pre-allocate a writeable shared memory map as a container for the
    # results of the parallel computation
    pixel_groups = list(
        range(0, Y.shape[0] - n_pixels_per_process + 1, n_pixels_per_process))

    if isinstance(Y, np.core.memmap):  # if input file is already memory mapped then find the filename
        Y_name = Y.filename

    else:
        if dview is not None:
            raise Exception('parallel processing requires memory mapped files')
        Y_name = Y

    argsin = [(Y_name, i, n_pixels_per_process, kwargs) for i in pixel_groups]
    pixels_remaining = Y.shape[0] % n_pixels_per_process
    if pixels_remaining > 0:
        argsin.append(
            (Y_name, Y.shape[0] - pixels_remaining, pixels_remaining, kwargs))

    if dview is None:
        print('Single Thread')
        results = list(map(fft_psd_multithreading, argsin))

    else:
        if 'multiprocessing' in str(type(dview)):
            results = dview.map_async(
                fft_psd_multithreading, argsin).get(4294967)
        else:
            ne = len(dview)
            print(('Running on %d engines.' % (ne)))
            if dview.client.profile == 'default':
                results = dview.map_sync(fft_psd_multithreading, argsin)

            else:
                print(('PROFILE:' + dview.client.profile))
                results = dview.map_sync(fft_psd_multithreading, argsin)

    _, _, psx_ = results[0]
    sn_s = np.zeros(Y.shape[0])
    psx_s = np.zeros((Y.shape[0], psx_.shape[-1]))
    for idx, sn, psx_ in results:
        sn_s[idx] = sn
        psx_s[idx, :] = psx_

    sn_s = np.array(sn_s)
    psx_s = np.array(psx_s)

    try:
        shutil.rmtree(folder)

    except:
        print(("Failed to delete: " + folder))
        raise

    return sn_s, psx_s
#%%

def fft_psd_multithreading(args):
    """helper function to parallelize get_noise_fft

    Args:
        Y: ndarray
            input movie (n_pixels x Time), can be also memory mapped file
        sn_s: ndarray (memory mapped)
            file where to store the results of computation.
        i: int
            pixel index start
        num_pixels: int
            number of pixel to select starting from i
        **kwargs: dict
            arguments to be passed to get_noise_fft

    Returns:
        idx: list
            list of the computed pixels
        res: ndarray(double)
            noise associated to each pixel
        psx: ndarray
            position of thoses pixels
    """

    (Y, i, num_pixels, kwargs) = args
    if isinstance(Y, basestring):
        Y, _, _ = cp_motioncorrection.load_memmap(Y)

    idxs = list(range(i, i + num_pixels))
    #print(len(idxs))
    res, psx = get_noise_fft(Y[idxs], **kwargs)

    return (idxs, res, psx)

def estimate_time_constant(Y, sn, p=None, lags=5, include_noise=False, pixels=None):
    """
    Estimating global time constants for the dataset Y through the autocovariance function (optional).
    The function is no longer used in the standard setting of the algorithm since every trace has its own
    time constant.

    Args:
        Y: np.ndarray (2D)
            input movie data with time in the last axis

        p: positive integer
            order of AR process, default: 2

        lags: positive integer
            number of lags in the past to consider for determining time constants. Default 5

        include_noise: Boolean
            Flag to include pre-estimated noise value when determining time constants. Default: False

        pixels: np.ndarray
            Restrict estimation to these pixels (e.g., remove saturated pixels). Default: All pixels

    Returns:
        g:  np.ndarray (p x 1)
            Discrete time constants
    """
    if p is None:
        raise Exception("You need to define p")
    if pixels is None:
        pixels = np.arange(old_div(np.size(Y), np.shape(Y)[-1]))

    from scipy.linalg import toeplitz
    npx = len(pixels)
    lags += p
    XC = np.zeros((npx, 2 * lags + 1))
    for j in range(npx):
        XC[j, :] = np.squeeze(axcov(np.squeeze(Y[pixels[j], :]), lags))

    gv = np.zeros(npx * lags)
    if not include_noise:
        XC = XC[:, np.arange(lags - 1, -1, -1)]
        lags -= p

    A = np.zeros((npx * lags, p))
    for i in range(npx):
        if not include_noise:
            A[i * lags + np.arange(lags), :] = toeplitz(np.squeeze(XC[i, np.arange(
                p - 1, p + lags - 1)]), np.squeeze(XC[i, np.arange(p - 1, -1, -1)]))
        else:
            A[i * lags + np.arange(lags), :] = toeplitz(np.squeeze(XC[i, lags + np.arange(
                lags)]), np.squeeze(XC[i, lags + np.arange(p)])) - (sn[i]**2) * np.eye(lags, p)
            gv[i * lags + np.arange(lags)] = np.squeeze(XC[i, lags + 1:])

    if not include_noise:
        gv = XC[:, p:].T
        gv = np.squeeze(np.reshape(gv, (np.size(gv), 1), order='F'))

    g = np.dot(np.linalg.pinv(A), gv)

    return g

def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag

    Args:
        data : array
            Array containing fluorescence data

        maxlag : int
            Number of lags to use in autocovariance calculation

    Returns:
        axcov : array
            Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(old_div(xcov, T))

def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).

    Args:
        value : int

    Returns:
        exponent : int
    """

    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent

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
            ds_mat = decimation_matrix(Y.shape[:2], ds[0])
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

def decimation_matrix(dims, sub):
    D = np.prod(dims)
    if sub == 2 and D <= 10000:  # faster for small matrices
        ind = np.arange(D) // 2 - \
            np.arange(dims[0], dims[0] + D) // (dims[0] * 2) * (dims[0] // 2) - \
            (dims[0] % 2) * (np.arange(D) % (2 * dims[0]) > dims[0]) * (np.arange(1, 1 + D) % 2)
    else:
        def create_decimation_matrix_bruteforce(dims, sub):
            dims_ds = tuple(1 + (np.array(dims) - 1) // sub)
            d_ds = np.prod(dims_ds)
            ds_matrix = np.eye(d_ds)
            ds_matrix = np.repeat(np.repeat(
                ds_matrix.reshape((d_ds,) + dims_ds, order='F'), sub, 1),
                sub, 2)[:, :dims[0], :dims[1]].reshape((d_ds, -1), order='F')
            ds_matrix /= ds_matrix.sum(1)[:, None]
            ds_matrix = csc_matrix(ds_matrix, dtype=np.float32)
            return ds_matrix
        tmp = create_decimation_matrix_bruteforce((dims[0], sub), sub).indices
        ind = np.concatenate([tmp] * (dims[1] // sub + 1))[:D] + \
            np.arange(D) // (dims[0] * sub) * ((dims[0] - 1) // sub + 1)
    data = 1. / np.unique(ind, return_counts=True)[1][ind]
    return csc_matrix((data, ind, np.arange(1 + D)), dtype=np.float32)

def decimate_last_axis(y, sub):
    q = y.shape[-1] // sub
    r = y.shape[-1] % sub
    Y_ds = np.zeros(y.shape[:-1] + (q + (r > 0),), dtype=y.dtype)
    Y_ds[..., :q] = y[..., :q * sub].reshape(y.shape[:-1] + (-1, sub)).mean(-1)
    if r > 0:
        Y_ds[..., -1] = y[..., -r:].mean(-1)
    return Y_ds

def local_correlations(Y, eight_neighbours: bool = True, swap_dim: bool = True, order_mean=1) -> np.ndarray:
    """Computes the correlation image for the input dataset Y

    Args:
        Y:  np.ndarray (3D or 4D)
            Input movie data in 3D or 4D format

        eight_neighbours: Boolean
            Use 8 neighbors if true, and 4 if false for 3D data (default = True)
            Use 6 neighbors for 4D data, irrespectively

        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

        order_mean: (undocumented)

    Returns:
        rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    rho = np.zeros(np.shape(Y)[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    # yapf: disable
    if order_mean == 0:
        rho = np.ones(np.shape(Y)[1:])
        rho_h = rho_h
        rho_w = rho_w
        rho[:-1, :] = rho[:-1, :] * rho_h
        rho[1:,  :] = rho[1:,  :] * rho_h
        rho[:, :-1] = rho[:, :-1] * rho_w
        rho[:,  1:] = rho[:,  1:] * rho_w
    else:
        rho[:-1, :] = rho[:-1, :] + rho_h**(order_mean)
        rho[1:,  :] = rho[1:,  :] + rho_h**(order_mean)
        rho[:, :-1] = rho[:, :-1] + rho_w**(order_mean)
        rho[:,  1:] = rho[:,  1:] + rho_w**(order_mean)

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
        rho[:, :, :-1] = rho[:, :, :-1] + rho_d
        rho[:, :, 1:] = rho[:, :, 1:] + rho_d

        neighbors = 6 * np.ones(np.shape(Y)[1:])
        neighbors[0]        = neighbors[0]        - 1
        neighbors[-1]       = neighbors[-1]       - 1
        neighbors[:,     0] = neighbors[:,     0] - 1
        neighbors[:,    -1] = neighbors[:,    -1] - 1
        neighbors[:,  :, 0] = neighbors[:,  :, 0] - 1
        neighbors[:, :, -1] = neighbors[:, :, -1] - 1

    else:
        if eight_neighbours:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:,]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:,]), axis=0)

            if order_mean == 0:
                rho_d1 = rho_d1
                rho_d2 = rho_d2
                rho[:-1, :-1] = rho[:-1, :-1] * rho_d2
                rho[1:,   1:] = rho[1:,   1:] * rho_d1
                rho[1:,  :-1] = rho[1:,  :-1] * rho_d1
                rho[:-1,  1:] = rho[:-1,  1:] * rho_d2
            else:
                rho[:-1, :-1] = rho[:-1, :-1] + rho_d2**(order_mean)
                rho[1:,   1:] = rho[1:,   1:] + rho_d1**(order_mean)
                rho[1:,  :-1] = rho[1:,  :-1] + rho_d1**(order_mean)
                rho[:-1,  1:] = rho[:-1,  1:] + rho_d2**(order_mean)

            neighbors = 8 * np.ones(np.shape(Y)[1:3])
            neighbors[0,   :] = neighbors[0,   :] - 3
            neighbors[-1,  :] = neighbors[-1,  :] - 3
            neighbors[:,   0] = neighbors[:,   0] - 3
            neighbors[:,  -1] = neighbors[:,  -1] - 3
            neighbors[0,   0] = neighbors[0,   0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1,  0] = neighbors[-1,  0] + 1
            neighbors[0,  -1] = neighbors[0,  -1] + 1
        else:
            neighbors = 4 * np.ones(np.shape(Y)[1:3])
            neighbors[0,  :]  = neighbors[0,  :] - 1
            neighbors[-1, :]  = neighbors[-1, :] - 1
            neighbors[:,  0]  = neighbors[:,  0] - 1
            neighbors[:, -1]  = neighbors[:, -1] - 1

    # yapf: enable
    if order_mean == 0:
        rho = np.power(rho, 1. / neighbors)
    else:
        rho = np.power(np.divide(rho, neighbors), 1 / order_mean)

    return rho

def estimate_components_quality(traces,
                                Y,
                                A,
                                C,
                                b,
                                f,
                                final_frate=30,
                                Npeaks=10,
                                r_values_min=.95,
                                fitness_min=-100,
                                fitness_delta_min=-100,
                                return_all: bool = False,
                                N=5,
                                remove_baseline=True,
                                dview=None,
                                robust_std=False,
                                Athresh=0.1,
                                thresh_C=0.3,
                                num_traces_per_group=20) -> Tuple[np.ndarray, ...]:
    """ Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode.
    The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.
    The algorithm also measures the reliability of the spatial mask by comparing the filters in A
     with the average of the movies over samples where exceptional events happen, after  removing (if possible)
    frames when neighboring neurons were active

    Args:
        Y: ndarray
            movie x,y,t

        A,C,b,f: various types
            outputs of cnmf

        traces: ndarray
            Fluorescence traces

        N: int
            N number of consecutive events probability multiplied

        Npeaks: int

        r_values_min: list
            minimum correlation between component and spatial mask obtained by averaging important points

        fitness_min: ndarray
            minimum acceptable quality of components (the lesser the better) on the raw trace

        fitness_delta_min: ndarray
            minimum acceptable the quality of components (the lesser the better) on diff(trace)

        thresh_C: float
            fraction of the maximum of C that is used as minimum peak height

    Returns:
        idx_components: ndarray
            the components ordered according to the fitness

        idx_components_bad: ndarray
            the components ordered according to the fitness

        fitness_raw: ndarray
            value estimate of the quality of components (the lesser the better) on the raw trace

        fitness_delta: ndarray
            value estimate of the quality of components (the lesser the better) on diff(trace)

        r_values: list
            float values representing correlation between component and spatial mask obtained by averaging important points

    """
    # TODO: Consider always returning it all and let the caller ignore what it does not want

    if 'memmap' not in str(type(Y)):
        logging.warning('NOT MEMORY MAPPED. FALLING BACK ON SINGLE CORE IMPLEMENTATION')
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, _ = \
            evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=remove_baseline,
                                N=N, robust_std=robust_std, Athresh=Athresh,
                                Npeaks=Npeaks, thresh_C=thresh_C)

    else:      # memory mapped case
        fitness_raw = []
        fitness_delta = []
        erfc_raw = []
        erfc_delta = []
        r_values = []
        Ncomp = A.shape[-1]

        if Ncomp > 0:
            groups = grouper(num_traces_per_group, range(Ncomp))
            params = []
            for g in groups:
                idx = list(g)
                # idx = list(filter(None.__ne__, idx))
                idx = list(filter(lambda a: a is not None, idx))
                params.append([
                    Y.filename, traces[idx],
                    A.tocsc()[:, idx], C[idx], b, f, final_frate, remove_baseline, N, robust_std, Athresh, Npeaks,
                    thresh_C
                ])

            if dview is None:
                res = map(evaluate_components_placeholder, params)
            else:
                logging.info('Component evaluation in parallel')
                if 'multiprocessing' in str(type(dview)):
                    res = dview.map_async(evaluate_components_placeholder, params).get(4294967)
                else:
                    res = dview.map_sync(evaluate_components_placeholder, params)

            for r_ in res:
                fitness_raw__, fitness_delta__, erfc_raw__, erfc_delta__, r_values__, _ = r_
                fitness_raw = np.concatenate([fitness_raw, fitness_raw__])
                fitness_delta = np.concatenate([fitness_delta, fitness_delta__])
                r_values = np.concatenate([r_values, r_values__])

                if len(erfc_raw) == 0:
                    erfc_raw = erfc_raw__
                    erfc_delta = erfc_delta__
                else:
                    erfc_raw = np.concatenate([erfc_raw, erfc_raw__], axis=0)
                    erfc_delta = np.concatenate([erfc_delta, erfc_delta__], axis=0)
        else:
            warnings.warn("There were no components to evaluate. Check your parameter settings.")

    idx_components_r = np.where(np.array(r_values) >= r_values_min)[0]         # threshold on space consistency
    idx_components_raw = np.where(np.array(fitness_raw) < fitness_min)[0]      # threshold on time variability
                                                                               # threshold on time variability (if nonsparse activity)
    idx_components_delta = np.where(np.array(fitness_delta) < fitness_delta_min)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    if return_all:
        return idx_components, idx_components_bad, np.array(fitness_raw), np.array(fitness_delta), np.array(r_values)
    else:
        return idx_components, idx_components_bad

def evaluate_components_placeholder(params):
    import caiman as cm
    fname, traces, A, C, b, f, final_frate, remove_baseline, N, robust_std, Athresh, Npeaks, thresh_C = params
    Yr, dims, T = cm.load_memmap(fname)
    Y = np.reshape(Yr, dims + (T,), order='F')
    fitness_raw, fitness_delta, _, _, r_values, significant_samples = \
        evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=remove_baseline,
                            N=N, robust_std=robust_std, Athresh=Athresh, Npeaks=Npeaks, thresh_C=thresh_C)

    return fitness_raw, fitness_delta, [], [], r_values, significant_samples

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def evaluate_components(Y: np.ndarray,
                        traces: np.ndarray,
                        A,
                        C,
                        b,
                        f,
                        final_frate,
                        remove_baseline: bool = True,
                        N: int = 5,
                        robust_std: bool = False,
                        Athresh: float = 0.1,
                        Npeaks: int = 5,
                        thresh_C: float = 0.3,
                        sigma_factor: float = 3.) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """ Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode.
    The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.
    The algorithm also measures the reliability of the spatial mask by comparing the filters in A
     with the average of the movies over samples where exceptional events happen, after  removing (if possible)
    frames when neighboring neurons were active

    Args:
        Y: ndarray
            movie x,y,t

        traces: ndarray
            Fluorescence traces

        A,C,b,f: various types
            outputs of cnmf

        final_frate: (undocumented)

        remove_baseline: bool
            whether to remove the baseline in a rolling fashion *(8 percentile)

        N: int
            N number of consecutive events probability multiplied


        Athresh: float
            threshold on overlap of A (between 0 and 1)

        Npeaks: int
            Number of local maxima to consider

        thresh_C: float
            fraction of the maximum of C that is used as minimum peak height

        sigma_factor: float
            multiplicative factor for noise

    Returns:
        idx_components: ndarray
            the components ordered according to the fitness

        fitness_raw: ndarray
            value estimate of the quality of components (the lesser the better) on the raw trace

        fitness_delta: ndarray
            value estimate of the quality of components (the lesser the better) on diff(trace)

        erfc_raw: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise on the raw trace

        erfc_raw: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise on diff(trace)

        r_values: list
            float values representing correlation between component and spatial mask obtained by averaging important points

        significant_samples: ndarray
            indexes of samples used to obtain the spatial mask by average
    """

    tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
    tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
    logging.info('tB:' + str(tB) + ',tA:' + str(tA))
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]

    Yr = np.reshape(Y, (np.prod(dims), T), order='F')

    logging.debug('Computing event exceptionality delta')
    fitness_delta, erfc_delta, _, _ = compute_event_exceptionality(np.diff(traces, axis=1),
                                                                   robust_std=robust_std,
                                                                   N=N,
                                                                   sigma_factor=sigma_factor)

    logging.debug('Removing Baseline')
    if remove_baseline:
        num_samps_bl = np.minimum(old_div(np.shape(traces)[-1], 5), 800)
        slow_baseline = False
        if slow_baseline:

            traces = traces - \
                scipy.ndimage.percentile_filter(
                    traces, 8, size=[1, num_samps_bl])

        else:                                                                                # fast baseline removal
            downsampfact = num_samps_bl
            elm_missing = int(np.ceil(T * 1.0 / downsampfact) * downsampfact - T)
            padbefore = int(np.floor(old_div(elm_missing, 2.0)))
            padafter = int(np.ceil(old_div(elm_missing, 2.0)))
            tr_tmp = np.pad(traces.T, ((padbefore, padafter), (0, 0)), mode='reflect')
            numFramesNew, num_traces = np.shape(tr_tmp)
                                                                                             #% compute baseline quickly
            logging.debug("binning data ...")
            tr_BL = np.reshape(tr_tmp, (downsampfact, numFramesNew // downsampfact, num_traces), order='F')
            tr_BL = np.percentile(tr_BL, 8, axis=0)
            logging.debug("interpolating data ...")
            logging.debug(tr_BL.shape)
            tr_BL = scipy.ndimage.zoom(np.array(tr_BL, dtype=np.float32), [downsampfact, 1],
                                       order=3,
                                       mode='constant',
                                       cval=0.0,
                                       prefilter=True)
            if padafter == 0:
                traces -= tr_BL.T
            else:
                traces -= tr_BL[padbefore:-padafter].T

    logging.debug('Computing event exceptionality')
    fitness_raw, erfc_raw, _, _ = compute_event_exceptionality(traces,
                                                               robust_std=robust_std,
                                                               N=N,
                                                               sigma_factor=sigma_factor)

    logging.debug('Evaluating spatial footprint')
    # compute the overlap between spatial and movie average across samples with significant events
    r_values, significant_samples = classify_components_ep(Yr,
                                                           A,
                                                           C,
                                                           b,
                                                           f,
                                                           Athresh=Athresh,
                                                           Npeaks=Npeaks,
                                                           tB=tB,
                                                           tA=tA,
                                                           thres=thresh_C)

    return fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples

def classify_components_ep(Y, A, C, b, f, Athresh=0.1, Npeaks=5, tB=-3, tA=10, thres=0.3) -> Tuple[np.ndarray, List]:
    """Computes the space correlation values between the detected spatial
    footprints and the original data when background and neighboring component
    activity has been removed.
    Args:
        Y: ndarray
            movie x,y,t

        A: scipy sparse array
            spatial components

        C: ndarray
            Fluorescence traces

        b: ndarrray
            Spatial background components

        f: ndarrray
            Temporal background components

        Athresh: float
            Degree of spatial overlap for a neighboring component to be
            considered overlapping

        Npeaks: int
            Number of peaks to consider for computing spatial correlation

        tB: int
            Number of frames to include before peak

        tA: int
            Number of frames to include after peak

        thres: float
            threshold value for computing distinct peaks

    Returns:
        rval: ndarray
            Space correlation values

        significant_samples: list
            Frames that were used for computing correlation values
    """

    K, _ = np.shape(C)
    A = csc_matrix(A)
    AA = (A.T * A).toarray()
    nA = np.sqrt(np.array(A.power(2).sum(0)))
    AA = old_div(AA, np.outer(nA, nA.T))
    AA -= np.eye(K)

    LOC = find_activity_intervals(C, Npeaks=Npeaks, tB=tB, tA=tA, thres=thres)
    rval = np.zeros(K)

    significant_samples: List[Any] = []
    for i in range(K):
        if (i + 1) % 200 == 0:         # Show status periodically
            logging.info('Components evaluated:' + str(i))
        if LOC[i] is not None:
            atemp = A[:, i].toarray().flatten()
            atemp[np.isnan(atemp)] = np.nanmean(atemp)
            ovlp_cmp = np.where(AA[:, i] > Athresh)[0]
            indexes = set(LOC[i])
            for _, j in enumerate(ovlp_cmp):
                if LOC[j] is not None:
                    indexes = indexes - set(LOC[j])

            if len(indexes) == 0:
                indexes = set(LOC[i])
                logging.warning('Component {0} is only active '.format(i) +
                                'jointly with neighboring components. Space ' +
                                'correlation calculation might be unreliable.')

            indexes = np.array(list(indexes)).astype(np.int)
            px = np.where(atemp > 0)[0]
            if px.size < 3:
                logging.warning('Component {0} is almost empty. '.format(i) + 'Space correlation is set to 0.')
                rval[i] = 0
                significant_samples.append({0})
            else:
                ysqr = np.array(Y[px, :])
                ysqr[np.isnan(ysqr)] = np.nanmean(ysqr)
                mY = np.mean(ysqr[:, indexes], axis=-1)
                significant_samples.append(indexes)
                rval[i] = scipy.stats.pearsonr(mY, atemp[px])[0]

        else:
            rval[i] = 0
            significant_samples.append(0)

    return rval, significant_samples

def find_activity_intervals(C, Npeaks: int = 5, tB=-3, tA=10, thres: float = 0.3) -> List:
    # todo todocument

    K, T = np.shape(C)
    L: List = []
    for i in range(K):
        if np.sum(np.abs(np.diff(C[i, :]))) == 0:
            L.append([])
            logging.debug('empty component at:' + str(i))
            continue
        indexes = peakutils.indexes(C[i, :], thres=thres)
        srt_ind = indexes[np.argsort(C[i, indexes])][::-1]
        srt_ind = srt_ind[:Npeaks]
        L.append(srt_ind)

    LOC = []
    for i in range(K):
        if len(L[i]) > 0:
            interval = np.kron(L[i], np.ones(int(np.round(tA - tB)), dtype=int)) + \
                np.kron(np.ones(len(L[i]), dtype=int), np.arange(tB, tA))
            interval[interval < 0] = 0
            interval[interval > T - 1] = T - 1
            LOC.append(np.array(list(set(interval))))
        else:
            LOC.append(None)

    return LOC

@profile
def compute_event_exceptionality(traces: np.ndarray,
                                 robust_std: bool = False,
                                 N: int = 5,
                                 use_mode_fast: bool = False,
                                 sigma_factor: float = 3.) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    """
    Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Args:
        Y: ndarray
            movie x,y,t

        A: scipy sparse array
            spatial components

        traces: ndarray
            Fluorescence traces

        N: int
            N number of consecutive events

        sigma_factor: float
            multiplicative factor for noise estimate (added for backwards compatibility)

    Returns:
        fitness: ndarray
            value estimate of the quality of components (the lesser the better)

        erfc: ndarray
            probability at each time step of observing the N consequtive actual trace values given the distribution of noise

        noise_est: ndarray
            the components ordered according to the fitness
    """

    T = np.shape(traces)[-1]
    if use_mode_fast:
        md = mode_robust_fast(traces, axis=1)
    else:
        md = mode_robust(traces, axis=1)

    ff1 = traces - md[:, None]

    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:

        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, _ in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(old_div(np.sum(ff1**2, 1), Ns))

    # compute z value
    z = old_div((traces - md[:, None]), (sigma_factor * sd_r[:, None]))

    # probability of observing values larger or equal to z given normal
    # distribution with mean md and std sd_r
    #erf = 1 - norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    #erf = np.log(erf)
    # compute with this numerically stable function
    erf = scipy.special.log_ndtr(-z)

    # moving sum
    erfc = np.cumsum(erf, 1)
    erfc[:, N:] -= erfc[:, :-N]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    return fitness, erfc, sd_r, md

def mode_robust_fast(inputData, axis=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:

        def fnc(x):
            return mode_robust_fast(x)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        data = inputData.ravel()
        # The data need to be sorted for this to work
        data = np.sort(data)
        # Find the mode
        dataMode = _hsm(data)

    return dataMode

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

