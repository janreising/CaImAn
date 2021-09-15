import cv2
import h5py
import logging
import numpy as np
import os
import pylab as pl
import scipy.ndimage
import scipy
from scipy.io import loadmat

import sys
import tifffile
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union, Optional
import itertools
from itertools import chain

import ipyparallel as parallel

from skimage.transform import warp as warp_sk
from skimage.transform import resize as resize_sk

from cv2 import dft as fftn
from cv2 import idft as ifftn

from numpy.fft import ifftshift

import pathlib

from past.utils import old_div

HAS_SIMA = False
HAS_CUDA = False
opencv = True

class MotionCorrect(object):
    """
        class implementing motion correction operations
       """

    def __init__(self, fname, min_mov=None, dview=None, max_shifts=(6, 6), niter_rig=1, splits_rig=14, num_splits_to_process_rig=None,
                 strides=(96, 96), overlaps=(32, 32), splits_els=14, num_splits_to_process_els=None,
                 upsample_factor_grid=4, max_deviation_rigid=3, shifts_opencv=True, nonneg_movie=True, gSig_filt=None,
                 use_cuda=False, border_nan=True, pw_rigid=False, num_frames_split=80, var_name_hdf5='mov',is3D=False,
                 indices=(slice(None), slice(None))):
        """
        Constructor class for motion correction operations

        Args:
           fname: str
               path to file to motion correct

           min_mov: int16 or float32
               estimated minimum value of the movie to produce an output that is positive

           dview: ipyparallel view object list
               to perform parallel computing, if NOne will operate in single thread

           max_shifts: tuple
               maximum allow rigid shift

           niter_rig':int
               maximum number of iterations rigid motion correction, in general is 1. 0
               will quickly initialize a template with the first frames

           splits_rig': int
            for parallelization split the movies in  num_splits chuncks across time

           num_splits_to_process_rig: list,
               if none all the splits are processed and the movie is saved, otherwise at each iteration
               num_splits_to_process_rig are considered

           strides: tuple
               intervals at which patches are laid out for motion correction

           overlaps: tuple
               overlap between pathes (size of patch strides+overlaps)

           pw_rigig: bool, default: False
               flag for performing motion correction when calling motion_correct

           splits_els':list
               for parallelization split the movies in  num_splits chuncks across time

           num_splits_to_process_els: list,
               if none all the splits are processed and the movie is saved  otherwise at each iteration
                num_splits_to_process_els are considered

           upsample_factor_grid:int,
               upsample factor of shifts per patches to avoid smearing when merging patches

           max_deviation_rigid:int
               maximum deviation allowed for patch with respect to rigid shift

           shifts_opencv: Bool
               apply shifts fast way (but smoothing results)

           nonneg_movie: boolean
               make the SAVED movie and template mostly nonnegative by removing min_mov from movie

           use_cuda : bool, optional
               Use skcuda.fft (if available). Default: False

           border_nan : bool or string, optional
               Specifies how to deal with borders. (True, False, 'copy', 'min')

           num_frames_split: int, default: 80
               Number of frames in each batch. Used when cosntructing the options
               through the params object

           var_name_hdf5: str, default: 'mov'
               If loading from hdf5, name of the variable to load

            is3D: bool, default: False
               Flag for 3D motion correction

            indices: tuple(slice), default: (slice(None), slice(None))
               Use that to apply motion correction only on a part of the FOV

       Returns:
           self

        """
        if 'ndarray' in str(type(fname)):
            # logging.info('Creating file for motion correction "tmp_mov_mot_corr.hdf5"')
            # cm.movie(fname).save('tmp_mov_mot_corr.hdf5')
            # fname = ['tmp_mov_mot_corr.hdf5']

            print("**to_julia** Don't think we need this")
            sys.exit(2)

        if not isinstance(fname, list):
            fname = [fname]

        if isinstance(gSig_filt, tuple):
            gSig_filt = list(gSig_filt) # There are some serializers down the line that choke otherwise

        self.fname = fname
        self.dview = dview
        self.max_shifts = max_shifts
        self.niter_rig = niter_rig
        self.splits_rig = splits_rig
        self.num_splits_to_process_rig = num_splits_to_process_rig
        self.strides = strides
        self.overlaps = overlaps
        self.splits_els = splits_els
        self.num_splits_to_process_els = num_splits_to_process_els
        self.upsample_factor_grid = upsample_factor_grid
        self.max_deviation_rigid = max_deviation_rigid
        self.shifts_opencv = bool(shifts_opencv)
        self.min_mov = min_mov
        self.nonneg_movie = nonneg_movie
        self.gSig_filt = gSig_filt
        self.use_cuda = False  # **to_julia** don't think we need this
        self.border_nan = border_nan
        self.pw_rigid = bool(pw_rigid)
        self.var_name_hdf5 = var_name_hdf5
        self.is3D = bool(is3D)
        self.indices = indices
        # if self.use_cuda and not HAS_CUDA:  # **to_julia** don't think we need this
        #     logging.debug("pycuda is unavailable. Falling back to default FFT.")

    def motion_correct(self, template=None, save_movie=False):
        """general function for performing all types of motion correction. The
        function will perform either rigid or piecewise rigid motion correction
        depending on the attribute self.pw_rigid and will perform high pass
        spatial filtering for determining the motion (used in 1p data) if the
        attribute self.gSig_filt is not None. A template can be passed, and the
        output can be saved as a memory mapped file.

        Args:
            template: ndarray, default: None
                template provided by user for motion correction

            save_movie: bool, default: False
                flag for saving motion corrected file(s) as memory mapped file(s)

        Returns:
            self
        """
        # TODO: Review the docs here, and also why we would ever return self
        #       from a method that is not a constructor

        assert self.gSig_filt is not None, "**to_julia** removed"

        # get minimum of first 400 frames!?
        self.min_mov = np.array([high_pass_filter_space(m_, self.gSig_filt)
            for m_ in load(self.fname[0], var_name_hdf5=self.var_name_hdf5,
                              subindices=slice(400))]).min()  #TODO why is the 400 hardcoded here?

        if self.pw_rigid:
            self.motion_correct_pwrigid(template=template, save_movie=save_movie)
            assert self.is3D is None, "**to_julia** removed"

            b0 = np.ceil(np.maximum(np.max(np.abs(self.x_shifts_els)),
                                np.max(np.abs(self.y_shifts_els))))
        else:
            # -> we go here
            self.motion_correct_rigid(template=template, save_movie=save_movie)
            b0 = np.ceil(np.max(np.abs(self.shifts_rig)))

        self.border_to_0 = b0.astype(np.int)
        self.mmap_file = self.fname_tot_els if self.pw_rigid else self.fname_tot_rig
        return self

    def motion_correct_rigid(self, template=None, save_movie=False) -> None:
        """
        Perform rigid motion correction

        Args:
            template: ndarray 2D (or 3D)
                if known, one can pass a template to register the frames to

            save_movie_rigid:Bool
                save the movies vs just get the template

        Important Fields:
            self.fname_tot_rig: name of the mmap file saved

            self.total_template_rig: template updated by iterating  over the chunks

            self.templates_rig: list of templates. one for each chunk

            self.shifts_rig: shifts in x and y (and z if 3D) per frame
        """
        logging.debug('Entering Rigid Motion Correction')
        logging.debug(-self.min_mov)  # XXX why the minus?
        self.total_template_rig = template
        self.templates_rig:List = []
        self.fname_tot_rig:List = []
        self.shifts_rig:List = []

        for fname_cur in self.fname:
            _fname_tot_rig, _total_template_rig, _templates_rig, _shifts_rig = motion_correct_batch_rigid(
                fname_cur,
                self.max_shifts,
                dview=self.dview,
                splits=self.splits_rig,
                num_splits_to_process=self.num_splits_to_process_rig,
                num_iter=self.niter_rig,
                template=self.total_template_rig,
                shifts_opencv=self.shifts_opencv,
                save_movie_rigid=save_movie,
                add_to_movie=-self.min_mov,
                nonneg_movie=self.nonneg_movie,
                gSig_filt=self.gSig_filt,
                use_cuda=self.use_cuda,
                border_nan=self.border_nan,
                var_name_hdf5=self.var_name_hdf5,
                is3D=self.is3D,
                indices=self.indices)

            if template is None:
                self.total_template_rig = _total_template_rig

            self.templates_rig += _templates_rig
            self.fname_tot_rig += [_fname_tot_rig]
            self.shifts_rig += _shifts_rig

    def motion_correct_pwrigid(self, save_movie:bool=True, template:np.ndarray=None, show_template:bool=False) -> None:
        """Perform pw-rigid motion correction

        Args:
            save_movie:Bool
                save the movies vs just get the template

            template: ndarray 2D (or 3D)
                if known, one can pass a template to register the frames to

            show_template: boolean
                whether to show the updated template at each iteration

        Important Fields:
            self.fname_tot_els: name of the mmap file saved
            self.templates_els: template updated by iterating  over the chunks
            self.x_shifts_els: shifts in x per frame per patch
            self.y_shifts_els: shifts in y per frame per patch
            self.z_shifts_els: shifts in z per frame per patch (if 3D)
            self.coord_shifts_els: coordinates associated to the patch for
            values in x_shifts_els and y_shifts_els (and z_shifts_els if 3D)
            self.total_template_els: list of templates. one for each chunk

        Raises:
            Exception: 'Error: Template contains NaNs, Please review the parameters'
        """

        num_iter = 1
        if template is None:
            logging.info('Generating template by rigid motion correction')
            self.motion_correct_rigid()
            self.total_template_els = self.total_template_rig.copy()
        else:
            self.total_template_els = template

        self.fname_tot_els:List = []
        self.templates_els:List = []
        self.x_shifts_els:List = []
        self.y_shifts_els:List = []
        if self.is3D:
            self.z_shifts_els:List = []

        self.coord_shifts_els:List = []
        for name_cur in self.fname:
            _fname_tot_els, new_template_els, _templates_els,\
                _x_shifts_els, _y_shifts_els, _z_shifts_els, _coord_shifts_els = motion_correct_batch_pwrigid(
                    name_cur, self.max_shifts, self.strides, self.overlaps, -self.min_mov,
                    dview=self.dview, upsample_factor_grid=self.upsample_factor_grid,
                    max_deviation_rigid=self.max_deviation_rigid, splits=self.splits_els,
                    num_splits_to_process=None, num_iter=num_iter, template=self.total_template_els,
                    shifts_opencv=self.shifts_opencv, save_movie=save_movie, nonneg_movie=self.nonneg_movie, gSig_filt=self.gSig_filt,
                    use_cuda=self.use_cuda, border_nan=self.border_nan, var_name_hdf5=self.var_name_hdf5, is3D=self.is3D,
                    indices=self.indices)
            if not self.is3D:
                if show_template:
                    pl.imshow(new_template_els)
                    pl.pause(.5)
            if np.isnan(np.sum(new_template_els)):
                raise Exception(
                    'Template contains NaNs, something went wrong. Reconsider the parameters')

            if template is None:
                self.total_template_els = new_template_els

            self.fname_tot_els += [_fname_tot_els]
            self.templates_els += _templates_els
            self.x_shifts_els += _x_shifts_els
            self.y_shifts_els += _y_shifts_els
            if self.is3D:
                self.z_shifts_els += _z_shifts_els
            self.coord_shifts_els += _coord_shifts_els

    def apply_shifts_movie(self, fname, rigid_shifts:bool=None, save_memmap:bool=False,
                           save_base_name:str='MC', order:str='F', remove_min:bool=True):
        """
        Applies shifts found by registering one file to a different file. Useful
        for cases when shifts computed from a structural channel are applied to a
        functional channel. Currently only application of shifts through openCV is
        supported. Returns either cm.movie or the path to a memory mapped file.

        Args:
            fname: str of List[str]
                name(s) of the movie to motion correct. It should not contain
                nans. All the loadable formats from CaImAn are acceptable

            rigid_shifts: bool (True)
                apply rigid or pw-rigid shifts (must exist in the mc object)
                deprectated (read directly from mc.pw_rigid)

            save_memmap: bool (False)
                flag for saving the resulting file in memory mapped form

            save_base_name: str ['MC']
                base name for memory mapped file name

            order: 'F' or 'C' ['F']
                order of resulting memory mapped file

            remove_min: bool (True)
                If minimum value is negative, subtract it from the data

        Returns:
            m_reg: caiman movie object
                caiman movie object with applied shifts (not memory mapped)
        """

        Y = load(fname).astype(np.float32)
        if remove_min:
            ymin = Y.min()
            if ymin < 0:
                Y -= Y.min()

        if rigid_shifts is not None:
            logging.warning('The rigid_shifts flag is deprecated and it is ' +
                            'being ignored. The value is read directly from' +
                            ' mc.pw_rigid and is current set to the opposite' +
                            ' of {}'.format(self.pw_rigid))

        if self.pw_rigid is False:
            if self.is3D:
                m_reg = [apply_shifts_dft(img, (sh[0], sh[1], sh[2]), 0,
                                          is_freq=False, border_nan=self.border_nan)
                         for img, sh in zip(Y, self.shifts_rig)]
            elif self.shifts_opencv:
                m_reg = [apply_shift_iteration(img, shift, border_nan=self.border_nan)
                         for img, shift in zip(Y, self.shifts_rig)]
            else:
                m_reg = [apply_shifts_dft(img, (
                    sh[0], sh[1]), 0, is_freq=False, border_nan=self.border_nan) for img, sh in zip(
                    Y, self.shifts_rig)]
        else:
            if self.is3D:
                xyz_grid = [(it[0], it[1], it[2]) for it in sliding_window_3d(
                            Y[0], self.overlaps, self.strides)]
                dims_grid = tuple(np.add(xyz_grid[-1], 1))
                shifts_x = np.stack([np.reshape(_sh_, dims_grid, order='C').astype(
                    np.float32) for _sh_ in self.x_shifts_els], axis=0)
                shifts_y = np.stack([np.reshape(_sh_, dims_grid, order='C').astype(
                    np.float32) for _sh_ in self.y_shifts_els], axis=0)
                shifts_z = np.stack([np.reshape(_sh_, dims_grid, order='C').astype(
                    np.float32) for _sh_ in self.z_shifts_els], axis=0)
                dims = Y.shape[1:]
                x_grid, y_grid, z_grid = np.meshgrid(np.arange(0., dims[1]).astype(
                    np.float32), np.arange(0., dims[0]).astype(np.float32),
                    np.arange(0., dims[2]).astype(np.float32))
                m_reg = [warp_sk(img, np.stack((resize_sk(shiftX.astype(np.float32), dims) + y_grid,
                                 resize_sk(shiftY.astype(np.float32), dims) + x_grid,
                                 resize_sk(shiftZ.astype(np.float32), dims) + z_grid), axis=0),
                                 order=3, mode='constant')
                         for img, shiftX, shiftY, shiftZ in zip(Y, shifts_x, shifts_y, shifts_z)]
                                 # borderValue=add_to_movie)
            else:
                xy_grid = [(it[0], it[1]) for it in sliding_window(Y[0], self.overlaps, self.strides)]
                dims_grid = tuple(np.max(np.stack(xy_grid, axis=1), axis=1) - np.min(
                    np.stack(xy_grid, axis=1), axis=1) + 1)
                shifts_x = np.stack([np.reshape(_sh_, dims_grid, order='C').astype(
                    np.float32) for _sh_ in self.x_shifts_els], axis=0)
                shifts_y = np.stack([np.reshape(_sh_, dims_grid, order='C').astype(
                    np.float32) for _sh_ in self.y_shifts_els], axis=0)
                dims = Y.shape[1:]
                x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(
                    np.float32), np.arange(0., dims[0]).astype(np.float32))
                m_reg = [cv2.remap(img, -cv2.resize(shiftY, dims[::-1]) + x_grid,
                                   -cv2.resize(shiftX, dims[::-1]) + y_grid,
                                   cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                         for img, shiftX, shiftY in zip(Y, shifts_x, shifts_y)]
        m_reg = np.stack(m_reg, axis=0)
        if save_memmap:
            dims = m_reg.shape
            fname_tot = memmap_frames_filename(save_base_name, dims[1:], dims[0], order)
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                        shape=prepare_shape((np.prod(dims[1:]), dims[0])), order=order)
            big_mov[:] = np.reshape(m_reg.transpose(1, 2, 0), (np.prod(dims[1:]), dims[0]), order='F')
            big_mov.flush()
            del big_mov
            return fname_tot
        else:
            return movie(m_reg)

def load_iter(file_name, subindices=None, var_name_hdf5: str = 'mov', outtype=np.float32):
    """
    load iterator over movie from file. Supports a variety of formats. tif, hdf5, avi.

    Args:
        file_name: string
            name of file. Possible extensions are tif, avi and hdf5

        subindices: iterable indexes
            for loading only a portion of the movie

        var_name_hdf5: str
            if loading from hdf5 name of the variable to load

        outtype: The data type for the movie

    Returns:
        iter: iterator over movie

    Raises:
        Exception 'Subindices not implemented'

        Exception 'sima module unavailable'

        Exception 'Unknown file type'

        Exception 'File not found!'
    """
    if os.path.exists(file_name):
        extension = os.path.splitext(file_name)[1].lower()
        if extension in ('.tif', '.tiff', '.btf'):
            Y = tifffile.TiffFile(file_name).pages
            if subindices is not None:
                if isinstance(subindices, range):
                    subindices = slice(subindices.start, subindices.stop, subindices.step)
                Y = Y[subindices]
            for y in Y:
                yield y.asarray().astype(outtype)
        elif extension in ('.avi', '.mkv'):
            cap = cv2.VideoCapture(file_name)
            if subindices is None:
                while True:
                    ret, frame = cap.read()
                    if ret:
                        yield frame[..., 0].astype(outtype)
                    else:
                        cap.release()
                        return
                        #raise StopIteration
            else:
                if isinstance(subindices, slice):
                    subindices = range(
                        subindices.start,
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if subindices.stop is None else subindices.stop,
                        1 if subindices.step is None else subindices.step)
                t = 0
                for ind in subindices:
#                    cap.set(1, ind)
#                    ret, frame = cap.read()
#                    if ret:
#                        yield frame[..., 0]
#                    else:
#                        raise StopIteration
                    while t <= ind:
                        ret, frame = cap.read()
                        t += 1
                    if ret:
                        yield frame[..., 0].astype(outtype)
                    else:
                        return
                        #raise StopIteration
                cap.release()

                return
                #raise StopIteration
        elif extension in ('.hdf5', '.h5', '.nwb', '.mat'):
            with h5py.File(file_name, "r") as f:
                Y = f.get('acquisition/' + var_name_hdf5 + '/data'
                           if extension == '.nwb' else var_name_hdf5)
                if subindices is None:
                    for y in Y:
                        yield y.astype(outtype)
                else:
                    if isinstance(subindices, slice):
                        subindices = range(subindices.start,
                                           len(Y) if subindices.stop is None else subindices.stop,
                                           1 if subindices.step is None else subindices.step)
                    for ind in subindices:
                        yield Y[ind].astype(outtype)
        else:  # fall back to memory inefficient version
            for y in load(file_name, var_name_hdf5=var_name_hdf5,
                          subindices=subindices, outtype=outtype):
                yield y
    else:
        logging.error(f"File request:[{file_name}] not found!")
        raise Exception('File not found!')

def high_pass_filter_space(img_orig, gSig_filt=None, freq=None, order=None):
    """
    Function for high passing the image(s) with centered Gaussian if gSig_filt
    is specified or Butterworth filter if freq and order are specified

    Args:
        img_orig: 2-d or 3-d array
            input image/movie

        gSig_filt:
            size of the Gaussian filter

        freq: float
            cutoff frequency of the Butterworth filter

        order: int
            order of the Butterworth filter

    Returns:
        img: 2-d array or 3-d movie
            image/movie after filtering
    """
    if freq is None or order is None:  # Gaussian
        ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
        ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
        ker2D = ker.dot(ker.T)
        nz = np.nonzero(ker2D >= ker2D[:, 0].max())
        zz = np.nonzero(ker2D < ker2D[:, 0].max())
        ker2D[nz] -= ker2D[nz].mean()
        ker2D[zz] = 0
        if img_orig.ndim == 2:  # image
            return cv2.filter2D(np.array(img_orig, dtype=np.float32),
                                -1, ker2D, borderType=cv2.BORDER_REFLECT)
        else:  # movie
            return movie(np.array([cv2.filter2D(np.array(img, dtype=np.float32),
                                -1, ker2D, borderType=cv2.BORDER_REFLECT) for img in img_orig]))
    else:  # Butterworth
        rows, cols = img_orig.shape[-2:]
        xx, yy = np.meshgrid(np.arange(cols, dtype=np.float32) - cols / 2,
                             np.arange(rows, dtype=np.float32) - rows / 2, sparse=True)
        H = np.fft.ifftshift(1 - 1 / (1 + ((xx**2 + yy**2)/freq**2)**order))
        if img_orig.ndim == 2:  # image
            return cv2.idft(cv2.dft(img_orig, flags=cv2.DFT_COMPLEX_OUTPUT) *
                            H[..., None])[..., 0] / (rows*cols)
        else:  # movie
            return movie(np.array([cv2.idft(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT) *
                            H[..., None])[..., 0] for img in img_orig]) / (rows*cols))

def load(file_name: Union[str, List[str]],
         fr: float = 30,
         start_time: float = 0,
         meta_data: Dict = None,
         subindices=None,
         shape: Tuple[int, int] = None,
         var_name_hdf5: str = 'mov',
         in_memory: bool = False,
         is_behavior: bool = False,
         bottom=0,
         top=0,
         left=0,
         right=0,
         channel=None,
         outtype=np.float32,
         is3D: bool = False) -> Any:
    """
    load movie from file. Supports a variety of formats. tif, hdf5, npy and memory mapped. Matlab is experimental.

    Args:
        file_name: string or List[str]
            name of file. Possible extensions are tif, avi, npy, (npz and hdf5 are usable only if saved by calblitz)

        fr: float
            frame rate

        start_time: float
            initial time for frame 1

        meta_data: dict
            dictionary containing meta information about the movie

        subindices: iterable indexes
            for loading only portion of the movie

        shape: tuple of two values
            dimension of the movie along x and y if loading from a two dimensional numpy array

        var_name_hdf5: str
            if loading from hdf5 name of the variable to load

        in_memory: (undocumented)

        is_behavior: (undocumented)

        bottom,top,left,right: (undocumented)

        channel: (undocumented)

        outtype: The data type for the movie

    Returns:
        mov: caiman.movie

    Raises:
        Exception 'Subindices not implemented'

        Exception 'Subindices not implemented'

        Exception 'sima module unavailable'

        Exception 'Unknown file type'

        Exception 'File not found!'
    """
    # case we load movie from file
    if max(top, bottom, left, right) > 0 and isinstance(file_name, str):
        file_name = [file_name]        # type: ignore # mypy doesn't like that this changes type

    if isinstance(file_name, list):
        if shape is not None:
            logging.error('shape parameter not supported for multiple movie input')

        return load_movie_chain(file_name,
                                fr=fr,
                                start_time=start_time,
                                meta_data=meta_data,
                                subindices=subindices,
                                bottom=bottom,
                                top=top,
                                left=left,
                                right=right,
                                channel=channel,
                                outtype=outtype,
                                var_name_hdf5=var_name_hdf5,
                                is3D=is3D)

    elif isinstance(file_name,tuple):
        print('**** PROCESSING AS SINGLE FRAMES *****')
        if shape is not None:
            logging.error('shape not supported for multiple movie input')
        else:
            return load_movie_chain(tuple([iidd for iidd in np.array(file_name)[subindices]]),
                     fr=fr, start_time=start_time,
                     meta_data=meta_data, subindices=None,
                     bottom=bottom, top=top, left=left, right=right,
                     channel = channel, outtype=outtype)

    if max(top, bottom, left, right) > 0:
        logging.error('top bottom etc... not supported for single movie input')

    if channel is not None:
        logging.error('channel not supported for single movie input')

    if os.path.exists(file_name):
        _, extension = os.path.splitext(file_name)[:2]

        extension = extension.lower()
        if extension == '.mat':
            logging.warning('Loading a *.mat file. x- and y- dimensions ' +
                            'might have been swapped.')
            byte_stream, file_opened = scipy.io.matlab.mio._open_file(file_name, appendmat=False)
            mjv, mnv = scipy.io.matlab.mio.get_matfile_version(byte_stream)
            if mjv == 2:
                extension = '.h5'

        if extension in ['.tif', '.tiff', '.btf']:  # load tif file
            with tifffile.TiffFile(file_name) as tffl:
                multi_page = True if tffl.series[0].shape[0] > 1 else False
                if len(tffl.pages) == 1:
                    logging.warning('Your tif file is saved a single page' +
                                    'file. Performance will be affected')
                    multi_page = False
                if subindices is not None:
                    # if isinstance(subindices, (list, tuple)): # is list or tuple:
                    if isinstance(subindices, list):  # is list or tuple:
                        if multi_page:
                            if len(tffl.series[0].shape) < 4:
                                input_arr = tffl.asarray(key=subindices[0])[:, subindices[1], subindices[2]]
                            else:  # 3D
                                shape = tffl.series[0].shape
                                ts = np.arange(shape[0])[subindices[0]]
                                input_arr = tffl.asarray(key=np.ravel(ts[:, None] * shape[1] +
                                                                      np.arange(shape[1]))
                                                         ).reshape((len(ts),) + shape[1:])[
                                    :, subindices[1], subindices[2], subindices[3]]
                        else:
                            input_arr = tffl.asarray()[tuple(subindices)]

                    else:
                        if multi_page:
                            if len(tffl.series[0].shape) < 4:
                                input_arr = tffl.asarray(key=subindices)
                            else:  # 3D
                                shape = tffl.series[0].shape
                                ts = np.arange(shape[0])[subindices]
                                input_arr = tffl.asarray(key=np.ravel(ts[:, None] * shape[1] +
                                                                      np.arange(shape[1]))
                                                         ).reshape((len(ts),) + shape[1:])
                        else:
                            input_arr = tffl.asarray()
                            input_arr = input_arr[subindices]

                else:
                    input_arr = tffl.asarray()

                input_arr = np.squeeze(input_arr)

        elif extension in ('.avi', '.mkv'):      # load video file
            cap = cv2.VideoCapture(file_name)

            try:
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except:
                logging.info('Roll back to opencv 2')
                length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

            cv_failed = False
            dims = [length, height, width]                     # type: ignore # a list in one block and a tuple in another
            if length == 0 or width == 0 or height == 0:       #CV failed to load
                cv_failed = True
            if subindices is not None:
                if not isinstance(subindices, list):
                    subindices = [subindices]
                for ind, sb in enumerate(subindices):
                    if isinstance(sb, range):
                        subindices[ind] = np.r_[sb]
                        dims[ind] = subindices[ind].shape[0]
                    elif isinstance(sb, slice):
                        if sb.start is None:
                            sb = slice(0, sb.stop, sb.step)
                        if sb.stop is None:
                            sb = slice(sb.start, dims[ind], sb.step)
                        subindices[ind] = np.r_[sb]
                        dims[ind] = subindices[ind].shape[0]
                    elif isinstance(sb, np.ndarray):
                        dims[ind] = sb.shape[0]

                start_frame = subindices[0][0]
            else:
                subindices = [np.r_[range(dims[0])]]
                start_frame = 0
            if not cv_failed:
                input_arr = np.zeros((dims[0], height, width), dtype=np.uint8)
                counter = 0
                cap.set(1, start_frame)
                current_frame = start_frame
                while True and counter < dims[0]:
                    # Capture frame-by-frame
                    if current_frame != subindices[0][counter]:
                        current_frame = subindices[0][counter]
                        cap.set(1, current_frame)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    input_arr[counter] = frame[:, :, 0]
                    counter += 1
                    current_frame += 1

                if len(subindices) > 1:
                    input_arr = input_arr[:, subindices[1]]
                if len(subindices) > 2:
                    input_arr = input_arr[:, :, subindices[2]]
            else:      #use pims to load movie
                import pims

                def rgb2gray(rgb):
                    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

                pims_movie = pims.Video(file_name)
                length = len(pims_movie)
                height, width = pims_movie.frame_shape[0:2]    #shape is (h,w,channels)
                input_arr = np.zeros((length, height, width), dtype=np.uint8)
                for i in range(len(pims_movie)):               #iterate over frames
                    input_arr[i] = rgb2gray(pims_movie[i])

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

        elif extension == '.npy':      # load npy file
            if fr is None:
                fr = 30
            if in_memory:
                input_arr = np.load(file_name)
            else:
                input_arr = np.load(file_name, mmap_mode='r')

            if subindices is not None:
                input_arr = input_arr[subindices]

            if input_arr.ndim == 2:
                if shape is not None:
                    _, T = np.shape(input_arr)
                    d1, d2 = shape
                    input_arr = np.transpose(np.reshape(input_arr, (d1, d2, T), order='F'), (2, 0, 1))
                else:
                    input_arr = input_arr[np.newaxis, :, :]

        elif extension == '.mat':      # load npy file
            input_arr = loadmat(file_name)['data']
            input_arr = np.rollaxis(input_arr, 2, -3)
            if subindices is not None:
                input_arr = input_arr[subindices]

        elif extension == '.npz':      # load movie from saved file
            if subindices is not None:
                raise Exception('Subindices not implemented')
            with np.load(file_name) as f:
                return movie(**f).astype(outtype)

        elif extension in ('.hdf5', '.h5', '.nwb'):
            if is_behavior:
                with h5py.File(file_name, "r") as f:
                    kk = list(f.keys())
                    kk.sort(key=lambda x: np.int(x.split('_')[-1]))
                    input_arr = []
                    for trial in kk:
                        logging.info('Loading ' + trial)
                        input_arr.append(np.array(f[trial]['mov']))

                    input_arr = np.vstack(input_arr)

            else:
                with h5py.File(file_name, "r") as f:
                    fkeys = list(f.keys())
                    if len(fkeys) == 1:
                        var_name_hdf5 = fkeys[0]

                    if extension == '.nwb':
                        try:
                            fgroup = f[var_name_hdf5]['data']
                        except:
                            fgroup = f['acquisition'][var_name_hdf5]['data']
                    else:
                        fgroup = f[var_name_hdf5]

                    if var_name_hdf5 in f or var_name_hdf5 in f['acquisition']:
                        if subindices is None:
                            images = np.array(fgroup).squeeze()
                            #if images.ndim > 3:
                            #    images = images[:, 0]
                        else:
                            if type(subindices).__module__ == 'numpy':
                                subindices = subindices.tolist()
                            if len(fgroup.shape) > 3:
                                images = np.array(fgroup[subindices]).squeeze()
                            else:
                                images = np.array(fgroup[subindices]).squeeze()

                        #input_arr = images
                        return movie(images.astype(outtype))
                    else:
                        logging.debug('KEYS:' + str(f.keys()))
                        raise Exception('Key not found in hdf5 file')

        elif extension == '.mmap':

            filename = os.path.split(file_name)[-1]
            Yr, dims, T = load_memmap(
                os.path.join(                  # type: ignore # same dims typing issue as above
                    os.path.split(file_name)[0], filename))
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            if subindices is not None:
                images = images[subindices]

            if in_memory:
                logging.debug('loading mmap file in memory')
                images = np.array(images).astype(outtype)

            logging.debug('mmap')
            return movie(images, fr=fr)

        elif extension == '.sbx':
            logging.debug('sbx')
            if subindices is not None:
                return movie(sbxreadskip(file_name[:-4], subindices), fr=fr).astype(outtype)
            else:
                return movie(sbxread(file_name[:-4], k=0, n_frames=np.inf), fr=fr).astype(outtype)

        # elif extension == '.sima':
        #     if not HAS_SIMA:
        #         raise Exception("sima module unavailable")
        #
        #     dataset = sima.ImagingDataset.load(file_name)
        #     frame_step = 1000
        #     if subindices is None:
        #         input_arr = np.empty(
        #             (dataset.sequences[0].shape[0], dataset.sequences[0].shape[2], dataset.sequences[0].shape[3]),
        #             dtype=outtype)
        #         for nframe in range(0, dataset.sequences[0].shape[0], frame_step):
        #             input_arr[nframe:nframe + frame_step] = np.array(
        #                 dataset.sequences[0][nframe:nframe + frame_step, 0, :, :, 0]).astype(outtype).squeeze()
        #     else:
        #         input_arr = np.array(dataset.sequences[0])[subindices, :, :, :, :].squeeze()

        else:
            raise Exception('Unknown file type')
    else:
        logging.error(f"File request:[{file_name}] not found!")
        raise Exception(f'File {file_name} not found!')

    return movie(input_arr.astype(outtype),
                 fr=fr,
                 start_time=start_time,
                 file_name=os.path.split(file_name)[-1],
                 meta_data=meta_data)

#%%
def motion_correct_batch_rigid(fname, max_shifts, dview=None, splits=56, num_splits_to_process=None, num_iter=1,
                               template=None, shifts_opencv=False, save_movie_rigid=False, add_to_movie=None,
                               nonneg_movie=False, gSig_filt=None, subidx=slice(None, None, 1), use_cuda=False,
                               border_nan=True, var_name_hdf5='mov', is3D=False, indices=(slice(None), slice(None))):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file

    Args:
        fname: str
            name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable

        max_shifts: tuple
            x and y (and z if 3D) maximum allowed shifts

        dview: ipyparallel view
            used to perform parallel computing

        splits: int
            number of batches in which the movies is subdivided

        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed

        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.

        template: ndarray
            if a good approximation of the template to register is available, it can be used

        shifts_opencv: boolean
             toggle the shifts applied with opencv, if yes faster but induces some smoothing

        save_movie_rigid: boolean
             toggle save movie

        subidx: slice
            Indices to slice

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

        indices: tuple(slice), default: (slice(None), slice(None))
           Use that to apply motion correction only on a part of the FOV

    Returns:
         fname_tot_rig: str

         total_template:ndarray

         templates:list
              list of produced templates, one per batch

         shifts: list
              inferred rigid shifts to correct the movie

    Raises:
        Exception 'The movie contains nans. Nans are not allowed!'

    """

    ## get dimensions and load file to memory
    dims, T = get_file_size(fname, var_name_hdf5=var_name_hdf5)
    Ts = np.arange(T)[subidx].shape[0]
    step =  Ts // 50
    corrected_slicer = slice(subidx.start, subidx.stop, step + 1)
    m = load(fname, var_name_hdf5=var_name_hdf5, subindices=corrected_slicer)

    # if len(m.shape) < 3:
    #     m = load(fname, var_name_hdf5=var_name_hdf5)
    #     m = m[corrected_slicer]
    #     logging.warning("Your original file was saved as a single page " +
    #                     "file. Consider saving it in multiple smaller files" +
    #                     "with size smaller than 4GB (if it is a .tif file)")


    m = m[:, indices[0], indices[1]] # TODO somehow it only loads 48 frames instead of the full T length. Why? How?

    if template is None:
        if gSig_filt is not None:
            m = movie(np.array([high_pass_filter_space(m_, gSig_filt) for m_ in m]))

        if not m.flags['WRITEABLE']:
            m = m.copy()

        template = bin_median(m.motion_correct(max_shifts[1], max_shifts[0], template=None)[0])

    new_templ = template
    if add_to_movie is None:
        add_to_movie = -np.min(template)

    if np.isnan(add_to_movie):
        logging.error('The movie contains NaNs. NaNs are not allowed!')
        raise Exception('The movie contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    save_movie = False
    fname_tot_rig = None
    res_rig:List = []
    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()
        if iter_ == num_iter - 1:
            save_movie = save_movie_rigid
            logging.debug('saving!')

        if isinstance(fname, tuple):
            base_name=os.path.split(fname[0])[-1][:-4] + '_rig_'
        else:
            base_name=os.path.split(fname)[-1][:-4] + '_rig_'

        fname_tot_rig, res_rig = motion_correction_piecewise(fname, splits, strides=None, overlaps=None,
                                                             add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts, max_deviation_rigid=0,
                                                             dview=dview, save_movie=save_movie, base_name=base_name, subidx = subidx,
                                                             num_splits=num_splits_to_process, shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie, gSig_filt=gSig_filt,
                                                             use_cuda=use_cuda, border_nan=border_nan, var_name_hdf5=var_name_hdf5, is3D=is3D,
                                                             indices=indices)

        new_templ = np.nanmedian(np.dstack([r[-1] for r in res_rig]), -1)

        if gSig_filt is not None:
            new_templ = high_pass_filter_space(new_templ, gSig_filt)

    total_template = new_templ
    templates = []
    shifts:List = []
    for rr in res_rig:
        shift_info, idxs, tmpl = rr
        templates.append(tmpl)
        shifts += [sh[0] for sh in shift_info[:len(idxs)]]

    return fname_tot_rig, total_template, templates, shifts

def motion_correct_batch_pwrigid(fname, max_shifts, strides, overlaps, add_to_movie, newoverlaps=None, newstrides=None,
                                 dview=None, upsample_factor_grid=4, max_deviation_rigid=3,
                                 splits=56, num_splits_to_process=None, num_iter=1,
                                 template=None, shifts_opencv=False, save_movie=False, nonneg_movie=False, gSig_filt=None,
                                 use_cuda=False, border_nan=True, var_name_hdf5='mov', is3D=False,
                                 indices=(slice(None), slice(None))):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file

    Args:
        fname: str
            name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable

        strides: tuple
            strides of patches along x and y (and z if 3D)

        overlaps:
            overlaps of patches along x and y (and z if 3D). example: If strides = (64,64) and overlaps (32,32) patches will be (96,96)

        newstrides: tuple
            overlaps after upsampling

        newoverlaps: tuple
            strides after upsampling

        max_shifts: tuple
            x and y maximum allowed shifts (and z if 3D)

        dview: ipyparallel view
            used to perform parallel computing

        splits: int
            number of batches in which the movies is subdivided

        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed

        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.

        template: ndarray
            if a good approximation of the template to register is available, it can be used

        shifts_opencv: boolean
             toggle the shifts applied with opencv, if yes faster but induces some smoothing

        save_movie_rigid: boolean
             toggle save movie

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

        indices: tuple(slice), default: (slice(None), slice(None))
           Use that to apply motion correction only on a part of the FOV

    Returns:
        fname_tot_rig: str

        total_template:ndarray

        templates:list
            list of produced templates, one per batch

        shifts: list
            inferred rigid shifts to corrrect the movie

    Raises:
        Exception 'You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function'
    """
    if template is None:
        raise Exception('You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function')
    else:
        new_templ = template

    if np.isnan(add_to_movie):
        logging.error('The template contains NaNs. NaNs are not allowed!')
        raise Exception('The template contains NaNs. NaNs are not allowed!')
    else:
        logging.debug('Adding to movie ' + str(add_to_movie))

    for iter_ in range(num_iter):
        logging.debug(iter_)
        old_templ = new_templ.copy()

        if iter_ == num_iter - 1:
            save_movie = save_movie
            if save_movie:

                if isinstance(fname, tuple):
                    logging.debug(f'saving mmap of {fname[0]} to {fname[-1]}')
                else:
                    logging.debug(f'saving mmap of {fname}')

        if isinstance(fname, tuple):
            base_name=os.path.split(fname[0])[-1][:-4] + '_els_'
        else:
            base_name=os.path.split(fname)[-1][:-4] + '_els_'

        fname_tot_els, res_el = motion_correction_piecewise(fname, splits, strides, overlaps,
                                                            add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts,
                                                            max_deviation_rigid=max_deviation_rigid,
                                                            newoverlaps=newoverlaps, newstrides=newstrides,
                                                            upsample_factor_grid=upsample_factor_grid, order='F', dview=dview, save_movie=save_movie,
                                                            base_name=base_name, num_splits=num_splits_to_process,
                                                            shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie, gSig_filt=gSig_filt,
                                                            use_cuda=use_cuda, border_nan=border_nan, var_name_hdf5=var_name_hdf5, is3D=is3D,
                                                            indices=indices)

        new_templ = np.nanmedian(np.dstack([r[-1] for r in res_el]), -1)
        if gSig_filt is not None:
            new_templ = high_pass_filter_space(new_templ, gSig_filt)

    total_template = new_templ
    templates = []
    x_shifts = []
    y_shifts = []
    z_shifts = []
    coord_shifts = []
    for rr in res_el:
        shift_info_chunk, idxs_chunk, tmpl_chunk = rr
        templates.append(tmpl_chunk)
        for shift_info, _ in zip(shift_info_chunk, idxs_chunk):
            if is3D:
                total_shift, _, xyz_grid = shift_info
                x_shifts.append(np.array([sh[0] for sh in total_shift]))
                y_shifts.append(np.array([sh[1] for sh in total_shift]))
                z_shifts.append(np.array([sh[2] for sh in total_shift]))
                coord_shifts.append(xyz_grid)
            else:
                total_shift, _, xy_grid = shift_info
                x_shifts.append(np.array([sh[0] for sh in total_shift]))
                y_shifts.append(np.array([sh[1] for sh in total_shift]))
                coord_shifts.append(xy_grid)

    return fname_tot_els, total_template, templates, x_shifts, y_shifts, z_shifts, coord_shifts


def motion_correction_piecewise(fname, splits, strides, overlaps, add_to_movie=0, template=None,
                                max_shifts=(12, 12), max_deviation_rigid=3, newoverlaps=None, newstrides=None,
                                upsample_factor_grid=4, order='F', dview=None, save_movie=True,
                                base_name=None, subidx = None, num_splits=None, shifts_opencv=False, nonneg_movie=False, gSig_filt=None,
                                use_cuda=False, border_nan=True, var_name_hdf5='mov', is3D=False,
                                indices=(slice(None), slice(None))):
    """

    """
    # todo todocument
    if isinstance(fname, tuple):
        name, extension = os.path.splitext(fname[0])[:2]
    else:
        name, extension = os.path.splitext(fname)[:2]
    extension = extension.lower()
    is_fiji = False

    dims, T = get_file_size(fname, var_name_hdf5=var_name_hdf5)
    z = np.zeros(dims)
    dims = z[indices].shape
    logging.debug('Number of Splits: {}'.format(splits))
    if isinstance(splits, int):
        if subidx is None:
            rng = range(T)
        else:
            rng = range(T)[subidx]

        idxs = np.array_split(list(rng), splits)

    else:
        idxs = splits
        save_movie = False
    if template is None:
        raise Exception('Not implemented')

    shape_mov = (np.prod(dims), T)

    if num_splits is not None:
        idxs = np.array(idxs)[np.random.randint(0, len(idxs), num_splits)]
        save_movie = False
        #logging.warning('**** MOVIE NOT SAVED BECAUSE num_splits is not None ****')

    if save_movie:
        if base_name is None:
            base_name = os.path.split(fname)[1][:-4]
        fname_tot:Optional[str] = memmap_frames_filename(base_name, dims, T, order)
        if isinstance(fname, tuple):
            fname_tot = os.path.join(os.path.split(fname[0])[0], fname_tot)
        else:
            fname_tot = os.path.join(os.path.split(fname)[0], fname_tot)

        np.memmap(fname_tot, mode='w+', dtype=np.float32,
                  shape=prepare_shape(shape_mov), order=order)
        logging.info('Saving file as {}'.format(fname_tot))
    else:
        fname_tot = None

    pars = []
    for idx in idxs:
        logging.debug('Processing: frames: {}'.format(idx))
        # TASK list
        pars.append([fname, fname_tot, idx, shape_mov, template, strides, overlaps, max_shifts, np.array(
            add_to_movie, dtype=np.float32), max_deviation_rigid, upsample_factor_grid,
            newoverlaps, newstrides, shifts_opencv, nonneg_movie, gSig_filt, is_fiji,
            use_cuda, border_nan, var_name_hdf5, is3D, indices])

    if dview is not None:
        logging.info('** Starting parallel motion correction **')
        if HAS_CUDA and use_cuda:
            # res = dview.map(tile_and_correct_wrapper,pars)
            # dview.map(close_cuda_process, range(len(pars)))
            print("**to_julia** not implemented the moment")
            sys.exit(2)
            # **to_julia** not interesting for us for now
        elif 'multiprocessing' in str(type(dview)):
            res = dview.map_async(tile_and_correct_wrapper, pars).get(4294967)
        else:
            res = dview.map_sync(tile_and_correct_wrapper, pars)
        logging.info('** Finished parallel motion correction **')
    else:
        res = list(map(tile_and_correct_wrapper, pars))

    return fname_tot, res


def tile_and_correct(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None, upsample_factor_grid=4,
                     upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=False, gSig_filt=None,
                     use_cuda=False, border_nan=True):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Args:
        img: ndaarray 2D
            image to correct

        template: ndarray
            reference image

        strides: tuple
            strides of the patches in which the FOV is subdivided

        overlaps: tuple
            amount of pixel overlaping between patches along each dimension

        max_shifts: tuple
            max shifts in x and y

        newstrides:tuple
            strides between patches along each dimension when upsampling the vector fields

        newoverlaps:tuple
            amount of pixel overlaping between patches along each dimension when upsampling the vector fields

        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

        upsample_factor_fft: int
            resolution of fractional shifts

        show_movie: boolean whether to visualize the original and corrected frame during motion correction

        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)

        add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

        filt_sig_size: tuple
            standard deviation and size of gaussian filter to center filter data in case of one photon imaging data

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

        border_nan : bool or string, optional
            specifies how to deal with borders. (True, False, 'copy', 'min')

    Returns:
        (new_img, total_shifts, start_step, xy_grid)
            new_img: ndarray, corrected image


    """

    img = img.astype(np.float64).copy()
    template = template.astype(np.float64).copy()

    # High Pass Filter with gSig_filt
    if gSig_filt is not None:

        img_orig = img.copy()
        img = high_pass_filter_space(img_orig, gSig_filt)

    img = img + add_to_movie
    template = template + add_to_movie

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts, use_cuda=use_cuda)

    ## max_deviation_rigid is 3 for us
    if max_deviation_rigid == 0:

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            new_img = apply_shift_iteration(
                img, (-rigid_shts[0], -rigid_shts[1]), border_nan=border_nan)

        else:

            if gSig_filt is not None:
                raise Exception(
                    'The use of FFT and filtering options have not been tested. Set opencv=True')

            new_img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)

        return new_img - add_to_movie, (-rigid_shts[0], -rigid_shts[1]), None, None
    else:
        # extract patches
        templates = [
            it[-1] for it in sliding_window(template, overlaps=overlaps, strides=strides)]
        xy_grid = [(it[0], it[1]) for it in sliding_window(
            template, overlaps=overlaps, strides=strides)]
        num_tiles = np.prod(np.add(xy_grid[-1], 1))
        imgs = [it[-1]
                for it in sliding_window(img, overlaps=overlaps, strides=strides)]
        dim_grid = tuple(np.add(xy_grid[-1], 1))

        if max_deviation_rigid is not None:

            lb_shifts = np.ceil(np.subtract(
                rigid_shts, max_deviation_rigid)).astype(int)
            ub_shifts = np.floor(
                np.add(rigid_shts, max_deviation_rigid)).astype(int)

        else:

            lb_shifts = None
            ub_shifts = None

        # extract shifts for each patch
        shfts_et_all = [register_translation(
            a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts, use_cuda=use_cuda) for a, b, c in zip(
            imgs, templates, [upsample_factor_fft] * num_tiles)]
        shfts = [sshh[0] for sshh in shfts_et_all]
        diffs_phase = [sshh[2] for sshh in shfts_et_all]
        # create a vector field
        shift_img_x = np.reshape(np.array(shfts)[:, 0], dim_grid)
        shift_img_y = np.reshape(np.array(shfts)[:, 1], dim_grid)
        diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            dims = img.shape
            x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(
                np.float32), np.arange(0., dims[0]).astype(np.float32))
            m_reg = cv2.remap(img, cv2.resize(shift_img_y.astype(np.float32), dims[::-1]) + x_grid,
                              cv2.resize(shift_img_x.astype(np.float32), dims[::-1]) + y_grid,
                              cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                             # borderValue=add_to_movie)
            total_shifts = [
                    (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
            return m_reg - add_to_movie, total_shifts, None, None

        # create automatically upsample parameters if not passed
        if newoverlaps is None:
            newoverlaps = overlaps
        if newstrides is None:
            newstrides = tuple(
                np.round(np.divide(strides, upsample_factor_grid)).astype(np.int))

        newshapes = np.add(newstrides, newoverlaps)

        imgs = [it[-1]
                for it in sliding_window(img, overlaps=newoverlaps, strides=newstrides)]

        xy_grid = [(it[0], it[1]) for it in sliding_window(
            img, overlaps=newoverlaps, strides=newstrides)]

        start_step = [(it[2], it[3]) for it in sliding_window(
            img, overlaps=newoverlaps, strides=newstrides)]

        dim_new_grid = tuple(np.add(xy_grid[-1], 1))

        shift_img_x = cv2.resize(
            shift_img_x, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
        shift_img_y = cv2.resize(
            shift_img_y, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
        diffs_phase_grid_us = cv2.resize(
            diffs_phase_grid, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)

        num_tiles = np.prod(dim_new_grid)

        max_shear = np.percentile(
            [np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product(
                [shift_img_x, shift_img_y], [0, 1])], 75)

        total_shifts = [
            (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
        total_diffs_phase = [
            dfs for dfs in diffs_phase_grid_us.reshape(num_tiles)]

        if gSig_filt is not None:
            raise Exception(
                'The use of FFT and filtering options have not been tested. Set opencv=True')

        imgs = [apply_shifts_dft(im, (
            sh[0], sh[1]), dffphs, is_freq=False, border_nan=border_nan) for im, sh, dffphs in zip(
            imgs, total_shifts, total_diffs_phase)]

        normalizer = np.zeros_like(img) * np.nan
        new_img = np.zeros_like(img) * np.nan

        weight_matrix = create_weight_matrix_for_blending(
            img, newoverlaps, newstrides)

        if max_shear < 0.5:
            for (x, y), (_, _), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):

                prev_val_1 = normalizer[x:x + newshapes[0], y:y + newshapes[1]]

                normalizer[x:x + newshapes[0], y:y + newshapes[1]] = np.nansum(
                    np.dstack([~np.isnan(im) * 1 * weight_mat, prev_val_1]), -1)
                prev_val = new_img[x:x + newshapes[0], y:y + newshapes[1]]
                new_img[x:x + newshapes[0], y:y + newshapes[1]
                        ] = np.nansum(np.dstack([im * weight_mat, prev_val]), -1)

            new_img = old_div(new_img, normalizer)

        else:  # in case the difference in shift between neighboring patches is larger than 0.5 pixels we do not interpolate in the overlaping area
            half_overlap_x = np.int(newoverlaps[0] / 2)
            half_overlap_y = np.int(newoverlaps[1] / 2)
            for (x, y), (idx_0, idx_1), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):

                if idx_0 == 0:
                    x_start = x
                else:
                    x_start = x + half_overlap_x

                if idx_1 == 0:
                    y_start = y
                else:
                    y_start = y + half_overlap_y

                x_end = x + newshapes[0]
                y_end = y + newshapes[1]
                new_img[x_start:x_end,
                        y_start:y_end] = im[x_start - x:, y_start - y:]

        # if show_movie:
        #     img = apply_shifts_dft(
        #         sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)
        #     img_show = np.vstack([new_img, img])
        #
        #     img_show = cv2.resize(img_show, None, fx=1, fy=1)
        #
        #     cv2.imshow('frame', old_div(img_show, np.percentile(template, 99)))
        #     cv2.waitKey(int(1. / 500 * 1000))
        #
        # else:
        #     try:
        #         cv2.destroyAllWindows()
        #     except:
        #         pass

        return new_img - add_to_movie, total_shifts, start_step, xy_grid

def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb=None, shifts_ub=None, max_shifts=(10, 10),
                         use_cuda=False):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"

     ValueError "Error: images must really be same size for "
                         "register_translation"

     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    if HAS_CUDA and use_cuda:
        # from skcuda.fft import Plan
        # from skcuda.fft import fft as cudafft
        # from skcuda.fft import ifft as cudaifft
        # try:
        #     cudactx # type: ignore
        # except NameError:
        #     init_cuda_process()
        print("**to_julia** currently not implemented")
        sys.exit(2)


    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        if HAS_CUDA and use_cuda:
            # # src_image_cpx = np.array(src_image, dtype=np.complex128, copy=False)
            # # target_image_cpx = np.array(target_image, dtype=np.complex128, copy=False)
            #
            # image_gpu = gpuarray.to_gpu(np.stack((src_image, target_image)).astype(np.complex128))
            # freq_gpu = gpuarray.empty((2, src_image.shape[0], src_image.shape[1]), dtype=np.complex128)
            # # src_image_gpu = gpuarray.to_gpu(src_image_cpx)
            # # src_freq_gpu = gpuarray.empty(src_image_cpx.shape, np.complex128)
            #
            # # target_image_gpu = gpuarray.to_gpu(target_image_cpx)
            # # target_freq_gpu = gpuarray.empty(target_image_cpx.shape, np.complex128)
            #
            # plan = Plan(src_image.shape, np.complex128, np.complex128, batch=2)
            # # cudafft(src_image_gpu, src_freq_gpu, plan, scale=True)
            # # cudafft(target_image_gpu, target_freq_gpu, plan, scale=True)
            # cudafft(image_gpu, freq_gpu, plan, scale=True)
            # # src_freq = src_freq_gpu.get()
            # # target_freq = target_freq_gpu.get()
            # freq = freq_gpu.get()
            # src_freq = freq[0, :, :]
            # target_freq = freq[1, :, :]
            #
            # # del(src_image_gpu)
            # # del(src_freq_gpu)
            # # del(target_image_gpu)
            # # del(target_freq_gpu)
            # del(image_gpu)
            # del(freq_gpu)

            print("**to_julia** currently not implemented")
            sys.exit(2)

        elif opencv:
            src_freq_1 = fftn(
                src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq_1[:, :, 0] + 1j * src_freq_1[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)
            target_freq_1 = fftn(
                target_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            target_freq = target_freq_1[:, :, 0] + 1j * target_freq_1[:, :, 1]
            target_freq = np.array(
                target_freq, dtype=np.complex128, copy=False)
        else:
            src_image_cpx = np.array(
                src_image, dtype=np.complex128, copy=False)
            target_image_cpx = np.array(
                target_image, dtype=np.complex128, copy=False)
            src_freq = np.fft.fftn(src_image_cpx)
            target_freq = np.fft.fftn(target_image_cpx)

    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if HAS_CUDA and use_cuda:
        # image_product_gpu = gpuarray.to_gpu(image_product)
        # cross_correlation_gpu = gpuarray.empty(
        #     image_product.shape, np.complex128)
        # iplan = Plan(image_product.shape, np.complex128, np.complex128)
        # cudaifft(image_product_gpu, cross_correlation_gpu, iplan, scale=True)
        # cross_correlation = cross_correlation_gpu.get()
        print("**to_julia** currently not implemented")
        sys.exit(2)
    elif opencv:

        image_product_cv = np.dstack(
            [np.real(image_product), np.imag(image_product)])
        cross_correlation = fftn(
            image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
        cross_correlation = cross_correlation[:,
                                              :, 0] + 1j * cross_correlation[:, :, 1]
    else:
        cross_correlation = ifftn(image_product)

    # Locate maximum
    new_cross_corr = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):

        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0]:, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:, :shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0

        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    ###########################
    # centering shifts around midpoint
    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(old_div(axis_size, 2)) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:

        src_amp = old_div(np.sum(np.abs(src_freq) ** 2), src_freq.size)
        target_amp = old_div(
            np.sum(np.abs(target_freq) ** 2), target_freq.size)
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + old_div(maxima, upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)

def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True, border_nan=True):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """

    is3D = len(src_freq.shape) == 3
    if not is_freq:
        src_freq = np.dstack([np.real(src_freq), np.imag(src_freq)])
        src_freq = fftn(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        src_freq = src_freq[:, :, 0] + 1j * src_freq[:, :, 1]
        src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    if not is3D:
        shifts = shifts[::-1]
        nc, nr = np.shape(src_freq)
        Nr = ifftshift(np.arange(-np.fix(nr/2.), np.ceil(nr/2.)))
        Nc = ifftshift(np.arange(-np.fix(nc/2.), np.ceil(nc/2.)))
        Nr, Nc = np.meshgrid(Nr, Nc)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * 1. * Nr / nr - shifts[1] * 1. * Nc / nc))
    else:
        #shifts = np.array([*shifts[:-1][::-1],shifts[-1]])
        shifts = np.array(list(shifts[:-1][::-1]) + [shifts[-1]])
        nc, nr, nd = np.array(np.shape(src_freq), dtype=float)
        Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
        Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
        Nd = ifftshift(np.arange(-np.fix(nd / 2.), np.ceil(nd / 2.)))
        Nr, Nc, Nd = np.meshgrid(Nr, Nc, Nd)
        Greg = src_freq * np.exp(-1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc -
                                  shifts[2] * Nd / nd))

    Greg = Greg.dot(np.exp(1j * diffphase))

    Greg = np.dstack([np.real(Greg), np.imag(Greg)])
    new_img = ifftn(Greg)[:, :, 0]

    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shifts[:2])).astype(np.int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shifts[:2])).astype(np.int)
        if border_nan is True:
            new_img[:max_h, :] = np.nan
            if min_h < 0:
                new_img[min_h:, :] = np.nan
            new_img[:, :max_w] = np.nan
            if min_w < 0:
                new_img[:, min_w:] = np.nan
        elif border_nan == 'min':
            min_ = np.nanmin(new_img)
            new_img[:max_h, :] = min_
            if min_h < 0:
                new_img[min_h:, :] = min_
            new_img[:, :max_w] = min_
            if min_w < 0:
                new_img[:, min_w:] = min_
        elif border_nan == 'copy':
            new_img[:max_h] = new_img[max_h]
            if min_h < 0:
                new_img[min_h:] = new_img[min_h-1]
            if max_w > 0:
                new_img[:, :max_w] = new_img[:, max_w, np.newaxis]
            if min_w < 0:
                new_img[:, min_w:] = new_img[:, min_w-1, np.newaxis]

    return new_img

#%% in parallel
def tile_and_correct_wrapper(params):
    """Does motion correction on specified image frames

    Returns:
    shift_info:
    idxs:
    mean_img: mean over all frames of corrected image (to get individ frames, use out_fname to write them to disk)

    Notes:
    Also writes corrected frames to the mmap file specified by out_fname (if not None)

    """
    # todo todocument


    try:
        cv2.setNumThreads(0)
    except:
        pass  # 'Open CV is naturally single threaded'

    img_name, out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, \
        shifts_opencv, nonneg_movie, gSig_filt, is_fiji, use_cuda, border_nan, var_name_hdf5, \
        is3D, indices = params


    if isinstance(img_name, tuple):
        name, extension = os.path.splitext(img_name[0])[:2]
    else:
        name, extension = os.path.splitext(img_name)[:2]
    extension = extension.lower()
    shift_info = []

    imgs = load(img_name, subindices=idxs, var_name_hdf5=var_name_hdf5,is3D=is3D)
    imgs = imgs[(slice(None),) + indices]
    mc = np.zeros(imgs.shape, dtype=np.float32)
    if not imgs[0].shape == template.shape:
        template = template[indices]
    for count, img in enumerate(imgs):
        if count % 10 == 0:
            logging.debug(count)
        if is3D:
            mc[count], total_shift, start_step, xyz_grid = tile_and_correct_3d(img, template, strides, overlaps, max_shifts,
                                                                       add_to_movie=add_to_movie, newoverlaps=newoverlaps,
                                                                       newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid,
                                                                       upsample_factor_fft=10, show_movie=False,
                                                                       max_deviation_rigid=max_deviation_rigid,
                                                                       shifts_opencv=shifts_opencv, gSig_filt=gSig_filt,
                                                                       use_cuda=use_cuda, border_nan=border_nan)
            shift_info.append([tuple(-np.array(total_shift)), start_step, xyz_grid])

        else:
            mc[count], total_shift, start_step, xy_grid = tile_and_correct(img, template, strides, overlaps, max_shifts,
                                                                       add_to_movie=add_to_movie, newoverlaps=newoverlaps,
                                                                       newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid,
                                                                       upsample_factor_fft=10, show_movie=False,
                                                                       max_deviation_rigid=max_deviation_rigid,
                                                                       shifts_opencv=shifts_opencv, gSig_filt=gSig_filt,
                                                                       use_cuda=use_cuda, border_nan=border_nan)
            shift_info.append([total_shift, start_step, xy_grid])

    if out_fname is not None:
        outv = np.memmap(out_fname, mode='r+', dtype=np.float32,
                         shape=prepare_shape(shape_mov), order='F')
        if nonneg_movie:
            bias = np.float32(add_to_movie)
        else:
            bias = 0
        outv[:, idxs] = np.reshape(
            mc.astype(np.float32), (len(imgs), -1), order='F').T + bias
    new_temp = np.nanmean(mc, 0)
    new_temp[np.isnan(new_temp)] = np.nanmin(new_temp)
    return shift_info, idxs, new_temp

def apply_shift_iteration(img, shift, border_nan:bool=False, border_type=cv2.BORDER_REFLECT):
    # todo todocument

    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.nanmin(img), np.nanmax(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i),
                                 flags=cv2.INTER_CUBIC, borderMode=border_type), min_, max_)
    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shift)).astype(np.int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shift)).astype(np.int)
        if border_nan is True:
            img[:max_h, :] = np.nan
            if min_h < 0:
                img[min_h:, :] = np.nan
            img[:, :max_w] = np.nan
            if min_w < 0:
                img[:, min_w:] = np.nan
        elif border_nan == 'min':
            img[:max_h, :] = min_
            if min_h < 0:
                img[min_h:, :] = min_
            img[:, :max_w] = min_
            if min_w < 0:
                img[:, min_w:] = min_
        elif border_nan == 'copy':
            if max_h > 0:
                img[:max_h] = img[max_h]
            if min_h < 0:
                img[min_h:] = img[min_h-1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w-1, np.newaxis]

    return img

def sliding_window(image, overlaps, strides):
    """ efficiently and lazily slides a window across the image

    Args:
        img:ndarray 2D
            image that needs to be slices

        windowSize: tuple
            dimension of the patch

        strides: tuple
            stride in each dimension

     Returns:
         iterator containing five items
              dim_1, dim_2 coordinates in the patch grid
              x, y: bottom border of the patch in the original matrix

              patch: the patch
     """
    windowSize = np.add(overlaps, strides)
    range_1 = list(range(
        0, image.shape[0] - windowSize[0], strides[0])) + [image.shape[0] - windowSize[0]]
    range_2 = list(range(
        0, image.shape[1] - windowSize[1], strides[1])) + [image.shape[1] - windowSize[1]]
    for dim_1, x in enumerate(range_1):
        for dim_2, y in enumerate(range_2):
            # yield the current window
            yield (dim_1, dim_2, x, y, image[x:x + windowSize[0], y:y + windowSize[1]])

def prepare_shape(mytuple: Tuple) -> Tuple:
    """ This promotes the elements inside a shape into np.uint64. It is intended to prevent overflows
        with some numpy operations that are sensitive to it, e.g. np.memmap """
    if not isinstance(mytuple, tuple):
        raise Exception("Internal error: prepare_shape() passed a non-tuple")
    return tuple(map(lambda x: np.uint64(x), mytuple))

def memmap_frames_filename(basename: str, dims: Tuple, frames: int, order: str = 'F') -> str:
    # Some functions calling this have the first part of *their* dims Tuple be the number of frames.
    # They *must* pass a slice to this so dims is only X, Y, and optionally Z. Frames is passed separately.
    dimfield_0 = dims[0]
    dimfield_1 = dims[1]
    if len(dims) == 3:
        dimfield_2 = dims[2]
    else:
        dimfield_2 = 1
    return f"{basename}_d1_{dimfield_0}_d2_{dimfield_1}_d3_{dimfield_2}_order_{order}_frames_{frames}_.mmap"

def load_movie_chain(file_list: List[str],
                     fr: float = 30,
                     start_time=0,
                     meta_data=None,
                     subindices=None,
                     var_name_hdf5: str = 'mov',
                     bottom=0,
                     top=0,
                     left=0,
                     right=0,
                     z_top=0,
                     z_bottom=0,
                     is3D: bool = False,
                     channel=None,
                     outtype=np.float32) -> Any:
    """ load movies from list of file names

    Args:
        file_list: list
           file names in string format

        the other parameters as in load_movie except

        bottom, top, left, right, z_top, z_bottom : int
            to load only portion of the field of view

        is3D : bool
            flag for 3d data (adds a fourth dimension)

    Returns:
        movie: movie
            movie corresponding to the concatenation og the input files

    """
    mov = []
    for f in tqdm(file_list):
        m = load(f,
                 fr=fr,
                 start_time=start_time,
                 meta_data=meta_data,
                 subindices=subindices,
                 in_memory=True,
                 outtype=outtype,
                 var_name_hdf5=var_name_hdf5)
        if channel is not None:
            logging.debug(m.shape)
            m = m[channel].squeeze()
            logging.debug(f"Movie shape: {m.shape}")

        if not is3D:
            if m.ndim == 2:
                m = m[np.newaxis, :, :]

            _, h, w = np.shape(m)
            m = m[:, top:h - bottom, left:w - right]
        else:
            if m.ndim == 3:
                m = m[np.newaxis, :, :, :]

            _, h, w, d = np.shape(m)
            m = m[:, top:h - bottom, left:w - right, z_top:d - z_bottom]

        mov.append(m)
    return concatenate(mov, axis=0)


def load_memmap(filename: str, mode: str = 'r') -> Tuple[Any, Tuple, int]:
    """ Load a memory mapped file created by the function save_memmap

    Args:
        filename: str
            path of the file to be loaded
        mode: str
            One of 'r', 'r+', 'w+'. How to interact with files

    Returns:
        Yr:
            memory mapped variable

        dims: tuple
            frame dimensions

        T: int
            number of frames


    Raises:
        ValueError "Unknown file extension"

    """
    if pathlib.Path(filename).suffix != '.mmap':
        logging.error(f"Unknown extension for file {filename}")
        raise ValueError(f'Unknown file extension for file {filename} (should be .mmap)')
    # Strip path components and use CAIMAN_DATA/example_movies
    # TODO: Eventually get the code to save these in a different dir
    fn_without_path = os.path.split(filename)[-1]
    fpart = fn_without_path.split('_')[1:-1]  # The filename encodes the structure of the map
    d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]

    filename = fn_relocated(filename)
    Yr = np.memmap(filename, mode=mode, shape=prepare_shape((d1 * d2 * d3, T)), dtype=np.float32, order=order)
    if d3 == 1:
        return (Yr, (d1, d2), T)
    else:
        return (Yr, (d1, d2, d3), T)


def save_memmap(filenames: List[str],
                base_name: str = 'Yr',
                resize_fact: Tuple = (1, 1, 1),
                remove_init: int = 0,
                idx_xy: Tuple = None,
                order: str = 'F',
                var_name_hdf5: str = 'mov',
                xy_shifts: Optional[List] = None,
                is_3D: bool = False,
                add_to_movie: float = 0,
                border_to_0=0,
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

    if len(filenames) > 1:
        recompute_each_memmap = False
        for file__ in filenames:
            if ('order_' + order not in file__) or ('.mmap' not in file__):
                recompute_each_memmap = True


        if recompute_each_memmap or (remove_init>0) or (idx_xy is not None)\
                or (xy_shifts is not None) or (add_to_movie != 0) or (border_to_0>0)\
                or slices is not None:

            logging.debug('Distributing memory map over many files')
            # Here we make a bunch of memmap files in the right order. Same parameters
            fname_parts = save_memmap_each(filenames,
                                              base_name=base_name,
                                              order=order,
                                              border_to_0=border_to_0,
                                              dview=dview,
                                              var_name_hdf5=var_name_hdf5,
                                              resize_fact=resize_fact,
                                              remove_init=remove_init,
                                              idx_xy=idx_xy,
                                              xy_shifts=xy_shifts,
                                              is_3D=is_3D,
                                              slices=slices,
                                              add_to_movie=add_to_movie)
        else:
            fname_parts = filenames

        # The goal is to make a single large memmap file, which we do here
        if order == 'F':
            raise Exception('You cannot merge files in F order, they must be in C order')

        fname_new = save_memmap_join(fname_parts, base_name=base_name,
                                        dview=dview, n_chunks=n_chunks)

    else:
        # TODO: can be done online
        Ttot = 0
        for idx, f in enumerate(filenames):
            if isinstance(f, str):     # Might not always be filenames.
                logging.debug(f)

            if is_3D:
                Yr = f if not (isinstance(f, basestring)) else tifffile.imread(f)
                if Yr.ndim == 3:
                    Yr = Yr[None, ...]
                if slices is not None:
                    Yr = Yr[tuple(slices)]
                else:
                    if idx_xy is None:         #todo remove if not used, superceded by the slices parameter
                        Yr = Yr[remove_init:]
                    elif len(idx_xy) == 2:     #todo remove if not used, superceded by the slices parameter
                        Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
                    else:                      #todo remove if not used, superceded by the slices parameter
                        Yr = Yr[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

            else:
                if isinstance(f, (basestring, list)):
                    Yr = cm.load(caiman.paths.fn_relocated(f), fr=1, in_memory=True, var_name_hdf5=var_name_hdf5)
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

            if border_to_0 > 0:
                if slices is not None:
                    if isinstance(slices, list):
                        raise Exception(
                            'You cannot slice in x and y and then use add_to_movie: if you only want to slice in time do not pass in a list but just a slice object'
                        )

                min_mov = Yr.calc_min()
                Yr[:, :border_to_0, :] = min_mov
                Yr[:, :, :border_to_0] = min_mov
                Yr[:, :, -border_to_0:] = min_mov
                Yr[:, -border_to_0:, :] = min_mov

            fx, fy, fz = resize_fact
            if fx != 1 or fy != 1 or fz != 1:
                if 'movie' not in str(type(Yr)):
                    Yr = cm.movie(Yr, fr=1)
                Yr = Yr.resize(fx=fx, fy=fy, fz=fz)

            T, dims = Yr.shape[0], Yr.shape[1:]
            Yr = np.transpose(Yr, list(range(1, len(dims) + 1)) + [0])
            Yr = np.reshape(Yr, (np.prod(dims), T), order='F')
            Yr = np.ascontiguousarray(Yr, dtype=np.float32) + np.float32(0.0001) + np.float32(add_to_movie)

            if idx == 0:
                fname_tot = base_name + '_d1_' + str(
                    dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(1 if len(dims) == 2 else dims[2]) + '_order_' + str(
                        order) # TODO: Rewrite more legibly, move to caiman.paths
                if isinstance(f, str):
                    fname_tot = caiman.paths.fn_relocated(os.path.join(os.path.split(f)[0], fname_tot))
                if len(filenames) > 1:
                    big_mov = np.memmap(caiman.paths.fn_relocated(fname_tot),
                                        mode='w+',
                                        dtype=np.float32,
                                        shape=prepare_shape((np.prod(dims), T)),
                                        order=order)
                    big_mov[:, Ttot:Ttot + T] = Yr
                    del big_mov
                else:
                    logging.debug('SAVING WITH numpy.tofile()')
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
            Ttot = Ttot + T

        fname_new = caiman.paths.fn_relocated(fname_tot + f'_frames_{Ttot}_.mmap')
        try:
            # need to explicitly remove destination on windows
            os.unlink(fname_new)
        except OSError:
            pass
        os.rename(fname_tot, fname_new)

    return fname_new


def save_memmap_each(fnames: List[str],
                     dview=None,
                     base_name: str = None,
                     resize_fact=(1, 1, 1),
                     remove_init: int = 0,
                     idx_xy=None,
                     var_name_hdf5='mov',
                     xy_shifts=None,
                     is_3D=False,
                     add_to_movie: float = 0,
                     border_to_0: int = 0,
                     order: str = 'C',
                     slices=None) -> List[str]:
    """
    Create several memory mapped files using parallel processing

    Args:
        fnames: list of str
            list of path to the filenames

        dview: ipyparallel dview
            used to perform computation in parallel. If none it will be signle thread

        base_name str
            BaseName for the file to be creates. If not given the file itself is used

        resize_fact: tuple
            resampling factors for each dimension x,y,time. .1 = downsample 10X

        remove_init: int
            number of samples to remove from the beginning of each chunk

        idx_xy: slice operator
            used to perform slicing of the movie (to select a subportion of the movie)

        xy_shifts: list
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping

        is_3D: boolean
            whether it is 3D data

        add_to_movie: float
            if movie too negative will make it positive

        border_to_0: int
            number of pixels on the border to set to the minimum of the movie

        order: (undocumented)

        slices: (undocumented)

    Returns:
        fnames_tot: list
            paths to the created memory map files

    """

    pars = []
    if xy_shifts is None:
        xy_shifts = [None] * len(fnames)

    if not isinstance(resize_fact, list):
        resize_fact = [resize_fact] * len(fnames)

    for idx, f in enumerate(fnames):
        if base_name is not None:
            pars.append([
                fn_relocated(f),
                base_name + '{:04d}'.format(idx), resize_fact[idx], remove_init, idx_xy, order,
                var_name_hdf5, xy_shifts[idx], is_3D, add_to_movie, border_to_0, slices
            ])
        else:
            pars.append([
                fn_relocated(f),
                os.path.splitext(f)[0], resize_fact[idx], remove_init, idx_xy, order, var_name_hdf5,
                xy_shifts[idx], is_3D, add_to_movie, border_to_0, slices
            ])

    # Perform the job using whatever computing framework we're set to use
    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            fnames_new = dview.map_async(save_place_holder, pars).get(4294967)
        else:
            fnames_new = my_map(dview, save_place_holder, pars)
    else:
        fnames_new = list(map(save_place_holder, pars))

    return fnames_new

def my_map(dv, func, args) -> List:
    v = dv
    rc = v.client
    # scatter 'id', so id=0,1,2 on engines 0,1,2
    dv.scatter('id', rc.ids, flatten=True)
    logging.debug(dv['id'])
    amr = v.map(func, args)

    pending = set(amr.msg_ids)
    results_all: Dict = dict()
    counter = 0
    while pending:
        try:
            rc.wait(pending, 1e0)
        except parallel.TimeoutError:
            # ignore timeouterrors, since it means at least one isn't done
            pass
        # finished is the set of msg_ids that are complete
        finished = pending.difference(rc.outstanding)
        # update pending to exclude those that just finished
        pending = pending.difference(finished)
        if counter % 10 == 0:
            logging.debug(amr.progress)
        for msg_id in finished:
            # we know these are done, so don't worry about blocking
            ar = rc.get_result(msg_id)
            logging.debug(f"job id {msg_id} finished on engine {ar.engine_id}") # TODO: Abstract out the ugly bits
            logging.debug("with stdout:")
            logging.debug('    ' + ar.stdout.replace('\n', '\n    ').rstrip())
            logging.debug("and errors:")
            logging.debug('    ' + ar.stderr.replace('\n', '\n    ').rstrip())
                                                                                      # note that each job in a map always returns a list of length chunksize
                                                                                      # even if chunksize == 1
            results_all.update(ar.get_dict())
        counter += 1

    result_ordered = list(chain.from_iterable([results_all[k] for k in sorted(results_all.keys())]))
    del results_all
    return result_ordered

def save_memmap_join(mmap_fnames: List[str], base_name: str = None, n_chunks: int = 20, dview=None,
                     add_to_mov=0) -> str:
    """
    Makes a large file memmap from a number of smaller files

    Args:
        mmap_fnames: list of memory mapped files

        base_name: string, will be the first portion of name to be solved

        n_chunks: number of chunks in which to subdivide when saving, smaller requires more memory

        dview: cluster handle

        add_to_mov: (undocumented)

    """

    tot_frames = 0
    order = 'C'
    for f in mmap_fnames:
        cleaner_f = fn_relocated(f)
        Yr, dims, T = load_memmap(cleaner_f)
        logging.debug(f"save_memmap_join (loading data): {cleaner_f} {T}")
        tot_frames += T
        del Yr

    d = np.prod(dims)

    if base_name is None:
        base_name = mmap_fnames[0]
        base_name = base_name[:base_name.find('_d1_')] + f'-#-{len(mmap_fnames)}'

    fname_tot = memmap_frames_filename(base_name, dims, tot_frames, order)
    fname_tot = os.path.join(os.path.split(mmap_fnames[0])[0], fname_tot)
    fname_tot = fn_relocated(fname_tot)
    logging.info(f"Memmap file for fname_tot: {fname_tot}")

    big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32, shape=prepare_shape((d, tot_frames)), order='C')

    step = np.int(old_div(d, n_chunks))
    pars = []
    for ref in range(0, d - step + 1, step):
        pars.append([fname_tot, d, tot_frames, mmap_fnames, ref, ref + step, add_to_mov])

    if len(pars[-1]) != 7:
        raise Exception(
            'You cannot change the number of element in list without changing the statement below (pars[]..)')
    else:
        # last batch should include the leftover pixels
        pars[-1][-2] = d

    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            dview.map_async(save_portion, pars).get(4294967)
        else:
            my_map(dview, save_portion, pars)

    else:
        list(map(save_portion, pars))

    np.savez(fn_relocated(base_name + '.npz'), mmap_fnames=mmap_fnames, fname_tot=fname_tot)

    logging.info('Deleting big mov')
    del big_mov
    sys.stdout.flush()
    return fname_tot

def save_portion(pars) -> int:
    # todo: todocument
    use_mmap_save = False
    big_mov_fn, d, tot_frames, fnames, idx_start, idx_end, add_to_mov = pars
    big_mov_fn = fn_relocated(big_mov_fn)

    Ttot = 0
    Yr_tot = np.zeros((idx_end - idx_start, tot_frames), dtype=np.float32)
    logging.debug(f"Shape of Yr_tot is {Yr_tot.shape}")
    for f in fnames:
        full_f = fn_relocated(f)
        logging.debug(f"Saving portion to {full_f}")
        Yr, _, T = load_memmap(full_ff)  # TODO **to_julia** no idea where this is coming from or what it is supposed to be!?!?!?
        Yr_tot[:, Ttot:Ttot +
               T] = np.ascontiguousarray(Yr[idx_start:idx_end], dtype=np.float32) + np.float32(add_to_mov)
        Ttot = Ttot + T
        del Yr

    logging.debug(f"Index start and end are {idx_start} and {idx_end}")

    if use_mmap_save:
        big_mov = np.memmap(big_mov_fn, mode='r+', dtype=np.float32, shape=prepare_shape((d, tot_frames)), order='C')
        big_mov[idx_start:idx_end, :] = Yr_tot
        del big_mov
    else:
        with open(big_mov_fn, 'r+b') as f:
            idx_start = np.uint64(idx_start)
            tot_frames = np.uint64(tot_frames)
            f.seek(np.uint64(idx_start * np.uint64(Yr_tot.dtype.itemsize) * tot_frames))
            f.write(Yr_tot)
            computed_position = np.uint64(idx_end * np.uint64(Yr_tot.dtype.itemsize) * tot_frames)
            if f.tell() != computed_position:
                logging.critical(f"Error in mmap portion write: at position {f.tell()}")
                logging.critical(
                    f"But should be at position {idx_end} * {Yr_tot.dtype.itemsize} * {tot_frames} = {computed_position}"
                )
                f.close()
                raise Exception('Internal error in mmapping: Actual position does not match computed position')

    del Yr_tot
    logging.debug('done')
    return Ttot

def save_place_holder(pars: List) -> str:
    """ To use map reduce
    """
    # todo: todocument

    (f, base_name, resize_fact, remove_init, idx_xy, order, var_name_hdf5, xy_shifts, is_3D, add_to_movie, border_to_0, slices) = pars

    return save_memmap([f],
                       base_name=base_name,
                       resize_fact=resize_fact,
                       remove_init=remove_init,
                       idx_xy=idx_xy,
                       order=order,
                       var_name_hdf5=var_name_hdf5,
                       xy_shifts=xy_shifts,
                       is_3D=is_3D,
                       add_to_movie=add_to_movie,
                       border_to_0=border_to_0,
                       slices=slices)

def fn_relocated(fn:str) -> str:
    """ If the provided filename does not contain any path elements, this returns what would be its absolute pathname
        as located in get_tempdir(). Otherwise it just returns what it is passed.

        The intent behind this is to ease having functions that explicitly mention pathnames have them go where they want,
        but if all they think about is filenames, they go under CaImAn's notion of its temporary dir. This is under the
        principle of "sensible defaults, but users can override them".
    """
    if not 'CAIMAN_NEW_TEMPFILE' in os.environ: # XXX We will ungate this in a future version of caiman
        return fn
    if str(os.path.basename(fn)) == str(fn): # No path stuff
        return os.path.join(get_tempdir(), fn)
    else:
        return fn

def get_tempdir() -> str:
    """ Returns where CaImAn can store temporary files, such as memmap files. Controlled mainly by environment variables """
    # CAIMAN_TEMP is used to control where temporary files live.
    # If unset, uses default of a temp folder under caiman_datadir()
    # To get the old "store it where I am" behaviour, set CAIMAN_TEMP to a single dot.
    # If you prefer to store it somewhere different, provide a full path to that location.
    if 'CAIMAN_TEMP' in os.environ:
        if os.path.isdir(os.environ['CAIMAN_TEMP']):
            return os.environ['CAIMAN_TEMP']
        else:
            logging.warning(f"CAIMAN_TEMP is set to nonexistent directory {os.environment['CAIMAN_TEMP']}. Ignoring")
    temp_under_data = os.path.join(caiman_datadir(), "temp")
    if not os.path.isdir(temp_under_data):
        logging.warning(f"Default temporary dir {temp_under_data} does not exist, creating")
        os.makedirs(temp_under_data)
    return temp_under_data

def caiman_datadir() -> str:
    """
	The datadir is a user-configurable place which holds a user-modifiable copy of
	data the Caiman libraries need to function, alongside code code_base and other things.
	This is meant to be separate from the library install of Caiman, which may be installed
	into the global python library path (or into a conda path or somewhere else messy).
	"""
    if "CAIMAN_DATA" in os.environ:
        return os.environ["CAIMAN_DATA"]
    else:
        return os.path.join(os.path.expanduser("~"), "caiman_data")

def sbxread(filename: str, k: int = 0, n_frames=np.inf) -> np.ndarray:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    # Determine number of frames in whole file
    max_idx = os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1

    # Paramters
    N = max_idx + 1    # Last frame
    N = np.minimum(N, n_frames)

    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']

    # Open File
    fo = open(filename + '.sbx')

    # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones
    fo.seek(k * nSamples, 0)
    ii16 = np.iinfo(np.uint16)
    x = ii16.max - np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * N))
    x = x.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(N)), order='F')

    x = x[0, :, :, :]

    fo.close()

    return x.transpose([2, 1, 0])

def sbxreadskip(filename: str, subindices: slice) -> np.ndarray:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx

        slice: pass a slice to slice along the last dimension
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    # Determine number of frames in whole file
    max_idx = np.int(os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1)

    # Paramters
    if isinstance(subindices, slice):
        if subindices.start is None:
            start = 0
        else:
            start = subindices.start

        if subindices.stop is None:
            N = max_idx + 1    # Last frame
        else:
            N = np.minimum(subindices.stop, max_idx + 1).astype(np.int)

        if subindices.step is None:
            skip = 1
        else:
            skip = subindices.step

        iterable_elements = range(start, N, skip)

    else:

        N = len(subindices)
        iterable_elements = subindices
        skip = 0

    N_time = len(list(iterable_elements))

    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']
    assert nSamples >= 0

    # Open File
    fo = open(filename + '.sbx')

    # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones

    counter = 0

    if skip == 1:
        # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones
        assert start * nSamples > 0
        fo.seek(start * nSamples, 0)
        ii16 = np.iinfo(np.uint16)
        x = ii16.max - np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * (N - start)))
        x = x.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(N - start)),
                      order='F')

        x = x[0, :, :, :]

    else:
        for k in iterable_elements:
            assert k >= 0
            if counter % 100 == 0:
                print(f'Reading Iteration: {k}')
            fo.seek(k * nSamples, 0)
            ii16 = np.iinfo(np.uint16)
            tmp = ii16.max - \
                np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * 1))

            tmp = tmp.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer'])), order='F')
            if counter == 0:
                x = np.zeros((tmp.shape[0], tmp.shape[1], tmp.shape[2], N_time))

            x[:, :, :, counter] = tmp
            counter += 1

        x = x[0, :, :, :]
    fo.close()

    return x.transpose([2, 1, 0])

def loadmat_sbx(filename: str):
    """
    this wrapper should be called instead of directly calling spio.loadmat

    It solves the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to fix all entries
    which are still mat-objects
    """
    data_ = loadmat(filename, struct_as_record=False, squeeze_me=True)
    _check_keys(data_)
    return data_

def _check_keys(checkdict: Dict) -> None:
    """
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries.
    Modifies its parameter in-place.
    """

    for key in checkdict:
        if isinstance(checkdict[key], scipy.io.matlab.mio5_params.mat_struct):
            checkdict[key] = _todict(checkdict[key])

def _todict(matobj) -> Dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    ret = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            ret[strg] = _todict(elem)
        else:
            ret[strg] = elem
    return ret

def bin_median(self, window: int = 10) -> np.ndarray:
    """ compute median of 3D array in along axis o by binning values

    Args:
        mat: ndarray
            input 3D matrix, time along first dimension

        window: int
            number of frames in a bin

    Returns:
        img:
            median image

    """
    T, d1, d2 = np.shape(self)
    num_windows = np.int(old_div(T, window))
    num_frames = num_windows * window
    return np.nanmedian(np.nanmean(np.reshape(self[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

def bin_median_3d(self, window=10):
        """ compute median of 4D array in along axis o by binning values

        Args:
            mat: ndarray
                input 4D matrix, (T, h, w, z)

            window: int
                number of frames in a bin

        Returns:
            img:
                median image

        """
        T, d1, d2, d3 = np.shape(self)
        num_windows = np.int(old_div(T, window))
        num_frames = num_windows * window
        return np.nanmedian(np.nanmean(np.reshape(self[:num_frames], (window, num_windows, d1, d2, d3)), axis=0),
                            axis=0)

def get_file_size(file_name, var_name_hdf5='mov'):
    """ Computes the dimensions of a file or a list of files without loading
    it/them in memory. An exception is thrown if the files have FOVs with
    different sizes
        Args:
            file_name: str/filePath or various list types
                locations of file(s)

            var_name_hdf5: 'str'
                if loading from hdf5 name of the dataset to load

        Returns:
            dims: tuple
                dimensions of FOV

            T: int or tuple of int
                number of timesteps in each file
    """
    if isinstance(file_name, pathlib.Path):
        # We want to support these as input, but str has a broader set of operations that we'd like to use, so let's just convert.
	# (specifically, filePath types don't support subscripting)
        file_name = str(file_name)
    if isinstance(file_name, str):
        if os.path.exists(file_name):
            _, extension = os.path.splitext(file_name)[:2]
            extension = extension.lower()
            if extension == '.mat':
                byte_stream, file_opened = scipy.io.matlab.mio._open_file(file_name, appendmat=False)
                mjv, mnv = scipy.io.matlab.mio.get_matfile_version(byte_stream)
                if mjv == 2:
                    extension = '.h5'
            if extension in ['.tif', '.tiff', '.btf']:
                tffl = tifffile.TiffFile(file_name)
                siz = tffl.series[0].shape
                T, dims = siz[0], siz[1:]
            elif extension in ('.avi', '.mkv'):
                cap = cv2.VideoCapture(file_name)
                dims = [0, 0]
                try:
                    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    dims[1] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    dims[0] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                except():
                    print('Roll back to opencv 2')
                    T = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                    dims[1] = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                    dims[0] = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            elif extension == '.mmap':
                filename = os.path.split(file_name)[-1]
                Yr, dims, T = load_memmap(os.path.join(
                        os.path.split(file_name)[0], filename))
            elif extension in ('.h5', '.hdf5', '.nwb'):
                with h5py.File(file_name, "r") as f:
                    kk = list(f.keys())
                    if len(kk) == 1:
                        siz = f[kk[0]].shape
                    elif var_name_hdf5 in f:
                        if extension == '.nwb':
                            siz = f[var_name_hdf5]['data'].shape
                        else:
                            siz = f[var_name_hdf5].shape
                    elif var_name_hdf5 in f['acquisition']:
                        siz = f['acquisition'][var_name_hdf5]['data'].shape
                    else:
                        logging.error('The file does not contain a variable' +
                                      'named {0}'.format(var_name_hdf5))
                        raise Exception('Variable not found. Use one of the above')
                T, dims = siz[0], siz[1:]

            else:
                raise Exception('Unknown file type')
            dims = tuple(dims)
        else:
            raise Exception('File not found!')
    elif isinstance(file_name, tuple):
        dims = load(file_name[0], var_name_hdf5=var_name_hdf5).shape
        T = len(file_name)

    elif isinstance(file_name, list):
        if len(file_name) == 1:
            dims, T = get_file_size(file_name[0], var_name_hdf5=var_name_hdf5)
        else:
            dims, T = zip(*[get_file_size(fn, var_name_hdf5=var_name_hdf5)
                for fn in file_name])
            if len(set(dims)) > 1:
                raise Exception('Files have FOVs with different sizes')
            else:
                dims = dims[0]
    else:
        raise Exception('Unknown input type')
    return dims, T

def concatenate(*args, **kwargs):
    """
    Concatenate movies

    Args:
        mov: XMovie object
    """
    # todo: todocument return

    frRef = None
    for arg in args:
        for m in arg:
            if issubclass(type(m), timeseries):
                if frRef is None:
                    obj = m
                    frRef = obj.fr
                else:
                    obj.__dict__['file_name'].extend([ls for ls in m.file_name])
                    obj.__dict__['meta_data'].extend([ls for ls in m.meta_data])
                    if obj.fr != m.fr:
                        raise ValueError('Frame rates of input vectors \
                            do not match. You cannot concatenate movies with \
                            different frame rates.')
    try:
        return obj.__class__(np.concatenate(*args, **kwargs), **obj.__dict__)
    except:
        logging.debug('no meta information passed')
        return obj.__class__(np.concatenate(*args, **kwargs))

class timeseries(np.ndarray):
    """
    Class representing a time series.
    """

    def __new__(cls, input_arr, fr=30, start_time=0, file_name=None, meta_data=None):
        """
            Class representing a time series.

            Example of usage

            Args:
                input_arr: np.ndarray

                fr: frame rate

                start_time: time beginning movie

                meta_data: dictionary including any custom meta data

            Raises:
                Exception 'You need to specify the frame rate'
            """
        if fr is None:
            raise Exception('You need to specify the frame rate')

        obj = np.asarray(input_arr).view(cls)
        # add the new attribute to the created instance

        obj.start_time = np.double(start_time)
        obj.fr = np.double(fr)
        if isinstance(file_name, list):
            obj.file_name = file_name
        else:
            obj.file_name = [file_name]

        if isinstance(meta_data, list):
            obj.meta_data = meta_data
        else:
            obj.meta_data = [meta_data]

        return obj

class movie(timeseries):
    """
    Class representing a movie. This class subclasses timeseries,
    that in turn subclasses ndarray

    movie(input_arr, fr=None,start_time=0,file_name=None, meta_data=None)

    Example of usage:
        input_arr = 3d ndarray
        fr=33; # 33 Hz
        start_time=0
        m=movie(input_arr, start_time=0,fr=33);

    See https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for
    notes on objects that are descended from ndarray
    """

    def __new__(cls, input_arr, **kwargs):
        """
        Args:
            input_arr:  np.ndarray, 3D, (time,height,width)

            fr: frame rate

            start_time: time beginning movie, if None it is assumed 0

            meta_data: dictionary including any custom meta data

            file_name: name associated with the file (e.g. path to the original file)
        """
        if isinstance(input_arr, movie):
            return input_arr

        if (isinstance(input_arr, np.ndarray)) or \
           (isinstance(input_arr, h5py._hl.dataset.Dataset)) or \
           ('mmap' in str(type(input_arr))) or \
           ('tifffile' in str(type(input_arr))):
            return super().__new__(cls, input_arr, **kwargs)
        else:
            raise Exception('Input must be an ndarray, use load instead!')

def memmap_frames_filename(basename: str, dims: Tuple, frames: int, order: str = 'F') -> str:
    # Some functions calling this have the first part of *their* dims Tuple be the number of frames.
    # They *must* pass a slice to this so dims is only X, Y, and optionally Z. Frames is passed separately.
    dimfield_0 = dims[0]
    dimfield_1 = dims[1]
    if len(dims) == 3:
        dimfield_2 = dims[2]
    else:
        dimfield_2 = 1
    return f"{basename}_d1_{dimfield_0}_d2_{dimfield_1}_d3_{dimfield_2}_order_{order}_frames_{frames}_.mmap"


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data : 2D ndarray
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(old_div(data.shape[1], 2))).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(old_div(data.shape[0], 2)))
    )

    if data.ndim > 2:
        pln_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[2] * upsample_factor)) *
        (np.arange(upsampled_region_size[2])[:, None] - axis_offsets[2]).dot(
                ifftshift(np.arange(data.shape[2]))[None, :] -
                np.floor(old_div(data.shape[2], 2))))

    # output = np.tensordot(np.tensordot(row_kernel,data,axes=[1,0]),col_kernel,axes=[1,0])
    output = np.tensordot(row_kernel, data, axes = [1,0])
    output = np.tensordot(output, col_kernel, axes = [1,0])

    if data.ndim > 2:
        output = np.tensordot(output, pln_kernel, axes = [1,1])
    #output = row_kernel.dot(data).dot(col_kernel)
    return output

def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be zero if images are non-negative).

    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)

def create_weight_matrix_for_blending(img, overlaps, strides):
    """ create a matrix that is used to normalize the intersection of the stiched patches

    Args:
        img: original image, ndarray

        shapes, overlaps, strides:  tuples
            shapes, overlaps and strides of the patches

    Returns:
        weight_mat: normalizing weight matrix
    """
    shapes = np.add(strides, overlaps)

    max_grid_1, max_grid_2 = np.max(
        np.array([it[:2] for it in sliding_window(img, overlaps, strides)]), 0)

    for grid_1, grid_2, _, _, _ in sliding_window(img, overlaps, strides):

        weight_mat = np.ones(shapes)

        if grid_1 > 0:
            weight_mat[:overlaps[0], :] = np.linspace(
                0, 1, overlaps[0])[:, None]
        if grid_1 < max_grid_1:
            weight_mat[-overlaps[0]:,
                       :] = np.linspace(1, 0, overlaps[0])[:, None]
        if grid_2 > 0:
            weight_mat[:, :overlaps[1]] = weight_mat[:, :overlaps[1]
                                                     ] * np.linspace(0, 1, overlaps[1])[None, :]
        if grid_2 < max_grid_2:
            weight_mat[:, -overlaps[1]:] = weight_mat[:, -
                                                      overlaps[1]:] * np.linspace(1, 0, overlaps[1])[None, :]

        yield weight_mat
