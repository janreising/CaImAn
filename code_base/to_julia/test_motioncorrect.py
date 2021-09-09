import unittest
from cp_motioncorrection import MotionCorrect
import cp_params

class MotionCorrectionMethods(unittest.TestCase):

    def test_test(self):
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == "__main__":

    # unittest.main()

    # mc = CMotionCorrect(path=input_, verbose=3, delete_temp_files=delete_temp_files, on_server=on_server,
    #                     dview=None)
    # mc.run_motion_correction(ram_size_multiplier=ram_size_multiplier, frames_per_file=frames_per_file)

    files = "/media/STORAGE/delete/1-40X-loc1.zip.h5"

    opts_dict = {
        # 'fnames': [files],
        # 'fr': 10,  # sample rate of the movie
        'pw_rigid': False,  # flag for pw-rigid motion correction
        'max_shifts': (50, 50),  # 20, 20                             # maximum allowed rigid shift
        'gSig_filt': (20, 20),
        # 10,10   # size of filter, in general gSig (see below),  # change if alg doesnt work
        'strides': (48, 48),  # start a new patch for pw-rigid motion correction every x pixels
        'overlaps': (24, 24),  # overlap between pathes (size of patch strides+overlaps)
        'max_deviation_rigid': 3,  # maximum deviation allowed for patch with respect to rigid shifts
        'border_nan': 'copy',
        'use_cuda': False,
    }

    mc = MotionCorrect(files, dview=None, var_name_hdf5=f"test/ast", **opts_dict)
    mc.motion_correct(save_movie=True)
