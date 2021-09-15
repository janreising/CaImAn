import unittest
from cp_motioncorrection import MotionCorrect
import cp_params

class MotionCorrectionMethods(unittest.TestCase):

    def test_test(self):
        self.assertEqual('foo'.upper(), 'FOO')

    # Testing movie class
    def test_movie_bin_median(self):

        import numpy as np
        import cp_motioncorrection

        Z, X, Y = 40, 100, 100
        img = np.random.rand(Z, X, Y)  # self in the function
        window = 10

        # creates image of size X*Y
        self.assertEqual(
            cp_motioncorrection.movie.bin_median(img, window),
            False
        )

    def test_movie_extract_shifts(self):
        import numpy as np
        import cp_motioncorrection

        Z, X, Y = 40, 100, 100
        img = np.random.rand(40, 100, 100)  # self in the function
        max_shift_h = 50
        max_shift_w = 50
        method = "opencv"
        template = None  # TODO figure out how to find a good template

        # return shifts: list of x,y shifts as float and xcorrs: list of correlations eg. [0.75]
        self.assertEqual(
            cp_motioncorrection.movie.extract_shifts(img, max_shift_w, max_shift_h, template=template, method=method),
            False
        )

    def test_movie_apply_shifts(self):

        import numpy as np
        import cp_motioncorrection

        Z, X, Y = 40, 100, 100
        img = np.random.rand(Z, X, Y)  # self in the function
        shifts = np.random.rand(Z) # shifts from previous function
        interpolation = "cubic"
        method = "opencv"
        remove_blanks = False

        # returns shifted image
        self.assertEqual(
            cp_motioncorrection.movie.bin_median(img, shifts=shifts, interpolation=interpolation, method=method, remove_blanks=remove_blanks),
            False
        )

    def test_movie_motion_correct(self):
        import numpy as np
        import cp_motioncorrection

        Z, X, Y = 40, 100, 100
        img = np.random.rand(Z, X, Y)  # self in the function
        max_shift_w = 50
        max_shift_h = 50
        num_frames_template = None
        template = None
        method = "opencv"
        remove_blanks = False
        interpolation = "cubic"

        # returns img, shifts, xcorrs and template
        self.assertEqual(
            cp_motioncorrection.movie.bin_median(img, max_shift_w, max_shift_h, num_frames_template, template=template,
                                            method=method, remove_blanks=remove_blanks, interpolation=interpolation),
            False
        )

        return False

    # Testing auxiliary functions
    def test_high_pass_filter_space(self):

        import numpy as np
        import cp_motioncorrection

        img_org = np.random.rand(100, 100)  # Random image
        gSig_filt = (20, 20)
        freq = None
        order = None

        # We are using a gaussian filter so far
        # they have also implemented a butterworth filter
        # when freq or order are not None

        self.assertEqual(
            cp_motioncorrection.high_pass_filter_space(
            img_org=img_org, gSig_filt=gSig_filt, freq=freq, order=order
            ),
            None  # TODO implement test for high_pass_filter_space
        )

if __name__ == "__main__":

    # unittest.main()

    # mc = CMotionCorrect(path=input_, verbose=3, delete_temp_files=delete_temp_files, on_server=on_server,
    #                     dview=None)
    # mc.run_motion_correction(ram_size_multiplier=ram_size_multiplier, frames_per_file=frames_per_file)

    print("Starting testing ...")
    files = "/media/carmichael/LaCie SSD/delete/slice7/1-40X-loc1.zip.h5"

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

    mc = MotionCorrect(files, dview=None, var_name_hdf5=f"data/ast", **opts_dict)
    mc.motion_correct(save_movie=True)
