import dask.array as da
from dask.diagnostics import ProgressBar
import dask_image.imread as imread
import json
import os
import tifffile as tf
import sys
from skimage.transform import resize, rescale
from skimage.util import img_as_uint
import getopt
from pathlib import Path

def zip_to_h5(self, zip_file, out, channels=2,
                  channel_labels={1: "neu", 2: "ast"}, resize_factor=1,
                  compressor="gzip", shuffle=True,
                  dtype='i2', chunks=(100, 100, 100), verbose=0
                  ):

        import h5py as h5
        import numpy as np
        import pandas as pd
        from io import StringIO
        from PIL import Image
        from skimage.transform import rescale
        from skimage.util import img_as_uint

        assert os.path.isfile(zip_file), f"Couldn't find zip file: {zip_file}"
        assert not os.path.isfile(out), f"Output file already exists: {out}"

        # get files
        with zipfile.ZipFile(zip_file, mode="r") as zdir:

            # get files in order
            images = [item.filename for item in zdir.filelist]

            num = [int(x.split("_")[-1].split(".")[0]) for x in images]

            df = pd.DataFrame([images, num]).transpose()
            df.columns = ["path", "num"]
            df.sort_values(by="num", inplace=True)
            del df["num"]
            df.reset_index(inplace=True, drop=True)

            # get dimensions
            i0 = df.path.tolist()[0]
            i0 = tf.imread(zdir.open(i0))
            if resize_factor != 1:
                i0 = rescale(i0, resize_factor, anti_aliasing=True)

            X, Y = i0.shape

            # create h5 file
            with h5.File(out, "w") as f:

                for c in range(channels):

                    # select channel indices
                    indices = range(c, len(df), channels)
                    Z = len(indices)

                    # get channel label
                    if channel_labels is None:
                        channel_lbl = str(c)
                    else:
                        channel_lbl = channel_labels[c + 1]

                    # create group in h5 file
                    arr_disk = f.create_dataset("data/" + channel_lbl, dtype=dtype, shape=(Z, X, Y),
                                                chunks=chunks, compression=compressor, shuffle=shuffle)
                    print(f"New shape: Z:{Z} x X:{X} x Y:{Y}")

                    if verbose > 0: print("Saving channel: {} ({})".format(channel_lbl, c))

                    if chunks is None:

                        counter = 0
                        for i in tqdm(indices, position=0, leave=True):
                            img_path = df.iloc[i][0]
                            img = tf.imread(zdir.open(img_path))

                            if resize_factor != 1:
                                img = rescale(img, resize_factor, anti_aliasing=True)

                            arr_disk[counter, :, :] = img
                            counter += 1

                    else:
                        chunk_x, chunk_y, chunk_z = chunks
                        for big_counter in tqdm(range(0, len(indices), chunk_z), desc=f"chunks: {channel_lbl}"):

                            # z axis dimension
                            z0 = big_counter
                            z1 = min(big_counter + chunk_z, len(indices))

                            # load chunk of images
                            chunk = np.zeros((X, Y, z1 - z0)) # TODO wrong, no? Should be np.zeros((z1-z0, X, Y))?
                            chunk_counter = 0
                            for small_counter in range(big_counter, min(big_counter + chunk_z, len(indices))):
                                img_path = df.iloc[indices[small_counter]][0]
                                img = tf.imread(zdir.open(img_path))

                                if resize_factor != 1:
                                    img = rescale(img, resize_factor, anti_aliasing=True)
                                    img = img_as_uint(img)

                                chunk[:, :, chunk_counter] = img
                                chunk_counter += 1

                            # adjust axis order for subsequent loading down the pipeline
                            chunk = np.moveaxis(chunk, -1, 0)

                            # save chunk of images
                            for x1 in range(0, X, chunk_x):
                                for y1 in range(0, Y, chunk_y):
                                    x2 = min(x1 + chunk_x, X)
                                    y2 = min(y1 + chunk_y, Y)

                                    arr_disk[z0:z1, x1:x2, y1:y2] = chunk[:, x1:x2, y1:y2]

                # create meta information
                f.create_dataset("meta/resize_factor", data=resize_factor)

        return True


if __name__ == "__main__":

    # GET INPUT
    input_folder = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:", ["ifolder="])
    except getopt.GetoptError:
        print("calpack.py -i <inputfolder>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--ifolder"):
            input_folder = arg

    if not os.path.isdir(input_folder):

        if input_folder.endswith(".h5"):
            print(".h5 already provided")
            sys.exit(0)

        else:
            print("File type not recognized: ", input_folder)
            sys.exit(1)

    # create paths
    file = Path(input_folder).name
    base_dir = Path(input_folder).parent
    out = Path(input_folder).with_suffix(".h5")
    input_folder = Path(input_folder)

    print("input: ", input_folder)
    print("file: ", file)
    print("base_dir: ", base_dir)
    print("out: ", out)

    # # create meta data
    # info = {
    #     "preprocessing": {
    #         "channel": [0, 1],
    #         "channel_label": ["neu", "ast"],
    #         "empty_channels": 0,
    #         "resize": (512, 512), #0.2,
    #         "subtract_empty": True,
    #         "subtract_value": None,
    #         "chunks": (100, 100, 100),
    #         "compression": "lzf",
    #     }
    # }
    #
    # with open(dir+file.replace("zip", "json"), "w") as o:
    #     json.dump(info, o, indent=2, sort_keys=True)

    # load meta
    with open(out.with_suffix(".json").as_posix(), "r") as i:
        meta = json.load(i)

    print(meta)
    preprop = meta["preprocessing"]

    # progress bar
    pbar = ProgressBar()
    pbar.register()

    # rename for appropriate sorting
    fnames = os.listdir(input_folder.as_posix())
    num_digits = len(str(len(fnames)))
    print("#digits: ", num_digits)

    for fname in fnames:

        digit = fname.split("_")[-1].split(".")[0]
        s1 = "{:0"+str(num_digits)+"d}"
        digit = s1.format(int(digit))

        new_fname = "_".join(fname.split("_")[:-1]) + "_" + digit + ".tiff"

        if fname != new_fname:
            os.rename(input_folder.joinpath(fname).as_posix(), input_folder.joinpath(new_fname).as_posix())

    # load files
    data = imread.imread(input_folder.joinpath("*.tiff").as_posix())
    print(data)

    num_channels = len(preprop["channel"])
    assert num_channels < 3, "currently not more than two channels allowed"

    # TODO what happens if no subtract

    # get background information
    if preprop["subtract_empty"]:

        background_channel = preprop["empty_channels"]
        background = data[background_channel::num_channels]

        # TODO implement more than 2 channels
        data_channel = [c for c in preprop["channel"] if c != background_channel][0]

        print("Calculating trace")
        trace = da.mean(background, axis=(1, 2))
        da.to_hdf5(out.as_posix(), "/background/trace", trace)

        print("Calculating noise mean")
        xy_noise = da.mean(background, axis=0)
        da.to_hdf5(out.as_posix(), "/background/xy_noise", xy_noise)

        print("Calculating noise std")
        xy_noise_std = da.std(background, axis=0)
        da.to_hdf5(out.as_posix(), "/background/xy_noise_std", xy_noise_std)

        print("Subtracting ...")
        data[data_channel::num_channels] = data[data_channel::num_channels] - xy_noise

    elif preprop["subtract_value"] is not None:

        for ch in [c for c in preprop["channel"] if c != preprop["empty_channels"]]:
            data[ch::num_channels] = data[ch::num_channels] - preprop["subtract_value"]

    elif ("subtract_folder" in preprop.keys()) and (preprop["subtract_folder"] is not None):

        background = imread.imread(input_folder.joinpath(preprop["subtract_folder"]))
        xy_noise = da.mean(background, axis=0)
        da.to_hdf5(out.as_posix(), "/background/xy_noise", xy_noise)

        for ch in [c for c in preprop["channel"] if c != preprop["empty_channels"]]:
            data[ch::num_channels] = data[ch::num_channels] - xy_noise


    # prepare resizing
    apply_resize=False
    if not (preprop["resize"] is None or preprop["resize"] == 1):

        print("Resizing ...")
        R = preprop["resize"]

        # define output size
        if type(R) == float:
            chunks = (1, int(data.shape[1]*R), int(data.shape[2]*R))
        elif type(R) in [list, tuple]:
            chunks = (1, R[0], R[1])

        def downscale(arr):

            if R is None or R == 1:
                return arr

            assert type(R) in [float, list, tuple], "Please provide resize value as None, float, list or tuple"

            if type(R) == float:
                arr = rescale(arr, R, anti_aliasing=True)
            elif type(R) in [list, tuple]:
                arr = resize(arr, [arr.shape[0]] + R, anti_aliasing=True)

            return img_as_uint(arr)

        apply_resize=True

    # saving
    for ch in [c for c in preprop["channel"] if c != preprop["empty_channels"]]:

        img = data[ch::num_channels]
        if apply_resize:
            img = img.map_blocks(downscale, chunks=chunks)

        # img = img.astype("i2", casting="safe")

        print("Saving ...")
        loc = "/data/"+preprop["channel_label"][ch]
        da.to_hdf5(out.as_posix(), loc, img, chunks=tuple(preprop["chunks"]),
                   compression=preprop["compression"], shuffle=False)

    pbar.unregister()
    print("Done")
