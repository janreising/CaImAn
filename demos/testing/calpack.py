import numpy as np
import h5py
import os
import tifffile as tf
import zipfile
import pandas as pd
from PIL import Image
from tqdm import tqdm
import sys, getopt
import shutil
import re

class Converter():

    # legacy
    import tifffile as tif

    def __init__(self):

        self.file = None

    def convert_to_h5(self, out, folder, channels=2,
                      channel_labels={1: "neu", 2: "ast"},
                      compressor="gzip", shuffle=True,
                      dtype='i2', chunks=(100, 100, 100), verbose=0):

        """
        channel_labels: {int:string}
        """

        import h5py
        import numpy as np

        # clean path
        if folder[-1] == os.sep:
            folder = folder[:-1]

        print(folder)

        # get path to single tiffs
        df = self.get_files(folder)

        # get dimensions
        X, Y = self.tif.imread(df.loc[0][0]).shape

        # create h5 file
        f = h5py.File(out, "w")

        for c in range(channels):

            # select channel indices
            indices = range(c, len(df), channels)
            Z = len(indices)

            # get channel label
            if channel_labels is None:
                channel_lbl = str(c)
            else:
                channel_lbl = channel_labels[c+1]

            # create group in h5 file
            arr_disk = f.create_dataset("data/"+channel_lbl, dtype=dtype, shape=(Z, X, Y),
                                        chunks=chunks, compression=compressor, shuffle=shuffle)

            if verbose > 0: print("Saving channel: {} ({})".format(channel_lbl, c))

            if chunks is None:

                counter = 0
                for i in tqdm(indices, position=0, leave=True):
                    img_path = df.iloc[i][0]
                    arr_disk[counter, :, :] = self.tif.imread(img_path)
                    counter += 1

            else:
                chunk_x, chunk_y, chunk_z = chunks
                for big_counter in tqdm(range(0, len(indices), chunk_z), desc="chunks"):

                    # z axis dimension
                    z0 = big_counter
                    z1 = min(big_counter+chunk_z, len(indices))

                    # load chunk of images
                    chunk = np.zeros((X, Y, z1-z0))
                    chunk_counter = 0
                    for small_counter in range(big_counter, min(big_counter+chunk_z, len(indices))):
                        img_path = df.iloc[indices[small_counter]][0]
                        chunk[:, :, chunk_counter] = self.tif.imread(img_path)
                        chunk_counter += 1

                    # adjust axis order for subsequent loading down the pipeline
                    chunk = np.moveaxis(chunk, -1, 0)

                    # save chunk of images
                    for x1 in range(0, X, chunk_x):
                        for y1 in range(0, Y, chunk_y):

                            x2 = min(x1 + chunk_x, X)
                            y2 = min(y1 + chunk_y, Y)

                            arr_disk[z0:z1, x1:x2, y1:y2] = chunk[:, x1:x2, y1:y2]

        return self.check_h5(folder, out)

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

        # get dimensions

        # create h5 file

    # def load_chunks_sequential(self, data):
    #
    #     chunk_x, chunk_y, chunk_z = data.chunks
    #     X, Y, Z = data.shape
    #
    #     temp = np.zeros((chunk_x, chunk_y, chunk_z))
    #
    #     for x0 in tqdm(range(0, X, chunk_x), position=0, leave=True):
    #         for y0 in range(0, Y, chunk_y):
    #             for z0 in range(0, Z, chunk_z):
    #                 x1 = min(x0+chunk_x, X)
    #                 y1 = min(y0+chunk_y, Y)
    #                 z1 = min(z0+chunk_z, Z)
    #
    #                 temp = data[x0:x1, y0:y1, z0:z1]

    # def load_zarr(self, input):
    #
    #     import zarr
    #
    #     return zarr.open("/home/janrei1@ad.cmm.se/Desktop/delete/testing_zarr", mode="r")

    @staticmethod
    def get_files(folder):

        import pandas as pd
        import os

        files = [folder+"/"+f for f in os.listdir(folder) if not f.startswith(".")]
        num = [int(x.split("_")[-1].split(".")[0]) for x in files]

        df = pd.DataFrame([files, num]).transpose()
        df.columns = ["path", "num"]
        df.sort_values(by="num", inplace=True)
        del df["num"]
        df.reset_index(inplace=True, drop=True)

        return df

    def zip_folder(self, dirName, progress=True):

        if dirName[-1] != os.sep:
            dirName += os.sep

        parent = dirName.split(os.sep)[-2]
        out = dirName[:-1]+".zip"

        with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as zipObj:

            iterator = [img for img in os.listdir(dirName) if img.endswith(".tiff")]
            if progress: iterator = tqdm(iterator)
            for img in iterator:
                zipObj.write(dirName+img, parent+os.sep+img)

        return self.check_zip(dirName, out)

    # quality control
    def check_zip(self, raw, zp):

        # load zip
        zdir = zipfile.ZipFile(zp, mode='r')
        zip_names = [item.filename.replace(zp.split(os.sep)[-1].replace(".zip", os.sep), "") for item in zdir.filelist]

        all_accounted_for = True
        for f in [img for img in os.listdir(raw) if img.endswith(".tiff")]:
            if f not in zip_names:
                all_accounted_for = False
                print("{} missing".format(f))
                break

        return all_accounted_for

    def check_h5(self, path, h5path):

        neu = Video(h5path, channel="neu")
        ast = Video(h5path, channel="ast")

        if os.path.isdir(path):

            files = self.get_files(path)

            # Check image length
            if neu.Z + ast.Z == len(files):
                sorting = "xyz"
            elif ast.X + neu.X == len(files):
                sorting = "zxy"
            else:
                sorting = None

            assert sorting is not None, "Frames missing! Neu ({}) + Ast ({}) != Files ({})".format(
                "{}+{}!={} & {}+{}!={}".format(neu.Z, ast.Z, len(files), neu.X, ast.X, len(files)))

            # Check first frame
            if sorting == "xyz":
                assert np.array_equal(tf.imread(files.iloc[0].path), neu.data[:, :, 0]), \
                    "First neuron frame not the same"
                assert np.array_equal(tf.imread(files.iloc[1].path), ast.data[:, :, 0]), \
                    "First astrocyte frame not the same"
            else:
                assert np.array_equal(tf.imread(files.iloc[0].path), neu.data[0, :, :]), \
                    "First neuron frame not the same"
                assert np.array_equal(tf.imread(files.iloc[1].path), ast.data[0, :, :]), \
                    "First astrocyte frame not the same"

            # Check last frame
            if sorting == "xyz":
                assert np.array_equal(tf.imread(files.iloc[-2 +(len(files)%2)].path), neu.data[:, :, -1]), \
                    "Last neuron frame not the same"
                assert np.array_equal(tf.imread(files.iloc[-1 -(len(files)%2)].path), ast.data[:, :, -1]), \
                    "Last astrocyte frame not the same"
            else:
                assert np.array_equal(tf.imread(files.iloc[-2 + (len(files) % 2)].path), neu.data[-1, :, :]), \
                    "Last neuron frame not the same"
                assert np.array_equal(tf.imread(files.iloc[-1 - (len(files) % 2)].path), ast.data[-1, :, :]), \
                    "Last astrocyte frame not the same"



        else:

            # load zip
            zdir = zipfile.ZipFile(path, mode='r')

            # sort files
            zip_names = [file.filename for file in zdir.filelist[1:]]
            num = [int(x.split("_")[-1].split(".")[0]) for x in zip_names]

            files = pd.DataFrame([zip_names, num]).transpose()
            files.columns = ["path", "num"]
            files.sort_values(by="num", inplace=True)
            del files["num"]
            files.reset_index(inplace=True, drop=True)

            # check image length
            assert neu.Z + ast.Z == len(zip_names), "Frames missing! Neu ({}) + Ast ({}) != Files ({})".format(neu.Z, ast.Z, len(zip_names))

            # Check first frame
            n0 = np.array(Image.open(zdir.open(files.iloc[0].path)))
            a0 = np.array(Image.open(zdir.open(files.iloc[1].path)))
            assert np.array_equal(n0, neu.data[:, :, 0]), "First neuron frame not the same"
            assert np.array_equal(a0, ast.data[:, :, 0]), "First astrocyte frame not the same"

            # Check last frame
            n1 = np.array(Image.open(zdir.open(files.iloc[-2 +(len(files)%2)].path)))
            a1 = np.array(Image.open(zdir.open(files.iloc[-1 -(len(files)%2)].path)))
            assert np.array_equal(n1, neu.data[:, :, -1]), "Last neuron frame not the same"
            assert np.array_equal(a1, ast.data[:, :, -1]), "Last astrocyte frame not the same"

        return True

    def convert_folder(self, input_folder, del_folder=False):

        """
        # CREATE H5
        if input_folder[-1] == os.sep:
            output_file = input_folder[:-1] + ".h5"
        else:
            output_file = input_folder + ".h5"

        if os.path.isfile(output_file):

            if self.check_h5(input_folder, output_file):
                print("H5 file already exists and is correct.")
                h5_ret = True
            else:
                print("H5 file exists but is corrupted. Deleting ...")
                os.remove(output_file)
                h5_ret = self.convert_to_h5(output_file, input_folder)

        else:
            h5_ret = self.convert_to_h5(output_file, input_folder)

        """

        h5_ret=True

        # CREATE ZIP
        if input_folder[-1] == os.sep:
            output_file = input_folder[:-1] + ".zip"
        else:
            output_file = input_folder + ".zip"

        if os.path.isfile(output_file):

            if self.check_zip(input_folder, output_file):
                print("ZIP file already exists and is correct.")
                zip_ret = True
            else:
                print("ZIP file exists but is corrupted. Deleting ...")
                # print("IN: {}\nOUT: {}".format(input_folder, output_file))
                # print("EQUAL?: ", self.check_zip(input_folder, output_file))
                os.remove(output_file)
                zip_ret = self.zip_folder(input_folder)

        else:
            zip_ret = self.zip_folder(input_folder)

        # CHECK RESULT
        assert self.check_zip(input_folder, output_file), f"Something went wrong while zipping folder {input_folder}"

        # DELETE FOLDER
        if zip_ret and h5_ret and del_folder:
            #remove
            shutil.rmtree(input_folder, ignore_errors=True)
            # shutil.rmdir(input_folder)

        return output_file

    def rec_search(self, root, container=[], pattern="single_[0-9][0-9]*.tif+$"):

        if os.path.isdir(root):

            for item in os.scandir(root):

                child = item.path
                res, container_ = self.rec_search(child, container)

                if res == True:
                    container = container_.append(root)
                    break
        else:
            # check if conforms to file type
            if re.search(pattern, root) is not None:
                return True, container
            else:
                return False, container

        return False, container


if __name__ == "__main__":

    print("Calpack started ...\n")

    del_folder = True

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

    if os.path.isdir(input_folder):

        # Convert
        loader = Loader()
        pfolders = []
        print("Recognized folder")
        _, _, = loader.rec_search(input_folder, pfolders)

        for parent in pfolders:
            print("\n\nConverting: {}\n".format(parent))
            zip_file = loader.convert_folder(parent, del_folder=del_folder)
            loader.zip_to_h5(zip_file, zip_file+".h5", resize_factor=0.5)

    elif os.path.isfile(input_folder) and input_folder.endswith(".zip"):

        print("Recognized zip file")
        loader = Loader()
        loader.zip_to_h5(input_folder, input_folder+".h5", resize_factor=0.5)

