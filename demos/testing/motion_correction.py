import os, sys, psutil, shutil
import numpy as np
import h5py as h5
import tifffile as tf
from tqdm import tqdm
import cv2

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy.volparams import volparams


def deconstruct_path(path):

    base = os.sep.join(path.split(os.sep)[:-1])
    name = path.split(os.sep)[-1].split(".")[0]
    ext = path.split(".")[-1]

    if base[-1] != os.sep:
        base = base + os.sep

    return base, name, ext


def convert_xyz_to_zxy(path, loc0):

    base, name, ext = deconstruct_path(path)
    old_path = f"{base}{name}-xyz.h5"
    new_path = f"{base}{name}-zxy.h5"

    data = h5.File(path, "r")[loc0]

    d1, d2, d3 = data.shape
    if d2 == 1200 and d3 == 1200:
        print("Expected data shape found. Aborting")
        return True

    file = h5.File(path, "r")
    for loc in file["data/"].keys():

        data = file["data/"+loc]
        X, Y, Z = data.shape
        cx, cy, cz = data.chunks

        new_ = h5.File(new_path, "a")
        arr = new_.create_dataset(loc, dtype="i2", shape=(Z, X, Y),
                                  compression="gzip", chunks=(cx, cy, cz), shuffle=True)

        if "dummy" not in new_:
            _ = new_.create_dataset("dummy", dtype="i2", shape=(1, 1, 1))

        for start in tqdm(range(0, Z, cz)):

            stop = min(start+cz, Z)

            transformed = np.array(data[:, :, start:stop])
            transformed = np.swapaxes(transformed, 0, 2)
            transformed = np.swapaxes(transformed, 1, 2)

            arr[start:stop, :, :] = transformed

    # # move files
    shutil.move(path, old_path)
    shutil.move(new_path, path)


def split_h5_file(file_path, loc, split_file_size=2000):

    # Load file
    base, name, ext = deconstruct_path(file_path)
    data = h5.File(name=file_path)[loc]
    Z, X, Y = data.shape

    c = 0
    names = []
    splits = int(os.stat(file_path).st_size / (split_file_size * 1024 * 1024))
    split_size = int(Z/splits)
    for start in tqdm(range(0, Z, split_size), position=0, leave=True):

        stop = min(start+split_size, Z)

        name_out = f'{base}{c}-{name}_{c}.h5'
        if not os.path.isfile(name):
            chunk = data[start:stop, :, :]

            temp = h5.File(name_out, "w")
            chunk_drive = temp.create_dataset("/data/ast", shape=chunk.shape, dtype=data.dtype)
            chunk_drive[:, :, :] = chunk
            temp.create_dataset("/proc/dummy", shape=(1, 1, 1), dtype=data.dtype)

        c += 1
        names.append(name_out)

    return names


def get_mmaps(fnames):

    base, name, ext = deconstruct_path(fnames[0])
    dir_files = os.listdir(base)

    dims = []
    tasks = []
    if len(fnames) > 1:

        dir_files = os.listdir(base)

        for p in fnames:
            start = p.split(os.sep)[-1].split("_")[0]

            for df in dir_files:
                name = df.split(os.sep)[-1]

                if name.startswith(start) and name.endswith(".mmap"):
                    tasks.append(base + df)

                    dsplit = df.split("_")
                    dims.append([int(dsplit[-2]), int(dsplit[5]), int(dsplit[7])])

                    continue

    else:
        start = fnames[0].split(os.sep)[-1].split(".")[0][:-1]

        for df in dir_files:
            name = df.split(os.sep)[-1]

            if name.startswith(start) and name.endswith(".mmap"):
                tasks.append(base + df)

                dsplit = df.split("_")
                dims.append([int(dsplit[-2]), int(dsplit[4]), int(dsplit[6])])

                continue

    return tasks, dims


def save_memmap_to_h5(fnames, loc):

    # get mmap file names
    base, name, ext = deconstruct_path(fnames[0])
    tasks, dims = get_mmaps(fnames)

    # create output file
    Z = sum([dim[0] for dim in dims])
    _, X, Y = dims[0]
    shape = (Z, X, Y)

    output = h5.File(f"{base}{name}_out.h5", "r+")
    loc_out = "mc/"+loc.split("/")[-1]
    data = output.create_dataset(loc_out, shape=shape, dtype="i2",
                                 compression="gzip", chunks=(100, 100, 100), shuffle=True)

    # fill array
    c = 0
    for task in tqdm(tasks):
        mm = np.memmap(task, shape=shape, dtype=np.float32)
        data[c*Z:(c+1)*Z, :, :] = mm

        c += 1

    return f"{base}{name}_out.h5", loc_out


def save_split_tiff(file, loc, skip=None, downsize=0.5, subindices=None):

    from skimage.transform import resize

    arr = h5.File(file, "r")[loc]
    base, name, ext = deconstruct_path(file)
    lcout = loc.replace(os.sep, "-")
    out = f"{base}{name}_{lcout}.tiff"

    Z, X, Y = arr.shape

    if skip is None:
        skip = int(Z/100)

    z0 = 0
    if subindices is not None:
        z0, z1 = subindices
        Z = min(Z, z1)

    z, x, y = int((Z-z0)/skip), int(X*downsize), int(Y*downsize)

    tarr = np.zeros((z, x, y))
    c=0
    for i in tqdm(range(z0, Z, skip)):
        img = arr[i, :, :]

        if downsize != 1:
            img = resize(img, (x, y))

        tarr[c, :, :] = img
        c += 1

    tf.imsave(out, tarr)


def run_motion_correction(path, loc):

    ##################
    # File preparation

    # check array shape; convert if necessary
    convert_xyz_to_zxy(path, loc)

    # create task list dependent on available RAM
    file_size = os.stat(path).st_size
    ram_size = psutil.virtual_memory().total
    print("{:.2f}GB:{:.2f}GB ({:.2f}%)".format(file_size/1000/1024/1024, ram_size/1000/1024/1024, file_size/ram_size*100))

    if ram_size < file_size * 10:
        print("RAM not sufficient. Splitting file ...")
        f_names = split_h5_file(path, loc=loc)
    else:
        f_names = [path]

    # check if exists
    # TODO check if exists. Maybe complicated because the file names changes; code might have crashed previously
    # TODO actually impossible because different channels will use the same name for mmap files

    ############
    # Parameters

    opts_dict = {
        'fnames': f_names,
        'fr': 10,  # sample rate of the movie
        'pw_rigid': False,      # flag for pw-rigid motion correction
        'max_shifts': (50, 50),  # 20, 20                             # maximum allowed rigid shift
        'gSig_filt': (20, 20),  # 10,10   # size of filter, in general gSig (see below),  # change if alg doesnt work
        'strides': (48, 48),  # start a new patch for pw-rigid motion correction every x pixels
        'overlaps': (24, 24),  # overlap between pathes (size of patch strides+overlaps)
        'max_deviation_rigid': 3 ,  # maximum deviation allowed for patch with respect to rigid shifts
        'border_nan': 'copy',
    }

    opts = volparams(params_dict=opts_dict)

    ###################
    # Motion Correction

    # start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=7, single_thread=False)

    # Run correction
    print(f_names)
    mc = MotionCorrect(f_names, dview=dview, var_name_hdf5=loc, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)

    # stop cluster
    cm.stop_server(dview=dview)

    ####################
    # Convert mmap to h5
    path_out, loc_out = save_memmap_to_h5(f_names, loc=loc)

    #############
    # Save sample
    save_split_tiff(path_out, loc_out)

    ###################
    # delete temp files
    for delf in get_mmaps(f_names):
        os.remove(delf)


if __name__ == "__main__":
    run_motion_correction(path="/media/carmichael/LaCie SSD/JR/data/ca_imaging/28.06.21/slice3/1-40X-loc1.h5",
                          loc="data/ast")
