import numpy as np
import h5py as h5
import time
import tifffile as tf
import sys
from tqdm import tqdm
import os

path = "/media/carmichael/LaCie SSD/JR/data/ca_imaging/28.06.21/slice4/"

fnames = path + "2-40X-loc1.h5"  # file path to movie file (will download if not present)

# %%%%%%%%%%%

# def split_h5_file(fnames, out, split_size = 1000, loc="data/ast/"):
#     t0 = time.time()
#     data = h5.File(name=fnames)
#
#     ast = data[loc]
#     Z, X, Y = ast.shape
#
#     c = 0
#     names = []
#     for start in tqdm(range(0, Z, split_size), position=0, leave=True):
#
#         stop = min(start+split_size, Z)
#
#         name = f'{path}{c}-{out}_{c}.h5'
#         if not os.path.isfile(name):
#             chunk = ast[start:stop, :, :]
#
#             temp = h5.File(name, "w")
#             chunk_drive = temp.create_dataset("/data/ast", shape=chunk.shape, dtype=ast.dtype)
#             chunk_drive[:, :, :] = chunk
#             temp.create_dataset("/proc/dummy", shape=(1, 1, 1), dtype=ast.dtype)
#
#         c += 1
#         names.append(name)
#
#     t1 = time.time()
#     print("Runtime: {:.2f}".format((t1-t0)/60/1000))
#
#     return names

# split_names = split_h5_file(fnames, out="2-40X-loc1")

# %%%%%%%%%%%

def save_memmap_to_h5(fnames, out, shape): # TODO dynamic shape

    output = h5.File(out, "w")
    Z, X, Y = shape
    data = output.create_dataset("mc/ast", shape=(len(fnames)*Z, X, Y), dtype="i2",
                                 compression="gzip", chunks=(100, 100, 100), shuffle=True)

    c = 0
    # mm = np.zeros(shape)
    for f in tqdm(fnames):

        mm = np.memmap(f, shape=shape, dtype=np.float32)
        data[c*Z:(c+1)*Z, :, :] = mm

        c += 1

# N = 10
# core = "2-40X-loc1__rig__d1_1200_d2_1200_d3_1_order_F_frames_1000_"
# fnames = [f"{path}{c}-{core}.mmap" for c in range(N)]
#
# shape = (int(core.split("_")[-2]), 1200, 1200)
# out = path+"2-40X-loc1_out.h5"
# save_memmap_to_h5(fnames, out, shape)

# %%%%%%%%%%%%


def save_split_tiff(file, loc, out, skip=10, downsize=1, subindices=None):

    from skimage.transform import resize

    arr = h5.File(file, "r")[loc]
    print(arr.shape)

    Z, X, Y = arr.shape

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


# name = "2-40X-loc1_out"
# file = path+name + ".h5"
# arr_ = "mc/ast"
#
# skip = 100
# downsize = 0.25
# subindices = (0, 10000)
#
# out = path+f"{name}_{skip}_{downsize}_{subindices[0]}-{subindices[1]}.tiff"
#
#
# save_split_tiff(file, arr_, out, skip=skip, downsize=downsize, subindices=subindices)

# %%%%%%%%%%%%

# ast = np.swapaxes(ast, 0, 2)
# ast = np.swapaxes(ast, 1, 2)
# print(ast.shape)

# out = "/media/carmichael/LaCie SSD/JR/data/ca_imaging/28.06.21/slice1/1-40X-loc1.tiff"
# tf.imsave(out, data=ast)

# %%%%%%%%%%%%

# file_name = path + "1-40X-loc_rig__d1_1200_d2_1200_d3_1_order_F_frames_200_.mmap"
# print(file_name)
#
# arr = np.memmap(file_name, shape=(200, 1200, 1200), dtype=np.float32)
# print(arr.shape)
#
# # arr = np.swapaxes(arr, 0, 2)
# arr = np.swapaxes(arr, 1, 2)
#
# out = path + "delete.tiff"
# tf.imsave(out, data=arr)

# %%%%%%%%%%%%

# file_name = path + "1-40X-loc1.h5"
# arr_ = "data/ast"
#
# if file_name.endswith(".h5"):
#     data = h5.File(file_name, "r")[arr_]
#     print(data.shape)
#
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(1, 1)
# ax.imshow(data[:, :, 1])
# fig.savefig(path+"delete.png")


# %%%%%%%%%%%%%%%%

# import caiman as cm
# from caiman.motion_correction import MotionCorrect
#
# mv = cm.load(path+"1-40X-loc1.h5", shape=(1200, 1200))
# mv.save(path+"delete.tiff")


# %%%%%%%%%%%%%%%%%%%% SWITCH AXES

# name = "1-40X-loc1"
# file_name = path + name + ".h5"
# arr_ = "data/neu"
#
# if file_name.endswith(".h5"):
#     data = h5.File(file_name, "r")[arr_]
#     print(data.shape)
#
# X, Y, Z = data.shape
# cx, cy, cz = data.chunks
#
# new_ = h5.File(path+name+"-zxy.h5", "a")
# arr = new_.create_dataset(arr_, dtype="i2", shape=(Z, X, Y), compression="gzip", chunks=(cx, cy, cz), shuffle=True)
# if "dummy" not in new_:
#     _ = new_.create_dataset("dummy", dtype="i2", shape=(1, 1, 1))
#
# for start in tqdm(range(0, Z, cz)):
#
#     stop = min(start+cz, Z)
#
#     transformed = np.array(data[:, :, start:stop])
#     transformed = np.swapaxes(transformed, 0, 2)
#     transformed = np.swapaxes(transformed, 1, 2)
#
#     arr[start:stop, :, :] = transformed
#
# print(arr.shape)
#
# print("Done")


## delte
# base = "/media/carmichael/LaCie SSD/JR/data/ca_imaging/28.06.21/slice1/temp/"
# paths = [
#     base+"0-2-40X-loc1_0.h5",
#     base+"1-2-40X-loc1_1.h5",
#     base+"2-2-40X-loc1_2.h5",
#     base+"3-2-40X-loc1_3.h5",
#     base+"4-2-40X-loc1_4.h5",
# ]

