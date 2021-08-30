import numpy as np
import h5py as h5
from tqdm import tqdm
import sys
import os

"""

Z, X, Y = 2, 3, 3

img = np.zeros((Z, X, Y))

# print(img, "\n")

for z in range(Z):
    for x in range(X):
        for y in range(Y):
            img[z, x, y] = int(f"{z+1}{x}{y}")

print(img, "\n")
print(img.shape, "\n")

img2 = img.copy()

T, dims = img2.shape[0], img2.shape[1:]
print(T, dims)

img2 = np.transpose(img2, list(range(1, len(dims) + 1)) + [0])
print(img2)
print(img2.shape, "\n")

img2 = np.reshape(img2, (np.prod(dims), T), order='F')
print(img2)
print(img2.shape, "\n")

# img2 = np.ascontiguousarray(img2, dtype=np.float32) + np.float32(0.0001)
# print(img2)
# print(img2.shape, "\n")

img2 = img2.T.reshape((T,) + dims, order='C')
print(img2)
print(img2.shape, "\n")

"""

out_path = "/media/carmichael/LaCie SSD/JR/data/ca_imaging/delete/test.h5"
loc = "mc/ast"

if os.path.isfile(out_path):
    with h5.File(out_path, "a") as file:
        # data = file.create_dataset(loc, dtype="i2", compression=None,
        #                            chunks=(100, 100, 100), shape=img.shape)
        # data[:] = img

        data = file[loc]
        print("Shape: ", data.shape)
        img2 = data[:]

        T, dims = img2.shape[0], img2.shape[1:]
        img2 = np.transpose(img2, list(range(1, len(dims) + 1)) + [0])
        img2 = np.reshape(img2, (np.prod(dims), T), order='F')

else:

    Z, X, Y = 250, 400, 400

    img = np.zeros((Z, X, Y))

    for z in tqdm(range(Z), desc="generation"):
        for x in range(X):
            for y in range(Y):
                img[z, x, y] = int(f"{z+1}{x}{y}")

    # print(img, "\n")
    print(img.shape, "\n")

    img2 = img.copy()
    T, dims = img2.shape[0], img2.shape[1:]

    img2 = np.transpose(img2, list(range(1, len(dims) + 1)) + [0])
    img2 = np.reshape(img2, (np.prod(dims), T), order='F')
    # print(img2)
    print(img2.shape, "\n")

    with h5.File(out_path, "a") as file:
        data = file.create_dataset(loc, dtype="i2", compression=None,
                                   chunks=(100, 100, 100), shape=img.shape)
        data[:] = img

# if os.path.isfile(out_path):
#     os.remove(out_path)

# file = "/media/carmichael/LaCie SSD/JR/data/ca_imaging/delete/test.h5"
# loc = "mc/ast"
file = out_path

with h5.File(file, "r") as file:

    data = file[loc]
    Z, X, Y = data.shape
    cz, cx, cy = data.chunks

    out = np.memmap(out_path+".mmap", dtype=np.float32, mode="w+", shape=(X*Y, Z))

    # chunk = np.zeros((cz, cx, cy))
    # T = np.zeros((cx*cy, cz))
    print(list(range(0, Z, cz)))
    for z0 in tqdm(range(0, Z, cz), desc="saving"):
        for x0 in range(0, X, cx):
            for y0 in range(0, Y, cy):

                z1 = min(Z, z0+cz)
                x1 = min(X, x0+cx)
                y1 = min(Y, y0+cy)

                # chunk[0:z1-z0, 0:x1-x0, 0:y1-y0] = data[z0:z1, x0:x1, y0:y1]
                chunk = data[z0:z1, x0:x1, y0:y1]

                chz, chx, chy = chunk.shape
                #
                # print(chunk)

                for a0 in range(chz):
                    for c0 in range(chy): # TODO change to cy

                        # print(f"{a0}, {c0}, {chunk.shape}")
                        col_section = chunk[a0, :, c0]

                        ind0 = int(x0/cx*cx + y0*X + c0*X)
                        ind1 = ind0+chx

                        indx0 = int(z0/cz*cz + a0)
                        # print(f"({x0} {y0} {c0}, {z0}) ({ind0} {ind1} {indx0}) , {col_section}")
                        out[ind0:ind1, indx0] = col_section + np.float32(0.0001)

                        sys.stdout.flush()
                        # print(out)

    # print(out)

    assert np.allclose(out, img2), print(img2, "\n", out)

    print("Finished without errors!")




