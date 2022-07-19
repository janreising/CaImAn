import h5py as h5
import numpy as np
import sys, os, getopt, time
import caiman as cm
from tqdm import tqdm

def calculate_dFF(path, loc,
                  method="only_baseline", secsWindow = 5, quantilMin = 8):

    print(f"Calculating dFF for loc:{loc}")

    with h5.File(path, "r") as file:
        data = file[f"{loc}"]
        Z, X, Y = data.shape
        cz, cx, cy = data.chunks

    for x0  in tqdm(range(0, X, cx)):
        for y0 in range(0, Y, cy):

            x1 = min(x0+cx, X)
            y1 = min(y0+cy, Y)

            with h5.File(path, "r") as file:
                rec = cm.movie(file[f"{loc}"][:, x0:x1, y0:y1])
            rec = (rec + abs(np.min(rec)) + 1)

            dff, _ = rec.computeDFF(secsWindow=secsWindow, method=method, quantilMin=quantilMin)

            with h5.File(path, "a") as file:
                new_loc = loc.replace(loc.split("/")[0], "dff")
                if new_loc not in file:

                    chunks = (100, 100, 100) if Z > cz else None
                    data = file.create_dataset(f"{new_loc}", shape=(Z, X, Y), dtype="float32", chunks=chunks,
                                        compression="gzip", shuffle=True)
                else:
                    data = file[f"{new_loc}"]

                data[:, x0:x1, y0:y1] = dff[:, :, :]

if __name__ == "__main__":

    input_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:l:", ["ifolder=", "local="])
    except getopt.GetoptError:
        print("calpack.py -i <input_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input_file"):
            input_file = arg

    assert os.path.isfile(input_file), "input_file is not a file: {}".format(input_file)

    t0 = time.time()
    calculate_dFF(input_file, "cnmfe/ast")
    calculate_dFF(input_file, "cnmfe/neu")
    t1 = (time.time() - t0) / 60
    print("dFF finished in {:.2f}".format(t1))
