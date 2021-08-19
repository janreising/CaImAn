import getopt
import h5py as h5
import os, sys

def main(path):

    file = h5.File(path, "a")
    for key in list(file.keys()):

        if key not in ["data", "dummy", "meta"]:
            del file[key]

    print(file.keys())


if __name__ == "__main__":

    input_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:", ["ifolder="])
    except getopt.GetoptError:
        print("calpack.py -i <input_file>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input_file"):
            input_file = arg

    assert os.path.isfile(input_file), "input_file is not a file: {}".format(input_file)

    print("InputFile: ", input_file)

    main(input_file)
