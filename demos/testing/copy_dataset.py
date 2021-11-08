import getopt, sys, os
import h5py as h5


if __name__ == "__main__":

    input_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:l:", ["ifolder=", "loc="])
    except getopt.GetoptError:
        print("calpack.py -i <input_file>")
        sys.exit(2)

    loc = None
    for opt, arg in opts:
        if opt in ("-i", "--input_file"):
            input_file = arg

        if opt in ("-l", "--loc"):
            loc = arg


    assert os.path.isfile(input_file), "input_file is not a file: {}".format(input_file)
    assert loc is not None, "Please provide a location '--loc' or '-l'"

    print("InputFile: ", input_file)
    output_file = input_file+loc.replace("/", ".")+".copy"
    print("OutputFile: ", output_file)

    assert not os.path.isfile(output_file), "output file already exists"

    with h5.File(input_file, "r") as file:

        with h5.File(input_file+".copy", "a") as out:
            file.copy(loc, out, loc)
