import h5py as h5
import getopt, sys, os

def switch_channels(path):

    conversion_dict = {"ast":"neu", "neu":"ast"}
    loc_tmp = "temp/"

    with h5.File(path, "a") as file:

        for key1 in list(file.keys()):

            print(key1, type(file[key1]))

            try:
                container = []
                for key2 in file[key1].keys():

                    loc_old = f"{key1}/{key2}"
                    loc_temp = f"{loc_tmp}/{key2}"
                    loc_new = f"{key1}/{conversion_dict[key2]}"

                    file[loc_temp] = file[loc_old]
                    del file[loc_old]

                    print(f"Moving {loc_old} to {loc_temp}")
                    container.append((loc_temp, loc_new))

                for loc_temp, loc_new in container:

                    file[loc_new] = file[loc_temp]
                    del file[loc_temp]

                    print(f"Moving {loc_tmp} to {loc_new}")

                del file[loc_temp]

            except Exception:
                continue

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

    switch_channels(input_file)
