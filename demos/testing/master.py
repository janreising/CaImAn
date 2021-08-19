import cnmfe2 as cnmfe
import motion_correction as mc
from calpack import Converter
import getopt, sys, os

if __name__ == "__main__":

    # GET INPUT
    input_ = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:", ["input="])
    except getopt.GetoptError:
        print("master.py -i <input>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input_ = arg

    # check if folder
    del_folder = True
    if os.path.isdir(input_):
        print(f"*MASTER* converting folder to zip")

        loader = Converter()
        input_ = loader.convert_folder(input_, del_folder=del_folder)

    # check if zip
    resize_factor = 0.5
    if os.path.isfile(input_) and input_.endswith(".zip"):
        loader = Converter()
        output_ = f"{input_}.h5"
        loader.zip_to_h5(input_, output_, resize_factor=resize_factor)

    # check if mc exists


    # check if cnmfe exists

    # check if dFF exists
