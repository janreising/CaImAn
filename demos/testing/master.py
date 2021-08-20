import cnmfe2 as cnmfe
from motion_correction import CMotionCorrect
from calpack import Converter
import getopt, sys, os
import h5py as h5
import caiman as cm
import traceback

if __name__ == "__main__":

    ###########
    # GET INPUT
    input_ = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:l:", ["input=", "local="])
    except getopt.GetoptError:
        print("master.py -i <input> -l <local>")
        sys.exit(2)

    on_server = True
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input_ = arg

        if opt in ("-l", "--local"):
            on_server = False

    ################
    # Pre-Processing

    # check if folder
    del_folder = True
    if os.path.isdir(input_):
        print(f"*MASTER* converting folder to zip")

        loader = Converter()
        input_ = loader.convert_folder(input_, del_folder=del_folder)

    # check if zip
    resize_factor = 0.5
    if os.path.isfile(input_) and input_.endswith(".zip"):
        print(f"*MASTER* converting zip to h5")
        loader = Converter()
        output_ = f"{input_}.h5"
        loader.zip_to_h5(input_, output_, resize_factor=resize_factor)
        input_ = output_

    ############
    # Processing

    assert input_.endswith(".h5"), f"at this stage we should be working with an .h5 file. However: {input_}"
    keys = []
    with h5.File(input_, "r") as file:
        for key in list(file.keys()):
            keys.append(key)

            try:
                for key2 in file[key].keys():
                    keys.append(f"{key}/{key2}")
            except Exception:
                continue

    if not on_server:
        num_proc = 6
    else:
        num_proc = None

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=num_proc,  single_thread=False)

    try:
        # check if mc exists
        missing_mcs = [key for key in keys if
                       (key.startswith("data/") and key.replace("data/", "mc/") not in keys)]

        delete_temp_files = True
        frames_per_file = 500
        ram_size_multiplier = None
        if len(missing_mcs) > 0:
            print(f"*MASTER* motion correction")
            mc = CMotionCorrect(path=input_, verbose=3, delete_temp_files=delete_temp_files, on_server=on_server,
                                dview=dview)
            mc.run_motion_correction(ram_size_multiplier=ram_size_multiplier, frames_per_file=frames_per_file)

        # check if cnmfe exists
        missing_cnmfes = [key for key in keys if
                   (key.startswith("data/") and key.replace("data/", "cnmfe/") not in keys)]

        if not on_server:
            steps = 200
        else:
            steps = 400
        if len(missing_cnmfes) > 0:
            for loc in missing_cnmfes:

                with h5.File(input_, "r") as file:
                    data = file[loc]
                    z, x, y = data.shape

                for z0 in range(0, z, steps):
                    z1 = min(z, z0+steps)
                    print(f"*MASTER* CNMFE processing indices {z0}:{z1} for loc {loc}")

                    cnmfe.main(path=input_, loc=loc, dview=dview, n_processes=n_processes, indices=slice(z0, z1))

        # check if dFF exists
        # TODO implement

    except Exception as err:
        print(err)

        traceback.print_exc()
    finally:
        dview.terminate()
        cm.stop_server()

        print("*MASTER* All done!")

