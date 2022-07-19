import dff
import cnmfe2 as cnmfe
from motion_correction import CMotionCorrect
from calpack import Converter
from overview import main as overview
import getopt, sys, os
import h5py as h5
import caiman as cm
import traceback
import time
from just_inference import Inference

def get_keys(path):
    keys = []
    with h5.File(path, "r") as file:
        for key in list(file.keys()):
            keys.append(key)

            try:
                for key2 in file[key].keys():
                    keys.append(f"{key}/{key2}")
            except Exception:
                continue
    return keys


if __name__ == "__main__":

    ###########
    # GET INPUT
    input_ = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:l:r:", ["input=", "local=", "resize="])
    except getopt.GetoptError:
        print("master_inf.py -i <input> -l <local> -r <resize>")
        sys.exit(2)

    on_server = True
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input_ = arg

        if opt in ("-l", "--local"):
            on_server = False

    ############
    # Processing

    assert input_.endswith(".h5"), f"at this stage we should be working with an .h5 file. However: {input_}"

    if not on_server:
        num_proc = 6
    else:
        num_proc = None

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=num_proc,  single_thread=False)

    try:
        ####################
        # Motion Correction
        keys = get_keys(input_)
        print(f"pre mc keys: {keys}")
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
        else:
            print(f"*MASTER* motion correction found!")

        ############
        # Inference

        keys = get_keys(input_)
        # print(f"pre cnmfe keys: {keys}")
        missing_inf = [key for key in keys if
                          (key.startswith("mc/") and key.replace("mc/", "inf/") not in keys)]

        if len(missing_inf) > 0:
            for loc in missing_inf:
                INF = Inference(input_, "~/Private/starsmooth/res/ast_full/", loc=loc)
                INF.run()
        else:
            print("*MASTER* INF found!")

        #######
        # CNMFE
        keys = get_keys(input_)
        # print(f"pre cnmfe keys: {keys}")
        missing_cnmfes = [key for key in keys if
                          (key.startswith("inf/") and key.replace("inf/", "cnmfe/") not in keys)]
        # print(f"keys: {missing_cnmfes}")

        if not on_server:
            steps = 200
        else:
            steps = 400
        if len(missing_cnmfes) > 0:
            t0 = time.time()
            for loc in missing_cnmfes:

                with h5.File(input_, "r") as file:
                    data = file[loc]
                    z, x, y = data.shape

                for z0 in range(0, z, steps):
                    z1 = min(z, z0+steps)
                    print(f"*MASTER* CNMFE processing indices {z0}:{z1} for loc {loc}")

                    cnmfe.main(path=input_, loc=loc, dview=dview, n_processes=n_processes, indices=slice(z0, z1))

            t1 = time.time() - t0
            print("*MASTER* CNMFE finished in {:.2f} min".format(t1/60))
        else:
            print(f"*MASTER* CNMFE found!")

        #####
        # dFF
        keys = get_keys(input_)
        missing_dFF = [key for key in keys if
                          (key.startswith("cnmfe/") and key.replace("cnmfe/", "dff/") not in keys)]

        if len(missing_dFF) > 0:
            t0 = time.time()
            for loc in missing_dFF:
                #method='only_baseline','delta_f_over_f','delta_f_over_sqrt_f'
                dff.calculate_dFF(input_, loc, method="delta_f_over_sqrt_f")

            t1 = time.time() - t0
            print("*MASTER* dFF finished in {:.2f} min".format(t1/60))

        else:
            print(f"*MASTER* dFF found!")

        ########
        # traces
        keys = get_keys(input_)
        missing_trace = [key for key in keys if
                       (key.startswith("dff/") and key.replace("dff/", "proc/trace/") not in keys)]

        if len(missing_trace) > 0:

            converter = Converter()

            t0 = time.time()
            for loc in missing_trace:
                converter.save_trace(input_, channel=loc)

            t1 = time.time() - t0
            print("*MASTER* trace finished in {:.2f} min".format(t1 / 60))
        else:
            print(f"*MASTER* traces found!")

    except Exception as err:
        print(err)

        traceback.print_exc()
    finally:
        dview.terminate()
        cm.stop_server()

        print(f"*MASTER* printing structure of : {input_}")
        overview(input_)

        print("*MASTER* All done!")

