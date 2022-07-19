## install dependencies
# !pip install tifffile
# !pip install nibabel

## imports
from generic import JsonSaver, ClassLoader
import os
import glob
import time
import tempfile

class Inference:

    def __init__(self, input_file, model, loc="mc/ast",
                 frames = [0, -1], batch_size=1, pre_post_frame=5,
                 ):

        # quality control
        assert os.path.isfile(input_file), "input doesn't exist: "+ input_file
        assert os.path.isdir(model) or os.path.isfile(model), "model doesn't exist: "+ model

        # ##############
        # # INFERENCE ##
        # ##############

        print("**master** Setting up inference ...")

        generator_param = {}
        inferrence_param = {}

        # We are reusing the data generator for training here.
        generator_param["type"] = "generator"
        generator_param["name"] = "OphysGenerator"
        generator_param["pre_post_frame"] = pre_post_frame
        generator_param["pre_post_omission"] = pre_post_frame #"pre_post_omission"
        generator_param["steps_per_epoch"] = 1
        generator_param["loc"] = loc

        generator_param["train_path"] = input_file
        # TODO apparently I need this, which is very weird. WHY?

        generator_param["batch_size"] = batch_size
        generator_param["start_frame"] = frames[0]
        generator_param["end_frame"] = frames[1] # -1 to go until the end.
        generator_param["randomize"] = 0 # important to keep the order

        inferrence_param["type"] = "inferrence"
        inferrence_param["name"] = "core_inferrence"
        inferrence_param["loc"] = "inf/"+loc.split("/")[-1]
        inferrence_param["output_datatype"] = "i2"


        # Get last model
        if os.path.isdir(model):
            models = list(filter(os.path.isfile, glob.glob(model + "/*.h5")))
            models.sort(key=lambda x: os.path.getmtime(x))
            inferrence_param["model_path"] = models[0]
        else:
            inferrence_param["model_path"] = model

        inferrence_param["output_file"] = input_file

        print("**master** Output: ", inferrence_param["output_file"])

        inf_dir = tempfile.TemporaryDirectory()
        self.inf_dir = inf_dir

        path_generator = os.path.join(inf_dir.name, "generator.json")
        json_obj = JsonSaver(generator_param)
        json_obj.save_json(path_generator)

        path_infer = os.path.join(inf_dir.name, "inferrence.json")
        json_obj = JsonSaver(inferrence_param)
        json_obj.save_json(path_infer)

        generator_obj = ClassLoader(path_generator)
        data_generator = generator_obj.find_and_build()(path_generator)

        inferrence_obj = ClassLoader(path_infer)
        inferrence_class = inferrence_obj.find_and_build()(path_infer,
                                                           data_generator)
        self.inference_class = inferrence_class

        print("**master** runing inference ...")
        inferrence_class.run()

        print("**master** Inference finished")
        inf_dir.cleanup()
