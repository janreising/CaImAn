#conda activate caiman

path="/media/carmichael/1TB/delete/2-40X-loc1.h5"

python downsample.py -i $path
python motion_correction.py -i $path
python cnmfe2.py -i $path