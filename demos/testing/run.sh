#conda activate caiman

path='ca_imaging/ca_imaging/19.07.21/slice3/3-40X-loc1.h5.copy'

python downsample.py -i $path
python motion_correction.py -i $path
python cnmfe2.py -i $path
