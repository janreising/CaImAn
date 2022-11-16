import sys
import tifffile as tf
import h5py as h5
from pathlib import Path

path = Path(str(sys.argv[1]))
assert path.is_file(), "file doesnt exist ({})".format(path)

print("exporting tiff for {}".format(path))

with h5.File(path.as_posix(), "r") as file:
	
	if "dff" in file.keys():
		
		for key in file["dff"].keys():
			data = file[f"dff/{key}"][:]
			
			new_path = path.with_suffix(f".dff.{key}.tiff")
			
			if new_path.is_file():
				print(f"\tskipping {key}")
				continue

			tf.imwrite(new_path.as_posix(), data)
			print("\texported {} to {}".format(key, new_path))

print("done")
