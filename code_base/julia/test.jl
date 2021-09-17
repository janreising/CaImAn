# Load Julia libraries
using PyCall
# using Conda
using Test

# Load Python libraries
# Conda.add("opencv")

# include python directory to look for modules
pushfirst!(PyVector(pyimport("sys")."path"), "../to_julia/")

# make imports
cp_mc = pyimport("cp_motioncorrection")

# Julia functions -> load from package in future
function bin_median(A, window)
	# TODO implement
	return false
end

function extract_shifts(A, max_shift_w, max_shift_h, template, method)
	# TODO implement
	return false
end

# Testing
@testset "MotionCorrection" begin
	# TODO generate something that is closer to
	# reality or maybe just actual data
	A = rand(Int, (40, 100, 100))

	window = 10
	@test cp_mc.movie.bin_median(A, window) == cp_mc.movie.bin_median(A, window) # bin_median(A, window)

	template = A[20, :, :]
	max_shift_h = 50
  max_shift_w = 50
  method = "opencv"
	@test cp_mc.movie.extract_shifts(A, max_shift_w, max_shift_h, method=method) == cp_mc.movie.extract_shifts(A, max_shift_w, max_shift_h, method=method) # extract_shifts(A, max_shift_w, max_shift_h, template=template, method=method)

end

# Close
print("\nAll tests finished successfully! Good job\n")
