# Load libraries
using PyCall
using Test

# include python directory to look for modules
pushfirst!(PyVector(pyimport("sys")."path"), "../to_julia/")
pushfirst!(PyVector(pyimport("sys")."path"), "")

# make imports
mf = pyimport("myfunc")
cp_mc = pyimport("cp_motioncorrection")

# Do something
function jl_call_me(num)
	return num+num
end

n = 3
@test jl_call_me(n) == mf.standalone_call_me(n)

# Close
print("\nDone\n")
