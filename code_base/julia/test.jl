using PyCall

push!(pyimport("sys")["path"], pwd());
# test = pyimport("test")

#print("Import done\n")

#@show test

#x = test.TestClass.call_me_julia(3)
#print(x)

@pyimport test.TestClass as testClass

print("\nDone\n")
