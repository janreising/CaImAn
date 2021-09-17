using PyCall

# Change that to whatever packages you need.
const PACKAGES = ["numpy", "warnings"]

# Import pip
try
    @pyimport pip
catch
    # If it is not found, install it
    get_pip = joinpath(dirname(@__FILE__), "get-pip.py")
    download("https://bootstrap.pypa.io/get-pip.py", get_pip)
    run(`$(PyCall.python) $get_pip --user`)
end

for pack in PACKAGES
    pip.main(["install", "--user", pack])
end
