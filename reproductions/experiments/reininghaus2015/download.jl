include("common.jl")
using Downloads
mkpath(joinpath(ROOT,"data"))
file=joinpath(ROOT,"data","Uli_data.csv")
if !isfile(file)
    Downloads.download(SOURCE_URL,file)
end
bytes2hex(sha256(read(file)))==SOURCE_SHA256 || error("unexpected download hash")
println("Verified source CSV: ",SOURCE_SHA256)
