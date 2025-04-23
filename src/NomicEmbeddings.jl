module NomicEmbeddings

using CUDA
using cuDNN
using DataDeps
using ONNXRunTime
using JSON
using Unicode
using Unicode: normalize
using DoubleArrayTries
using WordTokenizers
using WordTokenizers: TokenBuffer, isdone, flush!, character, spaces, atoms

include("wordpiece.jl")
include("Nomic.jl")

function __init__()
    include("src/Utils.jl")
end


end # module NomicEmbeddings
