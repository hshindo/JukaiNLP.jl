module Tagging

importall ..JukaiNLP
using Merlin
using HDF5

export train, Tagger

include("token.jl")
include("model.jl")
include("tagger.jl")
include("train.jl")

end
