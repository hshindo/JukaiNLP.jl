using Base.Test
using JukaiNLP

files = ["iddict"]

for f in files
    path = joinpath(dirname(@__FILE__), "$f.jl")
    println("$path ...")
    include(path)
end
