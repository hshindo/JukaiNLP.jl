workspace()
using HDF5

function getdata()
    path = joinpath(Pkg.dir("JukaiNLP"), ".corpus")
    w = h5read("$(path)/nyt100.h5", "vec")
    ws = Vector{Float32}[]
    n = size(w,1)
    for i = 1:size(w,2)
        push!(ws, w[(i-1)*n+1:i*n])
    end
    ws
end

function bench()
    ws = getdata()
    for i = 1:1
        x = ws[i]
        map(w ws)
    end
end
