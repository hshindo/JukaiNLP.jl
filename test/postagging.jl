workspace()
using JukaiNLP
using JukaiNLP.Tagging
using Merlin
using HDF5

#=
path = "C:/Users/hshindo/Dropbox/tagging3"
vecs = open(readlines,"$(path)/nyt100.lst")
vecs = map(vecs) do v
    s = split(chomp(v))
    map(x -> parse(Float32,x), s)
end
vec = hcat(vecs...)
str = open(readlines,"$(path)/words.lst")
str = map(chomp, str)
h5open("enwords.h5", "w") do file
    write(file, "s", str)
    write(file, "v", vec)
end
throw("finish")
=#

path = joinpath(Pkg.dir("JukaiNLP"), ".corpus/postagging")
t = Tagger("$(path)/enwords.h5")

traindata = CoNLL.read("$(path)/wsj_00-18.conll", 2, 5)
testdata = CoNLL.read("$(path)/wsj_22-24.conll", 2, 5)

train(t, 10, traindata, testdata)
throw("finish")

modelpath = "C:/Users/hshindo/Desktop/postagger.h5"
h5save(modelpath, Dict("wordfun"=>t.model.wordfun,"charfun"=>t.model.charfun,"sentfun"=>t.model.sentfun))
t.model = nothing
path = "C:/Users/hshindo/Desktop/postagger.jld"
JLD.save(path, "postagger", t)
throw("finish")

modelpath = "C:/Users/hshindo/Desktop/postagger.h5"
m = h5load(modelpath)
model = Tagging.POSModel(m["wordfun"],m["charfun"],m["sentfun"])
path = "C:/Users/hshindo/Desktop/postagger.jld"
t = JLD.load(path, "postagger")
t.model = model
t(["I","have","a","pen","."]) |> println
