workspace()
using JukaiNLP
using JukaiNLP.Tagging
using Merlin

train(10, traindata, testdata)

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
