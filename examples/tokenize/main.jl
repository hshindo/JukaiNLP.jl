using JukaiNLP
using Merlin

seg = Segmenter()
trainpath = joinpath(dirname(@__FILE__),".data/webtreebank.train")
train(seg, trainpath, 10)
Merlin.save("seg_epoch10.h5", "seg"=>seg)

# load
modelpath = joinpath(dirname(@__FILE__),"seg_epoch10.h5")
seg = Merlin.load(modelpath, "seg")
