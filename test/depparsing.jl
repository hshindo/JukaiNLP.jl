workspace()
using JukaiNLP
using JukaiNLP.DepParsing
using HDF5

path = joinpath(Pkg.dir("JukaiNLP"), ".corpus/nyt100.h5")
forms = h5read(path, "str")
formdict = IdDict(forms)

conv(form, cat, head) = push!(formdict,form), push!(catdict,cat), parse(Int,head)

trainpath = joinpath(Pkg.dir("JukaiNLP"), ".corpus/wsj_02-21.conll")
testpath = joinpath(Pkg.dir("JukaiNLP"), ".corpus/wsj_23.conll")
traindata = CoNLL.read(trainpath, 2, 5, 7)
testdata = CoNLL.read(testpath, 2, 5, 7)

#=
# parser for unlabeled dependency tree
path = Pkg.dir("JukaiNLP")
parser = DepParser(Unlabeled, "$(path)/dict/en-word_nyt.dict")

# parser for labeled dependency tree
parser = DepParser(Labeled, "$(path)/dict/en-word_nyt.dict")
sents = readconll(parser, "$(path)/corpus/mini-training-set.conll")

# supports structured perceptron and feedforward
# neural network (Chen and Manning, 2014) models

n = div(length(sents), 10) * 8
trainsents, testsents = sents[1:n], sents[n+1:end]

#train2!(FeedForward, parser, trainsents, iter=20)
#modelpath = "C:/Users/hshindo/Desktop/depparser.jld"
#JLD.save(modelpath, "depparser", parser.model)

train!(FeedForward, parser, trainsents, iter=20)

#train!(parser, trainsents, nonlinear=tanh, hiddensizes=[200])

res = parser(testsents)
evaluate(parser, res)
=#
