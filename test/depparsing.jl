workspace()
using JLD
using JukaiNLP
using JukaiNLP.DepParsing
using JukaiNLP.DepParsing: Perceptron, FeedForward, DepParser, Unlabeled, Labeled
using JukaiNLP.DepParsing: readconll, train!, train2!, decode, evaluate

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
