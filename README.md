# JukaiNLP: NLP Toolkit based on Deep Learning

[![Build Status](https://travis-ci.org/hshindo/JukaiNLP.jl.svg?branch=master)](https://travis-ci.org/hshindo/JukaiNLP.jl)
<!-- [![Build status](https://ci.appveyor.com/api/projects/status/github/hshindo/JukaiNLP.jl?branch=master)](https://ci.appveyor.com/project/hshindo/jukaiNLP-jl/branch/master) -->

<p align="center"><img src="https://github.com/hshindo/JukaiNLP.jl/blob/master/JukaiNLP.gif"></p>

[Try online demo!](http://jukainlp.hshindo.com/)

`JukaiNLP` is a natural language processing toolkit in [Julia](http://julialang.org/) based on a deep learning framework: [Merlin](https://github.com/hshindo/Merlin.jl).

## Installation
First, install [Merlin](https://github.com/hshindo/Merlin.jl).
Then,
```julia
julia> Pkg.clone("https://github.com/hshindo/TransitionParser.jl.git")
julia> Pkg.clone("https://github.com/hshindo/JukaiNLP.jl.git")
julia> Pkg.update()
```

## Tokenization

```julia
using JukaiNLP
using JukaiNLP.Tokenization
using JLD

# setup tokenizer
dirpath = Pkg.dir("JukaiNLP")
t = Tokenizer()

# training
trainpath = "$(dirpath)/corpus/mini-training-set.conll"
data = readconll(trainpath, [2,11])
train(t, 100, data)
#modelpath = "C:/Users/shindo/Desktop/tokenizer_20.jld"
#JLD.save(modelpath, "tokenizer", t)

# testing
#t = JLD.load(modelpath, "tokenizer")
str = "Pierre Vinken, 61 years old, will join the board.\nI have a pen.\n"
result = t(str)
join(map(r -> str[r], result), " ")
```

## Dependency Parsing
This is alpha version.
```julia
using JukaiNLP: Perceptron FeedForward, DepParser, Unlabeled, Labeled
using JukaiNLP: readconll, train!, decode, evaluate

# parser for unlabeled dependency tree
parser = DepParser(Unlabeled, "dict/en-word_nyt.dict")

# parser for labeled dependency tree
parser = DepParser(Labeled, "dict/en-word_nyt.dict")
sents = readconll(parser, "corpus/mini-training-set.conll")

# supports structured perceptron and feedforward
# neural network (Chen and Manning, 2014) models
initmodel!(parser, FeedForward)
initmodel!(parser, Perceptron)

n = div(length(sents), 10) * 8
trainsents, testsents = sents[1:n], sents[n+1:end]
train!(parser, trainsents)

actually train! has many keyword arguments

train!(parser, trainsents, nonlinear=tanh, hiddensizes=[200])

res = parser(testsents)
evaluate(parser, res)
```
